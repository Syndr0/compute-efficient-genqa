"""P2 - Shared Qwen-LoRA supervised fine-tuning pipeline with an exact training-token budget.

Budget accounting: a run consumes examples until the sum of their (prompt + answer) token
lengths reaches `token_budget`; the budget is split equally across the training sources, so
single- and multi-source runs see the same number of tokens (RQ2). Each run makes exactly one
pass over its sampled examples, so "tokens seen" == budget (+ at most one example).
"""
from __future__ import annotations

import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import load_split
from .modeling import count_params, load_lora_model, load_tokenizer
from .prompts import build_prompt
from .utils import (append_jsonl, ensure_dir, fmt_tokens, get_device, gpu_info, gpu_price, peak_memory_gb,
                    pick_dtypes, read_json, reset_peak_memory, set_seed, write_json)


def run_name(sources: list[str], token_budget: int, seed: int) -> str:
    return f"lora_{'+'.join(sources)}_{fmt_tokens(token_budget)}_s{seed}"


# --------------------------------------------------------------------------- features
def encode_example(tok, ex: dict, max_context_tokens: int, for_training: bool = True) -> dict:
    prompt, _ = build_prompt(tok, ex, max_context_tokens, for_training=for_training)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(ex["answers"][0], add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids, "id": ex["id"]}


def sample_to_budget(cfg: dict, tok, sources: list[str], token_budget: int, seed: int) -> tuple[list[dict], dict]:
    """Draw examples from each source's pool until its equal share of the token budget is met."""
    rng = random.Random(seed)
    per_source = token_budget / len(sources)
    feats, stats = [], {}
    for src in sources:
        pool = load_split(cfg, "train", src)
        rng.shuffle(pool)
        used = 0
        n = 0
        for ex in pool:
            if used >= per_source:
                break
            f = encode_example(tok, ex, cfg["model"]["max_context_tokens"])
            feats.append(f)
            used += len(f["input_ids"])
            n += 1
        if used < per_source:
            print(f"  warning: {src} pool exhausted at {used} tokens (< {per_source:.0f})")
        stats[src] = {"examples": n, "tokens": used}
    rng.shuffle(feats)
    return feats, stats


def collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pad a micro-batch for training."""
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


# --------------------------------------------------------------------------- loss
def answer_loss(model, batch: dict, reduction: str = "mean") -> torch.Tensor:
    """Causal-LM loss on answer tokens only.

    Projects only the supervised positions through the LM head, which avoids materialising
    a [batch, seq, 151k-vocab] logits tensor (the main memory cost on a 16 GB T4).
    """
    causal = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = causal.get_decoder()(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    labels = batch["labels"][:, 1:]
    sel = labels != -100
    h = hidden[:, :-1][sel]
    logits = causal.get_output_embeddings()(h).float()
    return F.cross_entropy(logits, labels[sel], reduction=reduction)


@torch.no_grad()
def dev_loss(model, feats: list[dict], pad_id: int, device, autocast_dtype, batch_size: int = 8) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(feats), batch_size):
        batch = {k: v.to(device) for k, v in collate(feats[i: i + batch_size], pad_id).items()}
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            loss = answer_loss(model, batch, reduction="sum")
        total += loss.item()
        count += int((batch["labels"][:, 1:] != -100).sum())
    model.train()
    return total / max(1, count)


# --------------------------------------------------------------------------- training
def train_lora(cfg: dict, sources: list[str], token_budget: int, seed: int, runs_dir: str | Path) -> Path:
    """Train one LoRA adapter; returns its run directory. Skips if already finished."""
    tcfg, mcfg = cfg["train"], cfg["model"]
    run_dir = ensure_dir(Path(runs_dir) / run_name(sources, token_budget, seed))
    if (run_dir / "train_summary.json").exists() and (run_dir / "adapter").exists():
        trained_on = read_json(run_dir / "train_summary.json").get("gpu")
        if trained_on == gpu_info()["gpu"] or tcfg.get("reuse_across_gpus", False):
            print(f"[skip] {run_dir.name} already trained")
            return run_dir
        # timing/cost must come from one GPU type: retrain (and re-evaluate) on the current one
        print(f"[retrain] {run_dir.name} was trained on {trained_on}; retraining on {gpu_info()['gpu']}")
        shutil.rmtree(run_dir)
        run_dir = ensure_dir(run_dir)
    log_path = run_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    set_seed(seed)
    device = get_device()
    _, autocast_dtype = pick_dtypes(device)
    tok = load_tokenizer(mcfg["name"], padding_side="right")
    pad_id = tok.pad_token_id

    feats, sample_stats = sample_to_budget(cfg, tok, sources, token_budget, seed)
    dev_feats = {
        src: [encode_example(tok, ex, mcfg["max_context_tokens"])
              for ex in load_split(cfg, "dev", src, limit=tcfg["dev_examples_per_source"])]
        for src in cfg["data"]["train_sources"]
    }

    model = load_lora_model(mcfg["name"], cfg["lora"], device, tcfg.get("gradient_checkpointing", False))
    model.train()
    params = count_params(model)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    mbs, accum = tcfg["micro_batch_size"], tcfg["grad_accum"]
    total_steps = math.ceil(len(feats) / (mbs * accum))
    warmup = max(1, int(tcfg["warmup_ratio"] * total_steps))
    from transformers import get_cosine_schedule_with_warmup

    sched = get_cosine_schedule_with_warmup(optim, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_dtype == torch.float16)
    eval_points = sorted({max(1, round(total_steps * k / tcfg["num_evals"])) for k in range(1, tcfg["num_evals"] + 1)})

    def log_dev(step: int, tokens: int, elapsed: float) -> dict:
        losses = {src: dev_loss(model, f, pad_id, device, autocast_dtype) for src, f in dev_feats.items()}
        row = {"type": "dev", "step": step, "tokens": tokens, "elapsed_s": elapsed,
               "dev_loss": losses, "dev_loss_mean": sum(losses.values()) / len(losses)}
        append_jsonl(row, log_path)
        print(f"  step {step}/{total_steps} tokens {tokens:,} dev_loss {row['dev_loss_mean']:.4f}")
        return row

    print(f"[train] {run_dir.name}: {len(feats)} examples, {total_steps} optimizer steps, "
          f"{params['trainable']:,} trainable params")
    reset_peak_memory()
    first = log_dev(0, 0, 0.0)
    tokens_seen, target_tokens, running = 0, 0, []
    train_time = 0.0
    step = 0
    t0 = time.perf_counter()
    for start in range(0, len(feats), mbs * accum):
        chunk = feats[start: start + mbs * accum]
        micro = [chunk[i: i + mbs] for i in range(0, len(chunk), mbs)]
        for mb in micro:
            batch = {k: v.to(device) for k, v in collate(mb, pad_id).items()}
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                loss = answer_loss(model, batch) / len(micro)
            scaler.scale(loss).backward()
            running.append(loss.item() * len(micro))
            tokens_seen += int(batch["attention_mask"].sum())
            target_tokens += int((batch["labels"] != -100).sum())
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        if step % tcfg["log_every_steps"] == 0 or step == total_steps:
            elapsed = time.perf_counter() - t0
            append_jsonl({"type": "train", "step": step, "tokens": tokens_seen, "loss": sum(running) / len(running),
                          "lr": sched.get_last_lr()[0], "elapsed_s": elapsed,
                          "tokens_per_s": tokens_seen / elapsed}, log_path)
            running = []
        if step in eval_points:
            train_time += time.perf_counter() - t0
            log_dev(step, tokens_seen, train_time)
            t0 = time.perf_counter()
    train_time += time.perf_counter() - t0
    final = read_last_dev(log_path)

    model.save_pretrained(str(run_dir / "adapter"))
    info = gpu_info()
    gpu_hours = train_time / 3600
    summary = {
        "run": run_dir.name,
        "sources": sources,
        "token_budget": token_budget,
        "seed": seed,
        "examples": len(feats),
        "sample_stats": sample_stats,
        "tokens_seen": tokens_seen,
        "target_tokens": target_tokens,
        "optimizer_steps": total_steps,
        "train_time_s": train_time,
        "tokens_per_s": tokens_seen / max(train_time, 1e-9),
        "gpu_hours": gpu_hours,
        "est_cost_usd": gpu_hours * gpu_price(info["gpu"]),
        # forward (2N) + activation-gradient backward (2N) per token; frozen base needs no weight grads
        "est_train_flops": 4 * params["total"] * tokens_seen,
        "peak_mem_gb": peak_memory_gb(),
        "params": params,
        "initial_dev_loss": first["dev_loss_mean"],
        "final_dev_loss": final["dev_loss_mean"],
        "final_dev_loss_by_source": final["dev_loss"],
        **info,
        "config": {"model": mcfg, "lora": cfg["lora"], "train": tcfg},
    }
    write_json(summary, run_dir / "train_summary.json")
    del model, optim
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_dir


def read_last_dev(log_path: Path) -> dict:
    from .utils import read_jsonl

    return [r for r in read_jsonl(log_path) if r["type"] == "dev"][-1]


def train_and_evaluate(cfg: dict, sources: list[str], token_budget: int, seed: int, runs_dir: str | Path) -> dict:
    """Train (or reuse) an adapter, then evaluate it on all eval sets."""
    from .evaluate import evaluate_system
    from .modeling import load_for_eval

    run_dir = train_lora(cfg, sources, token_budget, seed, runs_dir)

    def loader():
        return load_for_eval(cfg, run_dir / "adapter")

    metrics = evaluate_system(cfg, run_dir, loader=loader)
    return {"run_dir": str(run_dir), "summary": read_json(run_dir / "train_summary.json"), "metrics": metrics}
