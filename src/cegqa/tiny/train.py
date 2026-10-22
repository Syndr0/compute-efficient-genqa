"""P5 - Train decoder-only Transformers from scratch on context-grounded QA.

For each model size we train one run up to `token_budget` tokens and measure validation
cross-entropy at log-spaced token counts (the CE-vs-tokens learning curve), then report
greedy-decoding EM/F1 on the SQuAD eval set for comparison with Qwen3-0.6B.
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data import load_split
from ..metrics import aggregate, score
from ..utils import (append_jsonl, ensure_dir, get_device, gpu_info, gpu_price, peak_memory_gb, pick_dtypes,
                     read_jsonl, reset_peak_memory, set_seed, write_json, write_jsonl)
from .model import TinyConfig, TinyGPT

SPECIALS = ["<pad>", "<eos>", "<ctx>", "<q>", "<a>"]
PAD, EOS, CTX, Q, A = range(len(SPECIALS))


# --------------------------------------------------------------------------- tokenizer
def get_tokenizer(texts: list[str], vocab_size: int, path: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if path.exists():
        return Tokenizer.from_file(str(path))
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tk.train_from_iterator(texts, trainer)
    ensure_dir(path.parent)
    tk.save(str(path))
    return tk


def encode_qa(tk, ex: dict, max_len: int, reserve_answer: int, for_training: bool) -> tuple[list[int], list[int]]:
    """Return (prompt_ids, answer_ids) with the context truncated to fit `max_len`."""
    q_ids = tk.encode(ex["question"]).ids
    a_ids = tk.encode(ex["answers"][0]).ids[: reserve_answer - 1] + [EOS] if for_training else []
    room = max_len - len(q_ids) - reserve_answer - 3
    enc = tk.encode(ex["context"])
    ctx_ids = enc.ids
    if len(ctx_ids) > room:
        start = 0
        if for_training:
            pos = ex["context"].lower().find(ex["answers"][0].lower())
            if pos >= 0:
                ans_tok = next((i for i, (s, e) in enumerate(enc.offsets) if e > pos), 0)
                if ans_tok >= room - 8:
                    start = max(0, min(ans_tok - room // 2, len(ctx_ids) - room))
        ctx_ids = ctx_ids[start: start + max(room, 0)]
    return [CTX] + ctx_ids + [Q] + q_ids + [A], a_ids


# --------------------------------------------------------------------------- batching
def make_batch(seqs: list[tuple[list[int], list[int]]], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L = max(len(p) + len(a) for p, a in seqs)
    x = torch.full((len(seqs), L), PAD, dtype=torch.long)
    ans_mask = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, (p, a) in enumerate(seqs):
        x[i, : len(p) + len(a)] = torch.tensor(p + a)
        ans_mask[i, len(p): len(p) + len(a)] = True
    return x.to(device), (x != PAD).to(device), ans_mask.to(device)


def lm_losses(model, x, mask, ans_mask, autocast_dtype, device):
    """Per-token CE for next-token prediction; returns (loss_all, loss_answer, n_all, n_answer)."""
    with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        logits = model(x[:, :-1])
    ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="none")
    ce = ce.view(x.size(0), -1)
    m_all, m_ans = mask[:, 1:].float(), ans_mask[:, 1:].float()
    return (ce * m_all).sum(), (ce * m_ans).sum(), m_all.sum(), m_ans.sum()


@torch.no_grad()
def evaluate_ce(model, data, batch_size, autocast_dtype, device) -> dict:
    model.eval()
    tot = torch.zeros(4, device=device)
    for i in range(0, len(data), batch_size):
        x, mask, am = make_batch(data[i: i + batch_size], device)
        tot += torch.stack(lm_losses(model, x, mask, am, autocast_dtype, device))
    model.train()
    return {"val_ce": (tot[0] / tot[2]).item(), "val_answer_ce": (tot[1] / tot[3]).item()}


@torch.no_grad()
def evaluate_generation(model, tk, exs: list[dict], max_len: int, max_new: int, device) -> tuple[dict, list[dict]]:
    model.eval()
    rows = []
    for ex in exs:
        prompt, _ = encode_qa(tk, ex, max_len, max_new, for_training=False)
        out = model.generate(torch.tensor([prompt], device=device), max_new, EOS)
        pred = tk.decode(out).strip()
        rows.append({"id": ex["id"], "question": ex["question"], "answers": ex["answers"], "prediction": pred,
                     **score(pred, ex["answers"])})
    model.train()
    return aggregate(rows), rows


# --------------------------------------------------------------------------- training
def train_tiny(cfg: dict, out_dir: str | Path) -> list[dict]:
    tcfg = cfg["tiny"]
    out_dir = ensure_dir(out_dir)
    device = get_device()
    _, autocast_dtype = pick_dtypes(device)
    set_seed(tcfg["seed"])

    train_exs = [e for s in tcfg["sources"] for e in load_split(cfg, "train", s, limit=tcfg.get("max_train_examples"))]
    val_exs = [e for s in tcfg["sources"] for e in load_split(cfg, "dev", s)]
    gen_exs = load_split(cfg, "test", tcfg["eval_source"], limit=tcfg["gen_eval_examples"])
    tk = get_tokenizer([e["context"] + "\n" + e["question"] + "\n" + e["answers"][0] for e in train_exs[:40000]],
                       tcfg["vocab_size"], out_dir / "tokenizer.json")
    L, R = tcfg["max_seq_len"], tcfg["answer_reserve_tokens"]
    train_data = [encode_qa(tk, e, L, R, True) for e in train_exs]
    val_data = [encode_qa(tk, e, L, R, True) for e in val_exs]
    tokens_per_epoch = sum(len(p) + len(a) for p, a in train_data)
    print(f"[tiny] {len(train_data)} train seqs, {tokens_per_epoch / 1e6:.1f}M tokens/epoch, vocab {tk.get_vocab_size()}")

    summaries = []
    for spec in tcfg["models"]:
        run_dir = ensure_dir(out_dir / spec["name"])
        if (run_dir / "summary.json").exists():
            from ..utils import read_json

            print(f"[skip] tiny/{spec['name']}")
            summaries.append(read_json(run_dir / "summary.json"))
            continue
        set_seed(tcfg["seed"])
        mcfg = TinyConfig(vocab_size=tk.get_vocab_size(), max_seq_len=L, d_model=spec["d_model"],
                          n_layers=spec["n_layers"], n_heads=spec["n_heads"], dropout=tcfg.get("dropout", 0.0))
        model = TinyGPT(mcfg).to(device)
        params = model.param_counts()
        budget = int(tcfg["token_budget"])
        bs = tcfg["batch_size"]
        est_steps = math.ceil(budget / (tokens_per_epoch / len(train_data) * bs))
        warmup = max(1, int(0.02 * est_steps))
        optim = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], betas=(0.9, 0.95), weight_decay=0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda s: min(1.0, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / est_steps)))))
        scaler = torch.amp.GradScaler("cuda", enabled=autocast_dtype == torch.float16)
        checkpoints = sorted({int(x) for x in np.geomspace(budget / tcfg["num_ce_points_ratio"], budget, tcfg["num_ce_points"])})
        log_path = run_dir / "log.jsonl"
        if log_path.exists():
            log_path.unlink()
        print(f"[tiny] {spec['name']}: {params['total'] / 1e6:.2f}M params "
              f"({params['non_embedding'] / 1e6:.2f}M non-embedding), budget {budget / 1e6:g}M tokens")

        rng = random.Random(tcfg["seed"])
        order: list[int] = []
        tokens, step, ck, train_time = 0, 0, 0, 0.0
        reset_peak_memory()
        append_jsonl({"tokens": 0, "step": 0, **evaluate_ce(model, val_data, 64, autocast_dtype, device)}, log_path)
        t0 = time.perf_counter()
        while tokens < budget:
            if len(order) < bs:
                perm = list(range(len(train_data)))
                rng.shuffle(perm)
                order += perm
            idx, order = order[:bs], order[bs:]
            x, mask, am = make_batch([train_data[i] for i in idx], device)
            l_all, l_ans, n_all, n_ans = lm_losses(model, x, mask, am, autocast_dtype, device)
            loss = (l_ans / n_ans) if tcfg["loss_on"] == "answer" else (l_all / n_all)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            sched.step()
            tokens += int(mask.sum())
            step += 1
            while ck < len(checkpoints) and tokens >= checkpoints[ck]:
                train_time += time.perf_counter() - t0
                row = {"tokens": tokens, "step": step, "train_loss": loss.item(), "elapsed_s": train_time,
                       **evaluate_ce(model, val_data, 64, autocast_dtype, device)}
                append_jsonl(row, log_path)
                print(f"  {tokens / 1e6:7.2f}M tok  val CE {row['val_ce']:.3f}  answer CE {row['val_answer_ce']:.3f}")
                ck += 1
                t0 = time.perf_counter()
        gen_metrics, preds = evaluate_generation(model, tk, gen_exs, L, tcfg["answer_reserve_tokens"], device)
        write_jsonl(preds, run_dir / f"predictions_{tcfg['eval_source']}.jsonl")
        last = read_jsonl(log_path)[-1]
        info = gpu_info()
        summary = {
            "name": spec["name"], **spec, "params": params, "tokens": tokens, "steps": step,
            "tokens_per_param": tokens / params["total"],
            "train_time_s": train_time, "tokens_per_s": tokens / max(train_time, 1e-9),
            "gpu_hours": train_time / 3600, "est_cost_usd": train_time / 3600 * gpu_price(info["gpu"]),
            "est_train_flops": 6 * params["non_embedding"] * tokens,
            "peak_mem_gb": peak_memory_gb(), "final_val_ce": last["val_ce"], "final_val_answer_ce": last["val_answer_ce"],
            "eval_source": tcfg["eval_source"], "em": gen_metrics["em"], "f1": gen_metrics["f1"], **info,
        }
        write_json(summary, run_dir / "summary.json")
        torch.save(model.state_dict(), run_dir / "model.pt")
        print(f"  -> EM {summary['em']:.1f}  F1 {summary['f1']:.1f}")
        summaries.append(summary)
        del model, optim
    write_json(summaries, out_dir / "summary.json")
    return summaries
