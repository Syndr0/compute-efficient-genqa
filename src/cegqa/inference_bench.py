"""P4 - Inference efficiency: batch size vs throughput, latency, GPU memory (and quality).

Every configuration decodes the same fixed set of prompts with the same fine-tuned model.
Grid (see `rq3` in the config):
  * main sweep      batch_size in batch_sizes, KV cache on, length-sorted batches
  * sort ablation   same batch sizes with unsorted (random-length) batches -> padding cost
  * KV-cache off    a few batch sizes on a smaller subset (optional experiment)
  * precision       fp32 vs fp16/bf16 at a few batch sizes
"""
from __future__ import annotations

import gc
from pathlib import Path

import torch

from .data import load_split
from .generation import generate
from .metrics import aggregate, score
from .modeling import load_inference_model, load_tokenizer
from .prompts import build_prompt, extract_answer
from .utils import (append_jsonl, ensure_dir, get_device, gpu_info, read_json, read_jsonl, resolve_inference_dtype,
                    write_json)



def build_grid(rcfg: dict) -> list[dict]:
    grid = []
    for bs in rcfg["batch_sizes"]:
        grid.append({"group": "batch_sweep", "batch_size": bs, "use_cache": True, "sort_by_length": True, "dtype": "auto"})
    if rcfg.get("sort_ablation", True):
        for bs in rcfg["batch_sizes"]:
            if bs > 1:
                grid.append({"group": "unsorted", "batch_size": bs, "use_cache": True, "sort_by_length": False, "dtype": "auto"})
    for bs in rcfg.get("kv_cache_off_batch_sizes", []):
        grid.append({"group": "kv_cache_off", "batch_size": bs, "use_cache": False, "sort_by_length": True, "dtype": "auto"})
        grid.append({"group": "kv_cache_on_subset", "batch_size": bs, "use_cache": True, "sort_by_length": True, "dtype": "auto"})
    for dt in rcfg.get("extra_dtypes", []):
        for bs in rcfg.get("extra_dtype_batch_sizes", [1, 16]):
            grid.append({"group": f"dtype_{dt}", "batch_size": bs, "use_cache": True, "sort_by_length": True, "dtype": dt})
    return grid


def _key(c: dict) -> str:
    return f"{c['group']}|bs{c['batch_size']}|cache{int(c['use_cache'])}|sort{int(c['sort_by_length'])}|{c['dtype']}"


def resolve_adapter(cfg: dict, runs_dir: Path) -> Path | None:
    choice = cfg["rq3"].get("adapter_run", "auto")
    if choice in (None, "none", "base"):
        return None
    if choice == "auto":  # largest-budget multi-source adapter (seed 0), else any adapter
        cands = [c for c in runs_dir.glob("lora_*_s0/adapter") if (c.parent / "train_summary.json").exists()]
        if not cands:
            return None
        budget = lambda c: read_json(c.parent / "train_summary.json")["token_budget"]  # noqa: E731
        return max(cands, key=lambda c: ("+" in c.parent.name, budget(c)))
    path = runs_dir / choice / "adapter"
    return path if path.exists() else None


def run_inference_bench(cfg: dict, runs_dir: str | Path, out_dir: str | Path) -> list[dict]:
    rcfg, mcfg = cfg["rq3"], cfg["model"]
    runs_dir, out_dir = Path(runs_dir), ensure_dir(out_dir)
    res_path = out_dir / "results.jsonl"
    done = {r["key"] for r in read_jsonl(res_path)} if res_path.exists() else set()
    grid = [c for c in build_grid(rcfg) if _key(c) not in done]
    if not grid:
        print("[skip] RQ3 benchmark complete")
        return read_jsonl(res_path)

    device = get_device()
    adapter = resolve_adapter(cfg, runs_dir)
    tok = load_tokenizer(mcfg["name"])
    exs = load_split(cfg, "test", rcfg["source"], limit=rcfg["num_examples"])
    prompts = [build_prompt(tok, ex, mcfg["max_context_tokens"])[0] for ex in exs]
    write_json({"adapter": str(adapter) if adapter else None, "source": rcfg["source"], "n": len(exs), **gpu_info()},
               out_dir / "setup.json")
    print(f"[rq3] adapter={adapter}, {len(grid)} configurations to run")

    loaded: dict = {}
    for c in grid:
        dtype = resolve_inference_dtype(mcfg.get("inference_dtype", "auto") if c["dtype"] == "auto" else c["dtype"], device)
        if loaded.get("dtype") != dtype:
            loaded.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = load_inference_model(mcfg["name"], adapter, device, dtype)
            static_mem = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            loaded = {"dtype": dtype, "model": model, "static_mem_gb": static_mem}
        model = loaded["model"]
        n = rcfg.get("kv_cache_subset_examples", 64) if c["group"].startswith("kv_cache") else len(prompts)
        ps, refs = prompts[:n], exs[:n]
        row = {"key": _key(c), **c, "dtype_resolved": str(dtype).replace("torch.", ""),
               "model_mem_gb": loaded["static_mem_gb"]}
        try:
            generate(model, tok, ps[: c["batch_size"]], batch_size=c["batch_size"],  # warm-up
                     max_new_tokens=mcfg["max_new_tokens"], use_cache=c["use_cache"])
            reps = []
            for _ in range(rcfg["repeats"]):
                outs, stats = generate(model, tok, ps, batch_size=c["batch_size"], max_new_tokens=mcfg["max_new_tokens"],
                                       use_cache=c["use_cache"], sort_by_length=c["sort_by_length"])
                reps.append(stats)
            scored = [score(extract_answer(o), ex["answers"]) for o, ex in zip(outs, refs)]
            q = aggregate(scored)
            keys = ["examples_per_s", "gen_tokens_per_s", "batch_latency_mean_s", "batch_latency_p50_s",
                    "batch_latency_p95_s", "total_time_s", "peak_mem_gb", "padding_fraction"]
            for k in keys:
                vals = [r[k] for r in reps]
                row[k] = sum(vals) / len(vals)
                row[k + "_std"] = (sum((v - row[k]) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
            row.update({"n": len(ps), "em": q["em"], "f1": q["f1"], "oom": False,
                        "latency_per_example_ms": 1000 * row["batch_latency_mean_s"]})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row.update({"n": len(ps), "oom": True})
        append_jsonl(row, res_path)
        msg = "OOM" if row["oom"] else (f"{row['examples_per_s']:.1f} ex/s, {row['batch_latency_mean_s'] * 1000:.0f} ms/batch, "
                                         f"{row['peak_mem_gb']:.2f} GB, F1 {row['f1']:.1f}")
        print(f"  {row['key']}: {msg}")
    return read_jsonl(res_path)
