"""Batched greedy generation with timing / memory instrumentation."""
from __future__ import annotations

import time

import torch

from .modeling import stop_token_ids
from .utils import peak_memory_gb, reset_peak_memory, sync


@torch.no_grad()
def generate(model, tok, prompts: list[str], batch_size: int = 16, max_new_tokens: int = 32,
             use_cache: bool = True, sort_by_length: bool = True, progress: bool = False) -> tuple[list[str], dict]:
    """Greedy-decode every prompt. Returns (outputs in input order, timing stats)."""
    device = next(model.parameters()).device
    stops = stop_token_ids(tok)
    lengths = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: -lengths[i]) if sort_by_length else list(range(len(prompts)))

    outputs: list[str | None] = [None] * len(prompts)
    batch_latencies, gen_tokens, prompt_tokens, padded_tokens = [], 0, 0, 0
    reset_peak_memory()
    sync(device)
    t_start = time.perf_counter()
    batches = [order[i: i + batch_size] for i in range(0, len(order), batch_size)]
    iterator = batches
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(batches, desc=f"generate bs={batch_size}", leave=False)
    for idx in iterator:
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        t0 = time.perf_counter()
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=use_cache,
            eos_token_id=stops,
            pad_token_id=tok.pad_token_id,
        )
        sync(device)
        batch_latencies.append(time.perf_counter() - t0)
        new = out[:, enc["input_ids"].shape[1]:]
        for row, i in zip(new.tolist(), idx):
            n = next((k for k, t in enumerate(row) if t in stops), len(row))
            gen_tokens += n
            outputs[i] = tok.decode(row[:n], skip_special_tokens=True)
        prompt_tokens += int(enc["attention_mask"].sum())
        padded_tokens += int(enc["attention_mask"].numel())
    total = time.perf_counter() - t_start
    stats = {
        "n": len(prompts),
        "batch_size": batch_size,
        "use_cache": use_cache,
        "sort_by_length": sort_by_length,
        "total_time_s": total,
        "examples_per_s": len(prompts) / total,
        "gen_tokens": gen_tokens,
        "gen_tokens_per_s": gen_tokens / total,
        "prompt_tokens": prompt_tokens,
        "padding_fraction": 1 - prompt_tokens / max(1, padded_tokens),
        "batch_latency_mean_s": sum(batch_latencies) / len(batch_latencies),
        "batch_latency_p50_s": sorted(batch_latencies)[len(batch_latencies) // 2],
        "batch_latency_p95_s": sorted(batch_latencies)[min(len(batch_latencies) - 1, int(0.95 * len(batch_latencies)))],
        "peak_mem_gb": peak_memory_gb(),
    }
    return outputs, stats
