"""Move the compact experiment results (everything `cegqa.cli report` needs, without per-question
predictions or adapters) between machines as one JSON document.

    python scripts/bundle_results.py pack   <output_dir> > results_bundle.json
    python scripts/bundle_results.py unpack results_bundle.json <output_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

METRIC_KEYS = ("n", "em", "f1", "em_ci", "f1_ci", "answer_in_prompt_rate")
SUMMARY_KEYS = ("run", "sources", "token_budget", "seed", "examples", "sample_stats", "tokens_seen", "target_tokens",
                "optimizer_steps", "train_time_s", "tokens_per_s", "gpu_hours", "est_cost_usd", "est_train_flops",
                "peak_mem_gb", "params", "initial_dev_loss", "final_dev_loss", "final_dev_loss_by_source", "gpu",
                "gpu_mem_gb")
RQ3_KEYS = ("key", "group", "batch_size", "use_cache", "sort_by_length", "dtype", "dtype_resolved", "model_mem_gb",
            "n", "oom", "em", "f1", "examples_per_s", "examples_per_s_std", "gen_tokens_per_s", "batch_latency_mean_s",
            "batch_latency_p50_s", "batch_latency_p95_s", "peak_mem_gb", "padding_fraction", "latency_per_example_ms")


def _round(x):
    if isinstance(x, float):
        return round(x, 5)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pack(root: Path) -> dict:
    files: dict[str, object] = {}
    rep = root / "data" / "data_report.json"
    if rep.exists():
        files["data/data_report.json"] = json.loads(rep.read_text())
    for run in sorted((root / "runs").glob("*")):
        m = run / "metrics.json"
        if m.exists():
            files[f"runs/{run.name}/metrics.json"] = {
                s: {k: v for k, v in d.items() if k in METRIC_KEYS} for s, d in json.loads(m.read_text()).items()}
        s = run / "train_summary.json"
        if s.exists():
            files[f"runs/{run.name}/train_summary.json"] = {
                k: v for k, v in json.loads(s.read_text()).items() if k in SUMMARY_KEYS}
        log = run / "train_log.jsonl"
        if log.exists():
            files[f"runs/{run.name}/train_log.jsonl"] = [r for r in _jsonl(log) if r["type"] == "dev"]
    stats = root / "analysis" / "rq2_stats.json"
    if stats.exists():
        files["analysis/rq2_stats.json"] = json.loads(stats.read_text())
    if (root / "rq3" / "results.jsonl").exists():
        files["rq3/results.jsonl"] = [{k: v for k, v in r.items() if k in RQ3_KEYS}
                                      for r in _jsonl(root / "rq3" / "results.jsonl")]
        files["rq3/setup.json"] = json.loads((root / "rq3" / "setup.json").read_text())
    return _round(files)


def unpack(bundle: dict, root: Path) -> None:
    for rel, content in bundle.items():
        path = (root / rel).resolve()
        if root.resolve() not in path.parents:
            raise ValueError(f"unsafe path in bundle: {rel}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".jsonl"):
            path.write_text("".join(json.dumps(r) + "\n" for r in content))
        else:
            path.write_text(json.dumps(content, indent=2))
    print(f"unpacked {len(bundle)} files into {root}")


if __name__ == "__main__":
    if sys.argv[1] == "pack":
        print(json.dumps(pack(Path(sys.argv[2])), separators=(",", ":")))
    else:
        unpack(json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3]))
