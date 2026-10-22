"""Copy the lightweight, committable artifacts of an output directory into results/.

Keeps: data report, run metrics, predictions, training summaries/logs, RQ2 statistics, RQ3
benchmark rows, tiny-Transformer logs/summaries/predictions, figures and report.md.
Skips: processed data, LoRA adapters and model checkpoints.

    python scripts/export_results.py --src outputs --dst results
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

KEEP_SUFFIXES = {".json", ".jsonl", ".png", ".md"}
SKIP_DIRS = {"adapter", "data"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs")
    ap.add_argument("--dst", default="results")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    n = 0
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        if not f.is_file() or f.suffix not in KEEP_SUFFIXES or SKIP_DIRS & set(rel.parts[:-1]):
            continue
        if f.name == "tokenizer.json":
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / rel)
        n += 1
    data_report = src / "data" / "data_report.json"
    if data_report.exists():
        shutil.copy2(data_report, dst / "data_report.json")
        n += 1
    print(f"copied {n} files to {dst}")


if __name__ == "__main__":
    main()
