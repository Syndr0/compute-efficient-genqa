"""Command-line entry point.

    python -m cegqa.cli <stage> [<stage> ...] [--config configs/default.yaml ...] [--set key=value ...]

Stages: prepare, baselines, rq1, rq2, rq3, report, all
"""
from __future__ import annotations

import argparse
import os

# reduce CUDA memory fragmentation from variable-length batches (must be set before torch initialises CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .experiments import STAGES
from .utils import REPO_ROOT, load_config

ORDER = ["prepare", "baselines", "rq1", "rq2", "rq3", "report"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=ORDER + ["all"])
    ap.add_argument("--config", action="append", default=None,
                    help="YAML config(s), merged in order (default: configs/default.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    ap.add_argument("--force", action="store_true", help="re-run data preparation even if present")
    args = ap.parse_args(argv)

    cfg = load_config(args.config or [REPO_ROOT / "configs" / "default.yaml"], args.set)
    stages = ORDER if "all" in args.stages else args.stages
    for st in stages:
        print(f"\n===== stage: {st} =====")
        if st == "prepare":
            STAGES[st](cfg, force=args.force)
        else:
            STAGES[st](cfg)


if __name__ == "__main__":
    main()
