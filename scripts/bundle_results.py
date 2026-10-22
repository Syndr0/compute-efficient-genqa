"""Print the compact experiment results (metrics, training logs, statistics, benchmarks) as one
base64-encoded tar.gz, or unpack such a bundle.

    python scripts/bundle_results.py pack   <output_dir>              # prints the bundle
    python scripts/bundle_results.py unpack <bundle.b64> <output_dir>
"""
from __future__ import annotations

import base64
import io
import sys
import tarfile
from pathlib import Path

PATTERNS = [
    "data/data_report.json",
    "runs/*/metrics.json",
    "runs/*/train_summary.json",
    "runs/*/train_log.jsonl",
    "analysis/*.json",
    "rq3/*.json",
    "rq3/*.jsonl",
]


def pack(root: Path) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for pattern in PATTERNS:
            for f in sorted(root.glob(pattern)):
                tar.add(f, arcname=str(f.relative_to(root)))
    return base64.b64encode(buf.getvalue()).decode()


def unpack(b64_path: Path, root: Path) -> None:
    data = base64.b64decode(b64_path.read_text().strip())
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                raise ValueError(f"unsafe path in bundle: {m.name}")
        tar.extractall(root)
    print(f"unpacked into {root}")


if __name__ == "__main__":
    if sys.argv[1] == "pack":
        print(pack(Path(sys.argv[2])))
    else:
        unpack(Path(sys.argv[2]), Path(sys.argv[3]))
