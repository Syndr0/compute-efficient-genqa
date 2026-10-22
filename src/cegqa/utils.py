"""Shared helpers: config loading, seeding, JSON I/O, timing and GPU accounting."""
from __future__ import annotations

import copy
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- config
def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(v: str) -> Any:
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def load_config(paths: Iterable[str | os.PathLike], overrides: Iterable[str] = ()) -> dict:
    """Load and deep-merge YAML configs in order, then apply `a.b.c=value` overrides."""
    cfg: dict = {}
    for p in paths:
        with open(p) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    for ov in overrides:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(val)
    out_dir = os.environ.get("CEGQA_OUTPUT_DIR")
    if out_dir:
        cfg["output_dir"] = out_dir
    cfg.setdefault("output_dir", "outputs")
    cfg.setdefault("data_dir", os.path.join(cfg["output_dir"], "data"))
    return cfg


# --------------------------------------------------------------------------- misc
def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(p: str | os.PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path: str | os.PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def write_jsonl(rows: Iterable[dict], path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(row: dict, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fmt_tokens(n: float) -> str:
    """1_000_000 -> '1M', 250_000 -> '250k'."""
    if n >= 1e6:
        s = f"{n / 1e6:g}M"
    elif n >= 1e3:
        s = f"{n / 1e3:g}k"
    else:
        s = str(int(n))
    return s


# --------------------------------------------------------------------------- hardware
def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    # Opt-in only: on Apple MPS, batched generation with left padding yields NaN logits for padded rows.
    if os.environ.get("CEGQA_USE_MPS") == "1" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtypes(device) -> tuple:
    """Return (weight_dtype, autocast_dtype) suitable for the device.

    bf16-capable GPUs (A100/L4/H100) keep everything in bf16. Older GPUs such as the
    Colab T4 lack bf16, and Qwen activations can overflow in pure fp16, so we keep the
    weights in fp32 and use fp16 autocast with a GradScaler.
    """
    import torch

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, torch.bfloat16
        return torch.float32, torch.float16
    return torch.float32, None


def inference_dtype(device):
    import torch

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def resolve_inference_dtype(name: str | None, device):
    """Map a config value ('auto', 'fp32', 'fp16', 'bf16') to a torch dtype."""
    import torch

    if name in (None, "auto"):
        return inference_dtype(device)
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def gpu_info() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"gpu": "cpu", "gpu_mem_gb": 0.0}
    props = torch.cuda.get_device_properties(0)
    return {"gpu": props.name, "gpu_mem_gb": round(props.total_memory / 2**30, 2)}


# Rough on-demand cloud prices (USD / GPU-hour) used only to put GPU time on a common scale.
GPU_PRICE_PER_HOUR = {"T4": 0.35, "L4": 0.70, "A100": 2.00, "H100": 3.50, "V100": 1.00, "cpu": 0.0}


def gpu_price(gpu_name: str) -> float:
    for key, price in GPU_PRICE_PER_HOUR.items():
        if key.lower() in gpu_name.lower():
            return price
    return 1.0


def reset_peak_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> float:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**30
    return 0.0


def sync(device=None) -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device is not None and device.type == "mps":
        torch.mps.synchronize()


@contextmanager
def timer():
    """Usage: `with timer() as t: ...; t()` -> elapsed seconds."""
    start = time.perf_counter()
    end = None

    def elapsed() -> float:
        return (end if end is not None else time.perf_counter()) - start

    try:
        yield elapsed
    finally:
        end = time.perf_counter()
