"""Generate the Colab notebooks in notebooks/ (python scripts/make_notebooks.py)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = "Syndr0/compute-efficient-genqa"
ROOT = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def header(title: str, name: str) -> dict:
    return md(f"# {title}\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
              f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name})")


SETUP = code(f'''
import os
import sys

import torch

USE_DRIVE = True
SMOKE_TEST = False
GPU_PROFILE = "auto"  # auto | t4 | a100

REPO_DIR = "/content/compute-efficient-genqa"
if not os.path.exists(REPO_DIR):
    !git clone -q https://github.com/{REPO}.git {{REPO_DIR}}
else:
    !git -C {{REPO_DIR}} pull -q
%cd {{REPO_DIR}}
!pip install -q -e .
sys.path.insert(0, f"{{REPO_DIR}}/src")

if USE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/cegqa_outputs"
else:
    OUT = "/content/cegqa_outputs"

GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
if GPU_PROFILE == "auto":
    GPU_PROFILE = "a100" if GPU_MEM_GB >= 39 else "t4"
CONFIG_FILES = ["configs/default.yaml"]
if GPU_PROFILE == "a100":
    CONFIG_FILES.append("configs/a100.yaml")
    OUT += "_a100"
if SMOKE_TEST:
    CONFIG_FILES.append("configs/smoke.yaml")
    OUT += "_smoke"
os.environ["CEGQA_OUTPUT_DIR"] = OUT
CONFIG = " ".join(f"--config {{f}}" for f in CONFIG_FILES)
print(f"GPU: {{GPU}} ({{GPU_MEM_GB:.0f}} GB) | profile: {{GPU_PROFILE}} | outputs: {{OUT}}")
''')

LOAD_CFG = code('''
import json

import pandas as pd
from IPython.display import Image, Markdown, display

from cegqa.data import eval_sources
from cegqa.utils import load_config, read_jsonl

cfg = load_config(CONFIG_FILES)
''')

PREPARE = code("!python -m cegqa.cli prepare $CONFIG")
REPORT = code("!python -m cegqa.cli report $CONFIG")


def figures(*names: str) -> dict:
    return code("for f in " + repr(list(names)) + ":\n    display(Image(f\"{cfg['output_dir']}/figures/{f}\"))")


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


NOTEBOOKS = {
    "00_data.ipynb": [
        header("Data preparation and evaluation", "00_data.ipynb"),
        SETUP,
        PREPARE,
        LOAD_CFG,
        code('''
rep = json.load(open(f"{cfg['data_dir']}/data_report.json"))
rows = []
for split in ("train", "dev", "test"):
    for src, v in rep[split].items():
        rows.append({"split": split, "source": src, "n": v["n"],
                     "ctx_words": round(v["context_words"]["mean"]), "ctx_words_p90": v["context_words"]["p90"],
                     "answer_words": round(v["answer_words"]["mean"], 2),
                     "answer_in_context": round(100 * v["answer_in_context_rate"], 1)})
pd.DataFrame(rows)
'''),
        code('''
display(pd.DataFrame(rep["overlap_before_filter"]).T)
display(pd.DataFrame(rep["overlap_after_filter"]).T)
'''),
        code("!python -m pytest -q tests"),
    ],
    "01_rq1_training_budget.ipynb": [
        header("Prompting baselines and RQ1: training-token budget", "01_rq1_training_budget.ipynb"),
        SETUP,
        PREPARE,
        code("!python -m cegqa.cli baselines $CONFIG"),
        code("!python -m cegqa.cli rq1 $CONFIG"),
        REPORT,
        LOAD_CFG,
        code('''
from pathlib import Path

from cegqa.train_lora import run_name

runs = Path(cfg["output_dir"]) / "runs"
rows = []
for mode in cfg["baselines"]["modes"]:
    m = json.load(open(runs / f"baseline_{mode}" / "metrics.json"))
    rows.append({"system": mode, **{s: round(m[s]["f1"], 1) for s in eval_sources(cfg)}})
for b in cfg["rq1"]["token_budgets"]:
    d = runs / run_name(cfg["rq1"]["sources"], int(b), cfg["rq1"]["seeds"][0])
    m, s = json.load(open(d / "metrics.json")), json.load(open(d / "train_summary.json"))
    rows.append({"system": f"LoRA {b:,}", "gpu_min": round(s["train_time_s"] / 60, 1),
                 "dev_loss": round(s["final_dev_loss"], 3), **{x: round(m[x]["f1"], 1) for x in eval_sources(cfg)}})
pd.DataFrame(rows)
'''),
        figures("rq1_budget_curves.png", "rq1_dev_loss.png", "rq1_f1_vs_gpu_minutes.png"),
    ],
    "02_rq2_generalization.ipynb": [
        header("RQ2: single- vs multi-source fine-tuning", "02_rq2_generalization.ipynb"),
        SETUP,
        PREPARE,
        code("!python -m cegqa.cli rq2 $CONFIG"),
        REPORT,
        LOAD_CFG,
        code('''
st = json.load(open(f"{cfg['output_dir']}/analysis/rq2_stats.json"))
base, treat = list(st["conditions"])
rows = []
for src, e in st["by_source"].items():
    rows.append({"eval": src, "ood": e["ood"],
                 f"f1_{base}": f"{e[base]['f1']['mean']:.1f} ± {e[base]['f1']['std']:.1f}",
                 f"f1_{treat}": f"{e[treat]['f1']['mean']:.1f} ± {e[treat]['f1']['std']:.1f}",
                 "delta_f1": round(e["paired_f1"]["diff"], 2),
                 "ci95": f"[{e['paired_f1']['lo']:+.1f}, {e['paired_f1']['hi']:+.1f}]",
                 "p": round(e["paired_f1"]["p_two_sided"], 4)})
pd.DataFrame(rows)
'''),
        code('''
pd.DataFrame([{"eval": src, "condition": cond, **{k: round(100 * v, 1) for k, v in e[cond]["errors"]["categories"].items()}}
              for src, e in st["by_source"].items() for cond in (base, treat)])
'''),
        figures("rq2_f1_bars.png"),
    ],
    "03_rq3_inference.ipynb": [
        header("RQ3: inference efficiency", "03_rq3_inference.ipynb"),
        SETUP,
        PREPARE,
        code("!python -m cegqa.cli rq3 $CONFIG"),
        REPORT,
        LOAD_CFG,
        code('''
df = pd.DataFrame(read_jsonl(f"{cfg['output_dir']}/rq3/results.jsonl"))
cols = ["group", "batch_size", "dtype_resolved", "examples_per_s", "gen_tokens_per_s", "batch_latency_mean_s",
        "batch_latency_p95_s", "peak_mem_gb", "padding_fraction", "em", "f1", "oom"]
df[[c for c in cols if c in df]].round(3)
'''),
        figures("rq3_batch_sweep.png", "rq3_tradeoff.png"),
    ],
    "04_report.ipynb": [
        header("Results and report", "04_report.ipynb"),
        SETUP,
        REPORT,
        code('''
os.chdir(OUT)
display(Markdown(open("report.md").read()))
os.chdir(REPO_DIR)
'''),
        code('''
!python scripts/export_results.py --src $OUT --dst /content/results
!cd /content && zip -qr cegqa_results.zip results
from google.colab import files

files.download("/content/cegqa_results.zip")
'''),
    ],
    "run_all.ipynb": [
        header("Reproduce all experiments", "run_all.ipynb"),
        SETUP,
        code("!python -m cegqa.cli all $CONFIG"),
        code('''
os.chdir(OUT)
display(Markdown(open("report.md").read()))
os.chdir(REPO_DIR)
'''),
    ],
}


def main() -> None:
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.ipynb"):
        if old.name not in NOTEBOOKS:
            old.unlink()
    for name, cells in NOTEBOOKS.items():
        (out / name).write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False) + "\n")
        print("wrote", out / name)


if __name__ == "__main__":
    main()
