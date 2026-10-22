"""Generate the Colab notebooks in notebooks/ (run: python scripts/make_notebooks.py)."""
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


def badge(name: str) -> str:
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name})")


SETUP = code(f'''
#@title Setup: clone the repo, install, choose where outputs go
#@markdown Outputs (data, adapters, predictions, metrics) persist on Google Drive so a
#@markdown disconnected runtime can resume: every stage skips work that is already saved.
USE_DRIVE = True  #@param {{type:"boolean"}}
SMOKE_TEST = False  #@param {{type:"boolean"}}
GPU_PROFILE = "auto"  #@param ["auto", "t4", "a100"]
import os, sys
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
import torch
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
if GPU_PROFILE == "auto":  # large-GPU profile for A100 / H100 / G4 (RTX PRO 6000) and anything with >= 40 GB
    GPU_PROFILE = "a100" if GPU_MEM_GB >= 39 else "t4"
CONFIG_FILES = ["configs/default.yaml"]
if GPU_PROFILE == "a100":
    CONFIG_FILES.append("configs/a100.yaml")
    OUT += "_a100"  # different budgets/sizes: keep results of the two profiles apart
if SMOKE_TEST:
    CONFIG_FILES.append("configs/smoke.yaml")
    OUT += "_smoke"
os.environ["CEGQA_OUTPUT_DIR"] = OUT
CONFIG = " ".join(f"--config {{f}}" for f in CONFIG_FILES)
print(f"GPU: {{GPU}} ({{GPU_MEM_GB:.0f}} GB) | profile: {{GPU_PROFILE}} | outputs -> {{OUT}}")
print("configs:", CONFIG_FILES)
''')

LOAD_CFG = code('''
from cegqa.utils import load_config
cfg = load_config(CONFIG_FILES)
''')

PREPARE = code("!python -m cegqa.cli prepare $CONFIG")


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


NOTEBOOKS = {
    "00_data_and_evaluation.ipynb": [
        md(f"""
# 00 · Data preparation & evaluation pipeline (P1)
{badge("00_data_and_evaluation.ipynb")}

Builds every dataset used by the project from the MRQA release on the Hugging Face Hub:

* **training pools** – SQuAD, NewsQA, TriviaQA-web (MRQA *train* split)
* **dev sets** – held-out *passages* from each training pool (validation loss only)
* **ID test sets** – MRQA *validation* split of the three training sources
* **OOD test sets** – BioASQ and DuoRC-ParaphraseRC (MRQA *test* split)

then validates them, checks train/eval overlap, and unit-tests the EM/F1 metric.
Runtime: ~5 min (CPU is enough for this notebook).
"""),
        SETUP,
        PREPARE,
        md("## Data report: sizes, lengths, answer-in-passage rate"),
        LOAD_CFG,
        code('''
import json, pandas as pd
rep = json.load(open(f"{cfg['data_dir']}/data_report.json"))
rows = []
for split in ("train", "dev", "test"):
    for src, v in rep[split].items():
        rows.append({"split": split, "source": src, "n": v["n"],
                     "ctx_words_mean": round(v["context_words"]["mean"]), "ctx_words_p90": v["context_words"]["p90"],
                     "answer_words_mean": round(v["answer_words"]["mean"], 2),
                     "answer_in_context": f"{100*v['answer_in_context_rate']:.1f}%"})
pd.DataFrame(rows)
'''),
        md("## Overlap checks (exact normalised match against the training pools)"),
        code('''
display(pd.DataFrame(rep["overlap_before_filter"]).T.rename_axis("before filtering"))
display(pd.DataFrame(rep["overlap_after_filter"]).T.rename_axis("after filtering"))
'''),
        md("## Inspect examples and the prompt format"),
        code('''
from cegqa.data import load_split, load_fewshot
from cegqa.modeling import load_tokenizer
from cegqa.prompts import build_prompt
tok = load_tokenizer(cfg["model"]["name"])
ex = load_split(cfg, "test", "bioasq", limit=1)[0]
print("Q:", ex["question"], "| gold:", ex["answers"])
prompt, truncated = build_prompt(tok, ex, cfg["model"]["max_context_tokens"])
print(prompt[:1500])
print("few-shot demos:", [d["question"] for d in load_fewshot(cfg)])
'''),
        md("## Metric unit tests (official SQuAD/MRQA normalisation)"),
        code("!pip install -q pytest && python -m pytest -q tests"),
        code('''
from cegqa.metrics import score
for pred, golds in [("the Eiffel Tower", ["Eiffel Tower"]), ("Paris, France", ["Paris"]), ("1945", ["in 1945"])]:
    print(f"{pred!r:22} vs {golds} -> {score(pred, golds)}")
'''),
    ],
    "01_baselines_and_rq1_training_budget.ipynb": [
        md(f"""
# 01 · Prompting baselines & RQ1: training-token budget (P2)
{badge("01_baselines_and_rq1_training_budget.ipynb")}

1. **Zero-shot** and **3-shot** prompting of Qwen3-0.6B on all five eval sets.
2. **LoRA SFT** on the SQuAD + NewsQA + TriviaQA mixture at budgets of 0.25M, 1M and 4M
   training tokens (configurable in `configs/default.yaml → rq1`), each evaluated on all eval sets.

Logged per run: training loss, dev loss at 7 points, tokens/s, GPU time, estimated cost and FLOPs,
peak memory. Runtime on a T4: ~1 h.
"""),
        SETUP,
        PREPARE,
        md("## Prompting baselines"),
        code("!python -m cegqa.cli baselines $CONFIG"),
        md("## LoRA fine-tuning at each token budget"),
        code("!python -m cegqa.cli rq1 $CONFIG"),
        md("## Results"),
        LOAD_CFG,
        code('''
import json, pandas as pd
from pathlib import Path
from cegqa.data import eval_sources
from cegqa.train_lora import run_name
runs = Path(cfg["output_dir"]) / "runs"
rows = []
for mode in cfg["baselines"]["modes"]:
    m = json.load(open(runs / f"baseline_{mode}" / "metrics.json"))
    rows.append({"system": mode, **{f"{s} F1": round(m[s]["f1"], 1) for s in eval_sources(cfg)}})
for b in cfg["rq1"]["token_budgets"]:
    d = runs / run_name(cfg["rq1"]["sources"], int(b), cfg["rq1"]["seeds"][0])
    m, s = json.load(open(d / "metrics.json")), json.load(open(d / "train_summary.json"))
    rows.append({"system": f"LoRA {b:,} tok", "GPU-min": round(s["train_time_s"] / 60, 1),
                 "dev loss": round(s["final_dev_loss"], 3), **{f"{x} F1": round(m[x]["f1"], 1) for x in eval_sources(cfg)}})
pd.DataFrame(rows)
'''),
        code('''
!python -m cegqa.cli report $CONFIG
from IPython.display import Image, display
for f in ["rq1_budget_curves.png", "rq1_dev_loss.png", "rq1_f1_vs_gpu_minutes.png"]:
    display(Image(f"{cfg['output_dir']}/figures/{f}"))
'''),
    ],
    "02_rq2_generalization.ipynb": [
        md(f"""
# 02 · RQ2: single- vs multi-source fine-tuning and OOD generalization (P3)
{badge("02_rq2_generalization.ipynb")}

Holds the base model, LoRA configuration and **total training tokens (1M)** fixed and compares

* `single`: SQuAD only
* `multi`: SQuAD + NewsQA + TriviaQA (equal token share)

over **3 seeds** (data sample + LoRA init). Evaluation on ID sets and the unseen BioASQ and DuoRC
sets with seed mean ± std, paired bootstrap CIs over questions, and an error analysis.
The multi-source seed-0 run is shared with RQ1. Runtime on a T4: ~45 min.
"""),
        SETUP,
        PREPARE,
        code("!python -m cegqa.cli rq2 $CONFIG"),
        md("## Statistical comparison"),
        LOAD_CFG,
        code('''
import json, pandas as pd
st = json.load(open(f"{cfg['output_dir']}/analysis/rq2_stats.json"))
base, treat = list(st["conditions"])
rows = []
for src, e in st["by_source"].items():
    rows.append({"eval set": src, "OOD": e["ood"],
                 f"F1 {base}": f"{e[base]['f1']['mean']:.1f} ± {e[base]['f1']['std']:.1f}",
                 f"F1 {treat}": f"{e[treat]['f1']['mean']:.1f} ± {e[treat]['f1']['std']:.1f}",
                 "ΔF1": round(e["paired_f1"]["diff"], 2),
                 "95% CI": f"[{e['paired_f1']['lo']:+.1f}, {e['paired_f1']['hi']:+.1f}]",
                 "p (2-sided)": round(e["paired_f1"]["p_two_sided"], 4),
                 "ΔEM": round(e["paired_em"]["diff"], 2)})
pd.DataFrame(rows)
'''),
        md("## Error analysis"),
        code('''
cats = []
for src, e in st["by_source"].items():
    for cond in (base, treat):
        cats.append({"eval set": src, "condition": cond, **{k: round(100 * v, 1) for k, v in e[cond]["errors"]["categories"].items()}})
display(pd.DataFrame(cats))
for src, e in st["by_source"].items():
    if e["ood"]:
        print(f"\\n== {src}: questions where the two conditions disagree ==")
        display(pd.DataFrame(e["examples"]))
'''),
        code('''
!python -m cegqa.cli report $CONFIG
from IPython.display import Image
Image(f"{cfg['output_dir']}/figures/rq2_f1_bars.png")
'''),
    ],
    "03_rq3_inference_efficiency.ipynb": [
        md(f"""
# 03 · RQ3: inference batch size vs throughput, latency and memory (P4)
{badge("03_rq3_inference_efficiency.ipynb")}

Decodes the same 256 SQuAD prompts with the same fine-tuned model (merged LoRA; by default the
largest-budget multi-source adapter, otherwise the base model) under controlled configurations:

* batch size 1 → 64 (length-sorted batches, KV cache on)
* unsorted batches (padding cost)
* KV cache off (optional experiment, smaller subset)
* fp32 weights vs fp16/bf16

Each configuration: one warm-up batch, 3 timed repeats; quality (EM/F1) is recorded to confirm
it does not change. Run notebook 01 first so an adapter exists. Runtime on a T4: ~20 min.
"""),
        SETUP,
        PREPARE,
        code("!python -m cegqa.cli rq3 $CONFIG"),
        LOAD_CFG,
        code('''
import pandas as pd
from cegqa.utils import read_jsonl
df = pd.DataFrame(read_jsonl(f"{cfg['output_dir']}/rq3/results.jsonl"))
cols = ["group", "batch_size", "dtype_resolved", "examples_per_s", "gen_tokens_per_s", "batch_latency_mean_s",
        "batch_latency_p95_s", "peak_mem_gb", "padding_fraction", "em", "f1", "oom"]
df[[c for c in cols if c in df]].round(3)
'''),
        code('''
!python -m cegqa.cli report $CONFIG
from IPython.display import Image, display
for f in ["rq3_batch_sweep.png", "rq3_tradeoff.png"]:
    display(Image(f"{cfg['output_dir']}/figures/{f}"))
'''),
    ],
    "04_tiny_transformer.ipynb": [
        md(f"""
# 04 · Tiny decoder-only Transformer from scratch (P5)
{badge("04_tiny_transformer.ipynb")}

A GPT-style pre-LN Transformer (`src/cegqa/tiny/model.py`) with a byte-level BPE tokenizer (8k vocab)
trained on SQuAD in the format `<ctx> passage <q> question <a> answer <eos>`.
Three sizes (≈1M, 4M and 17M parameters) are trained to 20M tokens each; validation cross-entropy
is measured at log-spaced token counts (all tokens and answer tokens only), and greedy EM/F1 is
compared against Qwen3-0.6B. Runtime on a T4: ~25 min.
"""),
        SETUP,
        PREPARE,
        code("!python -m cegqa.cli tiny $CONFIG"),
        LOAD_CFG,
        code('''
import json, pandas as pd
s = json.load(open(f"{cfg['output_dir']}/tiny/summary.json"))
pd.DataFrame([{"model": x["name"], "params (M)": round(x["params"]["total"] / 1e6, 2),
               "non-emb (M)": round(x["params"]["non_embedding"] / 1e6, 2), "tokens (M)": round(x["tokens"] / 1e6, 1),
               "tokens/param": round(x["tokens_per_param"], 1), "FLOPs": f"{x['est_train_flops']:.2e}",
               "minutes": round(x["train_time_s"] / 60, 1), "val CE": round(x["final_val_ce"], 3),
               "answer CE": round(x["final_val_answer_ce"], 3), "EM": round(x["em"], 1), "F1": round(x["f1"], 1)} for x in s])
'''),
        code('''
!python -m cegqa.cli report $CONFIG
from IPython.display import Image
Image(f"{cfg['output_dir']}/figures/tiny_ce_curves.png")
'''),
        code('''
# sample predictions from the largest model
from cegqa.utils import read_jsonl
name = cfg["tiny"]["models"][-1]["name"]
pd.DataFrame(read_jsonl(f"{cfg['output_dir']}/tiny/{name}/predictions_{cfg['tiny']['eval_source']}.jsonl")[:15])[["question", "answers", "prediction", "f1"]]
'''),
    ],
    "05_report.ipynb": [
        md(f"""
# 05 · Aggregate results and build the report
{badge("05_report.ipynb")}

Collects every saved result into `report.md` (tables + figures), `analysis/summary.json`, and a
zip of the lightweight artifacts (configs, metrics, predictions, logs, figures; adapters excluded)
that can be committed to the repository's `results/` folder.
"""),
        SETUP,
        code("!python -m cegqa.cli report $CONFIG"),
        code('''
from IPython.display import Markdown, display
import os
os.chdir(OUT)
display(Markdown(open("report.md").read()))
os.chdir(REPO_DIR)
'''),
        code('''
!python scripts/export_results.py --src $OUT --dst /content/results
!cd /content && zip -qr cegqa_results.zip results && ls -lh /content/cegqa_results.zip
from google.colab import files
files.download("/content/cegqa_results.zip")
'''),
    ],
    "run_all.ipynb": [
        md(f"""
# Run everything
{badge("run_all.ipynb")}

Runs every stage in order (data → baselines → RQ1 → RQ2 → RQ3 → tiny Transformer → report).
Tick `SMOKE_TEST` for a ~10-minute end-to-end check on a handful of examples.
Full run: about 2–3 h on a T4, or 1.5–2 h on an A100 with the larger A100 profile
(`configs/a100.yaml`, picked automatically for any GPU with ≥ 40 GB: A100, H100, G4). Every stage resumes from what is already on Drive,
so after a disconnect just run the cells again.
"""),
        SETUP,
        code("!python -m cegqa.cli all $CONFIG"),
        code('''
from IPython.display import Markdown, display
import os
os.chdir(OUT)
display(Markdown(open("report.md").read()))
os.chdir(REPO_DIR)
'''),
    ],
}


def main() -> None:
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        (out / name).write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False) + "\n")
        print("wrote", out / name)


if __name__ == "__main__":
    main()
