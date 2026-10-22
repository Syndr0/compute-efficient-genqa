# Compute-Efficient Generative Question Answering

Adapting **Qwen3-0.6B** to context-grounded, single-turn question answering with **LoRA
supervised fine-tuning** under a small compute budget, using the standardized
[MRQA](https://huggingface.co/datasets/mrqa-workshop/mrqa) datasets.
LLM Semester Project · EN.737.

| | |
|---|---|
| Train | SQuAD, NewsQA, TriviaQA (MRQA train split) |
| ID eval | SQuAD, NewsQA, TriviaQA (MRQA validation split) |
| OOD eval | BioASQ, DuoRC (MRQA test split, never trained on) |
| Metrics | Exact Match, token F1, bootstrap CIs, paired bootstrap tests |

**Research questions**

* **RQ1 — Training efficiency:** QA quality vs LoRA training-token budget (0.25M / 1M / 4M tokens) against zero- and few-shot prompting, with learning curves, validation loss and GPU cost.
* **RQ2 — Generalization:** SQuAD-only vs SQuAD + NewsQA + TriviaQA at a fixed 1M-token budget, 3 seeds, evaluated on unseen BioASQ and DuoRC with paired bootstrap CIs and error analysis.
* **RQ3 — Inference efficiency:** batch size vs throughput, latency and GPU memory for the same fine-tuned model (plus length-sorting, KV-cache and precision ablations).
* **Tiny Transformer:** decoder-only Transformers (~1–17M params) trained from scratch; cross-entropy vs parameters and training tokens.

Project proposal: [docs/proposal.pdf](docs/proposal.pdf). Hypotheses and quantitative predictions are fixed in [docs/hypotheses.md](docs/hypotheses.md).

## Run on Google Colab

Pick a GPU runtime (**Runtime → Change runtime type**), open a notebook, and run all cells.
Every stage resumes from what is already saved on Drive.

| GPU | profile (auto-detected) | budgets (RQ1) | RQ2 budget | eval questions / set | outputs | full run |
|---|---|---|---|---|---|---|
| T4 (free tier) | `configs/default.yaml` | 0.25M, 1M, 4M | 1M | 500 | `MyDrive/cegqa_outputs` | ~2.5 h |
| A100 / H100 (Pro) | `default.yaml` + `configs/a100.yaml` | 0.25M, 1M, 4M, 16M | 4M | 1,000 | `MyDrive/cegqa_outputs_a100` | ~1.5–2 h |

Times in the notebook table below are for the T4 profile.

| Notebook | Owner | What it does | T4 time |
|---|---|---|---|
| [00 Data & evaluation](notebooks/00_data_and_evaluation.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Syndr0/compute-efficient-genqa/blob/main/notebooks/00_data_and_evaluation.ipynb) | P1 | MRQA download, splits, validation, overlap checks, metric tests | ~5 min |
| [01 Baselines & RQ1](notebooks/01_baselines_and_rq1_training_budget.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Syndr0/compute-efficient-genqa/blob/main/notebooks/01_baselines_and_rq1_training_budget.ipynb) | P2 | zero/few-shot prompting, LoRA token-budget sweep | ~1 h |
| [02 RQ2 generalization](notebooks/02_rq2_generalization.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Syndr0/compute-efficient-genqa/blob/main/notebooks/02_rq2_generalization.ipynb) | P3 | single vs multi-source, 3 seeds, statistics, error analysis | ~45 min |
| [03 RQ3 inference](notebooks/03_rq3_inference_efficiency.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Syndr0/compute-efficient-genqa/blob/main/notebooks/03_rq3_inference_efficiency.ipynb) | P4 | batch-size / sorting / KV-cache / precision sweeps | ~20 min |
| [04 Tiny Transformer](notebooks/04_tiny_transformer.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Syndr0/compute-efficient-genqa/blob/main/notebooks/04_tiny_transformer.ipynb) | P5 | from-scratch decoder-only Transformer, CE curves | ~25 min |
| [05 Report](notebooks/05_report.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Syndr0/compute-efficient-genqa/blob/main/notebooks/05_report.ipynb) | all | tables, figures, `report.md`, results zip | ~1 min |
| [Run all](notebooks/run_all.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Syndr0/compute-efficient-genqa/blob/main/notebooks/run_all.ipynb) | | every stage in order (tick `SMOKE_TEST` for a 10-minute check) | ~2.5 h |

Run 00 first (it prepares the data on Drive). After that, 01, 02 and 04 can run in parallel in separate Colab sessions sharing the same Drive folder; run 03 after 01 so it benchmarks the fine-tuned adapter.

## Run locally / from the command line

```bash
pip install -e ".[dev]"
python -m cegqa.cli all                                   # full pipeline (GPU recommended)
python -m cegqa.cli all --config configs/default.yaml --config configs/smoke.yaml   # quick end-to-end check
python -m cegqa.cli rq1 --set rq1.token_budgets=[500000,2000000]                    # override any config value
python -m pytest -q tests
```

Stages: `prepare`, `baselines`, `rq1`, `rq2`, `rq3`, `tiny`, `report`, `all`. Set `CEGQA_OUTPUT_DIR` to change where outputs go.

## Repository layout

```
configs/default.yaml        frozen experiment configuration (smoke.yaml: tiny end-to-end check)
src/cegqa/
  data.py                   P1  MRQA loading, cleaning, passage-level splits, validation, overlap checks
  metrics.py, evaluate.py   P1  EM/F1 (official normalisation), prediction logging
  prompts.py, modeling.py   P2  chat prompt, passage truncation, answer parsing, model loading
  train_lora.py             P2  LoRA SFT with exact token budgets, dev-loss curves, cost accounting
  experiments.py            P2/P3  baselines, RQ1, RQ2 orchestration + RQ2 statistics
  stats.py, error_analysis.py   P3  bootstrap CIs, paired bootstrap, error categories
  generation.py, inference_bench.py   P4  instrumented batched decoding, RQ3 sweeps
  tiny/                     P5  decoder-only Transformer from scratch + training
  report.py                 tables, figures and report.md
notebooks/                  Colab notebooks (generated by scripts/make_notebooks.py)
docs/                       hypotheses, report template, contribution statement
results/                    exported metrics, predictions, logs and figures (scripts/export_results.py)
```

## Outputs

Each LoRA run directory (`runs/lora_<sources>_<budget>_s<seed>/`) contains `train_summary.json`
(tokens, examples, steps, GPU time, cost, FLOPs, peak memory, dev losses), `train_log.jsonl`, the
`adapter/`, `predictions/<eval set>.jsonl` (gold answers, raw output, parsed answer, EM, F1,
truncation flags) and `metrics.json` (EM/F1 with 95% CIs). `python -m cegqa.cli report` turns
everything into `report.md`, `figures/` and `analysis/summary.json`.

## Deliverables

* Reproducible Colab notebooks and this repository
* Experiment configuration: [configs/default.yaml](configs/default.yaml), pre-registration: [docs/hypotheses.md](docs/hypotheses.md)
* Saved predictions and evaluation results: `results/` (exported after the Colab runs)
* Research report: generated `report.md` from [docs/report_template.md](docs/report_template.md)
* Team contribution statement: [docs/contributions.md](docs/contributions.md)

## License

MIT
