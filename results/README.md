# Results

Results of the final runs (large-GPU profile, `configs/default.yaml` + `configs/a100.yaml`, one NVIDIA RTX PRO
6000 Blackwell). Regenerate figures, tables and `report.md` from these files with

```bash
CEGQA_OUTPUT_DIR=results python -m cegqa.cli report --config configs/default.yaml --config configs/a100.yaml
```

| Path | Contents |
|---|---|
| `report.md` | all tables and figures with short descriptions |
| `runs/<run>/metrics.json` | EM/F1 per eval set with 95% bootstrap CIs |
| `runs/<run>/train_summary.json` | tokens, examples, steps, GPU time, FLOPs, peak memory, dev losses |
| `runs/<run>/train_log.jsonl` | dev-loss curve during training |
| `analysis/rq2_stats.json` | seed statistics, paired bootstrap, error analysis, example disagreements |
| `analysis/summary.json` | everything above in one file |
| `rq3/results.jsonl`, `rq3/setup.json` | inference benchmark rows |
| `data/data_report.json` | dataset sizes, lengths, overlap checks |
| `figures/`, `tables/` | figures (PNG/PDF) and LaTeX tables used in the paper |
| `results_bundle.json` | the run, analysis and benchmark files above as one JSON document |

Run directories are named `lora_<sources>_<budget>_s<seed>`; `baseline_zero_shot` and `baseline_few_shot` are the
prompting baselines. Per-question predictions and LoRA adapters are produced by the pipeline but not stored here.
