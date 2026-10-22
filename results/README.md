# Results

Exported experiment artifacts (metrics, per-question predictions, training summaries and logs,
RQ2 statistics, RQ3 benchmark rows, tiny-Transformer logs, figures, `report.md`).
Populate after the Colab runs with notebook `05_report` (downloads `cegqa_results.zip`) or:

```bash
python scripts/export_results.py --src outputs --dst results
```

LoRA adapters and processed data are not committed (re-created by the notebooks).
