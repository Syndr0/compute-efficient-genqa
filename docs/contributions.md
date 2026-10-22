# Team contribution statement

Each member owns one technical component; the report and its interpretation are shared.
Replace the bracketed names before submission.

| Member | Role | Owned code | Notebook |
|---|---|---|---|
| **P1** [name] | Data & evaluation | `data.py` (MRQA loading, cleaning, passage-level train/dev split, overlap checks), `metrics.py`, `evaluate.py` (prediction logging), `tests/` | `00_data_and_evaluation` |
| **P2** [name] | Modeling & training | `prompts.py`, `modeling.py`, `train_lora.py` (token-budget sampling, LoRA SFT, cost accounting), baselines + RQ1 | `01_baselines_and_rq1_training_budget` |
| **P3** [name] | Generalization | RQ2 design (`experiments.stage_rq2`, `rq2_statistics`), `stats.py` (paired bootstrap), `error_analysis.py` | `02_rq2_generalization` |
| **P4** [name] | Inference efficiency | `generation.py` (instrumented batched decoding), `inference_bench.py` (batch / sorting / KV-cache / precision sweeps) | `03_rq3_inference_efficiency` |
| **P5** [name] | Tiny Transformer | `tiny/model.py` (decoder-only Transformer from scratch), `tiny/train.py` (BPE tokenizer, CE-vs-tokens curves, parameter counts) | `04_tiny_transformer` |
| All | Report | `report.py`, `docs/report_template.md`, `docs/hypotheses.md` | `05_report` |
