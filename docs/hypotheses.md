# Pre-registered hypotheses and quantitative predictions

Fixed **before** the main experiments (after the 1,000-example pilot). The experimental
configuration is frozen in `configs/default.yaml`; any later change must be recorded in the
deviations log at the end of this file.

## Fixed configuration

| item | value |
|---|---|
| base model | Qwen/Qwen3-0.6B, thinking disabled, greedy decoding, ≤ 32 new tokens |
| prompt | system instruction + `Passage: … Question: …`, passage ≤ 768 tokens |
| LoRA | r = 16, α = 32, dropout 0.05, all attention + MLP projections |
| optimisation | AdamW, lr 2e-4, 5% warm-up + cosine, 16 sequences/step, answer-only loss, one pass |
| token budget | prompt + answer tokens of the sampled training examples (equal share per source) |
| eval sets | 500 questions each: SQuAD, NewsQA, TriviaQA (ID); BioASQ, DuoRC (OOD) |
| metrics | EM, token F1 (official MRQA normalisation); 95% bootstrap CIs; paired bootstrap for comparisons |
| seeds | RQ1: 1 seed; RQ2: 3 seeds (0, 1, 2) |
| A100 profile | `configs/a100.yaml`: 16 sequences/step as 8×2 micro-batches, bf16; RQ1 budgets 0.25M/1M/4M/16M; RQ2 at 4M tokens; 1,000 questions per eval set; RQ3 batch sizes 1–256 |

## RQ1 — Training efficiency

* **H1a.** Even the smallest budget (0.25M tokens) improves F1 over the best prompting
  baseline on every ID set. *Prediction:* +15 F1 or more on SQuAD, mainly by removing verbose answers.
* **H1b.** Returns diminish: F1 is roughly linear in log(tokens). *Prediction:* going from 1M to 4M
  tokens adds less than half the F1 gained from 0.25M to 1M (A100 profile: 4M to 16M adds less still).
* **H1c.** Dev answer-loss decreases monotonically with budget and ranks the runs the same way as dev-set F1.
* **H1d.** Training cost scales linearly with tokens. *Prediction:* tokens/s stays within ±10% across budgets.

## RQ2 — Generalization

* **H2a.** At an equal 1M-token budget, multi-source training gives higher mean OOD F1
  (average of BioASQ and DuoRC) than SQuAD-only. *Prediction:* +2 to +5 F1, with a paired-bootstrap
  95% CI excluding 0 on at least one OOD set.
* **H2b.** The gain is largest on DuoRC (long narrative passages, closer to NewsQA/TriviaQA) and
  smallest on BioASQ (short, technical passages closer to SQuAD).
* **H2c.** SQuAD-only training is at least as good on SQuAD itself (difference within ±2 F1).
* **H2d.** Multi-source training mainly reduces `not_in_passage` and `wrong_span` errors on long passages.

## RQ3 — Inference efficiency

* **H3a.** Throughput (examples/s) increases sub-linearly with batch size and saturates by
  batch 32–64 on a T4. *Prediction:* ≥ 8× speed-up from batch 1 to the best batch size.
* **H3b.** Per-request latency grows with batch size; at batch 64 it is ≥ 3× that of batch 1.
* **H3c.** Peak memory grows roughly linearly with batch size (KV cache + activations) on top of a
  constant ~1.2 GB for fp16 weights, and stays below 16 GB up to batch 64.
* **H3d.** Quality (EM/F1) is unchanged across batch sizes to within ±0.5 F1; unsorted batches
  waste padding and lower throughput by ≥ 15% at large batch sizes; disabling the KV cache lowers
  throughput by ≥ 2×.

## Deviations log

* 2026-10-22: added the large-GPU profile (`configs/a100.yaml`) for the final runs, executed on an
  NVIDIA RTX PRO 6000 Blackwell (Colab G4). RQ2 therefore uses a 4M-token budget and 1,000 questions
  per eval set instead of 1M / 500; the prediction for H2a applies unchanged.
* 2026-10-22: the from-scratch tiny-Transformer study was dropped from the project scope.
