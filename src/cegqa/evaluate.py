"""P1 - Evaluation pipeline: generate answers for each eval set, score EM/F1, log predictions.

Each system gets a run directory with
  predictions/<source>.jsonl   one row per question (gold, raw output, parsed answer, EM, F1, flags)
  metrics.json                 {source: {n, em, f1, em_ci, f1_ci, ...}} plus generation stats
"""
from __future__ import annotations

from pathlib import Path

from .data import load_fewshot, load_split
from .generation import generate
from .metrics import aggregate, score
from .prompts import build_prompt, extract_answer
from .stats import bootstrap_ci
from .utils import read_json, read_jsonl, write_json, write_jsonl


def evaluate_source(cfg: dict, model, tok, source: str, run_dir: Path, fewshot: list[dict] | None = None) -> dict:
    ecfg = cfg["eval"]
    pred_path = run_dir / "predictions" / f"{source}.jsonl"
    exs = load_split(cfg, "test", source, limit=ecfg.get("max_examples_per_source"))
    prompts, truncated = [], []
    for ex in exs:
        p, t = build_prompt(tok, ex, cfg["model"]["max_context_tokens"], fewshot)
        prompts.append(p)
        truncated.append(t)
    raw, gen_stats = generate(model, tok, prompts, batch_size=ecfg["batch_size"],
                              max_new_tokens=cfg["model"]["max_new_tokens"], progress=True)
    rows = []
    for ex, out, trunc, prompt in zip(exs, raw, truncated, prompts):
        pred = extract_answer(out)
        s = score(pred, ex["answers"])
        rows.append({
            "id": ex["id"],
            "source": source,
            "question": ex["question"],
            "answers": ex["answers"],
            "raw_output": out,
            "prediction": pred,
            "em": s["em"],
            "f1": s["f1"],
            "context_truncated": trunc,
            "answer_in_prompt": any(a.lower() in prompt.lower() for a in ex["answers"]),
            "pred_in_context": bool(pred) and pred.lower() in ex["context"].lower(),
        })
    write_jsonl(rows, pred_path)
    return summarize(rows, gen_stats)


def summarize(rows: list[dict], gen_stats: dict | None = None) -> dict:
    m = aggregate(rows)
    m["em_ci"] = bootstrap_ci([r["em"] for r in rows])
    m["f1_ci"] = bootstrap_ci([r["f1"] for r in rows])
    m["answer_in_prompt_rate"] = sum(r["answer_in_prompt"] for r in rows) / max(1, len(rows))
    if gen_stats:
        m["generation"] = gen_stats
    return m


def evaluate_system(cfg: dict, run_dir: str | Path, model=None, tok=None, adapter_dir=None,
                    fewshot: bool = False, sources: list[str] | None = None, loader=None) -> dict:
    """Evaluate one system on every eval source, resuming from saved predictions.

    `model`/`tok` may be passed in; otherwise `loader()` is called lazily, only if some
    source still needs predictions (so re-running a finished experiment is free).
    """
    from .data import eval_sources

    run_dir = Path(run_dir)
    sources = sources or eval_sources(cfg)
    metrics_path = run_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    demos = load_fewshot(cfg) if fewshot else None
    for src in sources:
        pred_path = run_dir / "predictions" / f"{src}.jsonl"
        if src in metrics and pred_path.exists():
            continue
        if pred_path.exists():  # metrics lost but predictions saved: just re-score
            metrics[src] = summarize(read_jsonl(pred_path))
        else:
            if model is None:
                model, tok = loader()
            metrics[src] = evaluate_source(cfg, model, tok, src, run_dir, demos)
        write_json(metrics, metrics_path)
        print(f"  [{run_dir.name}] {src}: EM {metrics[src]['em']:.1f}  F1 {metrics[src]['f1']:.1f}")
    return metrics
