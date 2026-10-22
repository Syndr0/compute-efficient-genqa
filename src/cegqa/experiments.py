"""Experiment orchestration for the prompting baselines and RQ1-RQ3, plus the tiny Transformer.

All stages are resumable: finished training runs, prediction files and benchmark rows are
detected on disk and skipped, so a disconnected Colab session can simply re-run the cell.
"""
from __future__ import annotations

from pathlib import Path

from .data import data_ready, eval_sources, prepare_data
from .utils import ensure_dir, read_jsonl, write_json


def paths(cfg: dict) -> dict:
    root = Path(cfg["output_dir"])
    return {"runs": root / "runs", "rq3": root / "rq3", "tiny": root / "tiny", "analysis": root / "analysis",
            "figures": root / "figures"}


def stage_prepare(cfg: dict, force: bool = False) -> None:
    if data_ready(cfg) and not force:
        print(f"[skip] data already prepared in {cfg['data_dir']}")
        return
    prepare_data(cfg)


# --------------------------------------------------------------------------- baselines
def stage_baselines(cfg: dict) -> dict:
    from .evaluate import evaluate_system
    from .modeling import load_for_eval

    cache: dict = {}

    def loader():
        if not cache:
            cache["m"], cache["t"] = load_for_eval(cfg)
        return cache["m"], cache["t"]

    results = {}
    for mode in cfg["baselines"]["modes"]:
        run_dir = paths(cfg)["runs"] / f"baseline_{mode}"
        print(f"[baseline] {mode}")
        results[mode] = evaluate_system(cfg, run_dir, loader=loader, fewshot=(mode == "few_shot"))
    return results


# --------------------------------------------------------------------------- RQ1
def stage_rq1(cfg: dict) -> list[dict]:
    from .train_lora import train_and_evaluate

    r = cfg["rq1"]
    out = []
    for seed in r["seeds"]:
        for budget in r["token_budgets"]:
            out.append(train_and_evaluate(cfg, r["sources"], int(budget), seed, paths(cfg)["runs"]))
    return out


# --------------------------------------------------------------------------- RQ2
def stage_rq2(cfg: dict) -> dict:
    from .train_lora import train_and_evaluate

    r = cfg["rq2"]
    for seed in r["seeds"]:
        for cond, sources in r["conditions"].items():
            print(f"[rq2] condition={cond} seed={seed}")
            train_and_evaluate(cfg, sources, int(r["token_budget"]), seed, paths(cfg)["runs"])
    return rq2_statistics(cfg)


def rq2_statistics(cfg: dict) -> dict:
    """Mean/std over seeds, paired bootstrap (multi - single) per eval set, and error analysis."""
    import numpy as np

    from .error_analysis import disagreement_examples, error_breakdown
    from .stats import mean_std, paired_bootstrap
    from .train_lora import run_name

    r = cfg["rq2"]
    runs = paths(cfg)["runs"]
    conds = list(r["conditions"])
    assert len(conds) == 2, "RQ2 compares exactly two conditions"
    base, treat = conds
    report: dict = {"conditions": r["conditions"], "token_budget": r["token_budget"], "seeds": r["seeds"], "by_source": {}}
    for src in eval_sources(cfg):
        per_cond: dict = {}
        for cond in conds:
            seed_rows = []
            for seed in r["seeds"]:
                f = runs / run_name(r["conditions"][cond], int(r["token_budget"]), seed) / "predictions" / f"{src}.jsonl"
                if f.exists():
                    seed_rows.append(sorted(read_jsonl(f), key=lambda x: x["id"]))
            per_cond[cond] = seed_rows
        if not all(per_cond[c] for c in conds):
            continue
        n_seeds = min(len(per_cond[c]) for c in conds)
        entry: dict = {"n_seeds": n_seeds}
        for cond in conds:
            rows = per_cond[cond][:n_seeds]
            entry[cond] = {
                "em": mean_std([100 * np.mean([x["em"] for x in rs]) for rs in rows]),
                "f1": mean_std([100 * np.mean([x["f1"] for x in rs]) for rs in rows]),
                "errors": error_breakdown(rows[0]),
            }
        # seed-averaged per-item scores -> paired bootstrap on the same questions
        for metric in ("em", "f1"):
            a = np.mean([[x[metric] for x in rs] for rs in per_cond[base][:n_seeds]], axis=0)
            b = np.mean([[x[metric] for x in rs] for rs in per_cond[treat][:n_seeds]], axis=0)
            entry[f"paired_{metric}"] = paired_bootstrap(a, b, n_boot=cfg["stats"]["n_boot"])
            entry[f"paired_{metric}_per_seed"] = [
                paired_bootstrap([x[metric] for x in ra], [x[metric] for x in rb], n_boot=cfg["stats"]["n_boot"])
                for ra, rb in zip(per_cond[base][:n_seeds], per_cond[treat][:n_seeds])
            ]
        entry["examples"] = disagreement_examples(per_cond[base][0], per_cond[treat][0], k=5)
        entry["ood"] = src in cfg["data"]["ood_eval_sources"]
        report["by_source"][src] = entry
    write_json(report, ensure_dir(paths(cfg)["analysis"]) / "rq2_stats.json")
    return report


# --------------------------------------------------------------------------- RQ3 / tiny
def stage_rq3(cfg: dict) -> list[dict]:
    from .inference_bench import run_inference_bench

    return run_inference_bench(cfg, paths(cfg)["runs"], paths(cfg)["rq3"])


def stage_tiny(cfg: dict) -> list[dict]:
    from .tiny.train import train_tiny

    return train_tiny(cfg, paths(cfg)["tiny"])


def stage_report(cfg: dict) -> Path:
    from .report import build_report

    return build_report(cfg)


STAGES = {
    "prepare": stage_prepare,
    "baselines": stage_baselines,
    "rq1": stage_rq1,
    "rq2": stage_rq2,
    "rq3": stage_rq3,
    "tiny": stage_tiny,
    "report": stage_report,
}
