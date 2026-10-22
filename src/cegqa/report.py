"""Aggregate every saved result into tables, figures and the research report.

Writes to <output_dir>:
  analysis/summary.json   all numbers in one machine-readable file
  figures/*.{png,pdf}     learning curves, budget/cost curves, RQ2 bars, RQ3 sweeps
  tables/*.tex            LaTeX tables for the paper
  report.md               docs/report_template.md with the generated results section filled in
"""
from __future__ import annotations

from pathlib import Path

from .data import eval_sources
from .experiments import paths, rq2_statistics
from .train_lora import run_name
from .latex import write_tables
from .utils import REPO_ROOT, ensure_dir, fmt_tokens, read_json, read_jsonl, write_json

# Validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red); never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"squad": "SQuAD", "newsqa": "NewsQA", "triviaqa": "TriviaQA", "bioasq": "BioASQ", "duorc": "DuoRC"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
    })
    return plt


def _token_axis(ax) -> None:
    """Log-scaled x axis labelled 250k / 1M / 4M instead of overlapping scientific minor ticks."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_tokens(v)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _save(fig, figs: Path, name: str) -> None:
    fig.savefig(figs / f"{name}.png")
    fig.savefig(figs / f"{name}.pdf")


def _name(s: str) -> str:
    return NAMES.get(s, s)


def _md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _load_metrics(run_dir: Path) -> dict | None:
    p = run_dir / "metrics.json"
    return read_json(p) if p.exists() else None


# --------------------------------------------------------------------------- sections
def section_data(cfg: dict) -> str:
    p = Path(cfg["data_dir"]) / "data_report.json"
    if not p.exists():
        return "_Data not prepared yet._"
    rep = read_json(p)
    rows = []
    for split in ("train", "dev", "test"):
        for src, v in rep[split].items():
            rows.append([split, _name(src), v["n"], f"{v['context_words'].get('mean', 0):.0f}",
                         f"{v['answer_words'].get('mean', 0):.1f}", f"{100 * v['answer_in_context_rate']:.1f}%"])
    ov = [[k, v["n"], v["context_overlap"], v["question_overlap"], v["pair_overlap"]]
          for k, v in rep["overlap_after_filter"].items()]
    ovb = rep["overlap_before_filter"]
    removed = ", ".join(f"{_name(k)}: {v['context_overlap']}" for k, v in ovb.items() if not k.startswith("dev_") and v["context_overlap"])
    return "\n\n".join([
        "### Data",
        _md_table(["split", "source", "n", "ctx words (mean)", "answer words (mean)", "answer in context"], rows),
        "Overlap with the training pools after filtering (exact normalised match):",
        _md_table(["eval set", "n", "context overlap", "question overlap", "pair overlap"], ov),
        f"Eval items whose passage also appears in a training pool (removed before sub-sampling): {removed or 'none'}.",
    ])


def section_baselines_rq1(cfg: dict, plt, figs: Path, summary: dict) -> str:
    runs = paths(cfg)["runs"]
    srcs = eval_sources(cfg)
    lines = ["### Prompting baselines and RQ1 (training-token budget)"]
    table = []
    base = {}
    for mode in cfg["baselines"]["modes"]:
        m = _load_metrics(runs / f"baseline_{mode}")
        if m:
            base[mode] = m
            table.append([mode.replace("_", "-"), "-", "-"] + [f"{m[s]['em']:.1f} / {m[s]['f1']:.1f}" if s in m else "-" for s in srcs])
    r = cfg["rq1"]
    rq1 = []
    for seed in r["seeds"]:
        for b in r["token_budgets"]:
            d = runs / run_name(r["sources"], int(b), seed)
            m, s = _load_metrics(d), (d / "train_summary.json")
            if m and s.exists():
                s = read_json(s)
                rq1.append({"budget": int(b), "seed": seed, "metrics": m, "summary": s, "dir": d})
                table.append([f"LoRA {fmt_tokens(b)} (s{seed})", f"{s['train_time_s'] / 60:.1f} min",
                              f"{s['final_dev_loss']:.3f}"] + [f"{m[x]['em']:.1f} / {m[x]['f1']:.1f}" if x in m else "-" for x in srcs])
    summary["baselines"] = base
    summary["rq1"] = [{"budget": e["budget"], "seed": e["seed"], "metrics": e["metrics"],
                       "train": {k: e["summary"][k] for k in ("tokens_seen", "examples", "train_time_s", "gpu_hours",
                                                                "est_cost_usd", "est_train_flops", "peak_mem_gb",
                                                                "tokens_per_s", "final_dev_loss", "gpu")}} for e in rq1]
    if not table:
        return "\n\n".join(lines + ["_No results yet._"])
    lines.append(f"Training sources: {', '.join(_name(s) for s in r['sources'])}. Cells are EM / F1 (%).")
    lines.append(_md_table(["system", "train time", "dev loss"] + [_name(s) for s in srcs], table))
    if rq1:
        cost = [[fmt_tokens(e["budget"]), e["seed"], f"{e['summary']['tokens_seen']:,}", e["summary"]["examples"],
                 f"{e['summary']['tokens_per_s']:.0f}", f"{e['summary']['gpu_hours']:.3f}",
                 f"${e['summary']['est_cost_usd']:.3f}", f"{e['summary']['est_train_flops']:.2e}",
                 f"{e['summary']['peak_mem_gb']:.1f}"] for e in rq1]
        lines.append(f"Computational cost (GPU: {rq1[0]['summary']['gpu']}):")
        lines.append(_md_table(["budget", "seed", "tokens", "examples", "tok/s", "GPU-h", "est. $", "est. FLOPs", "peak GB"], cost))

        # F1 vs budget per eval set, baselines as reference lines
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
        for ax, metric in zip(axes, ("f1", "em")):
            for i, s in enumerate(srcs):
                pts = sorted((e["budget"], e["metrics"][s][metric]) for e in rq1 if s in e["metrics"])
                if pts:
                    ax.plot(*zip(*pts), marker="o", color=SERIES[i], label=_name(s))
                for mode, ls in (("zero_shot", ":"), ("few_shot", "--")):
                    if mode in base and s in base[mode]:
                        ax.axhline(base[mode][s][metric], color=SERIES[i], ls=ls, lw=1, alpha=0.7)
            _token_axis(ax)
            ax.set_xticks(sorted({e["budget"] for e in rq1}))
            ax.set_xlabel("LoRA training tokens")
            ax.set_ylabel(f"{metric.upper()} (%)")
            ax.set_title(f"{metric.upper()} vs training-token budget")
        axes[1].legend(fontsize=8, title="eval set\n(dotted: zero-shot,\ndashed: few-shot)", title_fontsize=7,
                       loc="upper left", bbox_to_anchor=(1.02, 1))
        _save(fig, figs, "rq1_budget_curves")
        plt.close(fig)

        # dev-loss learning curves
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for i, e in enumerate(sorted(rq1, key=lambda e: e["budget"])):
            dev = [x for x in read_jsonl(e["dir"] / "train_log.jsonl") if x["type"] == "dev"]
            ax.plot([max(x["tokens"], 1e3) for x in dev], [x["dev_loss_mean"] for x in dev], marker="o",
                    color=SERIES[i % len(SERIES)], label=f"{fmt_tokens(e['budget'])} (s{e['seed']})")
        _token_axis(ax)
        ax.set_xlabel("training tokens seen")
        ax.set_ylabel("dev answer loss (nats/token)")
        ax.set_title("Validation loss during LoRA SFT")
        ax.legend(fontsize=8, title="budget", loc="upper left", bbox_to_anchor=(1.02, 1))
        _save(fig, figs, "rq1_dev_loss")
        plt.close(fig)

        # quality vs GPU cost (mean F1 over ID and OOD sets)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for i, (label, group) in enumerate((("in-domain", cfg["data"]["id_eval_sources"]),
                                            ("out-of-domain", cfg["data"]["ood_eval_sources"]))):
            pts = sorted((e["summary"]["gpu_hours"] * 60, sum(e["metrics"][s]["f1"] for s in group) / len(group))
                         for e in rq1 if all(s in e["metrics"] for s in group))
            if pts:
                ax.plot(*zip(*pts), marker="o", color=SERIES[i], label=label)
        ax.set_xlabel("training GPU-minutes")
        ax.set_ylabel("mean F1 (%)")
        ax.set_title("Quality vs training compute")
        ax.legend(fontsize=8)
        _save(fig, figs, "rq1_f1_vs_gpu_minutes")
        plt.close(fig)
        lines.append("![RQ1 budget curves](figures/rq1_budget_curves.png)\n\n"
                     "![RQ1 dev loss](figures/rq1_dev_loss.png) ![RQ1 cost](figures/rq1_f1_vs_gpu_minutes.png)")
    return "\n\n".join(lines)


def section_rq2(cfg: dict, plt, figs: Path, summary: dict) -> str:
    rep = rq2_statistics(cfg)
    summary["rq2"] = rep
    lines = ["### RQ2 (single- vs multi-source at a fixed token budget)"]
    if not rep["by_source"]:
        return "\n\n".join(lines + ["_No results yet._"])
    base, treat = list(rep["conditions"])
    lines.append(f"Budget {fmt_tokens(rep['token_budget'])} tokens; seeds {rep['seeds']}. "
                 f"`{base}` = {'+'.join(rep['conditions'][base])}, `{treat}` = {'+'.join(rep['conditions'][treat])}. "
                 f"Means ± std over seeds; Δ = {treat} − {base} with a 95% paired-bootstrap CI over questions "
                 f"(per-question scores averaged over seeds).")
    rows = []
    for src, e in rep["by_source"].items():
        f = lambda c, m: f"{e[c][m]['mean']:.1f} ± {e[c][m]['std']:.1f}"  # noqa: E731
        pf, pe = e["paired_f1"], e["paired_em"]
        rows.append([_name(src), "OOD" if e["ood"] else "ID", f(base, "em"), f(treat, "em"),
                     f"{pe['diff']:+.1f} [{pe['lo']:+.1f}, {pe['hi']:+.1f}]", f(base, "f1"), f(treat, "f1"),
                     f"{pf['diff']:+.1f} [{pf['lo']:+.1f}, {pf['hi']:+.1f}]", f"{pf['p_two_sided']:.3f}"])
    lines.append(_md_table(["eval set", "type", f"EM {base}", f"EM {treat}", "ΔEM [95% CI]", f"F1 {base}",
                            f"F1 {treat}", "ΔF1 [95% CI]", "p (F1, 2-sided)"], rows))

    srcs = list(rep["by_source"])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    w = 0.38
    for i, cond in enumerate((base, treat)):
        xs = [k + (i - 0.5) * w for k in range(len(srcs))]
        ys = [rep["by_source"][s][cond]["f1"]["mean"] for s in srcs]
        err = [rep["by_source"][s][cond]["f1"]["std"] for s in srcs]
        ax.bar(xs, ys, width=w - 0.03, yerr=err, color=SERIES[i], label=cond, capsize=3, error_kw={"lw": 1, "ecolor": INK_2})
    ax.set_xticks(range(len(srcs)))
    ax.set_xticklabels([f"{_name(s)}\n({'OOD' if rep['by_source'][s]['ood'] else 'ID'})" for s in srcs])
    ax.set_ylabel("F1 (%)")
    ax.set_title("Single- vs multi-source LoRA at equal token budget")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    _save(fig, figs, "rq2_f1_bars")
    plt.close(fig)

    from .error_analysis import CATEGORIES

    err_rows = []
    for src, e in rep["by_source"].items():
        for cond in (base, treat):
            cats = e[cond]["errors"]["categories"]
            err_rows.append([_name(src), cond] + [f"{100 * cats[c]:.1f}" for c in CATEGORIES])
    lines.append("![RQ2 bars](figures/rq2_f1_bars.png)")
    lines.append("Error analysis (% of questions, seed 0):")
    lines.append(_md_table(["eval set", "condition"] + CATEGORIES, err_rows))
    qt_rows = []
    for src, e in rep["by_source"].items():
        if not e["ood"]:
            continue
        for qt, v in e[treat]["errors"]["f1_by_question_type"].items():
            b = e[base]["errors"]["f1_by_question_type"].get(qt)
            if v["n"] >= 20 and b:
                qt_rows.append([_name(src), qt, v["n"], f"{b['f1']:.1f}", f"{v['f1']:.1f}"])
    if qt_rows:
        lines.append("OOD F1 by question type (types with ≥ 20 questions):")
        lines.append(_md_table(["eval set", "question type", "n", f"F1 {base}", f"F1 {treat}"], qt_rows))
    ex_rows = []
    for src, e in rep["by_source"].items():
        if e["ood"]:
            for x in e["examples"][:4]:
                ex_rows.append([_name(src), x["question"][:80].replace("|", "/"), " / ".join(x["answers"])[:40].replace("|", "/"),
                                x["pred_a"][:40].replace("|", "/"), x["pred_b"][:40].replace("|", "/")])
    if ex_rows:
        lines.append(f"Example OOD disagreements (a = {base}, b = {treat}):")
        lines.append(_md_table(["eval set", "question", "gold", "pred a", "pred b"], ex_rows))
    return "\n\n".join(lines)


def section_rq3(cfg: dict, plt, figs: Path, summary: dict) -> str:
    res_path = paths(cfg)["rq3"] / "results.jsonl"
    lines = ["### RQ3 (inference batch size vs throughput, latency, memory)"]
    if not res_path.exists():
        return "\n\n".join(lines + ["_No results yet._"])
    rows = [r for r in read_jsonl(res_path)]
    setup = read_json(paths(cfg)["rq3"] / "setup.json")
    summary["rq3"] = {"setup": setup, "rows": rows}
    lines.append(f"Model: `{setup['adapter'] or 'base model'}` (merged), {setup['n']} {_name(setup['source'])} prompts, "
                 f"GPU {setup['gpu']}, greedy decoding, max {cfg['model']['max_new_tokens']} new tokens, "
                 f"{cfg['rq3']['repeats']} timed repeats per configuration.")
    table = []
    for r in rows:
        if r["oom"]:
            table.append([r["group"], r["batch_size"], r["dtype_resolved"], "OOM", "", "", "", "", ""])
            continue
        table.append([r["group"], r["batch_size"], r["dtype_resolved"], f"{r['examples_per_s']:.1f} ± {r['examples_per_s_std']:.1f}",
                      f"{r['gen_tokens_per_s']:.0f}", f"{1000 * r['batch_latency_mean_s']:.0f}",
                      f"{1000 * r['batch_latency_p95_s']:.0f}", f"{r['peak_mem_gb']:.2f}",
                      f"{r['em']:.1f} / {r['f1']:.1f}"])
    lines.append(_md_table(["group", "batch", "dtype", "examples/s", "gen tok/s", "latency/batch ms", "p95 ms",
                            "peak GB", "EM / F1"], table))

    ok = [r for r in rows if not r["oom"]]
    sweep = sorted((r for r in ok if r["group"] == "batch_sweep"), key=lambda r: r["batch_size"])
    unsorted = sorted((r for r in ok if r["group"] == "unsorted"), key=lambda r: r["batch_size"])
    if sweep:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
        bs = [r["batch_size"] for r in sweep]
        axes[0].plot(bs, [r["examples_per_s"] for r in sweep], marker="o", color=SERIES[0], label="length-sorted")
        if unsorted:
            axes[0].plot([r["batch_size"] for r in unsorted], [r["examples_per_s"] for r in unsorted], marker="o",
                         color=SERIES[1], label="unsorted")
        axes[0].set_ylabel("throughput (examples/s)")
        axes[0].legend(fontsize=8)
        axes[1].plot(bs, [1000 * r["batch_latency_mean_s"] for r in sweep], marker="o", color=SERIES[0], label="mean")
        axes[1].plot(bs, [1000 * r["batch_latency_p95_s"] for r in sweep], marker="o", color=SERIES[2], label="p95")
        axes[1].set_ylabel("latency per request (ms)")
        axes[1].legend(fontsize=8)
        axes[2].plot(bs, [r["peak_mem_gb"] for r in sweep], marker="o", color=SERIES[0], label="peak")
        axes[2].axhline(sweep[0]["model_mem_gb"], color=INK_2, ls="--", lw=1, label="weights")
        axes[2].set_ylabel("GPU memory (GB)")
        axes[2].legend(fontsize=8)
        for ax, t in zip(axes, ("Throughput", "Latency", "Peak memory")):
            ax.set_xscale("log", base=2)
            ax.set_xticks(bs)
            ax.set_xticklabels([str(b) for b in bs])
            ax.minorticks_off()
            ax.set_xlabel("batch size")
            ax.set_title(t)
        fig.subplots_adjust(wspace=0.35)
        _save(fig, figs, "rq3_batch_sweep")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.plot([1000 * r["batch_latency_mean_s"] for r in sweep], [r["examples_per_s"] for r in sweep], marker="o", color=SERIES[0])
        for r in sweep:
            ax.annotate(f"bs={r['batch_size']}", (1000 * r["batch_latency_mean_s"], r["examples_per_s"]),
                        textcoords="offset points", xytext=(4, -10), fontsize=7, color=INK_2)
        ax.set_xscale("log")
        ax.set_xlabel("latency per request (ms)")
        ax.set_ylabel("throughput (examples/s)")
        ax.set_title("Throughput-latency trade-off")
        _save(fig, figs, "rq3_tradeoff")
        plt.close(fig)
        lines.append("![RQ3 sweep](figures/rq3_batch_sweep.png) ![RQ3 trade-off](figures/rq3_tradeoff.png)")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- main
def build_report(cfg: dict) -> Path:
    plt = _plt()
    root = Path(cfg["output_dir"])
    figs = ensure_dir(paths(cfg)["figures"])
    summary: dict = {}
    sections = [
        section_data(cfg),
        section_baselines_rq1(cfg, plt, figs, summary),
        section_rq2(cfg, plt, figs, summary),
        section_rq3(cfg, plt, figs, summary),
    ]
    write_json(summary, ensure_dir(paths(cfg)["analysis"]) / "summary.json")
    data_report = Path(cfg["data_dir"]) / "data_report.json"
    write_tables(cfg, summary, read_json(data_report) if data_report.exists() else None, root / "tables")
    results_md = "\n\n".join(sections)
    template = (REPO_ROOT / "docs" / "report_template.md").read_text()
    report = template.replace("<!-- RESULTS -->", results_md)
    out = root / "report.md"
    out.write_text(report)
    print(f"Report written to {out}")
    return out
