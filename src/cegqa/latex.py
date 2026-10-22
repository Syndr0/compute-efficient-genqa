"""LaTeX tables for the paper, generated from the aggregated results (`report.build_report`)."""
from __future__ import annotations

from pathlib import Path

from .utils import ensure_dir, fmt_tokens

NAMES = {"squad": "SQuAD", "newsqa": "NewsQA", "triviaqa": "TriviaQA", "bioasq": "BioASQ", "duorc": "DuoRC"}


def _n(s: str) -> str:
    return NAMES.get(s, s)


def _bold_max(values: list[float], fmt: str = "{:.1f}") -> list[str]:
    best = max(values) if values else None
    return [("\\textbf{" + fmt.format(v) + "}") if v == best else fmt.format(v) for v in values]


def table_data(rep: dict, id_srcs: list[str], ood_srcs: list[str]) -> str:
    rows = []
    for src in id_srcs + ood_srcs:
        tr = rep["train"].get(src)
        te = rep["test"][src]
        rows.append(f"{_n(src)} & {'OOD' if src in ood_srcs else 'ID'} & "
                    f"{tr['n'] if tr else '--'} & {te['n']} & {te['context_words']['mean']:.0f} & "
                    f"{te['answer_words']['mean']:.1f} & {100 * te['answer_in_context_rate']:.1f} \\\\")
    return "\n".join([
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Dataset & Split & Train pool & Eval & Passage words & Answer words & Answer in passage (\%) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
    ])


def table_rq1(summary: dict, srcs: list[str], id_srcs: list[str], ood_srcs: list[str], metric: str = "f1") -> str:
    systems = []
    for mode, label in (("zero_shot", "Zero-shot"), ("few_shot", "3-shot")):
        m = summary["baselines"].get(mode)
        if m:
            systems.append((label, "--", "--", m))
    for e in sorted(summary["rq1"], key=lambda e: e["budget"]):
        systems.append((f"LoRA {fmt_tokens(e['budget'])}", f"{e['train']['train_time_s'] / 60:.1f}",
                        f"{e['train']['final_dev_loss']:.3f}", e["metrics"]))
    cols = {s: [sys[3][s][metric] for sys in systems] for s in srcs}
    id_avg = [sum(sys[3][s][metric] for s in id_srcs) / len(id_srcs) for sys in systems]
    ood_avg = [sum(sys[3][s][metric] for s in ood_srcs) / len(ood_srcs) for sys in systems]
    bold = {s: _bold_max(v) for s, v in cols.items()}
    bid, bood = _bold_max(id_avg), _bold_max(ood_avg)
    rows = []
    for i, (label, minutes, loss, _) in enumerate(systems):
        cells = [bold[s][i] for s in srcs]
        rows.append(f"{label} & {minutes} & {loss} & " + " & ".join(cells) + f" & {bid[i]} & {bood[i]} \\\\")
        if i == 1:
            rows.append(r"\midrule")
    head = " & ".join(_n(s) for s in srcs)
    return "\n".join([
        r"\begin{tabular}{lrr" + "r" * len(srcs) + "rr}", r"\toprule",
        r" & & & \multicolumn{" + str(len(id_srcs)) + r"}{c}{In-domain} & \multicolumn{" + str(len(ood_srcs))
        + r"}{c}{Out-of-domain} & \multicolumn{2}{c}{Mean} \\",
        r"\cmidrule(lr){4-" + str(3 + len(id_srcs)) + r"}\cmidrule(lr){" + str(4 + len(id_srcs)) + "-"
        + str(3 + len(srcs)) + r"}\cmidrule(lr){" + str(4 + len(srcs)) + "-" + str(5 + len(srcs)) + "}",
        r"System & GPU-min & Dev loss & " + head + r" & ID & OOD \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}",
    ])


def table_cost(summary: dict) -> str:
    rows = []
    for e in sorted(summary["rq1"], key=lambda e: e["budget"]):
        t = e["train"]
        rows.append(f"{fmt_tokens(e['budget'])} & {t['examples']:,} & {t['tokens_seen']:,} & {t['tokens_per_s']:,.0f} & "
                    f"{t['train_time_s'] / 60:.1f} & {t['est_train_flops']:.2e} & {t['peak_mem_gb']:.1f} \\\\")
    return "\n".join([
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Budget & Examples & Tokens seen & Tokens/s & GPU-min & Est.\ FLOPs & Peak mem (GB) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
    ])


def table_rq2(rq2: dict, srcs: list[str]) -> str:
    base, treat = list(rq2["conditions"])
    rows = []
    for src in srcs:
        e = rq2["by_source"].get(src)
        if not e:
            continue
        rows.append(" & ".join([
            _n(src), "OOD" if e["ood"] else "ID",
            f"{e[base]['em']['mean']:.1f} $\\pm$ {e[base]['em']['std']:.1f}",
            f"{e[treat]['em']['mean']:.1f} $\\pm$ {e[treat]['em']['std']:.1f}",
            f"{e[base]['f1']['mean']:.1f} $\\pm$ {e[base]['f1']['std']:.1f}",
            f"{e[treat]['f1']['mean']:.1f} $\\pm$ {e[treat]['f1']['std']:.1f}",
            f"{e['paired_f1']['diff']:+.2f}", f"[{e['paired_f1']['lo']:+.2f}, {e['paired_f1']['hi']:+.2f}]",
            f"{e['paired_f1']['p_two_sided']:.3f}",
        ]) + r" \\")
    return "\n".join([
        r"\begin{tabular}{llrrrrrcr}", r"\toprule",
        r" & & \multicolumn{2}{c}{EM} & \multicolumn{2}{c}{F1} & \multicolumn{3}{c}{$\Delta$F1 (multi $-$ single)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}",
        r"Eval set & Type & Single & Multi & Single & Multi & $\Delta$ & 95\% CI & $p$ \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}",
    ])


def table_errors(rq2: dict, srcs: list[str]) -> str:
    from .error_analysis import CATEGORIES

    base, treat = list(rq2["conditions"])
    labels = {"correct": "Correct", "partial": "Partial", "verbose": "Verbose", "wrong_span": "Wrong span",
              "not_in_passage": "Not in passage", "empty": "Empty", "answer_truncated": "Truncated"}
    rows = []
    for src in srcs:
        e = rq2["by_source"].get(src)
        if not e:
            continue
        for cond in (base, treat):
            c = e[cond]["errors"]["categories"]
            rows.append(f"{_n(src) if cond == base else ''} & {cond} & "
                        + " & ".join(f"{100 * c[k]:.1f}" for k in CATEGORIES) + r" \\")
    return "\n".join([
        r"\begin{tabular}{ll" + "r" * len(CATEGORIES) + "}", r"\toprule",
        r"Eval set & Train & " + " & ".join(labels[k] for k in CATEGORIES) + r" \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}",
    ])


def table_rq3(rq3: dict) -> str:
    rows_in = [r for r in rq3["rows"] if not r.get("oom")]
    sweep = sorted((r for r in rows_in if r["group"] == "batch_sweep"), key=lambda r: r["batch_size"])
    unsorted = {r["batch_size"]: r for r in rows_in if r["group"] == "unsorted"}
    rows = []
    for r in sweep:
        u = unsorted.get(r["batch_size"])
        rows.append(f"{r['batch_size']} & {r['examples_per_s']:.1f} & {r['gen_tokens_per_s']:.0f} & "
                    f"{1000 * r['batch_latency_mean_s']:.0f} & {1000 * r['batch_latency_p95_s']:.0f} & "
                    f"{r['peak_mem_gb']:.2f} & {100 * r['padding_fraction']:.1f} & "
                    f"{(format(u['examples_per_s'], '.1f') if u else '--')} & {r['f1']:.1f} \\\\")
    for r in rq3["rows"]:
        if r.get("oom") and r["group"] == "batch_sweep":
            rows.append(f"{r['batch_size']} & \\multicolumn{{8}}{{c}}{{out of memory}} \\\\")
    return "\n".join([
        r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
        r"Batch & Ex/s & Gen tok/s & Latency (ms) & p95 (ms) & Peak mem (GB) & Padding (\%) & Ex/s unsorted & F1 \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
    ])


def table_rq3_ablation(rq3: dict) -> str:
    rows = []
    ok = [r for r in rq3["rows"] if not r.get("oom")]
    on = {r["batch_size"]: r for r in ok if r["group"] == "kv_cache_on_subset"}
    for r in sorted((r for r in ok if r["group"] == "kv_cache_off"), key=lambda r: r["batch_size"]):
        o = on.get(r["batch_size"])
        if o:
            rows.append(f"KV cache off vs.\\ on & {r['batch_size']} & {r['examples_per_s']:.2f} & {o['examples_per_s']:.2f} & "
                        f"{o['examples_per_s'] / r['examples_per_s']:.1f}$\\times$ & {r['peak_mem_gb']:.2f} / {o['peak_mem_gb']:.2f} \\\\")
    auto = {r["batch_size"]: r for r in ok if r["group"] == "batch_sweep"}
    for r in sorted((r for r in ok if r["group"].startswith("dtype_")), key=lambda r: (r["group"], r["batch_size"])):
        a = auto.get(r["batch_size"])
        if a:
            rows.append(f"{r['dtype_resolved']} vs.\\ {a['dtype_resolved']} & {r['batch_size']} & {r['examples_per_s']:.2f} & "
                        f"{a['examples_per_s']:.2f} & {a['examples_per_s'] / r['examples_per_s']:.1f}$\\times$ & "
                        f"{r['peak_mem_gb']:.2f} / {a['peak_mem_gb']:.2f} \\\\")
    return "\n".join([
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Variant vs.\ default & Batch & Ex/s (variant) & Ex/s (default) & Default speed-up & Peak mem (GB) \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}",
    ])


def write_tables(cfg: dict, summary: dict, data_report: dict | None, out_dir: Path) -> None:
    out = ensure_dir(out_dir)
    id_srcs, ood_srcs = list(cfg["data"]["id_eval_sources"]), list(cfg["data"]["ood_eval_sources"])
    srcs = id_srcs + ood_srcs
    if data_report:
        (out / "data.tex").write_text(table_data(data_report, id_srcs, ood_srcs))
    if summary.get("rq1") and summary.get("baselines"):
        (out / "rq1_f1.tex").write_text(table_rq1(summary, srcs, id_srcs, ood_srcs, "f1"))
        (out / "rq1_em.tex").write_text(table_rq1(summary, srcs, id_srcs, ood_srcs, "em"))
        (out / "cost.tex").write_text(table_cost(summary))
    if summary.get("rq2", {}).get("by_source"):
        (out / "rq2.tex").write_text(table_rq2(summary["rq2"], srcs))
        (out / "errors.tex").write_text(table_errors(summary["rq2"], srcs))
    if summary.get("rq3"):
        (out / "rq3.tex").write_text(table_rq3(summary["rq3"]))
        (out / "rq3_ablation.tex").write_text(table_rq3_ablation(summary["rq3"]))
