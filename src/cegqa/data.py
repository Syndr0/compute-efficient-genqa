"""P1 - Data preparation: MRQA loading, cleaning, splits, validation and overlap checks.

Outputs (in `cfg.data_dir`):
  train_<src>.jsonl   training pool (MRQA train split minus held-out dev contexts)
  dev_<src>.jsonl     in-domain dev set (held-out contexts from MRQA train) for val loss
  test_<src>.jsonl    evaluation set: MRQA validation split (ID) or test split (OOD)
  fewshot.jsonl       fixed few-shot demonstrations (short SQuAD train examples)
  data_report.json    counts, length stats, answer-in-context rates, overlap checks
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path

from .utils import ensure_dir, read_jsonl, write_json, write_jsonl

# Short names used throughout the repo -> MRQA `subset` values.
SOURCES = {
    "squad": "SQuAD",
    "newsqa": "NewsQA",
    "triviaqa": "TriviaQA-web",
    "searchqa": "SearchQA",
    "hotpotqa": "HotpotQA",
    "nq": "NaturalQuestionsShort",
    "bioasq": "BioASQ",
    "duorc": "DuoRC.ParaphraseRC",
    "drop": "DROP",
    "race": "RACE",
    "relext": "RelationExtraction",
    "textbookqa": "TextbookQA",
}
# MRQA in-domain datasets have train + validation splits; out-of-domain ones only "test".
OOD_SOURCES = {"bioasq", "duorc", "drop", "race", "relext", "textbookqa"}

_COLUMNS = ["subset", "qid", "question", "context", "answers"]


# --------------------------------------------------------------------------- loading
def _read_split(repo: str, split: str, wanted_subsets: set[str]) -> list[dict]:
    """Read only the needed columns/rows of an MRQA parquet split straight from the Hub."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{repo}/plain_text/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files for {repo}:{split}")
    rows: list[dict] = []
    for path in files:
        subsets = pq.read_table(path, columns=["subset"], filesystem=fs).column("subset")
        present = set(subsets.unique().to_pylist())
        if not present & wanted_subsets:
            continue
        table = pq.read_table(path, columns=_COLUMNS, filesystem=fs)
        mask = pc.is_in(table.column("subset"), value_set=pa.array(sorted(wanted_subsets)))
        rows.extend(table.filter(mask).to_pylist())
        print(f"  read {path.split('/')[-1]}: kept {int(pc.sum(mask).as_py())} rows")
    return rows


_MARKUP = [("[PAR]", "\n"), ("[DOC]", ""), ("[TLE]", ""), ("[SEP]", " | ")]


def clean_context(text: str) -> str:
    for tag, rep in _MARKUP:
        text = text.replace(tag, rep)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*", "\n", text)
    return text.strip()


def _to_example(row: dict, short: str, split: str) -> dict:
    answers = [a.strip() for a in row["answers"] if a and a.strip()]
    return {
        "id": f"{short}:{row['qid']}",
        "source": short,
        "split": split,
        "context": clean_context(row["context"]),
        "question": row["question"].strip(),
        "answers": list(dict.fromkeys(answers)),  # dedupe, keep order
    }


# --------------------------------------------------------------------------- checks
def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def text_hash(text: str) -> str:
    return hashlib.md5(_norm(text).encode()).hexdigest()


def answer_in_context(ex: dict) -> bool:
    ctx = ex["context"].lower()
    return any(a.lower() in ctx for a in ex["answers"])


def _length_stats(values: list[int]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(q * len(v)))]  # noqa: E731
    return {"mean": sum(v) / len(v), "p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99), "max": v[-1]}


def validate(examples: list[dict]) -> dict:
    return {
        "n": len(examples),
        "empty_answers": sum(not e["answers"] for e in examples),
        "answer_in_context_rate": sum(answer_in_context(e) for e in examples) / max(1, len(examples)),
        "context_words": _length_stats([len(e["context"].split()) for e in examples]),
        "question_words": _length_stats([len(e["question"].split()) for e in examples]),
        "answer_words": _length_stats([len(e["answers"][0].split()) for e in examples if e["answers"]]),
    }


def overlap_report(train_sets: dict[str, list[dict]], eval_sets: dict[str, list[dict]]) -> dict:
    """Exact (normalised) context / question / pair overlap between train pools and eval sets."""
    train_ctx, train_q, train_pair = set(), set(), set()
    for exs in train_sets.values():
        for e in exs:
            ch, qh = text_hash(e["context"]), text_hash(e["question"])
            train_ctx.add(ch)
            train_q.add(qh)
            train_pair.add(ch + qh)
    report = {}
    for name, exs in eval_sets.items():
        ctx = sum(text_hash(e["context"]) in train_ctx for e in exs)
        q = sum(text_hash(e["question"]) in train_q for e in exs)
        pair = sum(text_hash(e["context"]) + text_hash(e["question"]) in train_pair for e in exs)
        report[name] = {"n": len(exs), "context_overlap": ctx, "question_overlap": q, "pair_overlap": pair}
    return report


# --------------------------------------------------------------------------- main entry
def prepare_data(cfg: dict) -> dict:
    dcfg = cfg["data"]
    out = ensure_dir(cfg["data_dir"])
    rng = random.Random(dcfg["seed"])
    repo = dcfg["hf_dataset"]

    train_srcs = list(dcfg["train_sources"])
    id_srcs = list(dcfg["id_eval_sources"])
    ood_srcs = list(dcfg["ood_eval_sources"])

    # ---- training pools + held-out dev (split by context so no passage leaks across) ----
    print("Loading MRQA train split ...")
    raw = _read_split(repo, "train", {SOURCES[s] for s in train_srcs})
    by_src: dict[str, list[dict]] = defaultdict(list)
    inv = {v: k for k, v in SOURCES.items()}
    for row in raw:
        by_src[inv[row["subset"]]].append(_to_example(row, inv[row["subset"]], "train"))
    del raw

    train_sets, dev_sets = {}, {}
    for src in train_srcs:
        exs = [e for e in by_src[src] if e["answers"]]
        groups: dict[str, list[dict]] = defaultdict(list)
        for e in exs:
            groups[text_hash(e["context"])].append(e)
        keys = sorted(groups)
        rng.shuffle(keys)
        dev, train, i = [], [], 0
        while len(dev) < dcfg["dev_per_source"] and i < len(keys):
            dev.extend(groups[keys[i]])
            i += 1
        for k in keys[i:]:
            train.extend(groups[k])
        rng.shuffle(train)
        dev = dev[: dcfg["dev_per_source"]]
        for e in dev:
            e["split"] = "dev"
        if dcfg.get("max_train_per_source"):
            train = train[: dcfg["max_train_per_source"]]
        train_sets[src], dev_sets[src] = train, dev

    # ---- evaluation sets ----
    eval_sets = {}
    for split, srcs in (("validation", id_srcs), ("test", ood_srcs)):
        if not srcs:
            continue
        print(f"Loading MRQA {split} split ...")
        rows = _read_split(repo, split, {SOURCES[s] for s in srcs})
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            short = inv[row["subset"]]
            grouped[short].append(_to_example(row, short, "test"))
        for src in srcs:
            exs = [e for e in grouped[src] if e["answers"]]
            exs.sort(key=lambda e: e["id"])
            rng.shuffle(exs)
            eval_sets[src] = exs

    # ---- overlap checks (and optional removal) before subsampling ----
    overlaps_before = overlap_report(train_sets, {**{f"dev_{k}": v for k, v in dev_sets.items()}, **eval_sets})
    if dcfg.get("drop_overlaps", True):
        train_ctx = {text_hash(e["context"]) for exs in train_sets.values() for e in exs}
        for src in eval_sets:
            eval_sets[src] = [e for e in eval_sets[src] if text_hash(e["context"]) not in train_ctx]
    for src in eval_sets:
        eval_sets[src] = eval_sets[src][: dcfg["eval_per_source"]]
    overlaps_after = overlap_report(train_sets, {**{f"dev_{k}": v for k, v in dev_sets.items()}, **eval_sets})

    # ---- few-shot demonstrations: short SQuAD training examples, fixed by seed ----
    demo_pool = [e for e in train_sets.get("squad", next(iter(train_sets.values())))
                 if len(e["context"].split()) <= dcfg.get("fewshot_max_context_words", 120) and answer_in_context(e)]
    fewshot = demo_pool[: dcfg.get("num_fewshot", 3)]
    fewshot_ids = {e["id"] for e in fewshot}
    for src in train_sets:
        train_sets[src] = [e for e in train_sets[src] if e["id"] not in fewshot_ids]

    # ---- write ----
    for src, exs in train_sets.items():
        write_jsonl(exs, out / f"train_{src}.jsonl")
    for src, exs in dev_sets.items():
        write_jsonl(exs, out / f"dev_{src}.jsonl")
    for src, exs in eval_sets.items():
        write_jsonl(exs, out / f"test_{src}.jsonl")
    write_jsonl(fewshot, out / "fewshot.jsonl")

    report = {
        "config": dcfg,
        "train": {k: validate(v) for k, v in train_sets.items()},
        "dev": {k: validate(v) for k, v in dev_sets.items()},
        "test": {k: validate(v) for k, v in eval_sets.items()},
        "overlap_before_filter": overlaps_before,
        "overlap_after_filter": overlaps_after,
        "fewshot_ids": sorted(fewshot_ids),
    }
    write_json(report, out / "data_report.json")
    print(f"Wrote data to {out}")
    return report


# --------------------------------------------------------------------------- access
def data_ready(cfg: dict) -> bool:
    return (Path(cfg["data_dir"]) / "data_report.json").exists()


def load_split(cfg: dict, split: str, source: str, limit: int | None = None) -> list[dict]:
    exs = read_jsonl(Path(cfg["data_dir"]) / f"{split}_{source}.jsonl")
    return exs[:limit] if limit else exs


def load_fewshot(cfg: dict) -> list[dict]:
    return read_jsonl(Path(cfg["data_dir"]) / "fewshot.jsonl")


def eval_sources(cfg: dict) -> list[str]:
    return list(cfg["data"]["id_eval_sources"]) + list(cfg["data"]["ood_eval_sources"])
