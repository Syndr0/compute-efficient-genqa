"""P3 - Error analysis over saved prediction files."""
from __future__ import annotations

from collections import Counter

from .metrics import normalize_answer

CATEGORIES = [
    "correct",          # EM = 1
    "partial",          # 0 < F1 < 1
    "verbose",          # prediction contains a gold answer but adds extra words (F1 < 1)
    "wrong_span",       # F1 = 0, prediction is copied from the passage
    "not_in_passage",   # F1 = 0, prediction does not occur in the passage (hallucination / paraphrase)
    "empty",            # no answer produced
    "answer_truncated", # gold answer not visible after context truncation (unanswerable as posed)
]


def categorize(row: dict) -> str:
    if row["em"] == 1.0:
        return "correct"
    if not row["prediction"].strip():
        return "empty"
    if not row.get("answer_in_prompt", True):
        return "answer_truncated"
    pred = normalize_answer(row["prediction"])
    if any(normalize_answer(g) and normalize_answer(g) in pred for g in row["answers"]):
        return "verbose"
    if row["f1"] > 0:
        return "partial"
    return "wrong_span" if row.get("pred_in_context") else "not_in_passage"


_WH = ["what", "which", "who", "whom", "whose", "when", "where", "why", "how many", "how much", "how"]


def question_type(q: str) -> str:
    ql = q.lower()
    for w in _WH:
        if ql.startswith(w) or f" {w} " in f" {ql[:40]} ":
            return w
    if ql.split()[:1] and ql.split()[0] in {"is", "are", "was", "were", "do", "does", "did", "can", "could"}:
        return "yes/no"
    return "other"


def error_breakdown(rows: list[dict]) -> dict:
    cats = Counter(categorize(r) for r in rows)
    n = max(1, len(rows))
    by_type: dict[str, list[float]] = {}
    for r in rows:
        by_type.setdefault(question_type(r["question"]), []).append(r["f1"])
    pred_len = [len(r["prediction"].split()) for r in rows]
    gold_len = [min(len(a.split()) for a in r["answers"]) for r in rows]
    return {
        "categories": {c: cats.get(c, 0) / n for c in CATEGORIES},
        "f1_by_question_type": {k: {"n": len(v), "f1": 100 * sum(v) / len(v)} for k, v in sorted(by_type.items())},
        "mean_pred_words": sum(pred_len) / n,
        "mean_gold_words": sum(gold_len) / n,
    }


def disagreement_examples(rows_a: list[dict], rows_b: list[dict], k: int = 10) -> list[dict]:
    """Items where system B is right (F1=1) and A is wrong (F1=0), and vice versa."""
    out = []
    for a, b in zip(rows_a, rows_b):
        if a["f1"] != b["f1"] and {a["f1"], b["f1"]} == {0.0, 1.0}:
            out.append({"id": a["id"], "question": a["question"], "answers": a["answers"][:3],
                        "pred_a": a["prediction"], "pred_b": b["prediction"],
                        "winner": "b" if b["f1"] > a["f1"] else "a"})
    wins_b = [o for o in out if o["winner"] == "b"][:k]
    wins_a = [o for o in out if o["winner"] == "a"][:k]
    return wins_b + wins_a
