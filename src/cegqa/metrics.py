"""Exact Match and token-level F1, following the official SQuAD / MRQA evaluation script."""
from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1(prediction: str, gold: str) -> float:
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def max_over_golds(fn, prediction: str, golds: list[str]) -> float:
    return max(fn(prediction, g) for g in golds) if golds else 0.0


def score(prediction: str, golds: list[str]) -> dict:
    return {"em": max_over_golds(exact_match, prediction, golds), "f1": max_over_golds(f1, prediction, golds)}


def aggregate(rows: list[dict]) -> dict:
    """Mean EM/F1 (in %) over rows that carry per-example `em` and `f1`."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "em": 0.0, "f1": 0.0}
    return {
        "n": n,
        "em": 100.0 * sum(r["em"] for r in rows) / n,
        "f1": 100.0 * sum(r["f1"] for r in rows) / n,
    }
