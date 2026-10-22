"""Bootstrap confidence intervals and paired bootstrap tests over per-example scores."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap CI of the mean (values are per-example scores in [0, 1])."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": 100 * x.mean(), "lo": 100 * lo, "hi": 100 * hi}


def paired_bootstrap(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired bootstrap for mean(b) - mean(a) over the same items.

    Returns the observed difference, its percentile CI, and a one-sided p-value for
    H0: b is not better than a (fraction of resamples with diff <= 0), plus the
    two-sided version.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired bootstrap requires aligned per-example scores"
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    diffs = d[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p_one = float((diffs <= 0).mean())
    p_two = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    return {
        "diff": 100 * d.mean(),
        "lo": 100 * lo,
        "hi": 100 * hi,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "n": int(len(d)),
    }


def mean_std(values) -> dict:
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "n": int(len(x))}
