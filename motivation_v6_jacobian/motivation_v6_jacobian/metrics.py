"""Correlation, top-k overlap, and enrichment metrics for v6."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr


def spearman_pearson(
    x: Sequence[float], y: Sequence[float]
) -> Dict[str, float]:
    """Return both correlations, with safe fallbacks when degenerate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.allclose(x.std(), 0) or np.allclose(y.std(), 0):
        return {
            "spearman_r": float("nan"), "spearman_p": float("nan"),
            "pearson_r": float("nan"), "pearson_p": float("nan"),
            "n": int(len(x)),
        }
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
    return {
        "spearman_r": float(sr) if sr is not None else float("nan"),
        "spearman_p": float(sp) if sp is not None else float("nan"),
        "pearson_r": float(pr) if pr is not None else float("nan"),
        "pearson_p": float(pp) if pp is not None else float("nan"),
        "n": int(len(x)),
    }


def topk_overlap(
    score_a: Sequence[float], score_b: Sequence[float], k: int
) -> Dict[str, float]:
    """Overlap between the top-k indices under each ranking.

    Reports observed overlap, expected overlap under random, and
    enrichment ratio (observed / expected).
    """
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    n = len(a)
    if n == 0 or k <= 0:
        return {"k": k, "n": n, "overlap": 0,
                "expected_random": 0.0, "enrichment": float("nan")}
    k = min(k, n)
    top_a = set(np.argsort(-a)[:k].tolist())
    top_b = set(np.argsort(-b)[:k].tolist())
    overlap = len(top_a & top_b)
    expected = (k * k) / n
    enrichment = overlap / expected if expected > 0 else float("nan")
    return {
        "k": int(k), "n": int(n),
        "overlap": int(overlap),
        "expected_random": float(expected),
        "enrichment": float(enrichment),
    }


def rank_of_top1(
    score_target: Sequence[float], score_other: Sequence[float]
) -> int:
    """1-indexed rank of the argmax of score_other under the ordering
    induced by score_target (higher = better)."""
    t = np.asarray(score_target, dtype=float)
    o = np.asarray(score_other, dtype=float)
    if len(t) == 0:
        return -1
    top_other_idx = int(np.argmax(o))
    order = np.argsort(-t)
    return int(np.where(order == top_other_idx)[0][0]) + 1


__all__ = ["spearman_pearson", "topk_overlap", "rank_of_top1"]
