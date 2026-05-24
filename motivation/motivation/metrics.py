"""Metrics used across M1 / M2 / M3.

Kept dependency-light (just numpy + scikit-learn) so they're cheap to
re-import from any driver script.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Action-distribution metrics
# ---------------------------------------------------------------------------


def action_match_rate(
    dist_a: Dict[str, float],
    dist_b: Dict[str, float],
) -> float:
    """Top-1 action match: 1.0 if argmax actions agree, else 0.0.

    Kept for backward compat / reporting. Coarse — collapses an entire
    16-sample distribution to a single argmax bit. Prefer
    :func:`action_overlap_rate` as the primary signal-vs-noise metric.

    Returns float for easy averaging across many probe states.
    """
    if not dist_a or not dist_b:
        return 0.0
    a = max(dist_a.items(), key=lambda kv: kv[1])[0]
    b = max(dist_b.items(), key=lambda kv: kv[1])[0]
    return 1.0 if a == b else 0.0


def action_overlap_rate(
    dist_a: Dict[str, float],
    dist_b: Dict[str, float],
) -> float:
    """Continuous behavioural overlap: ``1 - TV(dist_a, dist_b)``.

    Range [0, 1]. 1 = identical distributions, 0 = disjoint support.
    Smooth replacement for :func:`action_match_rate` when you need to
    detect *graded* policy similarity rather than a binary argmax hit.

    For agent-step / QA tasks both sides should already come from
    :func:`agents.action_distribution`, i.e. each is a multinomial
    over canonicalised actions summing to 1.0.

    Why this matters: with N=16 samples the argmax can flip on a
    single noisy pick (e.g. 8/16 vs 7/16 for a given top action), so
    binary action-match rate has very high variance and tends to
    return 0 even when the two distributions differ only mildly. TV
    is order-statistic free and varies smoothly, which is critical
    for the instance-noise hinge test (Path D) where signal rows are
    rare.
    """
    if not dist_a or not dist_b:
        return 0.0
    return 1.0 - total_variation(dist_a, dist_b)


def js_divergence(
    p: Dict[str, float],
    q: Dict[str, float],
    *,
    eps: float = 1e-6,
) -> float:
    """Jensen-Shannon divergence (symmetrised KL). Range [0, log 2]."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    norm_p = sum(p.values()) or 1.0
    norm_q = sum(q.values()) or 1.0
    m = {k: 0.5 * (p.get(k, 0.0) / norm_p + q.get(k, 0.0) / norm_q) for k in keys}
    pn = {k: p.get(k, 0.0) / norm_p for k in keys}
    qn = {k: q.get(k, 0.0) / norm_q for k in keys}
    # Reuse our smoothed kl_divergence so eps-handling is consistent.
    return 0.5 * kl_divergence(pn, m, eps=eps) + 0.5 * kl_divergence(qn, m, eps=eps)


def kl_divergence(
    p: Dict[str, float],
    q: Dict[str, float],
    *,
    eps: float = 1e-6,
) -> float:
    """Discrete KL(p || q). Smoothed with ``eps`` over the union of keys."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    norm_p = sum(p.values()) or 1.0
    norm_q = sum(q.values()) or 1.0
    total = 0.0
    for k in keys:
        pv = max(p.get(k, 0.0) / norm_p, eps)
        qv = max(q.get(k, 0.0) / norm_q, eps)
        total += pv * math.log(pv / qv)
    return total


def total_variation(
    p: Dict[str, float],
    q: Dict[str, float],
) -> float:
    """TV distance over the same key space."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


# ---------------------------------------------------------------------------
# Text / token overlap metrics
# ---------------------------------------------------------------------------


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def tokenize_simple(text: str) -> list[str]:
    """Whitespace + punctuation-stripped lowercased tokens."""
    import re
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def sentence_split(text: str) -> list[str]:
    if not text:
        return []
    # Simple sentence splitter — good enough for compressed-prompt overlap.
    import re
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def tfidf_cosine(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:                              # pragma: no cover
        return 0.0
    vec = TfidfVectorizer().fit_transform([a, b])
    return float(cosine_similarity(vec[0:1], vec[1:2])[0, 0])


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation; returns NaN on empty / constant input."""
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return float("nan")
    try:
        from scipy.stats import spearmanr
    except ImportError:                              # pragma: no cover
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho) if rho == rho else float("nan")


def linear_fit_r2(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(xs) == 0 or np.std(ys) == 0:
        return 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    ypred = slope * xs + intercept
    ss_res = np.sum((ys - ypred) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot else 0.0


# ---------------------------------------------------------------------------
# Sequence similarity
# ---------------------------------------------------------------------------


def edit_distance_norm(a: Sequence[str], b: Sequence[str]) -> float:
    """1 - normalized Levenshtein distance over two tool-call sequences.

    Higher = more similar.
    """
    if not a and not b:
        return 1.0
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return 1.0 - dp[n] / max(m, n)
