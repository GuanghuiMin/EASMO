"""Sentence-BERT helpers — used to (a) cluster N sampled actions into
semantically equivalent buckets so the action distribution isn't broken
by trivial paraphrasing, and (b) compute semantic-overlap between two
oracle memories so M2 doesn't have to rely on lexical Jaccard alone.

A single global ``SentenceTransformer`` is lazy-loaded the first time
any helper is called. Default checkpoint is the tiny but strong
``all-MiniLM-L6-v2`` (90 MB, 22M params, dim=384). Override via the
``SBERT_MODEL`` env var.
"""

from __future__ import annotations

import os
import threading
from typing import Iterable, List

import numpy as np

from .utils import setup_logging

_logger = setup_logging("motivation.semantic")

_DEFAULT_MODEL = os.environ.get("SBERT_MODEL", "all-MiniLM-L6-v2")

_lock = threading.Lock()
_model = None


def _get_model():
    """Lazy-load the SBERT model. Single instance shared across threads."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _logger.info("Loading SBERT model %r", _DEFAULT_MODEL)
            _model = SentenceTransformer(_DEFAULT_MODEL)
    return _model


def embed(texts: List[str]) -> np.ndarray:
    """Return an (N, D) numpy array of L2-normalised sentence embeddings."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    m = _get_model()
    vecs = m.encode(
        texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Semantic clustering of N sampled actions → buckets → distribution
# ---------------------------------------------------------------------------


def cluster_by_similarity(
    texts: List[str], *, sim_threshold: float = 0.80,
) -> List[int]:
    """Greedy clustering: walk texts in order, assign each to the first
    existing cluster whose centroid is within ``sim_threshold`` cosine
    similarity, else start a new cluster.

    Returns a list of cluster-ids (ints), one per input text.
    """
    if not texts:
        return []
    vecs = embed(texts)
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    cluster_ids: list[int] = []
    for v in vecs:
        if not centroids:
            centroids.append(v.copy())
            counts.append(1)
            cluster_ids.append(0)
            continue
        sims = np.array([float(np.dot(v, c)) for c in centroids])
        best = int(np.argmax(sims))
        if sims[best] >= sim_threshold:
            # Incremental centroid update (online mean, then re-normalise).
            counts[best] += 1
            n = counts[best]
            new_centroid = centroids[best] + (v - centroids[best]) / n
            norm = float(np.linalg.norm(new_centroid))
            centroids[best] = new_centroid / norm if norm > 0 else new_centroid
            cluster_ids.append(best)
        else:
            centroids.append(v.copy())
            counts.append(1)
            cluster_ids.append(len(centroids) - 1)
    return cluster_ids


def cluster_representatives(
    texts: List[str], cluster_ids: List[int]
) -> dict[int, str]:
    """For each cluster, pick the first-seen text as the representative
    (used as a stable hashable key for distribution computation)."""
    out: dict[int, str] = {}
    for cid, t in zip(cluster_ids, texts):
        if cid not in out:
            out[cid] = t
    return out


def semantic_action_distribution(
    actions: List[str], *, sim_threshold: float = 0.80,
) -> dict[str, float]:
    """Replacement for :func:`agents.action_distribution` that buckets
    semantically equivalent answers together before counting frequencies.
    Returns a {representative_text: prob} dict that's still hashable as
    discrete distribution support keys.
    """
    if not actions:
        return {}
    canon = [a.strip() for a in actions if a and a.strip()]
    if not canon:
        return {}
    cids = cluster_by_similarity(canon, sim_threshold=sim_threshold)
    reps = cluster_representatives(canon, cids)
    counts: dict[str, int] = {}
    for cid in cids:
        rep = reps[cid]
        counts[rep] = counts.get(rep, 0) + 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Memory-text overlap
# ---------------------------------------------------------------------------


def semantic_overlap(text_a: str, text_b: str) -> float:
    """Cosine similarity of mean-pooled SBERT embeddings of two strings.

    Returns 0.0 if either is empty.
    """
    if not text_a or not text_b:
        return 0.0
    vecs = embed([text_a, text_b])
    if vecs.shape[0] < 2:
        return 0.0
    return float(np.dot(vecs[0], vecs[1]))


def precompute_embeddings(texts: Iterable[str]) -> dict[str, np.ndarray]:
    """Embed many texts at once and return a {text: vec} dict.
    Useful so a downstream loop can do cheap O(1) lookups."""
    seen = []
    seen_set: set[str] = set()
    for t in texts:
        if t and t not in seen_set:
            seen.append(t)
            seen_set.add(t)
    if not seen:
        return {}
    vecs = embed(seen)
    return {t: v for t, v in zip(seen, vecs)}
