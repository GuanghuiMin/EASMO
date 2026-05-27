"""SVD / PCA over Jacobian-weighted activation vectors.

Per spec §6, an active vector is

    a_j = H_{L,j} ⊙ G_{L,j}      (per token)
    a_span = sum_{j ∈ span} a_j
    a_example = sum_{j ∈ context} a_j

We centre and run a randomised SVD on the resulting matrix, then
report the cumulative explained variance curve and the explained
variance at k ∈ {4, 8, 16, 32, 64}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.utils.extmath import randomized_svd


@dataclass
class SpectrumResult:
    matrix_name: str
    layer_index: int
    n_rows: int
    n_cols: int
    singular_values: np.ndarray
    explained: np.ndarray              # S^2 / sum(S^2)
    cumulative: np.ndarray             # cumsum

    def cumulative_at(self, k: int) -> float:
        if k <= 0:
            return 0.0
        if k > len(self.cumulative):
            return float(self.cumulative[-1])
        return float(self.cumulative[k - 1])


def spectrum(matrix: np.ndarray, *, n_components: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (singular_values, explained, cumulative) for a centred,
    randomised SVD of ``matrix`` (rows = samples)."""
    if matrix.size == 0:
        return np.array([]), np.array([]), np.array([])
    X = matrix - matrix.mean(axis=0, keepdims=True)
    k = min(n_components, min(X.shape) - 1)
    if k < 1:
        return np.array([]), np.array([]), np.array([])
    _, S, _ = randomized_svd(X, n_components=k, random_state=0)
    total = float((S ** 2).sum())
    if total <= 0:
        return S, np.zeros_like(S), np.zeros_like(S)
    explained = (S ** 2) / total
    cumulative = np.cumsum(explained)
    return S, explained, cumulative


def aggregate_example_vector(active_hidden: np.ndarray, active_grad: np.ndarray) -> np.ndarray:
    """Sum H ⊙ G over context tokens → (hidden_dim,)."""
    return (active_hidden * active_grad).sum(axis=0)


def aggregate_span_vector(
    active_hidden: np.ndarray,
    active_grad: np.ndarray,
    span_token_start: int,
    span_token_end: int,
) -> np.ndarray:
    """Sum H ⊙ G over the token range belonging to one span."""
    if span_token_end <= span_token_start:
        return np.zeros(active_hidden.shape[-1], dtype=active_hidden.dtype)
    return (
        active_hidden[span_token_start:span_token_end] *
        active_grad[span_token_start:span_token_end]
    ).sum(axis=0)


def compute_all_spectra(
    *,
    example_matrix: np.ndarray,
    span_matrix: np.ndarray,
    high_mask: np.ndarray,
    low_mask: np.ndarray,
    layer_index: int,
    n_components: int = 256,
) -> List[SpectrumResult]:
    """Run all four spectra requested by §6.5."""
    results: List[SpectrumResult] = []
    for name, M in [
        ("example", example_matrix),
        ("span", span_matrix),
        ("high_v4", span_matrix[high_mask] if span_matrix.size else span_matrix),
        ("low_v4", span_matrix[low_mask] if span_matrix.size else span_matrix),
    ]:
        S, expl, cum = spectrum(M, n_components=n_components)
        results.append(SpectrumResult(
            matrix_name=name,
            layer_index=layer_index,
            n_rows=int(M.shape[0]) if M.ndim == 2 else 0,
            n_cols=int(M.shape[1]) if M.ndim == 2 and M.shape[1] > 0 else 0,
            singular_values=S,
            explained=expl,
            cumulative=cum,
        ))
    return results


__all__ = [
    "SpectrumResult",
    "spectrum",
    "aggregate_example_vector",
    "aggregate_span_vector",
    "compute_all_spectra",
]
