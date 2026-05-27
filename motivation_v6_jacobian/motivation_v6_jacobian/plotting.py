"""Plotting helpers for motivation_v6 figures.

All figures use matplotlib's default backend and are saved to
``outputs/figures/`` as both PNG (for the report) and PDF (for
paper-quality vector). Each public function takes data, not paths,
so the same renderers can be reused by scripts and notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_FIGSIZE = (6.4, 4.0)


def _save_both(fig, basepath: Path) -> None:
    basepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(basepath.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(basepath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def scatter_jacobian_vs_v4(
    jacobian_scores: Sequence[float],
    v4_scores: Sequence[float],
    spearman: float,
    save_to: Path,
) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.scatter(v4_scores, jacobian_scores, s=14, alpha=0.45, edgecolor="none")
    ax.set_xlabel("v4 final_sensitivity")
    ax.set_ylabel("jacobian score (|g·e|, sqrt-len normalised)")
    ax.set_title(f"Jacobian vs v4 finite-difference (global Spearman={spearman:.3f})")
    ax.grid(True, linestyle=":", alpha=0.5)
    _save_both(fig, save_to)


def bar_topk_enrichment(
    rows: List[Dict[str, float]],
    save_to: Path,
) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ks = [r["k"] for r in rows]
    enr = [r["enrichment"] for r in rows]
    ax.bar([str(k) for k in ks], enr,
           color="#4c72b0", edgecolor="black")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("k")
    ax.set_ylabel("observed / expected top-k overlap")
    ax.set_title("Top-k overlap enrichment (Jacobian ∩ v4-sensitivity)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    _save_both(fig, save_to)


def plot_spectrum(
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    save_to: Path,
    title: str,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, (S, expl, cum) in spectra.items():
        if len(cum) == 0:
            continue
        ax1.plot(np.arange(1, len(expl) + 1), expl, label=name, lw=1.6)
        ax2.plot(np.arange(1, len(cum) + 1), cum, label=name, lw=1.6)
    ax1.set_xlabel("component index")
    ax1.set_ylabel("explained variance fraction")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax1.legend()
    ax2.set_xlabel("component index")
    ax2.set_ylabel("cumulative explained variance")
    ax2.set_xscale("log")
    ax2.set_xticks([1, 4, 8, 16, 32, 64, 128, 256])
    ax2.set_xticklabels(["1", "4", "8", "16", "32", "64", "128", "256"])
    ax2.set_ylim(0, 1.02)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.axhline(0.7, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)
    ax2.legend()
    fig.suptitle(title)
    _save_both(fig, save_to)


def plot_soft_token_recovery(
    ks: Sequence[int],
    recoveries: Dict[str, Sequence[float]],
    save_to: Path,
) -> None:
    """recoveries: {label: per-case recovery list aligned with ks}.

    Plots median + IQR per k.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    xs = np.arange(len(ks))
    width = 0.7
    medians = []
    for k_idx in range(len(ks)):
        vals = []
        for series in recoveries.values():
            v = series[k_idx]
            if np.isfinite(v):
                vals.append(v)
        if vals:
            medians.append(np.median(vals))
        else:
            medians.append(np.nan)
    ax.bar(xs, medians, width=width, color="#55a868", edgecolor="black")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax.axhline(0.7, color="gray", linestyle=":", alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("soft tokens k")
    ax.set_ylabel("median gap-recovery (full↔no gap)")
    ax.set_title("Soft-token oracle: gap recovery vs k")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    _save_both(fig, save_to)


def plot_soft_token_loss_vs_k(
    ks: Sequence[int],
    per_method: Dict[str, Sequence[float]],
    save_to: Path,
) -> None:
    """per_method maps method label → median loss across cases."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    # soft-token line
    soft_label = "soft tokens"
    if soft_label in per_method:
        ax.plot(ks, per_method[soft_label], marker="o", color="#55a868",
                label=soft_label)
    for name, vals in per_method.items():
        if name == soft_label:
            continue
        if all(np.isfinite(v) for v in vals[: max(1, len(ks))]):
            ax.axhline(vals[0], linestyle="--", alpha=0.7, label=name)
    ax.set_xlabel("soft tokens k")
    ax.set_ylabel("median target NLL (nats)")
    ax.set_title("Soft-token oracle vs textual baselines")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    _save_both(fig, save_to)


__all__ = [
    "scatter_jacobian_vs_v4",
    "bar_topk_enrichment",
    "plot_spectrum",
    "plot_soft_token_recovery",
    "plot_soft_token_loss_vs_k",
]
