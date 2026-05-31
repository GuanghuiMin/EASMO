"""Plot helpers for motivation_v9 (spec §9 figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_both(fig, basepath: Path) -> None:
    basepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(basepath.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(basepath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_best_of_n_pass_gain(df: pd.DataFrame, save_to: Path) -> None:
    """spec fig 1 — greedy vs best-of-N pass rate at C1 and CK,
    facet by compressor model."""
    if df.empty:
        return
    models = sorted(df["compressor_model"].unique())
    fig, axes = plt.subplots(1, max(1, len(models)),
                              figsize=(5 * max(1, len(models)), 4.0),
                              sharey=True)
    if len(models) == 1:
        axes = [axes]
    rounds = ["C1", "CK"]
    for ax, m in zip(axes, models):
        sub = df[df["compressor_model"] == m]
        xs = np.arange(len(rounds))
        width = 0.4
        greedy = [float(sub[sub["eval_context_round"] == r]["greedy_pass_rate"].iloc[0])
                  if (sub["eval_context_round"] == r).any() else 0.0 for r in rounds]
        best  = [float(sub[sub["eval_context_round"] == r]["best_of_n_pass_rate"].iloc[0])
                 if (sub["eval_context_round"] == r).any() else 0.0 for r in rounds]
        ax.bar(xs - width/2, greedy, width=width, label="greedy", color="#4c72b0")
        ax.bar(xs + width/2, best,   width=width, label="best-of-N", color="#dd8452")
        ax.set_xticks(xs); ax.set_xticklabels(rounds)
        ax.set_title(m); ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    axes[0].set_ylabel("AppWorld pass rate")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Best-of-N pass gain by eval round")
    _save_both(fig, save_to)


def fig_c1_ck_transition_matrix(transition: pd.DataFrame, save_to: Path) -> None:
    if transition.empty:
        return
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    cats = ["robust_pass", "fragile_pass", "stress_improved", "robust_fail"]
    counts = [int((transition["class"] == c).sum()) for c in cats]
    grid = np.array([[counts[0], counts[1]],   # C1=1 row
                     [counts[2], counts[3]]])  # C1=0 row
    im = ax.imshow(grid, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{grid[i,j]}", ha="center", va="center",
                    color="white" if grid[i, j] > grid.max() / 2 else "black",
                    fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["CK pass", "CK fail"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["C1 pass", "C1 fail"])
    ax.set_title("C1 vs CK transition matrix")
    fig.colorbar(im, ax=ax, label="n candidates")
    _save_both(fig, save_to)


def fig_c1_ck_pass_drop_by_model(fragility_df: pd.DataFrame, save_to: Path) -> None:
    if fragility_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    df = fragility_df.copy()
    df["label"] = df["compressor_model"].astype(str) + "/" + df["generation_type"].astype(str)
    xs = np.arange(len(df))
    width = 0.4
    ax.bar(xs - width/2, df["pass_rate_C1"], width=width, label="C1", color="#55a868")
    ax.bar(xs + width/2, df["pass_rate_CK"], width=width, label="CK", color="#c44e52")
    ax.set_xticks(xs); ax.set_xticklabels(df["label"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("AppWorld pass rate")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("C1 → CK pass-rate drop by (model, candidate type)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    _save_both(fig, save_to)


def fig_stress_pass_curve_by_round(df: pd.DataFrame, save_to: Path) -> None:
    """spec fig 3 — pass rate at each stress round r = 0..K.

    Expects df columns: compressor_model, generation_type, stress_round,
    pass_rate.
    """
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for keys, grp in df.groupby(["compressor_model", "generation_type"]):
        grp = grp.sort_values("stress_round")
        ax.plot(grp["stress_round"], grp["pass_rate"],
                marker="o", label=f"{keys[0]}/{keys[1]}")
    ax.set_xlabel("stress round (T^r)")
    ax.set_ylabel("AppWorld pass rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title("Pass rate over stress round")
    _save_both(fig, save_to)


def fig_chunk_advantage_by_type(df: pd.DataFrame, save_to: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    df = df.sort_values("mean_score_advantage", ascending=True)
    ax.barh(df["chunk_type"], df["mean_score_advantage"],
            color="#4c72b0", edgecolor="black")
    ax.axvline(0.0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("mean chunk score advantage  (score_full − score_minus_chunk)")
    ax.set_title("Chunk information advantage by type")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    _save_both(fig, save_to)


def fig_top_chunk_type_distribution(df: pd.DataFrame, save_to: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    df = df.sort_values("frac_top_advantage", ascending=False)
    ax.bar(df["chunk_type"], df["frac_top_advantage"],
            color="#55a868", edgecolor="black")
    ax.set_ylabel("fraction of chunks that are top-advantage (≥0.25)")
    ax.set_xticklabels(df["chunk_type"], rotation=30, ha="right", fontsize=8)
    ax.set_title("Top-advantage chunks by type")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    _save_both(fig, save_to)


__all__ = [
    "fig_best_of_n_pass_gain",
    "fig_c1_ck_transition_matrix",
    "fig_c1_ck_pass_drop_by_model",
    "fig_stress_pass_curve_by_round",
    "fig_chunk_advantage_by_type",
    "fig_top_chunk_type_distribution",
]
