"""Plot helpers for motivation_v7 (spec §17)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_FIGSIZE = (8.0, 4.5)
_TYPE_ORDER = (
    "NARRATIVE_GOAL", "NARRATIVE_PROGRESS", "HIGH_LEVEL_REASONING",
    "PENDING_SUBTASK", "COMPLETED_SUBTASK", "ENVIRONMENT_STATE",
    "STALE_OR_OVERWRITTEN_STATE",
    "RUNTIME_VARIABLE", "AUTH_OR_ACCESS_TOKEN", "EXACT_IDENTIFIER",
    "FILE_PATH_OR_RESOURCE_LOCATOR", "API_SCHEMA_OR_PARAMETER",
    "ACTION_OUTCOME", "NUMERIC_OR_DATE_LITERAL",
    "NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT",
    "OTHER_CONCRETE_DETAIL",
)


def _save_both(fig, basepath: Path) -> None:
    basepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(basepath.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(basepath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_need_effect_by_fact_type(df: pd.DataFrame, save_to: Path) -> None:
    """Figure 1 — Δ_need per fact type, hue=model.

    Expects columns: fact_type, compressor_model, delta_need, ci_low, ci_high.
    """
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    models = sorted(df["compressor_model"].unique())
    types = [t for t in _TYPE_ORDER if t in set(df["fact_type"].unique())]
    width = 0.8 / max(1, len(models))
    xs = np.arange(len(types))
    for i, m in enumerate(models):
        sub = df[df["compressor_model"] == m].set_index("fact_type").reindex(types)
        bar_x = xs + (i - (len(models) - 1) / 2) * width
        yerr = np.stack([
            (sub["delta_need"] - sub["ci_low"]).fillna(0).clip(lower=0),
            (sub["ci_high"] - sub["delta_need"]).fillna(0).clip(lower=0),
        ])
        ax.bar(bar_x, sub["delta_need"], width=width, label=m,
               yerr=yerr, capsize=2, alpha=0.85)
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(types, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(r"$\Delta_{\mathrm{need}}$  ($P_{\mathrm{retain}|need=1} - P_{\mathrm{retain}|need=0}$)")
    ax.set_title("Need effect by fact type")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    _save_both(fig, save_to)


def fig_surface_dominance_index(df: pd.DataFrame, save_to: Path) -> None:
    """Figure 2 — SDI bar plot."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    df = df.copy()
    df["label"] = (df["compressor_model"].astype(str) + " / " +
                    df["prompt_variant"].astype(str))
    df = df.sort_values("sdi", ascending=True)
    ax.barh(df["label"], df["sdi"], color="#4c72b0", edgecolor="black")
    ax.axvline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("SDI =  (R²_type - R²_need) / (R²_type + R²_need)")
    ax.set_title("Surface dominance — fact type vs need-label")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    _save_both(fig, save_to)


def fig_preference_inversion_rate(df: pd.DataFrame, save_to: Path) -> None:
    """Figure 3 — PIR bar plot per (model, prompt, budget)."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    df = df.copy()
    df["label"] = (df["compressor_model"].astype(str) + " / " +
                    df["prompt_variant"].astype(str) + " / " +
                    df["budget_chars"].astype(str))
    df = df.sort_values("preference_inversion_rate", ascending=True)
    yerr = np.stack([
        (df["preference_inversion_rate"] - df["ci_low"]).clip(lower=0),
        (df["ci_high"] - df["preference_inversion_rate"]).clip(lower=0),
    ])
    ax.barh(df["label"], df["preference_inversion_rate"], xerr=yerr,
            color="#dd8452", edgecolor="black", capsize=4)
    ax.set_xlabel("Preference Inversion Rate (unneeded-narrative kept "
                  "while needed-concrete dropped)")
    ax.set_title("Preference inversion rate")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    _save_both(fig, save_to)


def fig_iterative_survival_curves(surv: pd.DataFrame, save_to: Path) -> None:
    """Figure 4 — line plot, x=round, y=survival, hue=fact_type, facet=model."""
    if surv.empty:
        return
    models = sorted(surv["compressor_model"].unique())
    fig, axes = plt.subplots(1, max(1, len(models)),
                              figsize=(6 * max(1, len(models)), 4.5),
                              sharey=True)
    if len(models) == 1:
        axes = [axes]
    type_set = [t for t in _TYPE_ORDER if t in set(surv["fact_type"].unique())]
    cmap = plt.get_cmap("tab20")
    for ax, m in zip(axes, models):
        sub = surv[surv["compressor_model"] == m]
        for i, t in enumerate(type_set):
            ts = sub[sub["fact_type"] == t].sort_values("round")
            if ts.empty:
                continue
            ax.plot(ts["round"], ts["survival_rate"],
                    color=cmap(i % 20), label=t, lw=1.2, marker="o")
        ax.set_title(m)
        ax.set_xlabel("compression round")
        ax.set_ylabel("survival rate")
        ax.set_ylim(0, 1.02)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
        ax.grid(True, linestyle=":", alpha=0.4)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.suptitle("Iterative compression: survival by fact type")
    _save_both(fig, save_to)


def fig_survival_hierarchy_heatmap(surv: pd.DataFrame, save_to: Path) -> None:
    """Figure 5 — heatmap of survival by (fact_type, round) per model."""
    if surv.empty:
        return
    models = sorted(surv["compressor_model"].unique())
    rounds = sorted(surv["round"].unique())
    type_set = [t for t in _TYPE_ORDER if t in set(surv["fact_type"].unique())]
    fig, axes = plt.subplots(1, max(1, len(models)),
                              figsize=(2.5 + 1.0 * len(rounds) * max(1, len(models)),
                                       0.4 * len(type_set) + 1.0),
                              squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, models):
        pivot = (surv[surv["compressor_model"] == m]
                 .pivot_table(index="fact_type", columns="round",
                              values="survival_rate", aggfunc="mean")
                 .reindex(type_set))
        im = ax.imshow(pivot.values, vmin=0, vmax=1, aspect="auto",
                       cmap="viridis")
        ax.set_yticks(range(len(type_set)))
        ax.set_yticklabels(type_set, fontsize=7)
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels(rounds)
        ax.set_xlabel("round")
        ax.set_title(m, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.7, label="survival rate")
    _save_both(fig, save_to)


def fig_cross_model_hierarchy_rank(ranks: pd.DataFrame, save_to: Path) -> None:
    """Figure 6 — slope plot of fact-type rank across models."""
    if ranks.empty:
        return
    models = sorted(ranks["compressor_model"].unique())
    if len(models) < 2:
        return
    types = ranks["fact_type"].unique()
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(types) + 1.0))
    cmap = plt.get_cmap("tab20")
    for i, t in enumerate(types):
        sub = ranks[ranks["fact_type"] == t]
        xs = [models.index(m) for m in sub["compressor_model"]]
        ys = list(sub["rank"])
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, marker="o", color=cmap(i % 20), label=t)
    ax.invert_yaxis()
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("rank (1 = longest half-life)")
    ax.set_title("Cross-model hierarchy rank")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    _save_both(fig, save_to)


def fig_fixed_point_recall(conv: pd.DataFrame, save_to: Path) -> None:
    """Figure 7 — needed/narrative/executable recall at convergence,
    grouped by (model, prompt)."""
    if conv.empty:
        return
    cols = [
        "needed_fact_recall_at_convergence",
        "narrative_fact_recall_at_convergence",
        "executable_fact_recall_at_convergence",
    ]
    have = [c for c in cols if c in conv.columns]
    if not have:
        return
    grp = (conv.groupby(["compressor_model", "prompt_variant"], dropna=False)[have]
              .mean().reset_index())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.8 / max(1, len(have))
    xs = np.arange(len(grp))
    for i, c in enumerate(have):
        offset = (i - (len(have) - 1) / 2) * width
        ax.bar(xs + offset, grp[c], width=width, label=c.replace("_at_convergence", ""))
    ax.set_xticks(xs)
    ax.set_xticklabels(grp["compressor_model"] + " / " + grp["prompt_variant"],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("recall at convergence")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fixed-point recall by fact group")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    _save_both(fig, save_to)


__all__ = [
    "fig_need_effect_by_fact_type",
    "fig_surface_dominance_index",
    "fig_preference_inversion_rate",
    "fig_iterative_survival_curves",
    "fig_survival_hierarchy_heatmap",
    "fig_cross_model_hierarchy_rank",
    "fig_fixed_point_recall",
]
