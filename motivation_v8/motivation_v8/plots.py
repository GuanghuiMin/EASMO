"""Plotting helpers for motivation_v8 (spec §14)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    if df.empty:
        return
    families = sorted(df["prompt_family"].unique())
    fig, axes = plt.subplots(1, max(1, len(families)),
                              figsize=(7 * max(1, len(families)), 4.5),
                              sharey=True)
    if len(families) == 1:
        axes = [axes]
    types = [t for t in _TYPE_ORDER if t in set(df["fact_type"].unique())]
    models = sorted(df["model"].unique())
    width = 0.8 / max(1, len(models))
    xs = np.arange(len(types))
    for ax, fam in zip(axes, families):
        sub_fam = df[df["prompt_family"] == fam]
        for i, m in enumerate(models):
            sub = sub_fam[sub_fam["model"] == m].set_index("fact_type").reindex(types)
            bar_x = xs + (i - (len(models) - 1) / 2) * width
            yerr = np.stack([
                (sub["delta_need"] - sub["ci_low"]).fillna(0).clip(lower=0),
                (sub["ci_high"] - sub["delta_need"]).fillna(0).clip(lower=0),
            ])
            ax.bar(bar_x, sub["delta_need"], width=width, label=m,
                   yerr=yerr, capsize=2, alpha=0.85)
        ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(types, rotation=60, ha="right", fontsize=7)
        ax.set_title(fam)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    axes[0].set_ylabel(r"$\Delta_{\mathrm{need}}$")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Single-round Δ_need by fact type, per prompt family / model")
    _save_both(fig, save_to)


def fig_surface_dominance_index(df: pd.DataFrame, save_to: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    df = df.copy()
    df["label"] = df["model"].astype(str) + " / " + df["prompt_family"].astype(str)
    df = df.sort_values("sdi")
    ax.barh(df["label"], df["sdi"], color="#4c72b0", edgecolor="black")
    ax.axvline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("SDI = (R²_type - R²_need) / (R²_type + R²_need)")
    ax.set_title("Surface Dominance — general prompts (v8)")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    _save_both(fig, save_to)


def fig_preference_inversion_rate(df: pd.DataFrame, save_to: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    df = df.copy()
    df["label"] = df["model"].astype(str) + " / " + df["prompt_family"].astype(str)
    df = df.sort_values("preference_inversion_rate")
    yerr = np.stack([
        (df["preference_inversion_rate"] - df["ci_low"]).clip(lower=0),
        (df["ci_high"] - df["preference_inversion_rate"]).clip(lower=0),
    ])
    ax.barh(df["label"], df["preference_inversion_rate"], xerr=yerr,
            color="#dd8452", edgecolor="black", capsize=4)
    ax.set_xlabel("Preference Inversion Rate")
    ax.set_title("PIR — unneeded-narrative retained while needed-concrete dropped")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    _save_both(fig, save_to)


def fig_iterative_survival_curves(surv: pd.DataFrame, save_to: Path) -> None:
    if surv.empty:
        return
    # Facet by (model, prompt_family); RAW_FULL + needed only for clarity
    sub = surv[(surv["init_type"] == "RAW_FULL") &
                (surv["condition_type"].isin(["needed", "task_agnostic"]))]
    if sub.empty:
        sub = surv
    panels = sorted(sub.groupby(["model", "prompt_family"]).groups.keys())
    n = max(1, len(panels))
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    type_set = [t for t in _TYPE_ORDER if t in set(sub["fact_type"].unique())]
    cmap = plt.get_cmap("tab20")
    for ax, key in zip(axes, panels):
        m, fam = key
        s = sub[(sub["model"] == m) & (sub["prompt_family"] == fam)]
        for i, t in enumerate(type_set):
            ts = s[s["fact_type"] == t].sort_values("round")
            if ts.empty:
                continue
            ax.plot(ts["round"], ts["survival_rate"],
                    color=cmap(i % 20), label=t, lw=1.2, marker="o")
        ax.set_title(f"{m}\n{fam}", fontsize=9)
        ax.set_xlabel("round")
        ax.set_ylim(0, 1.02)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
        ax.grid(True, linestyle=":", alpha=0.4)
    axes[0].set_ylabel("survival rate")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)
    fig.suptitle("Iterative survival under general prompts (RAW_FULL init)")
    _save_both(fig, save_to)


def fig_fixed_point_composition(comp: pd.DataFrame, save_to: Path) -> None:
    if comp.empty:
        return
    # Aggregate over init_type / condition_type for headline
    agg = (comp.groupby(["model", "prompt_family", "fact_type"])["survival_rate_fixed"]
                .mean().reset_index())
    panels = sorted(agg.groupby(["model", "prompt_family"]).groups.keys())
    types = [t for t in _TYPE_ORDER if t in set(agg["fact_type"].unique())]
    n = max(1, len(panels))
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab20")
    for ax, key in zip(axes, panels):
        m, fam = key
        sub = agg[(agg["model"] == m) & (agg["prompt_family"] == fam)].set_index("fact_type").reindex(types)
        ax.bar(range(len(types)), sub["survival_rate_fixed"].fillna(0),
               color=[cmap(i % 20) for i in range(len(types))])
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(types, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{m}\n{fam}", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    axes[0].set_ylabel("retention at fixed point")
    fig.suptitle("Fixed-point retention by fact type (v8)")
    _save_both(fig, save_to)


def fig_fixed_point_need_shift(df: pd.DataFrame, save_to: Path) -> None:
    if df.empty:
        return
    families = sorted(df["prompt_family"].unique())
    fig, axes = plt.subplots(1, max(1, len(families)),
                              figsize=(7 * max(1, len(families)), 4.5),
                              sharey=True)
    if len(families) == 1:
        axes = [axes]
    types = [t for t in _TYPE_ORDER if t in set(df["fact_type"].unique())]
    models = sorted(df["model"].unique())
    width = 0.8 / max(1, len(models))
    xs = np.arange(len(types))
    for ax, fam in zip(axes, families):
        sub_fam = df[df["prompt_family"] == fam]
        for i, m in enumerate(models):
            sub = sub_fam[sub_fam["model"] == m].set_index("fact_type").reindex(types)
            bar_x = xs + (i - (len(models) - 1) / 2) * width
            ax.bar(bar_x, sub["delta_need_infty"], width=width, label=m, alpha=0.85)
        ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(types, rotation=60, ha="right", fontsize=7)
        ax.set_title(fam)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    axes[0].set_ylabel(r"$\Delta_{\mathrm{need}}^{\infty}$")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Fixed-point need shift by fact type")
    _save_both(fig, save_to)


def fig_basin_contraction(df: pd.DataFrame, save_to: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    # boxplot of contraction ratios per (model, prompt_family) — only fact-Jaccard
    df = df.copy()
    df["label"] = df["model"].astype(str) + "/" + df["prompt_family"].astype(str)
    groups = df.groupby("label")["contraction_fact_jaccard"].apply(list).to_dict()
    labels = sorted(groups.keys())
    data = [groups[l] for l in labels]
    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#55a868")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="no contraction")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="attractor threshold")
    ax.set_ylabel("contraction ratio  =  final / initial  fact-Jaccard distance")
    ax.set_title("Basin contraction across initialisations")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    _save_both(fig, save_to)


def fig_fixed_point_recall_groups(conv: pd.DataFrame, save_to: Path) -> None:
    if conv.empty:
        return
    cols = ["needed_fact_recall_fixed",
            "narrative_fact_recall_fixed",
            "executable_fact_recall_fixed",
            "control_fact_recall_fixed"]
    have = [c for c in cols if c in conv.columns]
    if not have:
        return
    grp = (conv.groupby(["model", "prompt_family"], dropna=False)[have]
              .mean().reset_index())
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    width = 0.8 / len(have)
    xs = np.arange(len(grp))
    for i, c in enumerate(have):
        offset = (i - (len(have) - 1) / 2) * width
        label = c.replace("_fact_recall_fixed", "").replace("_recall_fixed", "")
        ax.bar(xs + offset, grp[c], width=width, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels(grp["model"].astype(str) + "\n" + grp["prompt_family"].astype(str),
                       rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("recall at fixed point")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fixed-point recall by fact group")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    _save_both(fig, save_to)


def fig_cross_model_prompt_hierarchy_rank(ranks: pd.DataFrame, save_to: Path) -> None:
    if ranks.empty:
        return
    # Build a slope-plot across (model, prompt_family) for RAW_FULL/needed only
    sub = ranks
    if "init_type" in ranks.columns:
        sub = ranks[ranks["init_type"] == "RAW_FULL"]
    if "condition_type" in sub.columns:
        sub = sub[sub["condition_type"].isin(["needed", "task_agnostic"])]
    sub = sub.copy()
    sub["panel"] = sub["model"].astype(str) + "/" + sub["prompt_family"].astype(str)
    panels = sorted(sub["panel"].unique())
    if len(panels) < 2:
        return
    types = sub["fact_type"].unique()
    fig, ax = plt.subplots(figsize=(8.0, 0.35 * len(types) + 1.0))
    cmap = plt.get_cmap("tab20")
    for i, t in enumerate(types):
        sub_t = sub[sub["fact_type"] == t]
        xs = [panels.index(p) for p in sub_t["panel"]]
        ys = list(sub_t["rank"])
        if len(xs) < 2:
            continue
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o",
                color=cmap(i % 20), label=t)
    ax.invert_yaxis()
    ax.set_xticks(range(len(panels)))
    ax.set_xticklabels(panels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("rank (1 = longest half-life)")
    ax.set_title("Cross model / prompt-family hierarchy rank (RAW_FULL)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    _save_both(fig, save_to)


__all__ = [
    "fig_need_effect_by_fact_type",
    "fig_surface_dominance_index",
    "fig_preference_inversion_rate",
    "fig_iterative_survival_curves",
    "fig_fixed_point_composition",
    "fig_fixed_point_need_shift",
    "fig_basin_contraction",
    "fig_fixed_point_recall_groups",
    "fig_cross_model_prompt_hierarchy_rank",
]
