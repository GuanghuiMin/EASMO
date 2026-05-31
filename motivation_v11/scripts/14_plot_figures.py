"""Stage 12 — generate 5 required figures (spec §13).

Figures:
  fig_prompt_family_pass_c1_ck.{pdf,png}
  fig_distribution_quality_vs_calibration_gap.{pdf,png}  ★ main paper figure
  fig_pass_at_n_curve.{pdf,png}
  fig_serial_recompression_fragility.{pdf,png}
  fig_selector_recovery.{pdf,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from motivation_v11.data import ensure_outputs, table_path, figure_path  # noqa


def _safe_read(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _save(fig, name):
    fig.savefig(figure_path(name + ".pdf"), bbox_inches="tight")
    fig.savefig(figure_path(name + ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_prompt_family_pass(dqcg: pd.DataFrame):
    if dqcg.empty: return
    fams = dqcg["prompt_family"].tolist()
    x = np.arange(len(fams)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5*w, dqcg["greedy_pass_C1"], w, label="Greedy C1", color="#c0c4cc")
    ax.bar(x - 0.5*w, dqcg["greedy_pass_CK"], w, label="Greedy CK", color="#909399")
    ax.bar(x + 0.5*w, dqcg["Q_dist_C1"], w, label="Oracle Best-of-N C1", color="#67c23a")
    ax.bar(x + 1.5*w, dqcg["Q_dist_CK"], w, label="Oracle Best-of-N CK", color="#409eff")
    ax.set_xticks(x); ax.set_xticklabels(fams, rotation=20, ha="right")
    ax.set_ylabel("AppWorld Pass Rate"); ax.set_ylim(0, 1.0)
    ax.set_title("Prompt-family pass rate: greedy vs oracle best-of-N (C1/CK)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_prompt_family_pass_c1_ck")
    print("  wrote fig_prompt_family_pass_c1_ck.{pdf,png}")


def fig_dq_vs_cg(dqcg: pd.DataFrame):
    if dqcg.empty: return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(dqcg["Q_dist_CK"], dqcg["G_calib_CK"], s=200, c="#409eff", edgecolor="black")
    for _, r in dqcg.iterrows():
        ax.annotate(r["prompt_family"], (r["Q_dist_CK"], r["G_calib_CK"]),
                    xytext=(8, 8), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Q_dist (Best-of-N Pass@CK)")
    ax.set_ylabel("G_calib (Best-of-N - Greedy Pass@CK)")
    ax.set_title("Distribution quality vs decoding calibration gap (CK)")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="grey", linewidth=0.8)
    _save(fig, "fig_distribution_quality_vs_calibration_gap")
    print("  wrote fig_distribution_quality_vs_calibration_gap.{pdf,png}")


def fig_pass_at_n(curve: pd.DataFrame):
    if curve.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, rnd in zip(axes, ("C1", "CK")):
        sub = curve[curve["eval_round"] == rnd]
        for family, g in sub.groupby("prompt_family"):
            g = g.sort_values("N")
            ax.plot(g["N"], g["pass_at_N"], marker="o", label=family)
        ax.set_title(f"Pass@N — {rnd}"); ax.set_xlabel("N samples")
        ax.set_xscale("log", base=2); ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Pass@N")
    axes[0].legend(fontsize=8)
    _save(fig, "fig_pass_at_n_curve")
    print("  wrote fig_pass_at_n_curve.{pdf,png}")


def fig_fragility(stress: pd.DataFrame):
    if stress.empty: return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fams = sorted(stress["prompt_family"].unique())
    x = np.arange(len(fams))
    for gen, w in (("greedy", -0.2), ("sample", 0.2)):
        sub = stress[stress["selector"] == gen].sort_values("prompt_family")
        axes[0].bar(x + w, sub["delta_pass_C1_to_CK_pp"], 0.4, label=gen)
        axes[1].bar(x + w, sub["fragility_rate"], 0.4, label=gen)
        axes[2].bar(x + w, sub["length_drift_pct"], 0.4, label=gen)
    for ax, title in zip(axes, ("C1→CK pass Δ (pp)", "Fragility rate", "Length drift C1→CK")):
        ax.set_xticks(x); ax.set_xticklabels(fams, rotation=20, ha="right")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_serial_recompression_fragility")
    print("  wrote fig_serial_recompression_fragility.{pdf,png}")


def fig_selector_recovery(sel: pd.DataFrame):
    if sel.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, rnd in zip(axes, ("C1", "CK")):
        sub = sel[sel["eval_round"] == rnd]
        if sub.empty: continue
        sub = sub.sort_values(["selector", "prompt_family"])
        selectors = sub["selector"].unique().tolist()
        fams = sub["prompt_family"].unique().tolist()
        x = np.arange(len(selectors)); w = 0.8/max(len(fams), 1)
        for i, fam in enumerate(fams):
            f_sub = sub[sub["prompt_family"]==fam].set_index("selector").reindex(selectors)
            ax.bar(x + (i - len(fams)/2)*w, f_sub["oracle_recovery"].fillna(0).values,
                   w, label=fam)
        ax.set_xticks(x); ax.set_xticklabels(selectors, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Oracle recovery — {rnd}")
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="50%")
        ax.set_ylim(-0.5, 1.0); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Oracle recovery fraction")
    axes[0].legend(fontsize=7, loc="upper right")
    _save(fig, "fig_selector_recovery")
    print("  wrote fig_selector_recovery.{pdf,png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    ensure_outputs()

    dqcg = _safe_read(table_path("distribution_quality_calibration_gap.csv"))
    curve = _safe_read(table_path("pass_at_n_curve.csv"))
    stress = _safe_read(table_path("stress_invariance_by_prompt_selector.csv"))
    sel = _safe_read(table_path("selector_recovery_summary.csv"))

    print("[12] generating 5 figures ...")
    fig_prompt_family_pass(dqcg)
    fig_dq_vs_cg(dqcg)
    fig_pass_at_n(curve)
    fig_fragility(stress)
    fig_selector_recovery(sel)
    print("[12] all figures written.")


if __name__ == "__main__":
    main()
