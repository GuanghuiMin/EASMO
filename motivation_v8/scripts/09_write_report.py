"""Stage 09 — write the deterministic Markdown report (spec §18).

No LLM call; the report is fully derived from the CSV tables.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v8.data import (  # noqa: E402
    ensure_outputs, table_path, raw_path, read_jsonl, PROVENANCE,
)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        return f"{float(v):.3f}"
    except Exception:
        return str(v)


def _csv(name: str) -> pd.DataFrame:
    p = table_path(name)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_REPO / "outputs/reports/results_summary.md"))
    args = ap.parse_args()
    ensure_outputs()

    # Inputs
    ne   = _csv("need_effect_by_type.csv")
    sdr  = _csv("surface_dominance_regression.csv")
    pir  = _csv("preference_inversion.csv")
    half = _csv("fact_type_half_life.csv")
    ausc = _csv("ausc_by_type.csv")
    sim  = _csv("cross_model_prompt_hierarchy_similarity.csv")
    conv = _csv("convergence_by_chain.csv")
    comp = _csv("fixed_point_composition_by_type.csv")
    shift= _csv("fixed_point_need_shift.csv")
    basin= _csv("basin_similarity.csv")
    bc_s = _csv("budget_compliance_single_round.csv")
    bc_i = _csv("budget_compliance_iterative.csv")

    # counts
    n_cases = len(read_jsonl(_REPO / "data" / "cases.jsonl"))
    n_facts = len(read_jsonl(_REPO / "data" / "fact_bank_filtered.jsonl"))
    n_conds = len(read_jsonl(_REPO / "data" / "need_conditions_validated.jsonl"))
    n_single = (len(read_jsonl(raw_path("single_round_compressions.jsonl")))
                if raw_path("single_round_compressions.jsonl").exists() else 0)
    n_iter = (len(read_jsonl(raw_path("iterative_chains.jsonl")))
              if raw_path("iterative_chains.jsonl").exists() else 0)
    n_ret = (len(read_jsonl(raw_path("retention_scores.jsonl")))
             if raw_path("retention_scores.jsonl").exists() else 0)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")

    # Verdict heuristics (spec §17)
    def verdict_a() -> str:
        if sdr.empty: return "INCONCLUSIVE"
        flags = 0
        ta = sdr[sdr["prompt_family"] == "general_task_aware"]
        if not ta.empty and (ta["r2_need"] < 0.02).any() and (ta["r2_type"] >= 5 * ta["r2_need"]).any():
            flags += 1
        if (sdr["sdi"] > 0.5).any(): flags += 1
        if not ne.empty:
            exec_ne = ne[ne["coarse_group"] == "EXECUTABLE"]
            if not exec_ne.empty and exec_ne["delta_need"].mean(skipna=True) < 0.15:
                flags += 1
            concrete = ne[ne["coarse_group"].isin(["EXECUTABLE", "CONTROL"])]
            if not concrete.empty and (concrete["delta_need"].abs() <= 0.05).sum() >= 2:
                flags += 1
        if not pir.empty and (pir["preference_inversion_rate"] > 0.20).any():
            flags += 1
        return ("STRONG POSITIVE" if flags >= 3 else
                "PARTIAL POSITIVE" if flags >= 2 else "WEAK / NEGATIVE")

    def verdict_bc() -> str:
        if conv.empty or half.empty: return "INCONCLUSIVE"
        flags = 0
        if conv["converged"].mean() >= 0.60: flags += 1
        narr = half[half["coarse_group"] == "NARRATIVE"]["half_life"]
        exec_ = half[half["coarse_group"] == "EXECUTABLE"]["half_life"]
        if narr.size and exec_.size and narr.mean() > exec_.mean(): flags += 1
        if not comp.empty:
            narr_r = comp[comp["coarse_group"] == "NARRATIVE"]["survival_rate_fixed"].mean(skipna=True)
            exec_r = comp[comp["coarse_group"] == "EXECUTABLE"]["survival_rate_fixed"].mean(skipna=True)
            if pd.notna(narr_r) and pd.notna(exec_r) and (narr_r - exec_r) > 0.10: flags += 1
        if not ausc.empty:
            for m in ausc["model"].unique():
                sub = ausc[ausc["model"] == m].sort_values("ausc")
                bottom3 = set(sub.head(3)["fact_type"])
                if "AUTH_OR_ACCESS_TOKEN" in bottom3 or "API_SCHEMA_OR_PARAMETER" in bottom3:
                    flags += 1
                    break
        if not sim.empty and (sim["kendall_tau"] > 0.35).any(): flags += 1
        return ("STRONG POSITIVE" if flags >= 3 else
                "PARTIAL POSITIVE" if flags >= 2 else "WEAK / NEGATIVE")

    def verdict_d() -> str:
        if shift.empty: return "INCONCLUSIVE"
        flags = 0
        ta = shift[shift["prompt_family"] == "general_task_aware"]
        if not ta.empty:
            exec_shift = ta[ta["coarse_group"] == "EXECUTABLE"]
            if not exec_shift.empty and (exec_shift["delta_need_infty"].abs() < 0.15).all():
                flags += 1
        if not comp.empty:
            narr_r = comp[comp["coarse_group"] == "NARRATIVE"]["survival_rate_fixed"].mean(skipna=True)
            exec_r = comp[comp["coarse_group"] == "EXECUTABLE"]["survival_rate_fixed"].mean(skipna=True)
            ds = shift[shift["coarse_group"] == "EXECUTABLE"]["delta_need_infty"].abs().max(skipna=True)
            if pd.notna(ds) and pd.notna(narr_r - exec_r) and (narr_r - exec_r) > ds:
                flags += 1
        return ("STRONG POSITIVE" if flags >= 2 else
                "PARTIAL POSITIVE" if flags >= 1 else "WEAK / NEGATIVE")

    def verdict_e() -> str:
        if basin.empty: return "INCONCLUSIVE"
        flags = 0
        if basin["contraction_fact_jaccard"].mean(skipna=True) < 0.5: flags += 1
        # final - initial jaccard similarity (i.e., 1 - distance) — gap >= 0.2
        init_sim = (1.0 - basin["init_fact_jaccard_distance"]).mean(skipna=True)
        fin_sim  = (1.0 - basin["fin_fact_jaccard_distance"]).mean(skipna=True)
        if pd.notna(fin_sim) and pd.notna(init_sim) and (fin_sim - init_sim) >= 0.2:
            flags += 1
        return ("STRONG POSITIVE" if flags >= 2 else
                "PARTIAL POSITIVE" if flags >= 1 else "WEAK / NEGATIVE")

    lines: list = []
    lines.append("# motivation_v8 Results — Fixed Points of General LLM Compression")
    lines.append("")
    lines.append(f"> Auto-written by `scripts/09_write_report.py` at {ts}.")
    lines.append("")
    lines.append("## 1. Setup")
    lines.append(f"- `n_cases` = **{n_cases}** (reused from motivation_v7)")
    lines.append(f"- `n_facts` (filtered, substring-grounded) = **{n_facts}**")
    lines.append(f"- `n_conditions` (quality-passed pairs) = **{n_conds}**")
    lines.append(f"- `n_compressions` (single round) = **{n_single}**")
    lines.append(f"- `n_compressions` (iterative + basin) = **{n_iter}**")
    lines.append(f"- `n_retention_scores` = **{n_ret}**")
    if not bc_s.empty:
        v = bc_s["violation_rate"].mean(skipna=True)
        lines.append(f"- Single-round budget violation rate (mean across (model, prompt)) = {_fmt(v)}")
    if not bc_i.empty:
        v = bc_i["violation_rate"].mean(skipna=True)
        lines.append(f"- Iterative budget violation rate (mean) = {_fmt(v)}")
    lines.append("- Compressors: `qwen3-4b-instruct-2507`, `MiniMaxAI/MiniMax-M2.5`")
    lines.append("- Prompt families: `general_task_aware` (P1), `general_task_agnostic` (P2)")
    lines.append("- Budget: 1500 chars (primary)")
    lines.append("- ACON prompts: NOT USED in v8 (spec §1).")
    lines.append("")

    lines.append("## 2. Claim A: Single-Round Need Conditioning")
    lines.append("")
    lines.append(f"**Verdict A: {verdict_a()}**")
    lines.append("")
    if not sdr.empty:
        lines.append("Surface-dominance regression:")
        lines.append("")
        lines.append(sdr[["model", "prompt_family", "budget_chars", "n",
                          "r2_need", "r2_type", "r2_both", "sdi"]]
                     .round(3).to_markdown(index=False))
        lines.append("")
    if not pir.empty:
        lines.append("Preference Inversion Rate:")
        lines.append("")
        lines.append(pir.round(3).to_markdown(index=False))
        lines.append("")
    lines.append("Figures: `fig_need_effect_by_fact_type`, "
                 "`fig_surface_dominance_index`, `fig_preference_inversion_rate`.")
    lines.append("")

    lines.append("## 3. Claim B: Fixed-Point Convergence")
    lines.append("")
    if not conv.empty:
        rate = float(conv["converged"].mean())
        lines.append(f"Convergence rate (text sim ≥ 0.95 ∧ fact Jaccard ≥ 0.95 "
                     f"∧ |Δlen|/len ≤ 0.02): **{rate:.1%}** of {len(conv)} chains.")
        agg = conv.groupby(["model", "prompt_family"])["converged"].mean().round(3)
        lines.append("")
        lines.append("Convergence rate by (model, prompt):")
        lines.append("")
        lines.append(agg.to_frame("converged_rate").to_markdown())
        lines.append("")
    lines.append("")

    lines.append("## 4. Claim C: Fixed-Point Composition")
    lines.append("")
    if not comp.empty:
        agg = (comp.groupby(["model", "prompt_family", "coarse_group"])[
            "survival_rate_fixed"].mean().reset_index())
        lines.append("Mean retention at fixed point by coarse group:")
        lines.append("")
        lines.append(agg.round(3).to_markdown(index=False))
        lines.append("")
    if not ausc.empty:
        lines.append("Bottom-3 fact types by AUSC per (model, prompt_family):")
        lines.append("")
        for keys, grp in ausc.groupby(["model", "prompt_family"]):
            sub = grp.sort_values("ausc").head(3)
            bot = ", ".join(f"{r.fact_type} ({r.ausc:.2f})" for r in sub.itertuples())
            lines.append(f"- {keys[0]} / {keys[1]}: {bot}")
        lines.append("")
    lines.append("")

    lines.append("## 5. Claim D: Need-Conditioned Fixed-Point Shift")
    lines.append("")
    lines.append(f"**Verdict D: {verdict_d()}**")
    lines.append("")
    if not shift.empty:
        agg = (shift.groupby(["model", "prompt_family", "coarse_group"])
                    ["delta_need_infty"].mean().round(3).reset_index())
        lines.append("Mean Δ_need^∞ by (model, prompt, coarse_group):")
        lines.append("")
        lines.append(agg.to_markdown(index=False))
        lines.append("")
    lines.append("")

    lines.append("## 6. Claim E: Basin of Attraction")
    lines.append("")
    lines.append(f"**Verdict E: {verdict_e()}**")
    lines.append("")
    if not basin.empty:
        agg = (basin.groupby(["model", "prompt_family"])[[
            "init_fact_jaccard_distance",
            "fin_fact_jaccard_distance",
            "contraction_fact_jaccard",
            "init_type_l1_distance",
            "fin_type_l1_distance",
            "contraction_type_l1",
        ]].mean().round(3).reset_index())
        lines.append("Mean basin metrics (init → final pairwise distance, contraction = final/init):")
        lines.append("")
        lines.append(agg.to_markdown(index=False))
        lines.append("")
    lines.append("")

    lines.append("## 7. Cross-Model and Cross-Prompt Stability")
    lines.append("")
    if not sim.empty:
        lines.append("Top pairwise Kendall τ (rank correlation across (model, prompt, init, cond)):")
        lines.append("")
        top = sim.sort_values("kendall_tau", ascending=False).head(8)
        lines.append(top.round(3).to_markdown(index=False))
        lines.append("")
    lines.append(f"**Verdict B/C: {verdict_bc()}**")
    lines.append("")

    lines.append("## 8. Comparison to v7")
    lines.append("v7 used ACON UTCO prompts and reported SDI ≈ 0.96 (MiniMax) / 0.99 (Qwen), "
                 "cross-model Kendall τ = 0.49, AUTH_OR_ACCESS_TOKEN as universal repellor.")
    lines.append("v8 uses general (non-ACON) prompts. See §2 SDI and §7 τ above to compare.")
    lines.append("")

    lines.append("## 9. Negative Findings and Caveats")
    if not bc_s.empty:
        worst = bc_s.sort_values("violation_rate", ascending=False).head(3)
        lines.append("Budget violations (top-3 worst):")
        lines.append("")
        lines.append(worst.round(3).to_markdown(index=False))
        lines.append("")
    lines.append("- v7-derived case pool: no <15-step trajectories.")
    lines.append("- Plan B → no UT/P3 ablations; only P1 (task-aware) + P2 (task-agnostic).")
    lines.append("- Cross-model retention scoring (Qwen↔MiniMax) may inherit prompt-family-shaped biases.")
    lines.append("- Length tolerance for budget = 10 %.")
    lines.append("")

    lines.append("## 10. Paper-Level Interpretation")
    lines.append("(Filled in manually in `docs/04_results_summary.md` after the auto-report is reviewed.)")
    lines.append("")

    lines.append("## 11. Files of Record")
    lines.append("Tables (`outputs/tables/`):")
    for n in ("need_effect_by_type", "surface_dominance_regression",
              "surface_dominance_index", "preference_inversion",
              "condition_responsiveness", "survival_by_round_type",
              "fact_type_half_life", "ausc_by_type", "hazard_by_round_type",
              "hierarchy_rank_by_model_prompt",
              "cross_model_prompt_hierarchy_similarity",
              "convergence_by_chain", "fixed_point_composition_by_type",
              "fixed_point_need_shift", "basin_similarity", "basin_contraction",
              "budget_compliance_single_round", "budget_compliance_iterative"):
        if table_path(n + ".csv").exists():
            lines.append(f"- `{n}.csv`")
    lines.append("")
    lines.append("Figures (`outputs/figures/`):")
    for n in ("fig_need_effect_by_fact_type",
              "fig_surface_dominance_index",
              "fig_preference_inversion_rate",
              "fig_iterative_survival_curves",
              "fig_fixed_point_composition_by_type",
              "fig_fixed_point_need_shift",
              "fig_basin_contraction",
              "fig_fixed_point_recall_groups",
              "fig_cross_model_prompt_hierarchy_rank"):
        if (PROVENANCE.parent / "figures" / (n + ".png")).exists():
            lines.append(f"- `{n}.{{png,pdf}}`")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"[09] wrote {args.out}")


if __name__ == "__main__":
    main()
