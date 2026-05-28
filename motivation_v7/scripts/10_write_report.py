"""Stage 10 — write the v7 results summary report (spec §22)."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v7.data import (  # noqa: E402
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
    ap.add_argument("--out",
                    default=str(_REPO / "outputs/reports/motivation_v7_results_summary.md"))
    args = ap.parse_args()
    ensure_outputs()

    # ---- inputs ----
    ne   = _csv("need_effect_by_type.csv")
    sdr  = _csv("surface_dominance_regression.csv")
    pir  = _csv("preference_inversion.csv")
    crs  = _csv("condition_responsiveness.csv")
    half = _csv("fact_type_half_life.csv")
    sim  = _csv("cross_model_hierarchy_similarity.csv")
    conv = _csv("convergence_by_case.csv")

    # provenance
    prov = {}
    p_json = PROVENANCE / "acon_prompt_sha256.json"
    if p_json.exists():
        import json
        prov = json.loads(p_json.read_text())

    n_cases = len(read_jsonl(_REPO / "data" / "case_pool.jsonl"))
    n_facts = len(read_jsonl(_REPO / "data" / "fact_bank_filtered.jsonl"))
    n_conds = len(read_jsonl(_REPO / "data" / "need_conditions.jsonl"))
    n_comps = (
        len(read_jsonl(raw_path("single_round_compressions.jsonl")))
        if raw_path("single_round_compressions.jsonl").exists() else 0
    )
    n_scores = (
        len(read_jsonl(raw_path("fact_retention_scores_single_round.jsonl")))
        if raw_path("fact_retention_scores_single_round.jsonl").exists() else 0
    )
    n_iter = (
        len(read_jsonl(raw_path("iterative_compressions.jsonl")))
        if raw_path("iterative_compressions.jsonl").exists() else 0
    )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")

    # ---- verdicts ----
    verdict_A = "INCONCLUSIVE"
    if not sdr.empty:
        sdi_max = float(sdr["sdi"].max(skipna=True)) if "sdi" in sdr else float("nan")
        pir_max = float(pir["preference_inversion_rate"].max(skipna=True)) if "preference_inversion_rate" in pir else float("nan")
        ne_executable = ne[ne["coarse_group"] == "EXECUTABLE"] if "coarse_group" in ne else pd.DataFrame()
        mean_dne_exec = float(ne_executable["delta_need"].mean(skipna=True)) if len(ne_executable) else float("nan")
        # spec §18 — strong positive needs ≥3 of 5
        flags = 0
        if math.isfinite(mean_dne_exec) and mean_dne_exec < 0.15: flags += 1
        if math.isfinite(sdi_max) and sdi_max > 0.3: flags += 1
        if math.isfinite(pir_max) and pir_max > 0.25: flags += 1
        if not sdr.empty and (sdr["r2_type"] > sdr["r2_need"]).any(): flags += 1
        # criterion 5: ≥2 concrete types with similar retention
        ne_concrete = ne[ne["coarse_group"].isin(["EXECUTABLE", "CONTROL"])] if "coarse_group" in ne else pd.DataFrame()
        if not ne_concrete.empty and (ne_concrete["delta_need"].abs() < 0.10).sum() >= 2: flags += 1
        verdict_A = ("STRONG POSITIVE" if flags >= 3
                     else "PARTIAL POSITIVE" if flags >= 2
                     else "WEAK / NEGATIVE")

    verdict_B = "INCONCLUSIVE"
    if not half.empty:
        flags = 0
        if half["half_life"].nunique() >= 3: flags += 1
        narr_half = half[half["coarse_group"] == "NARRATIVE"]["half_life"]
        exec_half = half[half["coarse_group"] == "EXECUTABLE"]["half_life"]
        if narr_half.size and exec_half.size and narr_half.mean() > exec_half.mean(): flags += 1
        if not sim.empty and (sim["kendall_tau"] > 0.4).any(): flags += 1
        if not conv.empty and "converged" in conv and conv["converged"].mean() > 0.5: flags += 1
        verdict_B = ("STRONG POSITIVE" if flags >= 3
                     else "PARTIAL POSITIVE" if flags >= 2
                     else "WEAK / NEGATIVE")

    # ---- write report ----
    lines = []
    lines.append("# motivation_v7 Results: Abstraction Prior & Iterative Compression Dynamics")
    lines.append("")
    lines.append(f"> Auto-written by `scripts/10_write_report.py` at {ts}.")
    lines.append("")
    lines.append("## 0. Counts")
    lines.append(f"- `n_cases` = **{n_cases}**")
    lines.append(f"- `n_facts` (filtered) = **{n_facts}**")
    lines.append(f"- `n_conditions` = **{n_conds}**")
    lines.append(f"- `n_compressions` (single round) = **{n_comps}**")
    lines.append(f"- `n_compressions` (iterative) = **{n_iter}**")
    lines.append(f"- `n_retention_scores` (single round) = **{n_scores}**")
    lines.append("")
    lines.append("## 1. ACON prompt provenance")
    lines.append(f"- ACON repo commit: `{prov.get('acon_commit_hash', 'UNKNOWN')}`")
    for variant in ("UT", "UTCO"):
        r = prov.get(variant, {})
        lines.append(f"- **{variant}** template: `{r.get('source_path', '?')}` "
                     f"(sha256 `{r.get('sha256', '?')}`)")
    sys_meta = prov.get("system", {})
    lines.append(f"- ACON system prompt: `{sys_meta.get('source_path', '?')}` "
                 f"(sha256 `{sys_meta.get('sha256', '?')}`)")
    lines.append("")
    lines.append("## 2. Claim A: Is compression preference need-conditioned?")
    lines.append("")
    lines.append(f"**Verdict A: {verdict_A}**")
    lines.append("")
    if not ne.empty:
        lines.append("### Need effect Δ_need (binary retention) by fact type, per model")
        agg = (ne.groupby(["fact_type", "compressor_model"])["delta_need"]
                  .mean().reset_index()
                  .pivot(index="fact_type", columns="compressor_model", values="delta_need"))
        lines.append("")
        lines.append(agg.round(3).to_markdown())
        lines.append("")
    if not sdr.empty:
        lines.append("### Surface dominance regression")
        lines.append("")
        lines.append(sdr[["compressor_model", "prompt_variant", "budget_chars",
                          "n", "r2_need", "r2_type", "r2_both", "sdi"]]
                     .round(3).to_markdown(index=False))
        lines.append("")
    if not pir.empty:
        lines.append("### Preference Inversion Rate")
        lines.append("")
        lines.append(pir.round(3).to_markdown(index=False))
        lines.append("")
    lines.append("Figures: `figures/fig_need_effect_by_fact_type.{png,pdf}`, "
                 "`figures/fig_surface_dominance_index.{png,pdf}`, "
                 "`figures/fig_preference_inversion_rate.{png,pdf}`.")
    lines.append("")
    lines.append("## 3. Claim B: Is there a stable iterative information-loss hierarchy?")
    lines.append("")
    lines.append(f"**Verdict B: {verdict_B}**")
    lines.append("")
    if not half.empty:
        lines.append("### Half-life by fact type")
        lines.append("")
        lines.append(half[["compressor_model", "prompt_variant", "fact_type",
                          "coarse_group", "half_life", "half_life_censored",
                          "final_survival"]]
                     .round(3).to_markdown(index=False))
        lines.append("")
    if not sim.empty:
        lines.append("### Cross-model hierarchy similarity")
        lines.append("")
        lines.append(sim.round(3).to_markdown(index=False))
        lines.append("")
    if not conv.empty and "converged" in conv:
        lines.append("### Convergence")
        lines.append("")
        conv_rate = float(conv["converged"].mean())
        lines.append(f"Converged within {conv['final_round'].max()} rounds: "
                     f"**{conv_rate:.1%}** of chains.")
        for col in ("needed_fact_recall_at_convergence",
                    "narrative_fact_recall_at_convergence",
                    "executable_fact_recall_at_convergence"):
            if col in conv:
                v = conv[col].mean(skipna=True)
                lines.append(f"- mean {col} = {_fmt(v)}")
        lines.append("")
    lines.append("Figures: `figures/fig_iterative_survival_curves.{png,pdf}`, "
                 "`figures/fig_survival_hierarchy_heatmap.{png,pdf}`, "
                 "`figures/fig_cross_model_hierarchy_rank.{png,pdf}`, "
                 "`figures/fig_fixed_point_recall.{png,pdf}`.")
    lines.append("")
    lines.append("## 4. Caveats")
    lines.append("- v3-derived 30 cases are all medium-length (≥15 steps); no <15-step stratum.")
    lines.append("- Plan B scope: iterative compression uses 2 chains/case "
                 "(needed + unneeded for one representative EXECUTABLE fact), "
                 "not the spec's full sweep of all condition_tasks.")
    lines.append("- Single budget = 1500 chars (spec primary). Secondary budgets {800, 2500} not run.")
    lines.append("- Single prompt variant = UTCO (samples_4). UT ablation not run.")
    lines.append("- Cross-model retention scorer (Qwen-compressions scored by MiniMax, "
                 "MiniMax-compressions scored by Qwen) per spec §4.")
    lines.append("")
    lines.append("## 5. Files of record")
    lines.append("Tables under `outputs/tables/`:")
    for n in ("need_effect_by_type", "surface_dominance_regression",
              "surface_dominance_index", "preference_inversion",
              "condition_responsiveness", "survival_by_round_type",
              "fact_type_half_life", "hazard_by_round_type",
              "ausc_by_type", "hierarchy_rank_by_model",
              "cross_model_hierarchy_similarity", "convergence_by_case",
              "fact_bank_grounding", "need_condition_quality"):
        if table_path(n + ".csv").exists():
            lines.append(f"- `{n}.csv`")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"[10] wrote {args.out}")


if __name__ == "__main__":
    main()
