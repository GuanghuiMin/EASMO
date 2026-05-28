"""Stage 07 — compute all metric tables (spec §13)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v8.data import (  # noqa: E402
    ensure_outputs, read_jsonl, table_path, raw_path,
)
from motivation_v8.metrics import (  # noqa: E402
    need_effect_by_type,
    surface_dominance_regression,
    preference_inversion_rate,
    condition_responsiveness,
    survival_by_round_type,
    half_life_table,
    ausc_by_type,
    hazard_by_round_type,
    hierarchy_rank_by_model_prompt,
    cross_model_prompt_hierarchy_similarity,
    convergence_by_chain,
    fixed_point_composition_by_type,
    fixed_point_need_shift,
    basin_metrics,
    budget_compliance,
)


def _df(path: Path) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(path)) if path.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds_cap", type=int, default=6)
    args = ap.parse_args()
    ensure_outputs()

    print("[07] loading raw data")
    chains = _df(raw_path("iterative_chains.jsonl"))
    single = _df(raw_path("single_round_compressions.jsonl"))
    retention = _df(raw_path("retention_scores.jsonl"))

    ret_single = retention[retention["context_source"] == "single_round"] if not retention.empty else pd.DataFrame()
    ret_iter   = retention[retention["context_source"] == "iterative"]    if not retention.empty else pd.DataFrame()

    # ---- Single-round / Claim A ----
    if not ret_single.empty:
        ne = need_effect_by_type(ret_single)
        ne.to_csv(table_path("need_effect_by_type.csv"), index=False)
        print(f"  - need_effect_by_type: {len(ne)} rows")

        sdr = surface_dominance_regression(ret_single)
        sdr.to_csv(table_path("surface_dominance_regression.csv"), index=False)
        sdr[["model", "prompt_family", "budget_chars", "sdi"]].to_csv(
            table_path("surface_dominance_index.csv"), index=False
        )
        print(f"  - surface_dominance_regression: {len(sdr)} rows")

        pir = preference_inversion_rate(ret_single)
        pir.to_csv(table_path("preference_inversion.csv"), index=False)
        print(f"  - preference_inversion: {len(pir)} rows")

        crs = condition_responsiveness(ret_single)
        crs.to_csv(table_path("condition_responsiveness.csv"), index=False)
        print(f"  - condition_responsiveness: {len(crs)} rows")

    # ---- Iterative / Claim B+C+D+E ----
    if not ret_iter.empty:
        surv = survival_by_round_type(ret_iter)
        surv.to_csv(table_path("survival_by_round_type.csv"), index=False)
        print(f"  - survival_by_round_type: {len(surv)} rows")

        half = half_life_table(surv, rounds_cap=args.rounds_cap)
        half.to_csv(table_path("fact_type_half_life.csv"), index=False)
        print(f"  - fact_type_half_life: {len(half)} rows")

        ausc = ausc_by_type(surv)
        ausc.to_csv(table_path("ausc_by_type.csv"), index=False)
        print(f"  - ausc_by_type: {len(ausc)} rows")

        hazard = hazard_by_round_type(surv)
        hazard.to_csv(table_path("hazard_by_round_type.csv"), index=False)
        print(f"  - hazard_by_round_type: {len(hazard)} rows")

        ranks = hierarchy_rank_by_model_prompt(half)
        ranks.to_csv(table_path("hierarchy_rank_by_model_prompt.csv"), index=False)
        sim = cross_model_prompt_hierarchy_similarity(ranks)
        sim.to_csv(table_path("cross_model_prompt_hierarchy_similarity.csv"), index=False)
        print(f"  - hierarchy_rank_by_model_prompt: {len(ranks)} rows; cross-sim: {len(sim)} rows")

        if not chains.empty:
            conv = convergence_by_chain(chains, ret_iter)
            conv.to_csv(table_path("convergence_by_chain.csv"), index=False)
            print(f"  - convergence_by_chain: {len(conv)} rows")

            comp = fixed_point_composition_by_type(chains, ret_iter, conv)
            comp.to_csv(table_path("fixed_point_composition_by_type.csv"), index=False)
            print(f"  - fixed_point_composition_by_type: {len(comp)} rows")

            shift = fixed_point_need_shift(chains, ret_iter, conv)
            shift.to_csv(table_path("fixed_point_need_shift.csv"), index=False)
            print(f"  - fixed_point_need_shift: {len(shift)} rows")

            basin = basin_metrics(chains, ret_iter, conv)
            basin.to_csv(table_path("basin_similarity.csv"), index=False)
            # also a summary per (model, prompt, init pair)
            if not basin.empty:
                basin_summary = (basin.groupby(["model", "prompt_family",
                                                 "init_a", "init_b"])[
                    ["init_fact_jaccard_distance", "fin_fact_jaccard_distance",
                     "contraction_fact_jaccard",
                     "init_type_l1_distance", "fin_type_l1_distance",
                     "contraction_type_l1"]
                ].mean().reset_index())
                basin_summary.to_csv(table_path("basin_contraction.csv"), index=False)
            print(f"  - basin_similarity / basin_contraction: {len(basin)} rows")

    # ---- Budget compliance ----
    if not single.empty:
        bc_single = budget_compliance(single)
        bc_single.to_csv(table_path("budget_compliance_single_round.csv"), index=False)
    if not chains.empty:
        bc_iter = budget_compliance(
            chains[chains["round"] >= 1],
            length_col="context_chars",
            violation_col="budget_violation",
            group_cols=("model", "prompt_family", "budget_chars", "init_type"),
        )
        bc_iter.to_csv(table_path("budget_compliance_iterative.csv"), index=False)

    print(f"[07] all tables under {table_path('')}")


if __name__ == "__main__":
    main()
