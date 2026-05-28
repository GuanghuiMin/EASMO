"""Stage 08 — compute the metrics tables (spec §15 + §16)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v7.data import (  # noqa: E402
    ensure_outputs, read_jsonl, table_path, raw_path,
)
from motivation_v7.metrics import (  # noqa: E402
    need_effect_by_type,
    surface_dominance_regression,
    preference_inversion_rate,
    condition_responsiveness,
    survival_by_round_type,
    half_life_table,
    hazard_by_round_type,
    ausc_by_type,
    hierarchy_rank_by_model,
    cross_model_hierarchy_similarity,
    convergence_by_case,
)


def _to_df(path: Path) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(path)) if path.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single",
                    default=str(raw_path("fact_retention_scores_single_round.jsonl")))
    ap.add_argument("--iterative",
                    default=str(raw_path("fact_retention_scores_iterative.jsonl")))
    ap.add_argument("--iter_compress",
                    default=str(raw_path("iterative_compressions.jsonl")))
    ap.add_argument("--out_dir", default=str(table_path("")))
    ap.add_argument("--rounds_cap", type=int, default=5)
    args = ap.parse_args()
    ensure_outputs()

    print(f"[08] loading retention scores")
    df_single = _to_df(Path(args.single))
    df_iter   = _to_df(Path(args.iterative))
    df_iter_compress = _to_df(Path(args.iter_compress))

    # ---- Claim A metrics ----
    if not df_single.empty:
        ne = need_effect_by_type(df_single)
        ne.to_csv(table_path("need_effect_by_type.csv"), index=False)
        print(f"  - need_effect_by_type: {len(ne)} rows")

        sdr = surface_dominance_regression(df_single)
        sdr.to_csv(table_path("surface_dominance_regression.csv"), index=False)
        sdr[["compressor_model", "prompt_variant", "budget_chars", "sdi"]].to_csv(
            table_path("surface_dominance_index.csv"), index=False
        )
        print(f"  - surface_dominance_regression: {len(sdr)} rows")

        pir = preference_inversion_rate(df_single)
        pir.to_csv(table_path("preference_inversion.csv"), index=False)
        print(f"  - preference_inversion: {len(pir)} rows")

        crs = condition_responsiveness(df_single)
        crs.to_csv(table_path("condition_responsiveness.csv"), index=False)
        print(f"  - condition_responsiveness: {len(crs)} rows")
    else:
        print("[08] no single-round retention data — skipping Claim A metrics")

    # ---- Claim B metrics ----
    if not df_iter.empty:
        surv = survival_by_round_type(df_iter)
        surv.to_csv(table_path("survival_by_round_type.csv"), index=False)
        print(f"  - survival_by_round_type: {len(surv)} rows")

        half = half_life_table(surv, rounds_cap=args.rounds_cap)
        half.to_csv(table_path("fact_type_half_life.csv"), index=False)
        print(f"  - fact_type_half_life: {len(half)} rows")

        hazard = hazard_by_round_type(surv)
        hazard.to_csv(table_path("hazard_by_round_type.csv"), index=False)
        print(f"  - hazard_by_round_type: {len(hazard)} rows")

        ausc = ausc_by_type(surv)
        ausc.to_csv(table_path("ausc_by_type.csv"), index=False)
        print(f"  - ausc_by_type: {len(ausc)} rows")

        ranks = hierarchy_rank_by_model(half)
        ranks.to_csv(table_path("hierarchy_rank_by_model.csv"), index=False)
        sim = cross_model_hierarchy_similarity(ranks)
        sim.to_csv(table_path("cross_model_hierarchy_similarity.csv"), index=False)
        print(f"  - hierarchy_rank_by_model: {len(ranks)} rows, "
              f"cross_model_similarity: {len(sim)} rows")

        # Convergence requires iterative compressions joined with retention
        if not df_iter_compress.empty:
            conv = convergence_by_case(df_iter_compress, df_iter)
            conv.to_csv(table_path("convergence_by_case.csv"), index=False)
            print(f"  - convergence_by_case: {len(conv)} rows")
    else:
        print("[08] no iterative retention data — skipping Claim B metrics")

    print(f"[08] all tables under {table_path('')}")


if __name__ == "__main__":
    main()
