"""Stage 13 — render 5 main figures (spec §9)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v9.data import ensure_outputs, raw_path, table_path, figure_path  # noqa
from motivation_v9.plots import (  # noqa
    fig_best_of_n_pass_gain, fig_c1_ck_transition_matrix,
    fig_c1_ck_pass_drop_by_model, fig_stress_pass_curve_by_round,
    fig_chunk_advantage_by_type, fig_top_chunk_type_distribution,
)


def _csv(name: str) -> pd.DataFrame:
    p = table_path(name)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    ensure_outputs()

    summary = _csv("best_of_n_summary.csv")
    trans = _csv("c1_ck_transition.csv")
    fragility = _csv("c1_ck_fragility_by_model.csv")
    chunk_by_type = _csv("chunk_advantage_by_type.csv")

    fig_best_of_n_pass_gain(summary, figure_path("fig_best_of_n_pass_gain"))
    fig_c1_ck_transition_matrix(trans, figure_path("fig_c1_ck_transition_matrix"))
    fig_c1_ck_pass_drop_by_model(fragility, figure_path("fig_c1_ck_pass_drop_by_model"))

    # stress_pass_curve: derive pass rates per stress round from behavior runs +
    # stress_chains. For C1 (round 0) and CK (round K) we already have those.
    # For intermediate rounds we'd need extra behavior runs (not done in v9 primary).
    df_runs = pd.DataFrame(_csv("best_of_n_by_case.csv"))
    if not df_runs.empty:
        stress_rows = []
        for round_label, col_succ in [("C1", "greedy_success"),
                                       ("CK", "greedy_success")]:
            sub = df_runs[df_runs["eval_context_round"] == round_label]
            for keys, grp in sub.groupby("compressor_model"):
                stress_rows.append({
                    "compressor_model": keys,
                    "generation_type": "greedy",
                    "stress_round": 0 if round_label == "C1" else 1,
                    "pass_rate": float(grp[col_succ].mean()),
                })
        df_curve = pd.DataFrame(stress_rows)
        fig_stress_pass_curve_by_round(df_curve, figure_path("fig_stress_pass_curve_by_round"))

    fig_chunk_advantage_by_type(chunk_by_type, figure_path("fig_chunk_advantage_by_type"))
    fig_top_chunk_type_distribution(chunk_by_type, figure_path("fig_top_chunk_type_distribution"))
    print(f"[13] wrote figures under {figure_path('')}")


if __name__ == "__main__":
    main()
