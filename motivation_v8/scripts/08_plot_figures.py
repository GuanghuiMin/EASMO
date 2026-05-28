"""Stage 08 — render the 9 figures (spec §14)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v8.data import (  # noqa: E402
    ensure_outputs, table_path, figure_path,
)
from motivation_v8.plots import (  # noqa: E402
    fig_need_effect_by_fact_type,
    fig_surface_dominance_index,
    fig_preference_inversion_rate,
    fig_iterative_survival_curves,
    fig_fixed_point_composition,
    fig_fixed_point_need_shift,
    fig_basin_contraction,
    fig_fixed_point_recall_groups,
    fig_cross_model_prompt_hierarchy_rank,
)


def _csv(name: str) -> pd.DataFrame:
    p = table_path(name)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    ensure_outputs()

    fig_need_effect_by_fact_type(_csv("need_effect_by_type.csv"),
                                  figure_path("fig_need_effect_by_fact_type"))
    fig_surface_dominance_index(_csv("surface_dominance_regression.csv"),
                                 figure_path("fig_surface_dominance_index"))
    fig_preference_inversion_rate(_csv("preference_inversion.csv"),
                                   figure_path("fig_preference_inversion_rate"))
    fig_iterative_survival_curves(_csv("survival_by_round_type.csv"),
                                   figure_path("fig_iterative_survival_curves"))
    fig_fixed_point_composition(_csv("fixed_point_composition_by_type.csv"),
                                 figure_path("fig_fixed_point_composition_by_type"))
    fig_fixed_point_need_shift(_csv("fixed_point_need_shift.csv"),
                                figure_path("fig_fixed_point_need_shift"))
    fig_basin_contraction(_csv("basin_similarity.csv"),
                           figure_path("fig_basin_contraction"))
    fig_fixed_point_recall_groups(_csv("convergence_by_chain.csv"),
                                   figure_path("fig_fixed_point_recall_groups"))
    fig_cross_model_prompt_hierarchy_rank(_csv("hierarchy_rank_by_model_prompt.csv"),
                                           figure_path("fig_cross_model_prompt_hierarchy_rank"))
    print(f"[08] wrote 9 figures under {figure_path('')}")


if __name__ == "__main__":
    main()
