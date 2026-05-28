"""Stage 09 — render the 7 main figures (spec §17)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v7.data import (  # noqa: E402
    ensure_outputs, table_path, figure_path,
)
from motivation_v7.plots import (  # noqa: E402
    fig_need_effect_by_fact_type,
    fig_surface_dominance_index,
    fig_preference_inversion_rate,
    fig_iterative_survival_curves,
    fig_survival_hierarchy_heatmap,
    fig_cross_model_hierarchy_rank,
    fig_fixed_point_recall,
)


def _csv(name: str) -> pd.DataFrame:
    p = table_path(name)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(figure_path("")))
    args = ap.parse_args()
    ensure_outputs()

    ne   = _csv("need_effect_by_type.csv")
    sdr  = _csv("surface_dominance_regression.csv")
    pir  = _csv("preference_inversion.csv")
    surv = _csv("survival_by_round_type.csv")
    ranks = _csv("hierarchy_rank_by_model.csv")
    conv = _csv("convergence_by_case.csv")

    fig_need_effect_by_fact_type(ne, figure_path("fig_need_effect_by_fact_type"))
    fig_surface_dominance_index(sdr, figure_path("fig_surface_dominance_index"))
    fig_preference_inversion_rate(pir, figure_path("fig_preference_inversion_rate"))
    fig_iterative_survival_curves(surv, figure_path("fig_iterative_survival_curves"))
    fig_survival_hierarchy_heatmap(surv, figure_path("fig_survival_hierarchy_heatmap"))
    fig_cross_model_hierarchy_rank(ranks, figure_path("fig_cross_model_hierarchy_rank"))
    fig_fixed_point_recall(conv, figure_path("fig_fixed_point_recall"))
    print(f"[09] wrote 7 figures under {args.out_dir}")


if __name__ == "__main__":
    main()
