"""Stage 07 — final plot pass.

Stages 03 and 04 already produce per-experiment figures. This pass
adds the two soft-token figures and re-renders any figure that may
have been generated with stale data.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v6_jacobian.data import (  # noqa: E402
    ensure_outputs, table_path, figure_path,
)
from motivation_v6_jacobian.plotting import (  # noqa: E402
    plot_soft_token_recovery, plot_soft_token_loss_vs_k,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    ensure_outputs()

    csv_path = table_path("soft_token_oracle_losses.csv")
    if not csv_path.exists():
        print("[07] soft_token_oracle_losses.csv missing; skipping")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[07] empty soft-token CSV; skipping")
        return

    k_cols = sorted(
        (int(c.split("k")[-1]) for c in df.columns
         if c.startswith("soft_loss_k")),
    )
    if not k_cols:
        print("[07] no soft_loss_k* columns; skipping")
        return

    ks = k_cols
    recovery_series = {}
    soft_medians: List[float] = []
    for k in ks:
        rec_col = f"soft_gap_recovery_k{k}"
        loss_col = f"soft_loss_k{k}"
        recovery_series.setdefault("soft", []).append(
            float(pd.to_numeric(df[rec_col], errors="coerce")
                  .median(skipna=True)) if rec_col in df else float("nan")
        )
        soft_medians.append(
            float(pd.to_numeric(df[loss_col], errors="coerce")
                  .median(skipna=True)) if loss_col in df else float("nan")
        )

    plot_soft_token_recovery(
        ks=ks, recoveries=recovery_series,
        save_to=figure_path("fig_soft_token_gap_recovery"),
    )

    # Loss-vs-k plot — soft curve + horizontal lines for textual baselines
    baseline_medians = {}
    for label, col in [
        ("full context", "full_loss"),
        ("no context", "no_loss"),
        ("recent (last 5 spans)", "recent_loss"),
        ("ACON baseline", "acon_loss"),
    ]:
        if col in df:
            vals = pd.to_numeric(df[col], errors="coerce")
            m = float(vals.median(skipna=True))
            if math.isfinite(m):
                baseline_medians[label] = [m] * len(ks)
    baseline_medians["soft tokens"] = soft_medians
    plot_soft_token_loss_vs_k(
        ks=ks, per_method=baseline_medians,
        save_to=figure_path("fig_soft_token_loss_vs_k"),
    )
    print(f"[07] wrote soft-token figures")


if __name__ == "__main__":
    main()
