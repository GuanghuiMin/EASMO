"""Stage 08 — write outputs/reports/results_summary.md.

Pulls numbers from outputs/tables/*.csv (produced by stages 03–07) and
emits a paper-tier markdown report following the template in
user_feedback/motivation_v6_jacobian_active_subspace_experiment.md §10.
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v6_jacobian.data import (  # noqa: E402
    ensure_outputs, table_path, figure_path, raw_path, read_jsonl,
)


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except Exception:
        return str(v)
    if not math.isfinite(f):
        return "—"
    if abs(f) >= 100:
        return f"{f:.1f}"
    return f"{f:.3f}"


def _lookup(df: pd.DataFrame, section: str, metric: str) -> Optional[float]:
    sub = df[(df["section"] == section) & (df["metric"] == metric)]
    if sub.empty:
        return None
    v = float(sub["value"].iloc[0])
    return v if math.isfinite(v) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--model_path", default="?")
    ap.add_argument("--max_context_tokens", type=int, default=12000)
    ap.add_argument("--layer_index", type=int, default=None)
    args = ap.parse_args()
    ensure_outputs()

    dash_path = table_path("v6_dashboard.csv")
    if not dash_path.exists():
        print("[08] dashboard missing; run stage 06 first")
        return
    dash = pd.read_csv(dash_path)

    # Pull individual numbers
    A_global_s   = _lookup(dash, "A", "global_spearman_primary")
    A_global_p   = _lookup(dash, "A", "global_pearson_primary")
    A_med_pt_s   = _lookup(dash, "A", "median_per_task_spearman")
    A_corr_len   = _lookup(dash, "A", "corr_span_gxa_sqrtlen_vs_token_count")
    A_corr_rec   = _lookup(dash, "A", "corr_span_gxa_sqrtlen_vs_step_id")
    A_top1       = _lookup(dash, "A", "median_top1_enrichment")
    A_top3       = _lookup(dash, "A", "median_top3_enrichment")
    A_top5       = _lookup(dash, "A", "median_top5_enrichment")

    B_ex_k16  = _lookup(dash, "B", "example_cumvar_k16")
    B_ex_k32  = _lookup(dash, "B", "example_cumvar_k32")
    B_ex_k64  = _lookup(dash, "B", "example_cumvar_k64")
    B_sp_k16  = _lookup(dash, "B", "span_cumvar_k16")
    B_sp_k32  = _lookup(dash, "B", "span_cumvar_k32")
    B_hi_k16  = _lookup(dash, "B", "high_v4_cumvar_k16")
    B_lo_k16  = _lookup(dash, "B", "low_v4_cumvar_k16")

    C_full = _lookup(dash, "C", "median_full_loss")
    C_no   = _lookup(dash, "C", "median_no_loss")
    C_rec  = _lookup(dash, "C", "median_recent_loss")
    C_acn  = _lookup(dash, "C", "median_acon_loss")
    C_soft_l = {k: _lookup(dash, "C", f"median_soft_loss_k{k}")
                for k in (4, 8, 16, 32, 64)}
    C_soft_r = {k: _lookup(dash, "C", f"median_soft_gap_recovery_k{k}")
                for k in (4, 8, 16, 32, 64)}

    # Setup numbers
    case_summary = []
    case_path = raw_path("jacobian_case_summary.jsonl")
    if case_path.exists():
        case_summary = read_jsonl(case_path)
    n_tasks = len(case_summary)
    n_tokens_median = (
        int(np.median([c["n_context_tokens"] for c in case_summary]))
        if case_summary else None
    )

    # Verdict helpers
    def verdict_a() -> str:
        if A_med_pt_s is None:
            return "—"
        if A_med_pt_s >= 0.25 or (A_top3 is not None and A_top3 >= 2.0):
            return "**STRONG POSITIVE**"
        if A_top3 is not None and A_top3 >= 1.3:
            return "MEDIUM positive"
        return "WEAK / NEGATIVE"

    def verdict_b() -> str:
        if B_ex_k16 is not None and B_ex_k16 >= 0.5:
            return "**STRONG POSITIVE**"
        if B_ex_k32 is not None and B_ex_k32 >= 0.7:
            return "**STRONG POSITIVE**"
        if (B_hi_k16 is not None and B_lo_k16 is not None
                and B_hi_k16 - B_lo_k16 >= 0.1):
            return "MEDIUM positive"
        return "WEAK / NEGATIVE"

    def verdict_c() -> str:
        for k in (16, 32):
            if C_soft_r[k] is not None and C_soft_r[k] >= 0.7:
                return "**STRONG POSITIVE**"
        for k in (4, 8, 16, 32, 64):
            v = C_soft_l[k]
            if (v is not None and C_acn is not None and v < C_acn) or \
               (v is not None and C_rec is not None and v < C_rec):
                return "MEDIUM positive"
        return "WEAK / NEGATIVE"

    layer_str = (f"{args.layer_index}" if args.layer_index is not None
                 else "N/2 (default)")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")

    md = textwrap.dedent(f"""\
        # Motivation v6 Results: Jacobian Active-Subspace Diagnostics

        > Auto-written by `scripts/08_write_report.py` at {ts}.

        ## 1. Setup

        * model path: `{args.model_path}`
        * number of tasks (v4 reuse): **{n_tasks}**
        * max context tokens (white-box): **{args.max_context_tokens}**
        * median ctx tokens after tokenization: **{n_tokens_median}**
        * target type: canonical v4 reference decision-state JSON (teacher-forced)
        * mid-layer for active capture: **layer {layer_str}**

        ## 2. Experiment A — Jacobian saliency vs v4 finite-difference sensitivity

        | Metric | Value |
        |---|---|
        | Global Spearman (primary score = `span_gxa_sqrtlen`) | **{_fmt(A_global_s)}** |
        | Global Pearson | {_fmt(A_global_p)} |
        | Median per-task Spearman | **{_fmt(A_med_pt_s)}** |
        | Median top-1 enrichment over random | {_fmt(A_top1)}× |
        | Median top-3 enrichment over random | **{_fmt(A_top3)}×** |
        | Median top-5 enrichment over random | {_fmt(A_top5)}× |
        | Confounder: Jacobian ↔ token_count Spearman | {_fmt(A_corr_len)} |
        | Confounder: Jacobian ↔ step_id (recency) Spearman | {_fmt(A_corr_rec)} |
        | Confounder: v4 ↔ token_count Spearman | {_fmt(_lookup(dash, "A", "corr_v4_vs_token_count"))} |

        **Verdict A:** {verdict_a()}

        Figures: `figures/fig_jacobian_vs_v4_scatter.{{png,pdf}}`,
        `figures/fig_jacobian_rank_overlap.{{png,pdf}}`.

        ## 3. Experiment B — Active subspace spectrum (layer {layer_str})

        Cumulative explained variance of randomised SVD on
        Jacobian-weighted activations:

        | Matrix | k=4 | k=8 | k=16 | k=32 | k=64 |
        |---|---|---|---|---|---|
        | example  | {_fmt(_lookup(dash,'B','example_cumvar_k4'))} | {_fmt(_lookup(dash,'B','example_cumvar_k8'))} | **{_fmt(B_ex_k16)}** | {_fmt(B_ex_k32)} | {_fmt(B_ex_k64)} |
        | span     | {_fmt(_lookup(dash,'B','span_cumvar_k4'))} | {_fmt(_lookup(dash,'B','span_cumvar_k8'))} | **{_fmt(B_sp_k16)}** | {_fmt(B_sp_k32)} | {_fmt(_lookup(dash,'B','span_cumvar_k64'))} |
        | high_v4  | {_fmt(_lookup(dash,'B','high_v4_cumvar_k4'))} | {_fmt(_lookup(dash,'B','high_v4_cumvar_k8'))} | {_fmt(B_hi_k16)} | {_fmt(_lookup(dash,'B','high_v4_cumvar_k32'))} | {_fmt(_lookup(dash,'B','high_v4_cumvar_k64'))} |
        | low_v4   | {_fmt(_lookup(dash,'B','low_v4_cumvar_k4'))} | {_fmt(_lookup(dash,'B','low_v4_cumvar_k8'))} | {_fmt(B_lo_k16)} | {_fmt(_lookup(dash,'B','low_v4_cumvar_k32'))} | {_fmt(_lookup(dash,'B','low_v4_cumvar_k64'))} |

        High-vs-low cumulative variance gap at k=16: **{_fmt((B_hi_k16 or 0) - (B_lo_k16 or 0))}**.

        **Verdict B:** {verdict_b()}

        Figures: `figures/fig_active_subspace_spectrum_example.{{png,pdf}}`,
        `figures/fig_active_subspace_spectrum_span.{{png,pdf}}`,
        `figures/fig_high_vs_low_sensitivity_spectrum.{{png,pdf}}`.

        ## 4. Experiment C — Soft-token oracle

        Median target NLL across cases:

        | Method | median NLL | gap recovery |
        |---|---|---|
        | full context | **{_fmt(C_full)}** | 1.000 |
        | no context | {_fmt(C_no)} | 0.000 |
        | recent (last 5 spans) | {_fmt(C_rec)} | {_fmt(_lookup(dash, 'C', 'median_recent_loss'))} |
        | ACON baseline | {_fmt(C_acn)} | {_fmt(_lookup(dash, 'C', 'median_acon_loss'))} |
        | soft k=4 | {_fmt(C_soft_l[4])} | {_fmt(C_soft_r[4])} |
        | soft k=8 | {_fmt(C_soft_l[8])} | {_fmt(C_soft_r[8])} |
        | soft k=16 | **{_fmt(C_soft_l[16])}** | **{_fmt(C_soft_r[16])}** |
        | soft k=32 | **{_fmt(C_soft_l[32])}** | **{_fmt(C_soft_r[32])}** |
        | soft k=64 | {_fmt(C_soft_l[64])} | {_fmt(C_soft_r[64])} |

        **Verdict C:** {verdict_c()}

        Figures: `figures/fig_soft_token_gap_recovery.{{png,pdf}}`,
        `figures/fig_soft_token_loss_vs_k.{{png,pdf}}`.

        ## 5. Negative findings

        - Jacobian-vs-v4 correlation against token-length: {_fmt(A_corr_len)}; if
          this is large, Jacobian scores are partially a length proxy.
        - Jacobian-vs-step_id (recency) Spearman: {_fmt(A_corr_rec)}; large
          positive means scores covary with recency.
        - low_v4 spans cumulative variance at k=16 ≈ {_fmt(B_lo_k16)} — if not
          much smaller than high_v4 ({_fmt(B_hi_k16)}), then the active subspace
          isn't really concentrating on the spans v4 flagged.

        ## 6. Implications for method design

        (See interpretation rules §11 of the spec.)

        - If A & B positive: motivates an active-subspace-preserving compressor.
        - If B & C positive but A weak: motivates soft-memory / representation-
          level compression, NOT span selection.
        - If A positive but B/C weak: gradients are useful as a saliency signal
          but the low-rank claim is unsupported.

        ## 7. Files of record

        Tables:
        - `tables/jacobian_vs_v4_correlations.csv`
        - `tables/jacobian_topk_overlap.csv`
        - `tables/jacobian_top1_ranks.csv`
        - `tables/active_subspace_spectrum.csv`
        - `tables/soft_token_oracle_losses.csv`
        - `tables/v6_dashboard.csv`

        Figures:
        - `figures/fig_jacobian_vs_v4_scatter.{{png,pdf}}`
        - `figures/fig_jacobian_rank_overlap.{{png,pdf}}`
        - `figures/fig_active_subspace_spectrum_example.{{png,pdf}}`
        - `figures/fig_active_subspace_spectrum_span.{{png,pdf}}`
        - `figures/fig_high_vs_low_sensitivity_spectrum.{{png,pdf}}`
        - `figures/fig_soft_token_gap_recovery.{{png,pdf}}`
        - `figures/fig_soft_token_loss_vs_k.{{png,pdf}}`

        Raw artifacts:
        - `raw/cases.jsonl`
        - `raw/jacobian_span_scores.jsonl`
        - `raw/jacobian_case_summary.jsonl`
        - `raw/active_vectors_layer*.npz`
        - `raw/active_vector_metadata.jsonl`
        - `raw/soft_token_histories.jsonl`
    """)
    out = _REPO / "outputs" / "reports" / "results_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"[08] wrote {out}")


if __name__ == "__main__":
    main()
