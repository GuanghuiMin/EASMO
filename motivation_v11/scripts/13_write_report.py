"""Stage 13 — auto-write motivation_v11_results_summary.md (spec §16, §13.13)."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pandas as pd

from motivation_v11.data import ensure_outputs, table_path, REPORTS  # noqa


def _read(p):
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()


def _md(df, max_rows=20):
    if df.empty: return "*(no data)*\n"
    try: return df.head(max_rows).to_markdown(index=False) + "\n"
    except Exception:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |",
                 "|" + "|".join(["---"]*len(cols)) + "|"]
        for _, r in df.head(max_rows).iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPORTS / "motivation_v11_results_summary.md"))
    args = ap.parse_args()
    ensure_outputs()

    pool = _read(table_path("case_pool_summary.csv"))
    sum_pf = _read(table_path("prompt_family_behavior_summary.csv"))
    dqcg = _read(table_path("distribution_quality_calibration_gap.csv"))
    utvs = _read(table_path("ut_vs_utco_headroom.csv"))
    stress = _read(table_path("stress_invariance_by_prompt_selector.csv"))
    crosseval = _read(table_path("best_c1_vs_best_ck_cross_eval.csv"))
    sel = _read(table_path("selector_recovery_summary.csv"))
    curve = _read(table_path("pass_at_n_curve.csv"))

    lines = []
    lines.append("# motivation_v11 Results\n")
    lines.append(f"> Auto-written by `scripts/13_write_report.py` at "
                 f"{dt.datetime.utcnow().strftime('%Y-%m-%d %H:%MZ')}.\n")
    lines.append("> Hand-written paper-tier companion lives at "
                 "`docs/04_results_summary.md` (TBD).\n\n")

    lines.append("## 1. Case Pool\n")
    lines.append(_md(pool))

    lines.append("\n## 2. Structured vs Generic Compression\n")
    lines.append(_md(sum_pf, max_rows=40))

    lines.append("\n## 3. Distribution Quality vs Decoding Calibration\n")
    lines.append(_md(dqcg))

    lines.append("\n## 4. UT vs UTCO: Prompt Optimization vs Policy Headroom\n")
    lines.append(_md(utvs))

    lines.append("\n## 5. Serial Recompression Robustness\n")
    lines.append(_md(stress))
    lines.append("\n### Best-C1 vs Best-CK cross-evaluation\n")
    lines.append(_md(crosseval, max_rows=20))

    lines.append("\n## 6. Verbal Selectors Are Not Behavior Reward\n")
    lines.append(_md(sel, max_rows=60))

    lines.append("\n## 7. Main Figures for the Paper\n")
    for name in (
        "fig_prompt_family_pass_c1_ck",
        "fig_distribution_quality_vs_calibration_gap",
        "fig_pass_at_n_curve",
        "fig_serial_recompression_fragility",
        "fig_selector_recovery",
    ):
        lines.append(f"* `outputs/figures/{name}.pdf`")
    lines.append("\n### Pass@N curve\n")
    lines.append(_md(curve, max_rows=30))

    lines.append("\n## 8. What This Motivates for TRACE / Policy Optimization\n")
    lines.append("(See hand-written interpretation in `docs/04_results_summary.md`.)\n")

    lines.append("\n## 9. Negative Results and Caveats\n")
    lines.append("(See hand-written companion. Auto-report does not attempt to flag negatives.)\n")

    lines.append("\n## 10. Files of Record\n")
    for p in (
        "data/v11_baseline_runs.jsonl",
        "data/v11_primary_cases.jsonl",
        "data/v11_secondary_all_cases.jsonl",
        "outputs/raw/compression_candidates_c1.jsonl",
        "outputs/raw/stress_chains.jsonl",
        "outputs/raw/behavior_runs_c1_ck.jsonl",
        "outputs/raw/pointwise_verifier_scores.jsonl",
        "outputs/raw/pairwise_verifier_matches.jsonl",
        "outputs/raw/continuation_entropy_samples.jsonl",
        "outputs/data/full_dev_compression_candidate_bank.jsonl",
        "outputs/tables/*.csv",
        "outputs/figures/*.pdf",
    ):
        lines.append(f"* `{p}`")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"[13] wrote {args.out}")


if __name__ == "__main__":
    main()
