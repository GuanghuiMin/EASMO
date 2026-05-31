"""Stage 16 — auto-write motivation_v11_results_summary.md (spec §17, 13 sections)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pandas as pd

from motivation_v11.data import ensure_outputs, raw_path, table_path, REPORTS  # noqa


def _read_csv(p):
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()


def _read_jsonl(p):
    out = []
    if not Path(p).exists(): return out
    for line in open(p):
        line = line.strip()
        if line: out.append(json.loads(line))
    return out


def _md(df, max_rows=40):
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

    inventory = _read_csv(_REPO / "outputs" / "provenance" / "appworld_task_inventory.csv")
    transition = _read_csv(table_path("transition_matrix_by_prompt_selector_round.csv"))
    dqcg = _read_csv(table_path("distribution_quality_calibration_gap.csv"))
    sum_pf = _read_csv(table_path("prompt_family_behavior_summary.csv"))
    utvs = _read_csv(table_path("ut_vs_utco_headroom.csv"))
    stress = _read_csv(table_path("stress_invariance_by_prompt_selector.csv"))
    crosseval = _read_csv(table_path("best_c1_vs_best_ck_cross_eval.csv"))
    sel = _read_csv(table_path("selector_recovery_summary.csv"))
    sel_trans = _read_csv(table_path("selector_transition_summary.csv"))
    curve = _read_csv(table_path("pass_at_n_curve.csv"))
    bootstrap = _read_csv(table_path("bootstrap_confidence_intervals.csv"))

    full_runs = _read_jsonl(raw_path("full_context_runs.jsonl"))
    n_full = len(full_runs); n_full_pass = sum(1 for r in full_runs if r.get("full_success"))

    case_study_dir = REPORTS / "case_studies"
    case_files = list(case_study_dir.glob("*.md")) if case_study_dir.exists() else []

    lines = []
    lines.append("# Motivation v11 Results\n")
    lines.append(f"> Auto-written by `scripts/16_write_report.py` at "
                 f"{dt.datetime.utcnow().strftime('%Y-%m-%d %H:%MZ')}.\n")
    lines.append("> Hand-written paper-tier companion lives at "
                 "`docs/04_results_summary.md` (to be added after manual review).\n")
    lines.append("> Spec source: "
                 "`user_feedback/motivation_v11_final_train_dev_transition_experiment.md`\n\n")

    # 1. Setup and Scope
    lines.append("## 1. Setup and Scope\n")
    lines.append("* AppWorld task pool: train + dev (145 total).\n")
    lines.append("* 4 prompt families: general_task_agnostic, general_task_aware, ACON_UT, ACON_UTCO.\n")
    lines.append("* N = 8 stochastic samples per case (greedy + N seeds 1000..1007).\n")
    lines.append("* Stress depth K = 2 deterministic recompression rounds.\n")
    lines.append("* MiniMax-M2.5 for compressor, downstream agent, and all verbal selectors.\n")
    lines.append("* **Evaluation protocol**: trajectory-derived (spec §7.2 fallback) — the "
                 "local productive_agents runner does not support env-restore checkpoint continuation.\n\n")

    # 2. Full-Context Baseline and Task Inventory
    lines.append("## 2. Full-Context Baseline and Task Inventory\n")
    lines.append(f"* Tasks attempted: **{n_full}**\n")
    lines.append(f"* Baseline pass: **{n_full_pass}** / {n_full} = "
                 f"{100*n_full_pass/max(n_full,1):.1f} %\n")
    lines.append(f"* Task inventory provenance: `outputs/provenance/appworld_task_inventory.csv` "
                 f"({len(inventory)} rows)\n\n")

    # 3. Prompt Families: Generic vs Structured Compression
    lines.append("## 3. Prompt Families: Generic vs Structured Compression\n")
    lines.append("Per (split × family × selector × round). For paper figures use the "
                 "rows where `split = combined`.\n\n")
    lines.append(_md(sum_pf, max_rows=80))

    # 4. ★ Full-vs-Compressed Transition Analysis (THE HEADLINE)
    lines.append("\n## 4. Full-vs-Compressed Transition Analysis (★ headline)\n")
    lines.append("```\n"
                 "preserve_success  = P(F=1, C=1)\n"
                 "harm              = P(F=1, C=0)\n"
                 "rescue            = P(F=0, C=1)\n"
                 "both_fail         = P(F=0, C=0)\n"
                 "overall_gain_pp   = 100 * (compressed_pass_rate - full_pass_rate)\n"
                 "                  = 100 * (rescue_rate - harm_rate)\n"
                 "```\n\n")
    lines.append(_md(transition, max_rows=80))

    # 5. Distribution Quality vs Decoding Calibration Gap
    lines.append("\n## 5. Distribution Quality vs Decoding Calibration Gap (spec §2.3)\n")
    lines.append("`q_dist_preserve = P(BestN=1 | F=1)` and `q_dist_rescue = P(BestN=1 | F=0)` "
                 "decompose the headroom by what the full-context baseline did.\n\n")
    lines.append(_md(dqcg, max_rows=40))

    # 6. UT vs UTCO
    lines.append("\n## 6. UT vs UTCO: Does Compression Optimization Improve the Distribution or Selection?\n")
    lines.append(_md(utvs))

    # 7. Serial Recompression Robustness: C1 vs CK
    lines.append("\n## 7. Serial Recompression Robustness: C1 vs CK (spec §13.3)\n")
    lines.append(_md(stress, max_rows=40))
    lines.append("\n### 7.1 Best-C1 vs Best-CK cross-evaluation\n")
    lines.append(_md(crosseval, max_rows=20))

    # 8. Selector Analysis
    lines.append("\n## 8. Selector Analysis: Greedy, Verbal Proxies, and Oracle Best-of-N (spec §13.6)\n")
    lines.append(_md(sel_trans if not sel_trans.empty else sel, max_rows=60))

    # 9. Length and Pass-per-Token Analysis
    lines.append("\n## 9. Length and Pass-per-Token Analysis\n")
    lines.append("Pass@N curve (`outputs/tables/pass_at_n_curve.csv`):\n")
    lines.append(_md(curve, max_rows=40))

    # 10. Train vs Dev Split Consistency
    lines.append("\n## 10. Train vs Dev Split Consistency\n")
    lines.append("Every table above uses `split ∈ {train, dev, combined}` so trends "
                 "can be cross-checked. Bootstrap CIs over 2,000 paired resamples "
                 "for the 8 spec-required comparisons:\n\n")
    lines.append(_md(bootstrap, max_rows=40))

    # 11. Representative Case Studies
    lines.append("\n## 11. Representative Case Studies (spec §18)\n")
    if case_files:
        for f in sorted(case_files):
            lines.append(f"* `{f.relative_to(REPORTS.parent)}`")
        lines.append("")
    else:
        lines.append("*(no case studies yet — run stage 15 to populate)*\n")

    # 12. Paper-Facing Takeaways
    lines.append("\n## 12. Paper-Facing Takeaways\n")
    lines.append("(See hand-written `docs/04_results_summary.md` for the paper-quality "
                 "interpretation. The auto-report does not attempt narrative claims.)\n")

    # 13. Limitations and Failure Cases
    lines.append("\n## 13. Limitations and Failure Cases\n")
    lines.append("* Evaluation protocol is `trajectory_derived` (spec §7.2 fallback): "
                 "the local AppWorld runner does not support env-restore for the spec-preferred "
                 "online-checkpoint protocol. Rescue interpretations carry this caveat.\n")
    lines.append("* MiniMax-M2.5 has non-determinism at temperature 0.0 "
                 "(observed ~25 % run-to-run variance in v9/v10). Split-level statistics "
                 "and bootstrap CIs are therefore more reliable than any single greedy run.\n")
    lines.append("* Entropy selector initially runs only on ACON_UTCO under plan (β). "
                 "Plan (α) — entropy on all 4 families — is a +5 h incremental extension "
                 "via `--families general_task_agnostic,general_task_aware,ACON_UT,ACON_UTCO` "
                 "on stage 08c.\n")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"[16] wrote {args.out}")


if __name__ == "__main__":
    main()
