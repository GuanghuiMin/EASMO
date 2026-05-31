"""Stage 12 — auto-write motivation_v10_results_summary.md (spec §18).

Reads:
  outputs/tables/proxy_selection_summary.csv     (Claim 1)
  outputs/tables/grpo_readiness_summary.csv     (Claim 3)
  outputs/tables/proxy_by_case.csv               (per-case detail)
  outputs/tables/c1_ck_fragility_by_generation.csv (Claim 2 from stage 04 post-hoc)
  outputs/tables/chunk_advantage_revised.csv     (Claim 4, if stage 11 ran)

Writes:
  outputs/reports/motivation_v10_results_summary.md

Mirrors v9 stage 14 style — auto-generated tables + per-claim verdict
according to spec §19 Go/No-go thresholds. A separate hand-written
docs/04_results_summary.md will be added after manual review.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pandas as pd

from motivation_v10.data import ensure_outputs, table_path, REPORTS  # noqa


def _read_csv_safe(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "*(no data — table missing or empty)*\n"
    df = df.head(max_rows)
    try:
        return df.to_markdown(index=False) + "\n"
    except Exception:
        # tabulate not installed → ad-hoc
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines) + "\n"


def verdict_claim1(df_proxy):
    if df_proxy.empty:
        return "PENDING (proxy_selection_summary.csv missing)"
    ck = df_proxy[df_proxy["eval_round"] == "CK"]
    if ck.empty:
        return "PENDING"
    r = ck.iloc[0]
    gain_pp = float(r.get("proxy_gain_pp", 0.0))
    rec_gain = float(r.get("recovered_gain_proxy", 0.0))
    if gain_pp >= 10.0 or rec_gain >= 0.40:
        return f"PASS (proxy gain={gain_pp:.1f} pp on CK, recovered={rec_gain:.2f} of oracle gain)"
    return f"FAIL (proxy gain={gain_pp:.1f} pp on CK, recovered={rec_gain:.2f})"


def verdict_claim3(df_grpo):
    if df_grpo.empty:
        return "PENDING (grpo_readiness_summary.csv missing)"
    ck = df_grpo[df_grpo["variant"] == "Qwen-SFT-CK"]
    if ck.empty:
        return "PENDING (Qwen-SFT-CK row missing)"
    r = ck.iloc[0]
    win = float(r.get("oracle_win_rate_over_greedy", 0.0))
    spread = float(r.get("mean_within_case_std", 0.0))
    all_low = float(r.get("all_low_rate", 1.0))
    ok = (win >= 0.50) and (spread >= 0.15) and (all_low <= 0.15)
    return (f"PASS (oracle_win={win:.2f}, std={spread:.3f}, all_low={all_low:.2f})"
            if ok else f"FAIL (oracle_win={win:.2f}, std={spread:.3f}, all_low={all_low:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPORTS / "motivation_v10_results_summary.md"))
    args = ap.parse_args()
    ensure_outputs()

    df_proxy = _read_csv_safe(table_path("proxy_selection_summary.csv"))
    df_grpo  = _read_csv_safe(table_path("grpo_readiness_summary.csv"))
    df_pcase = _read_csv_safe(table_path("proxy_by_case.csv"))
    df_frag  = _read_csv_safe(table_path("c1_ck_fragility_by_generation.csv"))
    df_chunk = _read_csv_safe(table_path("v11_chunk_advantage_by_type.csv"))
    df_chunk_role = _read_csv_safe(table_path("v11_chunk_advantage_by_role.csv"))
    df_chunk_reg = _read_csv_safe(table_path("v11_chunk_advantage_regression.csv"))

    lines = []
    lines.append("# motivation_v10 results (auto-written)\n")
    lines.append(f"> Auto-written by `scripts/12_write_report.py` at "
                 f"{dt.datetime.utcnow().strftime('%Y-%m-%d %H:%MZ')}.\n")
    lines.append("> Honest hand-written counterpart: `docs/04_results_summary.md` (TBD).\n")
    lines.append("")

    lines.append("## TL;DR (per spec §19 Go / No-go)\n")
    lines.append(f"* **Claim 1 — proxy recovers best-of-N gain**: {verdict_claim1(df_proxy)}")
    lines.append(f"* **Claim 2 — stress-selected SFT targets better than one-step**: "
                 f"see §3 student eval table below for raw Qwen vs SFT-C1 vs SFT-CK comparison.")
    lines.append(f"* **Claim 3 — Qwen-SFT-CK has GRPO-trainable reward spread**: "
                 f"{verdict_claim3(df_grpo)}")
    lines.append(f"* **Claim 4 — chunk surface labels insufficient**: requires stage 11 (chunk reanalysis).")
    lines.append("")

    lines.append("## §1 Proxy selection summary (stage 06)\n")
    lines.append(_md_table(df_proxy))

    lines.append("## §2 Per-case proxy detail (head)\n")
    lines.append(_md_table(df_pcase, max_rows=15))

    lines.append("## §3 GRPO readiness summary (stage 10)\n")
    lines.append(_md_table(df_grpo))

    lines.append("## §4 C1-vs-CK fragility (stage 04 post-hoc)\n")
    lines.append(_md_table(df_frag))

    lines.append("## §5 Chunk advantage by type (stage 11c)\n")
    lines.append(_md_table(df_chunk, max_rows=12))

    lines.append("\n## §5b Chunk advantage by functional_role_guess (stage 11c)\n")
    lines.append(_md_table(df_chunk_role, max_rows=12))

    lines.append("\n## §5c Claim 4 regression (label-only R² vs full R²)\n")
    lines.append(_md_table(df_chunk_reg, max_rows=20))

    lines.append("\n## Files of record\n")
    for p in (
        "outputs/raw/v10_baseline_runs.jsonl",
        "outputs/raw/minimax_candidates.jsonl",
        "outputs/raw/stress_chains.jsonl",
        "outputs/raw/behavior_runs_candidates.jsonl",
        "outputs/raw/proxy_verifier_scores.jsonl",
        "outputs/raw/proxy_pairwise_scores.jsonl",
        "outputs/raw/student_compressions.jsonl",
        "outputs/raw/student_behavior_runs.jsonl",
        "outputs/raw/grpo_readiness_*.jsonl",
        "outputs/raw/chunks.jsonl",
        "outputs/raw/chunk_type_labels.jsonl",
        "outputs/raw/chunk_information_advantage.csv",
        "outputs/data/sft_targets_c1.jsonl",
        "outputs/data/sft_targets_ck.jsonl",
        "outputs/models/qwen_sft_c1/",
        "outputs/models/qwen_sft_ck/",
    ):
        lines.append(f"* `{p}`")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[12] wrote {args.out}")


if __name__ == "__main__":
    main()
