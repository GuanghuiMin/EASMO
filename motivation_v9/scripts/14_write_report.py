"""Stage 14 — write v9 deterministic results summary report."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v9.data import (  # noqa
    ensure_outputs, raw_path, table_path, read_jsonl, PROVENANCE,
)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        return f"{float(v):.3f}"
    except Exception:
        return str(v)


def _csv(name: str) -> pd.DataFrame:
    p = table_path(name)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=str(_REPO / "outputs/reports/motivation_v9_results_summary.md"))
    args = ap.parse_args()
    ensure_outputs()

    summary  = _csv("best_of_n_summary.csv")
    by_case  = _csv("best_of_n_by_case.csv")
    spread   = _csv("reward_spread_by_case.csv")
    fragility= _csv("c1_ck_fragility_by_model.csv")
    trans    = _csv("c1_ck_transition.csv")
    chunk_by_type = _csv("chunk_advantage_by_type.csv")
    chunk_adv = _csv("chunk_information_advantage.csv")

    cfg_path = PROVENANCE / "run_config.json"
    cfg = {}
    if cfg_path.exists():
        import json
        cfg = json.loads(cfg_path.read_text())

    prov_path = PROVENANCE / "prompt_sha256.json"
    prov = {}
    if prov_path.exists():
        import json
        prov = json.loads(prov_path.read_text())

    n_cases = (len(read_jsonl(_REPO / "data" / "v9_cases.jsonl"))
               if (_REPO / "data" / "v9_cases.jsonl").exists() else 0)
    n_cand = (len(read_jsonl(raw_path("candidate_compressions.jsonl")))
              if raw_path("candidate_compressions.jsonl").exists() else 0)
    n_runs = (len(read_jsonl(raw_path("behavior_runs_c1_ck.jsonl")))
              if raw_path("behavior_runs_c1_ck.jsonl").exists() else 0)
    n_chunks = (len(read_jsonl(raw_path("chunks.jsonl")))
                if raw_path("chunks.jsonl").exists() else 0)
    n_labels = (len(read_jsonl(raw_path("chunk_type_labels.jsonl")))
                if raw_path("chunk_type_labels.jsonl").exists() else 0)
    n_ablation = (len(read_jsonl(raw_path("chunk_ablation_behavior_runs.jsonl")))
                  if raw_path("chunk_ablation_behavior_runs.jsonl").exists() else 0)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    lines = []
    lines.append("# motivation_v9 Results — Behavioral Compression Stress and Chunk Information Advantage")
    lines.append("")
    lines.append(f"> Auto-written by `scripts/14_write_report.py` at {ts}.")
    lines.append("")
    lines.append("## TL;DR")
    lines.append("- Behavioral validation of v7/v8 surface-type abstraction prior findings.")
    lines.append("- Tests if ACON's greedy compression is optimal under its own distribution.")
    lines.append("- Tests if one-step compression survives repeated-compression stress.")
    lines.append("- Tests if natural-language chunks (causal/control) carry behavioral information beyond entity recall.")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- n_cases = **{n_cases}**")
    lines.append(f"- n_candidate_compressions = **{n_cand}**")
    lines.append(f"- n_behavior_runs (C1+CK) = **{n_runs}**")
    lines.append(f"- n_chunks = **{n_chunks}**")
    lines.append(f"- n_chunk_labels = **{n_labels}**")
    lines.append(f"- n_chunk_ablation_runs = **{n_ablation}**")
    lines.append(f"- ACON UTCO commit = `{prov.get('acon_repo_commit', '?')}`")
    lines.append(f"- ACON history prompt sha256 = `{prov.get('history_prompt_sha256', '?')}`")
    lines.append("")

    lines.append("## Claim 1: ACON Best-of-N Behavioral Gap")
    lines.append("")
    if not summary.empty:
        lines.append("Per (compressor_model, eval_context_round):")
        lines.append("")
        lines.append(summary.round(3).to_markdown(index=False))
        lines.append("")
        ck = summary[summary["eval_context_round"] == "CK"]
        if not ck.empty:
            owr = float(ck["oracle_win_rate"].max())
            best_gain = float(ck["pass_gain_pp"].max())
            verdict = ("STRONG POSITIVE" if (owr >= 0.25 or best_gain >= 10)
                       else "MODERATE / NEGATIVE")
            lines.append(f"**Verdict Claim 1:** {verdict} (oracle_win_rate_CK={owr:.2f}, best gain={best_gain:.1f} pp)")
    lines.append("")

    lines.append("## Claim 2: C1 vs CK Pass Fragility")
    lines.append("")
    if not fragility.empty:
        lines.append(fragility.round(3).to_markdown(index=False))
        lines.append("")
        max_frag = float(fragility["fragility_rate"].max())
        max_drop = float(fragility["stress_drop_pp"].max())
        verdict = ("STRONG POSITIVE" if (max_frag >= 0.20 or max_drop >= 10)
                   else "WEAK / NEGATIVE")
        lines.append(f"**Verdict Claim 2:** {verdict} (max fragility_rate={max_frag:.2f}, max stress drop={max_drop:.1f} pp)")
    lines.append("")

    lines.append("## Claim 3: Chunk Information Advantage")
    lines.append("")
    if not chunk_by_type.empty:
        lines.append(chunk_by_type.round(3).to_markdown(index=False))
        lines.append("")
        causal = chunk_by_type[chunk_by_type["chunk_type"].isin([
            "CAUSAL_PRECONDITION", "CONTROL_NEGATIVE_EVIDENCE", "ACTION_OUTCOME"
        ])]
        entity = chunk_by_type[chunk_by_type["chunk_type"] == "ENTITY_LIST_ONLY"]
        if not causal.empty and not entity.empty:
            causal_mean = float(causal["mean_score_advantage"].mean())
            entity_mean = float(entity["mean_score_advantage"].mean())
            verdict = ("STRONG POSITIVE" if causal_mean > entity_mean
                       else "WEAK / NEGATIVE")
            lines.append(f"**Verdict Claim 3:** {verdict} "
                         f"(causal/control mean adv={causal_mean:.3f} vs "
                         f"entity-only mean adv={entity_mean:.3f})")
    lines.append("")

    lines.append("## What This Motivates for RL")
    lines.append("- If Claim 1 + 2 pass: train compressor with reward = behavior_after_stress(T^K) - λ·length.")
    lines.append("- If Claim 3 passes: per-chunk information advantage supports IAPO-style natural-language credit assignment.")
    lines.append("")
    lines.append("## Negative / Null Results")
    lines.append("(Filled in after manual review.)")
    lines.append("")
    lines.append("## Files of record")
    for n in ("best_of_n_by_case", "best_of_n_summary", "reward_spread_by_case",
              "c1_ck_transition", "c1_ck_fragility_by_model",
              "stress_chain_convergence",
              "chunk_information_advantage", "chunk_advantage_by_type"):
        if table_path(n + ".csv").exists():
            lines.append(f"- `tables/{n}.csv`")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"[14] wrote {args.out}")


if __name__ == "__main__":
    main()
