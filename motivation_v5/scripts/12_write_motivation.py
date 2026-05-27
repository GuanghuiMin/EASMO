"""Stage 12 — write motivation_summary.md using aggregate stats + LLM.

Per spec §12, calls the aggregate-summary prompt with the
final aggregated stats and a small set of representative cases.
The output is a markdown report under outputs/reports/motivation_summary.md.

Falls back to a deterministic template if the LLM call fails.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _load_csv(p: Path):
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def main():
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v5.data import DATA, RAW, TABLES, REPORTS, read_jsonl
    from motivation_v5.audit import AGGREGATE_SUMMARY_TEMPLATE
    from motivation_v5.clients import chat_minimax, pack_prompt_for_qwen

    merged = read_jsonl(RAW / "merged_case_audits.jsonl")
    failure_rows = _load_csv(TABLES / "failure_mode_counts.csv")
    rtd_rows = _load_csv(TABLES / "recovered_then_dropped.csv")
    audit_added_rows = _load_csv(TABLES / "audit_added_facts.csv")
    agree_rows = _load_csv(TABLES / "model_agreement.csv")

    n = len(merged)
    n_comp = sum(1 for m in merged if m.get("is_compression_caused"))
    n_reason = sum(
        1 for m in merged if m["primary_failure_mode"]
        == "AGENT_REASONING_FAILURE_NOT_COMPRESSION"
    )
    n_grounded = sum(m["n_grounded_audit_added_items"] for m in merged)
    n_dropped = sum(m["n_recovered_then_dropped_items"] for m in merged)
    n_critical_dropped = sum(1 for m in merged if m["critical_recovered_then_dropped"])
    n_recompressed_succ = sum(
        1 for m in merged
        if m.get("final_after_recompression_success") is True
    )
    n_recompressed_fail = sum(
        1 for m in merged
        if m.get("final_after_recompression_success") is False
    )
    acon_success = sum(1 for m in merged if m["acon_success"])
    acon_fail = n - acon_success

    aggregate_stats = {
        "n_cases": n,
        "acon_success": acon_success,
        "acon_failure": acon_fail,
        "n_compression_caused_failures": n_comp,
        "compression_caused_rate": round(n_comp / max(n, 1), 4),
        "n_reasoning_failures": n_reason,
        "reasoning_failure_rate": round(n_reason / max(n, 1), 4),
        "n_grounded_audit_added_items_total": n_grounded,
        "n_recovered_then_dropped_items_total": n_dropped,
        "recovered_then_dropped_rate": round(
            n_dropped / max(n_grounded, 1), 4),
        "n_cases_with_critical_recovered_then_dropped": n_critical_dropped,
        "final_after_recompression_success": n_recompressed_succ,
        "final_after_recompression_failure": n_recompressed_fail,
        "failure_mode_distribution":
            {r["primary_failure_mode"]: int(r["n_cases"]) for r in failure_rows},
        "recovered_then_dropped_categories": dict(
            Counter(r["category"] for r in rtd_rows).most_common()
        ),
        "audit_added_categories": dict(
            Counter(r["category"] for r in audit_added_rows).most_common()
        ),
        "model_agreement": {r["metric"]: float(r["value"]) for r in agree_rows},
    }

    # Pick 3 representative cases per spec §12: prefer those with
    # critical_recovered_then_dropped=True and high reliability.
    candidates = sorted(
        [m for m in merged if m["critical_recovered_then_dropped"]],
        key=lambda m: (-m["qwen_reliability"], -m["n_recovered_then_dropped_items"]),
    )
    representative = candidates[:3]
    rep_payload = [
        {"case_id": m["case_id"], "task_id": m["task_id"],
         "primary_failure_mode": m["primary_failure_mode"],
         "n_recovered_then_dropped_items": m["n_recovered_then_dropped_items"],
         "qwen_reliability": m["qwen_reliability"],
         "final_failure_summary": m["final_failure_summary"]}
        for m in representative
    ]

    prompt = pack_prompt_for_qwen(  # reuse packer for safety
        AGGREGATE_SUMMARY_TEMPLATE,
        fields={
            "aggregate_stats_json": json.dumps(aggregate_stats, indent=2),
            "representative_cases_json": json.dumps(rep_payload, indent=2),
        },
        reserve_output_tokens=2048,
    )

    print("[12] calling MiniMax for motivation_summary.md ...")
    out_path = REPORTS / "motivation_summary.md"
    try:
        t0 = time.time()
        res = chat_minimax(prompt, temperature=0.0, max_tokens=4096)
        elapsed = time.time() - t0
        if res.text and len(res.text) > 100:
            out_path.write_text(res.text, encoding="utf-8")
            print(f"[12] wrote {out_path} ({len(res.text)} chars, {elapsed:.1f}s)")
            return
        print(f"[12] LLM returned empty/short text; falling back to deterministic template")
    except Exception as exc:
        print(f"[12] LLM call failed: {exc}; falling back to deterministic template")

    # Deterministic fallback
    md = []
    md.append("# Motivation Findings (deterministic fallback)")
    md.append("")
    md.append(f"_Auto-generated from aggregate stats on {n} audited cases. "
              f"LLM aggregator call failed or returned empty; this is the "
              f"deterministic template version._")
    md.append("")
    md.append("## Observation 1: failure-mode distribution")
    md.append("")
    md.append("Top failure modes (from Table failure_mode_counts.csv):")
    md.append("")
    for r in failure_rows[:5]:
        md.append(f"- `{r['primary_failure_mode']}`: {r['n_cases']} cases "
                  f"(easy={r['n_easy']}, medium={r['n_medium']}, hard={r['n_hard']})")
    md.append("")
    md.append("## Observation 2: recovered-then-dropped pattern")
    md.append("")
    md.append(f"- {n_grounded} grounded audit-added items across {n} cases.")
    md.append(f"- {n_dropped} items were dropped again by the recompressor.")
    md.append(f"- recovered_then_dropped_rate = "
              f"**{aggregate_stats['recovered_then_dropped_rate']*100:.0f}%** of grounded "
              f"audit additions are dropped again.")
    md.append(f"- {n_critical_dropped} cases have at least one CRITICAL "
              f"recovered-then-dropped item.")
    md.append("")
    md.append("## Observation 3: compression vs reasoning fault split")
    md.append("")
    md.append(f"- {aggregate_stats['compression_caused_rate']*100:.0f}% of cases are "
              f"compression-caused; {aggregate_stats['reasoning_failure_rate']*100:.0f}% are "
              f"agent-reasoning failures unrelated to compression.")
    md.append("")
    if n_recompressed_succ + n_recompressed_fail > 0:
        md.append("## Observation 4: recompression closed loop")
        md.append("")
        md.append(f"- Re-running the downstream agent with the recompressed_context: "
                  f"**{n_recompressed_succ} success / {n_recompressed_fail} fail** "
                  f"out of {n_recompressed_succ + n_recompressed_fail} cases.")
        md.append("")
    md.append("## Implications for Method Design")
    md.append("")
    md.append("- If recovered_then_dropped_rate is high, the bottleneck is "
              "compressor preservation of audit-recovered actionable state "
              "(motivating a method focused on preserving such state).")
    md.append("- If a specific category dominates (e.g. RUNTIME_VARIABLE or "
              "API_SCHEMA), a targeted compressor design is supported.")
    md.append("- If most failures are reasoning failures (not compression), "
              "do not build a new compressor on these cases.")
    md.append("")
    md.append("## Negative Results / What Not to Pursue")
    md.append("")
    md.append("(filled by reviewer after manual inspection of "
              "outputs/reports/per_case_markdown/*.md)")
    md.append("")
    md.append("## Representative Cases")
    md.append("")
    for r in representative:
        md.append(f"- **{r['case_id']}** ({r['primary_failure_mode']}): "
                  f"{r['final_failure_summary']}")
    md.append("")
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[12] wrote {out_path} (deterministic fallback)")


if __name__ == "__main__":
    main()
