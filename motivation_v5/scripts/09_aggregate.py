"""Stage 09 — aggregate the spec tables (§13.2/13.3 + 4 metrics tables).

Outputs (under outputs/tables/):
  failure_mode_counts.csv
  audit_added_facts.csv
  recovered_then_dropped.csv
  critical_info_loss.csv
  model_agreement.csv
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _write_csv(path: Path, rows: list, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def main():
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v5.data import DATA, RAW, TABLES, read_jsonl

    merged = read_jsonl(RAW / "merged_case_audits.jsonl")
    case_audits = {r["case_id"]: r for r in read_jsonl(RAW / "qwen_case_audits.jsonl")}
    add_audits = {r["case_id"]: r for r in read_jsonl(RAW / "qwen_addition_audits.jsonl")}
    rec_audits = {r["case_id"]: r for r in read_jsonl(RAW / "qwen_recompression_audits.jsonl")}
    cases = {c["case_id"]: c for c in read_jsonl(DATA / "sampled_cases.jsonl")}

    # ------------------------------------------------------------------
    # Table 1: failure_mode_counts.csv  (spec §15)
    # ------------------------------------------------------------------
    rows1 = []
    fmode_cnt = Counter(m["primary_failure_mode"] for m in merged)
    fmode_by_diff = Counter(
        (m["primary_failure_mode"], m.get("difficulty", "unknown")) for m in merged
    )
    fmode_by_budget = Counter(
        (m["primary_failure_mode"], m.get("budget_max_steps", "?")) for m in merged
    )
    for mode, n in fmode_cnt.most_common():
        rows1.append({
            "primary_failure_mode": mode,
            "n_cases": n,
            "n_easy":   fmode_by_diff.get((mode, "easy"), 0),
            "n_medium": fmode_by_diff.get((mode, "medium"), 0),
            "n_hard":   fmode_by_diff.get((mode, "hard"), 0),
            "n_cap15":  fmode_by_budget.get((mode, 15), 0),
            "n_cap8":   fmode_by_budget.get((mode, 8), 0),
        })
    n = _write_csv(
        TABLES / "failure_mode_counts.csv", rows1,
        ["primary_failure_mode", "n_cases", "n_easy", "n_medium", "n_hard",
         "n_cap15", "n_cap8"],
    )
    print(f"[09] failure_mode_counts.csv: {n} rows")

    # ------------------------------------------------------------------
    # Table 2: audit_added_facts.csv  (spec §13.2)
    # ------------------------------------------------------------------
    rows2 = []
    for cid, rec in add_audits.items():
        a = rec.get("audit", {})
        if a.get("parse_failed"):
            continue
        case = cases.get(cid, {})
        for item in (a.get("audit_added_items") or []):
            rows2.append({
                "task_id": case.get("task_id"),
                "case_id": cid,
                "difficulty": case.get("difficulty"),
                "category": item.get("category"),
                "added_item": (item.get("added_item") or "")[:240],
                "grounded_in_baseline": item.get("grounded_in_baseline"),
                "criticality": item.get("criticality"),
                "is_actionable": item.get("is_actionable"),
                "already_present_in_acon": item.get("already_present_in_acon"),
                "baseline_evidence": (item.get("baseline_evidence") or "")[:240],
                "audit_augmented_excerpt": (item.get("audit_augmented_excerpt") or "")[:240],
            })
    n = _write_csv(
        TABLES / "audit_added_facts.csv", rows2,
        ["task_id", "case_id", "difficulty", "category", "added_item",
         "grounded_in_baseline", "criticality", "is_actionable",
         "already_present_in_acon", "baseline_evidence", "audit_augmented_excerpt"],
    )
    print(f"[09] audit_added_facts.csv: {n} rows")

    # ------------------------------------------------------------------
    # Table 3: recovered_then_dropped.csv  (spec §13.3)
    # ------------------------------------------------------------------
    rows3 = []
    for cid, rec in rec_audits.items():
        a = rec.get("audit", {})
        if a.get("parse_failed"):
            continue
        case = cases.get(cid, {})
        for item in (a.get("recovered_then_dropped_items") or []):
            rows3.append({
                "task_id": case.get("task_id"),
                "case_id": cid,
                "difficulty": case.get("difficulty"),
                "category": item.get("category"),
                "item": (item.get("item") or "")[:240],
                "criticality": item.get("criticality"),
                "likely_reason_compressor_dropped_it":
                    item.get("likely_reason_compressor_dropped_it"),
                "baseline_evidence": (item.get("baseline_evidence") or "")[:240],
                "audit_augmented_excerpt": (item.get("audit_augmented_excerpt") or "")[:240],
                "recompressed_absent_or_changed_evidence":
                    (item.get("recompressed_absent_or_changed_evidence") or "")[:240],
                "expected_effect_on_agent": (item.get("expected_effect_on_agent") or "")[:200],
            })
    n = _write_csv(
        TABLES / "recovered_then_dropped.csv", rows3,
        ["task_id", "case_id", "difficulty", "category", "item",
         "criticality", "likely_reason_compressor_dropped_it",
         "baseline_evidence", "audit_augmented_excerpt",
         "recompressed_absent_or_changed_evidence",
         "expected_effect_on_agent"],
    )
    print(f"[09] recovered_then_dropped.csv: {n} rows")

    # ------------------------------------------------------------------
    # Table 4: critical_info_loss.csv (one row per merged case)
    # ------------------------------------------------------------------
    rows4 = []
    for m in merged:
        rows4.append({
            "case_id": m["case_id"],
            "task_id": m["task_id"],
            "difficulty": m["difficulty"],
            "budget_max_steps": m["budget_max_steps"],
            "acon_success": m["acon_success"],
            "final_after_recompression_success": m["final_after_recompression_success"],
            "primary_failure_mode": m["primary_failure_mode"],
            "n_missing_items": m["n_missing_items"],
            "n_audit_added_items": m["n_audit_added_items"],
            "n_grounded_audit_added_items": m["n_grounded_audit_added_items"],
            "n_critical_audit_added_items": m["n_critical_audit_added_items"],
            "n_recovered_then_dropped_items": m["n_recovered_then_dropped_items"],
            "critical_recovered_then_dropped": m["critical_recovered_then_dropped"],
            "compression_fault_probability": m["compression_fault_probability"],
            "qwen_reliability": m["qwen_reliability"],
            "rule_grounding_score": m["rule_grounding_score"],
        })
    n = _write_csv(
        TABLES / "critical_info_loss.csv", rows4,
        list(rows4[0].keys()) if rows4 else [],
    )
    print(f"[09] critical_info_loss.csv: {n} rows")

    # ------------------------------------------------------------------
    # Table 5: model_agreement.csv  (spec §14.5)
    # ------------------------------------------------------------------
    verified = [m for m in merged if m.get("minimax_verified")]
    n_total = len(verified)
    n_mode_agree = sum(
        1 for m in verified
        if m.get("minimax_verified_primary_mode") == m.get("primary_failure_mode")
    )
    n_compcaused_agree = sum(
        1 for m in verified
        if (m.get("minimax_verified_compression_caused") is not None
            and bool(m.get("minimax_verified_compression_caused"))
                == bool(m.get("is_compression_caused")))
    )
    n_rtd_agree = sum(
        1 for m in verified
        if (m.get("minimax_verified_recovered_then_dropped") is not None
            and bool(m.get("minimax_verified_recovered_then_dropped"))
                == bool(m.get("critical_recovered_then_dropped")))
    )
    rule_grounding_mean = (sum(m["rule_grounding_score"] for m in merged) / len(merged)
                           if merged else 0.0)
    rows5 = [
        {"metric": "qwen_minimax_primary_mode_agreement",
         "value": round(n_mode_agree / max(n_total, 1), 4),
         "n_total_verified": n_total},
        {"metric": "qwen_minimax_compression_causality_agreement",
         "value": round(n_compcaused_agree / max(n_total, 1), 4),
         "n_total_verified": n_total},
        {"metric": "qwen_minimax_recovered_then_dropped_agreement",
         "value": round(n_rtd_agree / max(n_total, 1), 4),
         "n_total_verified": n_total},
        {"metric": "rule_based_mean_grounding_score",
         "value": round(rule_grounding_mean, 4),
         "n_total_verified": len(merged)},
    ]
    n = _write_csv(
        TABLES / "model_agreement.csv", rows5,
        ["metric", "value", "n_total_verified"],
    )
    print(f"[09] model_agreement.csv: {n} rows")

    # ------------------------------------------------------------------
    # Pretty-print headline metrics (spec §14)
    # ------------------------------------------------------------------
    n_total_cases = len(merged)
    n_comp = sum(1 for m in merged if m.get("is_compression_caused"))
    n_reason = sum(1 for m in merged if m["primary_failure_mode"]
                   == "AGENT_REASONING_FAILURE_NOT_COMPRESSION")
    n_rec_dropped_total = sum(m["n_recovered_then_dropped_items"] for m in merged)
    n_grounded_added_total = sum(m["n_grounded_audit_added_items"] for m in merged)
    rtd_rate = (n_rec_dropped_total / max(n_grounded_added_total, 1))

    print()
    print("=== Headline metrics ===")
    print(f"  n_total_cases:                        {n_total_cases}")
    print(f"  compression_caused_rate:              {n_comp/max(n_total_cases,1)*100:.0f}%")
    print(f"  agent_reasoning_failure_rate:         {n_reason/max(n_total_cases,1)*100:.0f}%")
    print(f"  total grounded audit-added items:     {n_grounded_added_total}")
    print(f"  total recovered_then_dropped items:   {n_rec_dropped_total}")
    print(f"  recovered_then_dropped_rate:          {rtd_rate*100:.0f}%")
    print(f"  rule-based mean grounding score:      {rule_grounding_mean:.3f}")
    print(f"  qwen-minimax primary mode agreement:  {n_mode_agree/max(n_total,1)*100:.0f}% (n_verified={n_total})")


if __name__ == "__main__":
    main()
