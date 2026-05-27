"""Stage 08 — merge all 4 audit sources into one row per case.

Per spec §13.1, each merged record carries the aggregate fields
needed by the tables/figures stages.

Outputs:
  outputs/raw/merged_case_audits.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def main():
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v5.data import DATA, RAW, read_jsonl, write_jsonl

    cases = {c["case_id"]: c for c in read_jsonl(DATA / "sampled_cases.jsonl")}
    case_audits = {r["case_id"]: r for r in read_jsonl(RAW / "qwen_case_audits.jsonl")}
    add_audits = {r["case_id"]: r for r in read_jsonl(RAW / "qwen_addition_audits.jsonl")}
    rec_audits = {r["case_id"]: r for r in read_jsonl(RAW / "qwen_recompression_audits.jsonl")}
    verifications = {r["case_id"]: r for r in read_jsonl(RAW / "minimax_verifications.jsonl")} if (RAW / "minimax_verifications.jsonl").exists() else {}
    rule_records = {r["task_id"] + "_x": r for r in read_jsonl(RAW / "rule_based_grounding.jsonl")} if (RAW / "rule_based_grounding.jsonl").exists() else {}
    rule_by_case = {}
    if (RAW / "rule_based_grounding.jsonl").exists():
        for r in read_jsonl(RAW / "rule_based_grounding.jsonl"):
            # rule records keyed by task_id only; we need case_id-aware lookup
            # but our verify_all wrote task_id field. Fall back to first match.
            rule_by_case[r["task_id"]] = r

    merged = []
    for cid, case in cases.items():
        ca = (case_audits.get(cid) or {}).get("audit", {}) or {}
        ad = (add_audits.get(cid) or {}).get("audit", {}) or {}
        rc = (rec_audits.get(cid) or {}).get("audit", {}) or {}
        vf = (verifications.get(cid) or {}).get("verification", {}) or {}
        rule = rule_by_case.get(case["task_id"]) or {}

        n_missing = len(ca.get("missing_information") or [])
        n_added = len(ad.get("audit_added_items") or [])
        n_grounded_added = sum(
            1 for x in (ad.get("audit_added_items") or [])
            if x.get("grounded_in_baseline")
        )
        n_critical_added = sum(
            1 for x in (ad.get("audit_added_items") or [])
            if str(x.get("criticality", "")).lower() == "high"
            and x.get("grounded_in_baseline")
        )
        n_rec_dropped = len(rc.get("recovered_then_dropped_items") or [])
        n_critical_rec_dropped = sum(
            1 for x in (rc.get("recovered_then_dropped_items") or [])
            if str(x.get("criticality", "")).lower() == "high"
            and x.get("was_grounded_in_baseline", True)
        )

        record = {
            "task_id": case["task_id"],
            "case_id": cid,
            "difficulty": case.get("difficulty"),
            "budget_max_steps": case.get("budget_max_steps"),
            "compression_type": case.get("compression_type"),
            "acon_variant": case.get("acon_variant"),
            "baseline_success": case.get("baseline_success"),
            "acon_success": case.get("acon_success"),
            "audit_augmented_exists": bool(case.get("audit_augmented_context")),
            "recompressed_exists": bool(case.get("recompressed_context")),
            "final_after_recompression_success": case.get("final_after_recompression_success"),
            "primary_failure_mode": ca.get("primary_failure_mode") or "INSUFFICIENT_EVIDENCE",
            "secondary_failure_modes": ca.get("secondary_failure_modes") or [],
            "is_compression_caused": ca.get("is_compression_caused"),
            "compression_fault_probability": _safe_float(
                (ca.get("compression_vs_reasoning_judgment") or {}).get("compression_fault_probability"),
            ),
            "agent_reasoning_fault_probability": _safe_float(
                (ca.get("compression_vs_reasoning_judgment") or {}).get("agent_reasoning_fault_probability"),
            ),
            "n_missing_items": n_missing,
            "n_audit_added_items": n_added,
            "n_grounded_audit_added_items": n_grounded_added,
            "n_critical_audit_added_items": n_critical_added,
            "n_recovered_then_dropped_items": n_rec_dropped,
            "critical_recovered_then_dropped": n_critical_rec_dropped > 0,
            "qwen_reliability": _safe_float(ca.get("reliability_score"), 0.0),
            "minimax_verified": bool(vf),
            "minimax_supports_qwen": vf.get("qwen_audit_supported"),
            "minimax_verified_primary_mode": vf.get("verified_primary_failure_mode"),
            "minimax_verified_compression_caused": vf.get("verified_is_compression_caused"),
            "minimax_verified_recovered_then_dropped": vf.get("verified_recovered_then_dropped"),
            "minimax_confidence": _safe_float(vf.get("confidence"), 0.0),
            "rule_grounding_score": _safe_float(rule.get("overall_grounding_score"), 0.0),
            "rule_case_n_items": (rule.get("case_audit_grounding") or {}).get("n_items", 0),
            "rule_case_n_grounded": (rule.get("case_audit_grounding") or {}).get("n_grounded", 0),
            "rule_add_n_items": (rule.get("addition_audit_grounding") or {}).get("n_items", 0),
            "rule_add_n_grounded_baseline": (rule.get("addition_audit_grounding") or {}).get("n_grounded_baseline", 0),
            "rule_rec_n_items": (rule.get("recompression_audit_grounding") or {}).get("n_items", 0),
            "rule_rec_n_grounded_aug": (rule.get("recompression_audit_grounding") or {}).get("n_grounded_in_augmented", 0),
            "rule_rec_n_absent_from_recomp": (rule.get("recompression_audit_grounding") or {}).get("n_absent_from_recompressed", 0),
            "final_failure_summary": ca.get("concise_failure_mechanism_summary", ""),
        }
        merged.append(record)

    out = RAW / "merged_case_audits.jsonl"
    write_jsonl(out, merged)
    print(f"[08] wrote {len(merged)} merged audit rows -> {out}")


if __name__ == "__main__":
    main()
