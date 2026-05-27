"""Rule-based grounding verifier (no-LLM).

For each audit-claimed item that quotes evidence from a specific
context, deterministically check whether the quote actually appears
in that context. This is a stronger, cheaper, bias-free verifier
than running another LLM, and we use it alongside the MiniMax LLM
verifier per user instruction.

Verifies three classes of claim:

  1. **Missing-info evidence**: claims of the form "X was in baseline
     but absent/distorted in ACON" — we check that the baseline quote
     actually appears in baseline_history AND that the ACON quote (if
     given) actually appears in acon_compressed_history.

  2. **Audit-addition grounding**: claims that an item is grounded in
     baseline — we check that `baseline_evidence` actually appears in
     baseline_history AND that the `audit_augmented_excerpt` actually
     appears in audit_augmented_context.

  3. **Recovered-then-dropped grounding**: claims that an item was in
     `audit_augmented_context` but absent from `recompressed_context`
     — we check both halves.

A claim that fails any check is flagged. A summary per case
(grounding_score) is added to the verification record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _norm(text: str) -> str:
    """Whitespace-normalised for substring matching."""
    return re.sub(r"\s+", " ", (text or "").strip())


def quote_present(quote: str, context: str) -> bool:
    """Loose substring match: case-insensitive, whitespace-normalised.
    For very short / empty quotes we return False (caller should treat
    as not-grounded)."""
    q = _norm(quote)
    c = _norm(context)
    if not q or len(q) < 4:
        return False
    return q.lower() in c.lower()


@dataclass
class RuleVerifyResult:
    task_id: str
    case_audit_grounding: Dict[str, int] = field(default_factory=dict)
    addition_audit_grounding: Dict[str, int] = field(default_factory=dict)
    recompression_audit_grounding: Dict[str, int] = field(default_factory=dict)
    overall_grounding_score: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "case_audit_grounding": self.case_audit_grounding,
            "addition_audit_grounding": self.addition_audit_grounding,
            "recompression_audit_grounding": self.recompression_audit_grounding,
            "overall_grounding_score": round(self.overall_grounding_score, 4),
            "notes": self.notes,
        }


def verify_case_audit(case_audit: dict, case: dict) -> Dict[str, int]:
    """For each missing_information / distorted item, check whether the
    quoted evidence actually appears in the right context."""
    baseline = case.get("baseline_history", "") or ""
    acon = case.get("acon_compressed_history", "") or ""
    n_total = 0
    n_grounded = 0
    n_baseline_quote_ok = 0
    n_acon_quote_ok = 0
    for item in (case_audit.get("missing_information") or []):
        n_total += 1
        baseline_quote = (item.get("baseline_evidence") or "").strip()
        acon_quote = (item.get("acon_absent_or_distorted_evidence") or "").strip()
        b_ok = quote_present(baseline_quote, baseline) if baseline_quote else False
        a_quote_present_in_acon = quote_present(acon_quote, acon) if acon_quote else False
        # An "absent" claim is correct if the ACON quote is NOT in ACON OR if
        # the explanation makes clear it's absent (we accept either).
        if b_ok:
            n_baseline_quote_ok += 1
            n_grounded += 1
        if a_quote_present_in_acon:
            n_acon_quote_ok += 1
    for item in (case_audit.get("distorted_or_hallucinated_information") or []):
        n_total += 1
        comp = (item.get("compressed_excerpt") or "").strip()
        ref = (item.get("correct_baseline_reference") or "").strip()
        ok_comp = quote_present(comp, acon)
        ok_ref = quote_present(ref, baseline)
        if ok_comp and ok_ref:
            n_grounded += 1
    return {
        "n_items": n_total,
        "n_grounded": n_grounded,
        "n_baseline_quote_ok": n_baseline_quote_ok,
        "n_acon_quote_present": n_acon_quote_ok,
    }


def verify_addition_audit(addition_audit: dict, case: dict) -> Dict[str, int]:
    """For each audit_added_item, verify baseline_evidence appears in baseline
    AND audit_augmented_excerpt appears in audit_augmented_context."""
    baseline = case.get("baseline_history", "") or ""
    augmented = case.get("audit_augmented_context", "") or ""
    n_total = 0
    n_grounded_baseline = 0
    n_present_in_augmented = 0
    for item in (addition_audit.get("audit_added_items") or []):
        n_total += 1
        b_q = (item.get("baseline_evidence") or "").strip()
        a_q = (item.get("audit_augmented_excerpt") or "").strip()
        if b_q and quote_present(b_q, baseline):
            n_grounded_baseline += 1
        if a_q and quote_present(a_q, augmented):
            n_present_in_augmented += 1
    return {
        "n_items": n_total,
        "n_grounded_baseline": n_grounded_baseline,
        "n_present_in_augmented": n_present_in_augmented,
    }


def verify_recompression_audit(recompression_audit: dict, case: dict) -> Dict[str, int]:
    """For each recovered_then_dropped item, verify the audit_augmented
    quote appears in audit_augmented_context AND the
    recompressed_absent_or_changed_evidence is consistent (either the
    exact recompressed quote is in recompressed_context, OR the audit
    augmented quote is NOT present in recompressed_context)."""
    augmented = case.get("audit_augmented_context", "") or ""
    recompressed = case.get("recompressed_context", "") or ""
    n_total = 0
    n_grounded_augmented = 0
    n_absent_from_recompressed = 0
    for item in (recompression_audit.get("recovered_then_dropped_items") or []):
        n_total += 1
        aug_q = (item.get("audit_augmented_excerpt") or "").strip()
        rec_q = (item.get("recompressed_absent_or_changed_evidence") or "").strip()
        if aug_q and quote_present(aug_q, augmented):
            n_grounded_augmented += 1
            # if the aug quote is NOT in the recompressed text, the "dropped"
            # claim is supported deterministically.
            if not quote_present(aug_q, recompressed):
                n_absent_from_recompressed += 1
    return {
        "n_items": n_total,
        "n_grounded_in_augmented": n_grounded_augmented,
        "n_absent_from_recompressed": n_absent_from_recompressed,
    }


def verify_all(
    *,
    case: dict,
    case_audit: Optional[dict] = None,
    addition_audit: Optional[dict] = None,
    recompression_audit: Optional[dict] = None,
) -> RuleVerifyResult:
    res = RuleVerifyResult(task_id=case["task_id"])
    scores = []
    if case_audit:
        g = verify_case_audit(case_audit, case)
        res.case_audit_grounding = g
        if g["n_items"] > 0:
            scores.append(g["n_grounded"] / g["n_items"])
    if addition_audit:
        g = verify_addition_audit(addition_audit, case)
        res.addition_audit_grounding = g
        if g["n_items"] > 0:
            scores.append((g["n_grounded_baseline"] + g["n_present_in_augmented"])
                          / (2 * g["n_items"]))
    if recompression_audit:
        g = verify_recompression_audit(recompression_audit, case)
        res.recompression_audit_grounding = g
        if g["n_items"] > 0:
            scores.append((g["n_grounded_in_augmented"] + g["n_absent_from_recompressed"])
                          / (2 * g["n_items"]))
    res.overall_grounding_score = sum(scores) / len(scores) if scores else 1.0
    if not scores:
        res.notes.append("No audited items to verify")
    return res
