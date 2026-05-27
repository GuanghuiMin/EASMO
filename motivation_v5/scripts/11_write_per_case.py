"""Stage 11 — write per-case markdown reports for spec deliverable §21.10.

One markdown file per audited case under
outputs/reports/per_case_markdown/<case_id>.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def main():
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v5.data import DATA, PER_CASE, RAW, read_jsonl

    cases = {c["case_id"]: c for c in read_jsonl(DATA / "sampled_cases.jsonl")}
    case_audits = {r["case_id"]: r for r in read_jsonl(RAW / "qwen_case_audits.jsonl")}
    add_audits = {r["case_id"]: r for r in read_jsonl(RAW / "qwen_addition_audits.jsonl")}
    rec_audits = {r["case_id"]: r for r in read_jsonl(RAW / "qwen_recompression_audits.jsonl")}
    verifications = {r["case_id"]: r for r in read_jsonl(RAW / "minimax_verifications.jsonl")} if (RAW / "minimax_verifications.jsonl").exists() else {}
    rule_records = read_jsonl(RAW / "rule_based_grounding.jsonl") if (RAW / "rule_based_grounding.jsonl").exists() else []
    rule_by_task = {r["task_id"]: r for r in rule_records}

    PER_CASE.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for cid, case in cases.items():
        ca = (case_audits.get(cid) or {}).get("audit", {}) or {}
        ad = (add_audits.get(cid) or {}).get("audit", {}) or {}
        rc = (rec_audits.get(cid) or {}).get("audit", {}) or {}
        vf = (verifications.get(cid) or {}).get("verification", {}) or {}
        rule = rule_by_task.get(case["task_id"], {}) or {}

        lines = []
        push = lines.append
        push(f"# Case audit: {cid}")
        push("")
        push(f"- Task: `{case['task_id']}`  (budget=`max_steps={case['budget_max_steps']}`, difficulty=`{case.get('difficulty')}`)")
        push(f"- baseline_success: **{case['baseline_success']}**, baseline_env_steps: {case['baseline_env_steps']}")
        push(f"- acon_success: **{case['acon_success']}**, acon_env_steps: {case['acon_env_steps']}, step_ratio: {case['step_ratio']}")
        if case.get("final_after_recompression_success") is not None:
            push(f"- final_after_recompression_success: **{case['final_after_recompression_success']}**")
        push("")
        push(f"## User instruction")
        push("")
        push(f"> {case.get('user_instruction','')[:600]}")
        push("")
        push(f"## Qwen case-level audit (primary)")
        push("")
        if ca.get("parse_failed"):
            push("_Parse failed_")
        else:
            push(f"- primary_failure_mode: **{ca.get('primary_failure_mode')}**")
            push(f"- secondary_failure_modes: {ca.get('secondary_failure_modes', [])}")
            push(f"- is_compression_caused: **{ca.get('is_compression_caused')}**")
            push(f"- reliability_score: {ca.get('reliability_score')}")
            push(f"- mechanism: _{ca.get('concise_failure_mechanism_summary','')}_")
            mi = ca.get("missing_information") or []
            if mi:
                push("")
                push(f"### Missing information ({len(mi)})")
                for x in mi[:10]:
                    push(f"  - **{x.get('info_type')}** `{x.get('missing_item','')[:120]}` "
                         f"({x.get('criticality')}): {x.get('why_it_matters','')[:200]}")
            dist = ca.get("distorted_or_hallucinated_information") or []
            if dist:
                push("")
                push(f"### Distorted/hallucinated ({len(dist)})")
                for x in dist[:5]:
                    push(f"  - {x.get('issue','')[:180]}; impact: {x.get('impact','')[:120]}")
        push("")
        push(f"## Addition audit")
        push("")
        if ad.get("parse_failed"):
            push("_Parse failed or not applicable_")
        else:
            added = ad.get("audit_added_items") or []
            push(f"- {len(added)} audit-added items, "
                 f"{sum(1 for x in added if x.get('grounded_in_baseline'))} grounded.")
            hh = ad.get("audit_added_hallucinations_or_unverified_items") or []
            push(f"- {len(hh)} ungrounded/hallucinated additions.")
            net = ad.get("net_effect_of_audit") or {}
            push(f"- net effect: adds_grounded_critical_info={net.get('adds_grounded_critical_info')}, "
                 f"adds_noise={net.get('adds_noise_or_hallucination')}")
            for x in added[:8]:
                push(f"  - `{x.get('category')}`: {x.get('added_item','')[:120]} "
                     f"(grounded={x.get('grounded_in_baseline')}, "
                     f"criticality={x.get('criticality')})")
        push("")
        push(f"## Recompression-loss audit")
        push("")
        if rc.get("parse_failed"):
            push("_Parse failed or not applicable_")
        else:
            dropped = rc.get("recovered_then_dropped_items") or []
            push(f"- {len(dropped)} recovered-then-dropped items.")
            for x in dropped[:8]:
                push(f"  - `{x.get('category')}` (criticality={x.get('criticality')}, "
                     f"reason=`{x.get('likely_reason_compressor_dropped_it')}`):")
                push(f"    - item: {x.get('item','')[:160]}")
                push(f"    - effect on agent: {x.get('expected_effect_on_agent','')[:160]}")
        push("")
        push(f"## MiniMax verification")
        push("")
        if not vf:
            push("_Not selected for verification._")
        elif vf.get("parse_failed"):
            push("_Parse failed._")
        else:
            push(f"- supports_qwen: **{vf.get('qwen_audit_supported')}**")
            push(f"- verified_primary_failure_mode: {vf.get('verified_primary_failure_mode')}")
            push(f"- verified_compression_caused: {vf.get('verified_is_compression_caused')}")
            push(f"- confidence: {vf.get('confidence')}")
            push(f"- verdict: _{vf.get('one_sentence_verdict','')}_")
        push("")
        push(f"## Rule-based grounding check")
        push("")
        push(f"- overall_grounding_score: {rule.get('overall_grounding_score', 'n/a')}")
        push(f"- case_audit_grounding: {rule.get('case_audit_grounding', {})}")
        push(f"- addition_audit_grounding: {rule.get('addition_audit_grounding', {})}")
        push(f"- recompression_audit_grounding: {rule.get('recompression_audit_grounding', {})}")
        push("")

        out_path = PER_CASE / f"{cid}.md"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        n_written += 1
    print(f"[11] wrote {n_written} per-case markdown files -> {PER_CASE}")


if __name__ == "__main__":
    main()
