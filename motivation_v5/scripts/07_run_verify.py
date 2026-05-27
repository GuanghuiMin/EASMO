"""Stage 07 — verification with MiniMax-M2.5 + rule-based grounding.

Two verifiers, both run per spec §3.3 + user request:

  1. MiniMax LLM verifier (spec §11 / prompt 04) on cases where:
       * qwen reliability_score < 0.7
       * OR Qwen's recompression audit claims drops_critical_audit_recovered_info
       * OR a 20% random sample of remaining cases
  2. Rule-based grounding verifier (deterministic substring check)
     for ALL audited cases — no LLM, gives a hard-fact-check score.

Outputs:
  outputs/raw/minimax_verifications.jsonl
  outputs/raw/rule_based_grounding.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _should_verify(case_audit_record, recompression_audit_record,
                   *, force_random_pick: bool) -> bool:
    if force_random_pick:
        return True
    if case_audit_record:
        rel = case_audit_record.get("audit", {}).get("reliability_score", 1.0)
        try:
            if float(rel) < 0.7:
                return True
        except Exception:
            pass
    if recompression_audit_record:
        rec_judgment = (recompression_audit_record.get("audit", {})
                                                  .get("recompression_judgment", {}))
        if rec_judgment.get("drops_critical_audit_recovered_info"):
            return True
    return False


def _verify_one(case: dict, qwen_audit_json: dict):
    sys.path.insert(0, str(_REPO))
    from motivation_v5.audit import run_verifier
    from motivation_v5.clients import parse_json
    res = run_verifier(case, qwen_audit_json)
    parsed = parse_json(res.text) or {"parse_failed": True}
    return {"case_id": case["case_id"], "task_id": case["task_id"],
            "verification": parsed, "elapsed_s": res.elapsed_s,
            "model": res.model, "raw": res.text[:6000]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--random_verify_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v5.data import DATA, RAW, read_jsonl
    from motivation_v5.rule_verify import verify_all

    rng = random.Random(args.seed)
    cases = read_jsonl(DATA / "sampled_cases.jsonl")
    case_by_id = {c["case_id"]: c for c in cases}

    case_audits = read_jsonl(RAW / "qwen_case_audits.jsonl")
    add_audits = read_jsonl(RAW / "qwen_addition_audits.jsonl")
    rec_audits = read_jsonl(RAW / "qwen_recompression_audits.jsonl")

    case_audit_by_id = {r["case_id"]: r for r in case_audits}
    add_audit_by_id = {r["case_id"]: r for r in add_audits}
    rec_audit_by_id = {r["case_id"]: r for r in rec_audits}

    # ----- Rule-based grounding for ALL cases (cheap, no LLM) -----
    print(f"[07a] rule-based grounding for {len(cases)} cases (no LLM)")
    rule_records = []
    for c in cases:
        cid = c["case_id"]
        case_audit = (case_audit_by_id.get(cid) or {}).get("audit", {})
        add_audit = (add_audit_by_id.get(cid) or {}).get("audit", {})
        rec_audit = (rec_audit_by_id.get(cid) or {}).get("audit", {})
        r = verify_all(
            case=c,
            case_audit=case_audit if not case_audit.get("parse_failed") else None,
            addition_audit=add_audit if not add_audit.get("parse_failed") else None,
            recompression_audit=rec_audit if not rec_audit.get("parse_failed") else None,
        )
        rule_records.append(r.to_dict())
    rule_path = RAW / "rule_based_grounding.jsonl"
    with open(rule_path, "w") as f:
        for r in rule_records:
            f.write(json.dumps(r) + "\n")
    print(f"[07a] wrote {len(rule_records)} rows -> {rule_path}")
    grounding_scores = [r["overall_grounding_score"] for r in rule_records]
    if grounding_scores:
        print(f"      mean grounding_score: {sum(grounding_scores)/len(grounding_scores):.3f}")

    # ----- Decide which cases to verify with MiniMax -----
    selected_ids = set()
    random_pool = []
    for c in cases:
        cid = c["case_id"]
        case_rec = case_audit_by_id.get(cid)
        rec_rec = rec_audit_by_id.get(cid)
        if _should_verify(case_rec, rec_rec, force_random_pick=False):
            selected_ids.add(cid)
        else:
            random_pool.append(cid)
    n_random = max(1, int(args.random_verify_ratio * len(random_pool)))
    for cid in rng.sample(random_pool, min(n_random, len(random_pool))):
        selected_ids.add(cid)

    print(f"\n[07b] minimax verifier on {len(selected_ids)} cases "
          f"(low-confidence/critical-recovered + {args.random_verify_ratio:.0%} random sample); "
          f"workers={args.workers}")

    selected_cases = [case_by_id[cid] for cid in sorted(selected_ids)
                      if cid in case_by_id]
    t0 = time.time()
    out_records = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for c in selected_cases:
            qwen_audit = (case_audit_by_id.get(c["case_id"]) or {}).get("audit", {})
            futures[ex.submit(_verify_one, c, qwen_audit)] = c
        for fut in as_completed(futures):
            try:
                r = fut.result()
                out_records.append(r)
            except Exception as exc:
                cid = futures[fut]["case_id"]
                out_records.append({"case_id": cid,
                                    "verification": {"parse_failed": True,
                                                     "error": str(exc)}})
            n_done += 1
            elapsed = time.time() - t0
            eta = (len(selected_cases) - n_done) * elapsed / max(n_done, 1)
            print(f"  [{n_done:>3d}/{len(selected_cases)}] elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min",
                  flush=True)

    minimax_path = RAW / "minimax_verifications.jsonl"
    with open(minimax_path, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[07b] wrote {len(out_records)} rows -> {minimax_path}")


if __name__ == "__main__":
    main()
