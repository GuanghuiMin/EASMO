"""Stage 06 — run the 3 Qwen audit passes.

For each sampled case, runs:
  * case-level failure audit (spec §8 / prompt 01)
  * audit-addition audit (spec §9 / prompt 02), if audit_augmented exists
  * recompression-loss audit (spec §10 / prompt 03), if recompressed exists

Outputs:
  outputs/raw/qwen_case_audits.jsonl
  outputs/raw/qwen_addition_audits.jsonl
  outputs/raw/qwen_recompression_audits.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _run_case_audit(case):
    sys.path.insert(0, str(_REPO))
    from motivation_v5.audit import run_case_failure_audit
    from motivation_v5.clients import parse_json
    res = run_case_failure_audit(case)
    parsed = parse_json(res.text) or {"parse_failed": True}
    return {"case_id": case["case_id"], "task_id": case["task_id"],
            "audit": parsed, "elapsed_s": res.elapsed_s,
            "model": res.model, "finish_reason": res.finish_reason,
            "raw": res.text[:8000]}


def _run_addition_audit(case):
    sys.path.insert(0, str(_REPO))
    from motivation_v5.audit import run_addition_audit
    from motivation_v5.clients import parse_json
    if not case.get("audit_augmented_context"):
        return None
    res = run_addition_audit(case)
    parsed = parse_json(res.text) or {"parse_failed": True}
    return {"case_id": case["case_id"], "task_id": case["task_id"],
            "audit": parsed, "elapsed_s": res.elapsed_s,
            "model": res.model, "finish_reason": res.finish_reason,
            "raw": res.text[:8000]}


def _run_recompression_audit(case):
    sys.path.insert(0, str(_REPO))
    from motivation_v5.audit import run_recompression_audit
    from motivation_v5.clients import parse_json
    if not case.get("recompressed_context"):
        return None
    res = run_recompression_audit(case)
    parsed = parse_json(res.text) or {"parse_failed": True}
    return {"case_id": case["case_id"], "task_id": case["task_id"],
            "audit": parsed, "elapsed_s": res.elapsed_s,
            "model": res.model, "finish_reason": res.finish_reason,
            "raw": res.text[:8000]}


def _run_pool(cases, fn, label, workers, out_path):
    t0 = time.time()
    n_done = 0
    n_err = 0
    out_records = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, c): c for c in cases}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                if r is None:
                    continue
                out_records.append(r)
                if r["audit"].get("parse_failed"):
                    n_err += 1
            except Exception as exc:
                n_err += 1
                cid = futures[fut]["case_id"]
                out_records.append({"case_id": cid, "audit": {"parse_failed": True,
                                                              "error": str(exc)}})
            n_done += 1
            elapsed = time.time() - t0
            eta = (len(cases) - n_done) * elapsed / max(n_done, 1)
            print(f"  [{label} {n_done:>3d}/{len(cases)}] elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min err={n_err}",
                  flush=True)

    with open(out_path, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(out_records)} rows -> {out_path} (err={n_err})")
    return out_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v5.data import DATA, RAW, read_jsonl

    cases = read_jsonl(DATA / "sampled_cases.jsonl")
    print(f"[06] {len(cases)} sampled cases; workers={args.workers}")

    print()
    print("==== 06a case_failure_audit ====")
    _run_pool(cases, _run_case_audit, "case",
              args.workers, RAW / "qwen_case_audits.jsonl")

    print()
    print("==== 06b addition_audit ====")
    cases_with_aug = [c for c in cases if c.get("audit_augmented_context")]
    _run_pool(cases_with_aug, _run_addition_audit, "add",
              args.workers, RAW / "qwen_addition_audits.jsonl")

    print()
    print("==== 06c recompression_audit ====")
    cases_with_rec = [c for c in cases if c.get("recompressed_context")]
    _run_pool(cases_with_rec, _run_recompression_audit, "rec",
              args.workers, RAW / "qwen_recompression_audits.jsonl")


if __name__ == "__main__":
    main()
