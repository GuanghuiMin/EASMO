"""Stage 03 — build audit_augmented_context for each sampled case.

The audit model (Qwen3-4B) reads the baseline trajectory and the ACON
compressed summary, then emits a bracketed [AUDIT_AUGMENTATION] block
of grounded, actionable items that ACON dropped. We append that block
to the ACON summary to form `audit_augmented_context`.

Outputs:
  data/sampled_cases.jsonl  (overwritten with audit_augmented_context filled in)
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _build_one(case: dict):
    sys.path.insert(0, str(_REPO))
    from motivation_v5.augmenter import (
        build_augmented_context, merge_acon_and_augmentation,
    )
    res = build_augmented_context(
        user_instruction=case.get("user_instruction", ""),
        baseline_history=case.get("baseline_history", ""),
        acon_compressed_history=case.get("acon_compressed_history", ""),
    )
    merged = merge_acon_and_augmentation(
        case.get("acon_compressed_history", ""),
        res.text,
    )
    return {
        "task_id": case["task_id"],
        "case_id": case["case_id"],
        "audit_augmented_context": merged,
        "audit_augmented_raw_block": res.text,
        "augmenter_elapsed_s": res.elapsed_s,
        "augmenter_model": res.model,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v5.data import DATA, read_jsonl, write_jsonl

    sampled_path = DATA / "sampled_cases.jsonl"
    if not sampled_path.exists():
        sys.exit(f"missing {sampled_path}; run scripts/02_sample_cases.py first")
    cases = read_jsonl(sampled_path)
    print(f"[03] {len(cases)} cases; workers={args.workers}")

    t0 = time.time()
    results = {}
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_build_one, c): c for c in cases}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results[r["case_id"]] = r
            except Exception as exc:
                case = futures[fut]
                results[case["case_id"]] = {"error": str(exc),
                                            "audit_augmented_context": ""}
            n_done += 1
            elapsed = time.time() - t0
            eta = (len(cases) - n_done) * elapsed / max(n_done, 1)
            print(f"  [{n_done:>3d}/{len(cases)}] elapsed={elapsed:.0f}s ETA={eta:.0f}s",
                  flush=True)

    # Merge into cases
    for c in cases:
        r = results.get(c["case_id"]) or {}
        c["audit_augmented_context"] = r.get("audit_augmented_context", "")
        c["audit_augmented_raw_block"] = r.get("audit_augmented_raw_block", "")
        c["augmenter_elapsed_s"] = r.get("augmenter_elapsed_s", 0.0)
        c["augmenter_model"] = r.get("augmenter_model", "qwen3-4b")

    write_jsonl(sampled_path, cases)
    print(f"\n[03] augmented {len(results)} cases -> {sampled_path}")
    print(f"[03] total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
