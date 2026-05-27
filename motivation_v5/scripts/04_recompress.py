"""Stage 04 — recompress audit_augmented_context to obtain
`recompressed_context`.

We feed the audit-augmented text (ACON summary + [AUDIT_AUGMENTATION]
block) back into MiniMax with the same `acon_style_summary` prompt to
simulate "compressor runs again on a richer input." What gets dropped
in this pass is the headline 'recovered-then-dropped' signal.

Outputs:
  data/sampled_cases.jsonl  (overwritten with recompressed_context filled)
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _recompress_one(case: dict):
    sys.path.insert(0, str(_REPO))
    from motivation_v5.compressor import recompress
    res = recompress(
        case.get("user_instruction", ""),
        case.get("audit_augmented_context", ""),
        max_tokens=2048,
    )
    return {
        "case_id": case["case_id"],
        "recompressed_context": res.text,
        "recompressor_elapsed_s": res.elapsed_s,
        "recompressor_model": res.model,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v5.data import DATA, read_jsonl, write_jsonl

    sampled_path = DATA / "sampled_cases.jsonl"
    cases = read_jsonl(sampled_path)
    cases_with_aug = [c for c in cases if c.get("audit_augmented_context")]
    print(f"[04] recompressing {len(cases_with_aug)} cases; workers={args.workers}")

    t0 = time.time()
    results = {}
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_recompress_one, c): c for c in cases_with_aug}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results[r["case_id"]] = r
            except Exception as exc:
                cid = futures[fut]["case_id"]
                results[cid] = {"error": str(exc), "recompressed_context": ""}
            n_done += 1
            elapsed = time.time() - t0
            eta = (len(cases_with_aug) - n_done) * elapsed / max(n_done, 1)
            print(f"  [{n_done:>3d}/{len(cases_with_aug)}] elapsed={elapsed:.0f}s ETA={eta:.0f}s",
                  flush=True)

    for c in cases:
        r = results.get(c["case_id"]) or {}
        if r:
            c["recompressed_context"] = r.get("recompressed_context", "")
            c["recompressor_elapsed_s"] = r.get("recompressor_elapsed_s", 0.0)
            c["recompressor_model"] = r.get("recompressor_model",
                                            "MiniMaxAI/MiniMax-M2.5")

    write_jsonl(sampled_path, cases)
    print(f"\n[04] wrote {len(cases)} cases (with recompressed) -> {sampled_path}")
    print(f"[04] total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
