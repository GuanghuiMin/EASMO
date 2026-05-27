"""Stage 05 — rerun downstream agent with recompressed_context.

For each case with a recompressed_context, run the same MiniMax-M2.5
downstream agent and record final_after_recompression_success.

Outputs:
  data/sampled_cases.jsonl (overwritten with the success field)
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _run_one(args_tuple):
    (task_id, case_id, recompressed_context, max_steps) = args_tuple
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v5.runner import run_recompressed_downstream
    res = run_recompressed_downstream(
        task_id,
        recompressed_context=recompressed_context,
        max_steps=max_steps,
    )
    return {
        "case_id": case_id,
        "task_id": task_id,
        "final_after_recompression_success": bool(res.success),
        "final_after_recompression_iters": int(res.iterations),
        "final_after_recompression_input_tokens": int(res.input_tokens),
        "final_after_recompression_output_dir": res.output_dir,
        "final_after_recompression_termination": res.termination_reason,
        "final_after_recompression_error": res.error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v5.data import DATA, read_jsonl, write_jsonl

    sampled_path = DATA / "sampled_cases.jsonl"
    cases = read_jsonl(sampled_path)
    cells = [
        (c["task_id"], c["case_id"], c.get("recompressed_context", ""),
         int(c.get("budget_max_steps", 15)))
        for c in cases if c.get("recompressed_context")
    ]
    print(f"[05] rerunning {len(cells)} downstream cells; workers={args.workers}")

    t0 = time.time()
    results = {}
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one, c): c for c in cells}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results[r["case_id"]] = r
            except Exception as exc:
                cid = futures[fut][1]
                results[cid] = {"final_after_recompression_success": False,
                                "final_after_recompression_error": str(exc)}
            n_done += 1
            elapsed = time.time() - t0
            eta = (len(cells) - n_done) * elapsed / max(n_done, 1)
            print(f"  [{n_done:>3d}/{len(cells)}] elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min",
                  flush=True)

    for c in cases:
        r = results.get(c["case_id"]) or {}
        for k, v in r.items():
            if k != "case_id" and k != "task_id":
                c[k] = v

    write_jsonl(sampled_path, cases)
    print(f"\n[05] wrote {len(cases)} cases (with recompression outcome) -> {sampled_path}")
    print(f"[05] total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
