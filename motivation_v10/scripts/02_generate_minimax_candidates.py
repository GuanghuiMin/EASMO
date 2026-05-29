"""Stage 02 — generate greedy + N stochastic MiniMax candidates per case.

Mirrors v9 stage 02 but reads from v10's `data/v10_cases.jsonl` (which
includes 4 splits: teacher_train, dev_proxy, test_behavior, legacy_v9)
and emits `outputs/raw/minimax_candidates.jsonl`.

v10 is MiniMax-only by spec §4 for the candidate compressor pool —
Qwen students are evaluated separately in stage 09.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v10.data import ensure_outputs, read_jsonl, raw_path  # noqa
from motivation_v10.clients import make_client                        # noqa
from motivation_v10.acon_prompt_loader import load_utco_bundle        # noqa
from motivation_v10.compress import compress_once                     # noqa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "v10_cases.jsonl"))
    ap.add_argument("--out", default=str(raw_path("minimax_candidates.jsonl")))
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--target_max_chars", type=int, default=1500)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature_sample", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--splits", default="teacher_train,dev_proxy,test_behavior,legacy_v9",
                    help="Comma list of v10 splits to include (default = all four).")
    args = ap.parse_args()
    ensure_outputs()

    cases = read_jsonl(Path(args.cases))
    wanted_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    cases = [c for c in cases if c.get("split") in wanted_splits]
    print(f"[02] loaded {len(cases)} cases from {args.cases} matching splits={sorted(wanted_splits)}")
    if not cases:
        print("[02] no cases — exiting")
        return

    bundle = load_utco_bundle()
    client = make_client("minimax")

    # Build work units
    work: List[Tuple] = []
    for case in cases:
        work.append((case, "greedy", "greedy", 0.0, 42))
        for i in range(args.n_samples):
            work.append((case, "sample", f"sample_{i:02d}",
                          args.temperature_sample, 1000 + i))

    print(f"[02] {len(work)} candidate compressions "
          f"(cases={len(cases)}, model=MiniMax-M2.5, N={args.n_samples})")

    # Resumability — append-only, skip already-done candidate_ids
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line).get("candidate_id"))
            except Exception:
                pass
    print(f"[02] {len(done)} already done; running remaining")

    def _run_one(item):
        case, gen_type, sample_id, temperature, seed = item
        cid = f"{case['case_id']}__minimax__{sample_id}"
        if cid in done:
            return None
        res = compress_once(
            client=client, model_name="minimax", bundle=bundle,
            user_instruction=case["user_instruction"],
            history=case["full_trajectory_text"],
            max_chars=args.target_max_chars,
            temperature=temperature, seed=seed, sample_id=sample_id,
            max_tokens=args.max_tokens,
        )
        return {
            "candidate_id":          cid,
            "case_id":               case["case_id"],
            "split":                 case.get("split", "unknown"),
            "compressor_model":      "minimax",
            "prompt_variant":        bundle.variant,
            "generation_type":       gen_type,
            "sample_id":             sample_id,
            "temperature":           temperature,
            "seed":                  seed,
            "compressed_text":       res.compressed_text,
            "compressed_chars":      res.output_chars,
            "compressed_tokens_est": max(1, res.output_chars // 4),
            "text_sha256":           res.text_sha256,
            "elapsed_s":             res.elapsed_s,
            "prompt_tokens":         res.prompt_tokens,
            "completion_tokens":     res.completion_tokens,
            "error":                 res.error,
        }

    t0 = time.time()
    n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, w): w for w in work}
            for fut in as_completed(futs):
                rec = fut.result()
                if rec is None:
                    continue
                if rec.get("error"):
                    n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                n_done += 1
                if n_done % 30 == 0 or n_done <= 3:
                    eta = (time.time() - t0) / n_done * (len(work) - n_done - len(done))
                    print(f"  [{n_done:>4d}/{len(work)-len(done)}] "
                          f"err={n_err:<3d} eta={eta:.0f}s",
                          flush=True)
    print(f"[02] wrote {n_done} new candidates ({n_err} errors) -> {out_path}")
    print(f"[02] total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
