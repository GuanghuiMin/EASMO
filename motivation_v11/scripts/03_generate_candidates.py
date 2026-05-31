"""Stage 03 — generate C1 candidates: 4 families × cases × (1 greedy + N samples).

Output: outputs/raw/compression_candidates_c1.jsonl

Resumable: skips already-generated candidate_ids.
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

from motivation_v11.data import (  # noqa
    ensure_outputs, read_jsonl, raw_path, sha256_text,
)
from motivation_v11.clients import make_client, chat                  # noqa
from motivation_v11 import prompt_families as pf                       # noqa


def _read_jsonl(p):
    out = []
    if not Path(p).exists():
        return out
    for line in open(p):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases",
                    default=str(_REPO / "data" / "v11_primary_cases.jsonl"),
                    help="Default = primary cases. Use --cases v11_secondary_all_cases.jsonl "
                         "to include baseline-fail cases too (for compression rescue analysis).")
    ap.add_argument("--out", default=str(raw_path("compression_candidates_c1.jsonl")))
    ap.add_argument("--families", default="general_task_agnostic,general_task_aware,ACON_UT,ACON_UTCO")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--max_chars", type=int, default=2000)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature_sample", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ensure_outputs()

    cases = read_jsonl(Path(args.cases))
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    print(f"[03] {len(cases)} cases × {len(families)} families × "
          f"(1 greedy + {args.n_samples} samples) = "
          f"{len(cases) * len(families) * (1 + args.n_samples)} compressions")

    client = make_client("minimax")
    bundles = {f: pf.get_bundle(f) for f in families}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for r in _read_jsonl(out_path):
            done.add(r["candidate_id"])
    print(f"[03] {len(done)} already done")

    # Build work list: (case, family, candidate_type, sample_id, temperature, seed)
    work: List[Tuple] = []
    for case in cases:
        for fam in families:
            cid = f"{case['task_id']}__{fam}__greedy"
            if cid not in done:
                work.append((case, fam, "greedy", -1, 0.0, 42))
            for i in range(args.n_samples):
                cid = f"{case['task_id']}__{fam}__sample_{i:02d}"
                if cid not in done:
                    work.append((case, fam, "sample", i, args.temperature_sample, 1000 + i))
    print(f"[03] {len(work)} pending compressions")

    def _do(item):
        case, fam, candidate_type, sample_id, temperature, seed = item
        b = bundles[fam]
        user = pf.render(b, task=case["user_instruction"],
                          history=case["full_trajectory_text"],
                          max_chars=args.max_chars)
        cid_suffix = "greedy" if candidate_type == "greedy" else f"sample_{sample_id:02d}"
        cid = f"{case['task_id']}__{fam}__{cid_suffix}"
        try:
            res = chat(name="minimax", user=user, system=b.system_text,
                        temperature=temperature, max_tokens=args.max_tokens,
                        seed=seed, client=client, json_mode=False)
            text = res.text
            return {
                "task_id":              case["task_id"],
                "prompt_family":        fam,
                "candidate_id":         cid,
                "candidate_type":       candidate_type,
                "sample_id":            sample_id,
                "temperature":          temperature,
                "seed":                 seed,
                "task_instruction":     case["user_instruction"],
                "source_context_chars": len(case["full_trajectory_text"]),
                "c1_text":              text,
                "c1_chars":             len(text),
                "c1_tokens_est":        max(1, len(text) // 4),
                "text_sha256":          sha256_text(text),
                "elapsed_s":            res.elapsed_s,
                "prompt_tokens":        res.prompt_tokens,
                "completion_tokens":    res.completion_tokens,
                "generation_error":     res.error,
                "prompt_sha256":        b.sha256_user,
                "model":                "MiniMaxAI/MiniMax-M2.5",
            }
        except Exception as e:
            return {
                "task_id":          case["task_id"],
                "prompt_family":    fam,
                "candidate_id":     cid,
                "candidate_type":   candidate_type,
                "sample_id":        sample_id,
                "c1_text":          "",
                "c1_chars":         0,
                "generation_error": str(e),
                "model":            "MiniMaxAI/MiniMax-M2.5",
            }

    t0 = time.time(); n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_do, w): w for w in work}
            for fut in as_completed(futs):
                rec = fut.result()
                if rec.get("generation_error"): n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                n_done += 1
                if n_done % 50 == 0 or n_done <= 3:
                    eta = (time.time()-t0)/n_done * (len(work)-n_done)
                    print(f"  [{n_done}/{len(work)}] err={n_err} eta={eta:.0f}s",
                          flush=True)
    print(f"[03] done: {n_done} new ({n_err} errors); elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
