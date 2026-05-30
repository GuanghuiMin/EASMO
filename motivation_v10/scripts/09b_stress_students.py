"""Stage 09b — stress every student greedy compression via MiniMax T^K.

Takes outputs/raw/student_compressions.jsonl (written by stage 09
phase A) and writes outputs/raw/student_stress_chains.jsonl with K=2
recompression chains per (variant, case_id).

This lets stage 09 phase B evaluate students on both C1 (their own
one-shot compression) and CK (the MiniMax-recompressed K=2 text),
matching spec §15's primary student-eval criteria.

Runs in EASMO/.venv (calls MiniMax-M2.5).
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

from motivation_v10.data import (  # noqa
    ensure_outputs, read_jsonl, raw_path,
)
from motivation_v10.clients import make_client                # noqa
from motivation_v10.acon_prompt_loader import load_utco_bundle # noqa
from motivation_v10.stress import stress_chain                 # noqa


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
    ap.add_argument("--cases", default=str(_REPO / "data" / "v10_cases.jsonl"))
    ap.add_argument("--compressions",
                    default=str(raw_path("student_compressions.jsonl")))
    ap.add_argument("--out",
                    default=str(raw_path("student_stress_chains.jsonl")))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ensure_outputs()

    cases = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    comps = _read_jsonl(args.compressions)
    comps = [c for c in comps
             if c.get("compressed_text") and not c.get("error")]
    print(f"[09b] {len(comps)} student compressions to stress (K={args.rounds})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out_path.exists():
        for r in _read_jsonl(out_path):
            done_keys.add((r["variant"], r["case_id"]))
    pending = [c for c in comps if (c["variant"], c["case_id"]) not in done_keys]
    print(f"[09b] {len(done_keys)} already stressed; {len(pending)} pending")

    bundle = load_utco_bundle()
    client = make_client("minimax")

    def _do(c):
        case = cases.get(c["case_id"])
        if not case:
            return []
        rows = stress_chain(
            candidate_id=f"{c['variant']}__{c['case_id']}",
            case_id=c["case_id"],
            client=client,
            model_name="minimax",
            bundle=bundle,
            user_instruction=case["user_instruction"],
            c0=c["compressed_text"],
            rounds=args.rounds,
            max_chars=1500,
            max_tokens=2048,
        )
        return [{
            "variant":          c["variant"],
            "case_id":          c["case_id"],
            "split":            c.get("split", "unknown"),
            "round":            r.round,
            "context_text":     r.context_text,
            "chars":            r.chars,
            "tokens_est":       r.tokens_est,
            "text_hash":        r.text_hash,
            "elapsed_s":        r.elapsed_s,
            "error":            r.error,
        } for r in rows]

    t0 = time.time(); n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_do, c): c for c in pending}
            for fut in as_completed(futs):
                rows = fut.result()
                for r in rows:
                    if r.get("error"): n_err += 1
                    f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                f_out.flush()
                n_done += 1
                if n_done % 10 == 0 or n_done <= 3:
                    eta = (time.time()-t0)/n_done * (len(pending)-n_done)
                    print(f"  [{n_done}/{len(pending)}] err={n_err} eta={eta:.0f}s",
                          flush=True)
    print(f"[09b] wrote {n_done} new chains ({n_err} errors); "
          f"elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
