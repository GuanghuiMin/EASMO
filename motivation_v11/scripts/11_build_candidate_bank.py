"""Stage 11 — assemble outputs/data/full_dev_compression_candidate_bank.jsonl (spec §14)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v11.data import ensure_outputs, raw_path, data_out_path, sha256_text  # noqa


def _read_jsonl(p):
    out = []
    if not Path(p).exists(): return out
    for line in open(p):
        line = line.strip()
        if line: out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(raw_path("compression_candidates_c1.jsonl")))
    ap.add_argument("--stress", default=str(raw_path("stress_chains.jsonl")))
    ap.add_argument("--behavior", default=str(raw_path("behavior_runs_c1_ck.jsonl")))
    ap.add_argument("--out", default=str(data_out_path("full_dev_compression_candidate_bank.jsonl")))
    args = ap.parse_args()
    ensure_outputs()

    cands = {c["candidate_id"]: c for c in _read_jsonl(args.candidates)}
    stress = _read_jsonl(args.stress)
    behavior = _read_jsonl(args.behavior)

    # group stress by candidate_id, sorted by round
    by_cid = defaultdict(list)
    for r in stress:
        by_cid[r["candidate_id"]].append(r)
    for k in by_cid:
        by_cid[k].sort(key=lambda x: x["round"])

    # pass/chars per (cid, round)
    pass_round = {}; chars_round = {}; output_dirs = {}; scores = {}
    for r in behavior:
        if r.get("error"): continue
        k = (r["candidate_id"], r["eval_round"])
        pass_round[k] = bool(r.get("success"))
        chars_round[k] = r.get("compressed_context_chars", 0)
        output_dirs[k] = r.get("output_dir")
        scores[k] = r.get("score", 0.0)

    out_rows = []
    for cid, c in cands.items():
        chain_rows = by_cid.get(cid, [])
        ck_text = chain_rows[-1]["context_text"] if chain_rows else c["c1_text"]
        out_rows.append({
            "task_id":              c["task_id"],
            "prompt_family":        c["prompt_family"],
            "candidate_id":         cid,
            "candidate_type":       c["candidate_type"],
            "sample_id":            c["sample_id"],
            "temperature":          c.get("temperature"),
            "seed":                 c.get("seed"),
            "task_instruction":     c["task_instruction"],
            "source_context_hash":  sha256_text(c.get("task_instruction","")),
            "c1_text":              c["c1_text"],
            "stress_chain":         [r["context_text"] for r in chain_rows],
            "ck_text":              ck_text,
            "pass_c1":              pass_round.get((cid, "C1"), False),
            "pass_ck":              pass_round.get((cid, "CK"), False),
            "score_c1":             scores.get((cid, "C1"), 0.0),
            "score_ck":             scores.get((cid, "CK"), 0.0),
            "chars_c1":             chars_round.get((cid, "C1"), c["c1_chars"]),
            "chars_ck":             chars_round.get((cid, "CK"), len(ck_text)),
            "output_dir_c1":        output_dirs.get((cid, "C1")),
            "output_dir_ck":        output_dirs.get((cid, "CK")),
            "prompt_sha256":        c.get("prompt_sha256"),
        })

    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[11] wrote candidate bank -> {args.out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
