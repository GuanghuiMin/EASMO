"""Stage 10 — Pass@N curve + better-than-greedy mass estimate (spec §11.8, §11.9)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pandas as pd
import numpy as np

from motivation_v11.data import ensure_outputs, raw_path, table_path  # noqa


def _read_jsonl(p):
    out = []
    if not Path(p).exists(): return out
    for line in open(p):
        line = line.strip()
        if line: out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default=str(raw_path("behavior_runs_c1_ck.jsonl")))
    ap.add_argument("--candidates", default=str(raw_path("compression_candidates_c1.jsonl")))
    ap.add_argument("--out", default=str(table_path("pass_at_n_curve.csv")))
    ap.add_argument("--Ns", default="1,2,4,8")
    args = ap.parse_args()
    ensure_outputs()

    Ns = [int(n.strip()) for n in args.Ns.split(",")]
    behavior = _read_jsonl(args.behavior)
    cands = {c["candidate_id"]: c for c in _read_jsonl(args.candidates)}

    # group: (task, family) -> {round -> {sample_id -> pass, greedy_pass}}
    by_tf = defaultdict(lambda: {"C1": {"greedy": None, "samples": {}},
                                   "CK": {"greedy": None, "samples": {}}})
    chars_by = defaultdict(lambda: {"C1": [], "CK": []})
    for r in behavior:
        if r.get("error"): continue
        c = cands.get(r["candidate_id"])
        if not c: continue
        key = (c["task_id"], c["prompt_family"]); rnd = r["eval_round"]
        if c["candidate_type"] == "greedy":
            by_tf[key][rnd]["greedy"] = bool(r.get("success"))
        else:
            by_tf[key][rnd]["samples"][c["sample_id"]] = bool(r.get("success"))
        chars_by[key][rnd].append(r.get("compressed_context_chars", 0))

    rows = []
    for family in sorted({c["prompt_family"] for c in cands.values()}):
        for rnd in ("C1", "CK"):
            family_keys = [k for k in by_tf if k[1] == family]
            for N in Ns:
                pass_at_n_count = 0
                oracle_win_count = 0
                mean_chars = []
                n_eval = 0
                for k in family_keys:
                    d = by_tf[k][rnd]
                    if d["greedy"] is None: continue
                    samples_in_order = [d["samples"].get(i) for i in range(N)]
                    samples_in_order = [s for s in samples_in_order if s is not None]
                    if len(samples_in_order) < N: continue
                    n_eval += 1
                    if any(samples_in_order):
                        pass_at_n_count += 1
                    if any(samples_in_order) and (not d["greedy"]):
                        oracle_win_count += 1
                    elif any(samples_in_order) and d["greedy"]:
                        # tie; not strict win
                        pass
                    # mean_chars across the N selected (or just the winning one)
                    if chars_by[k][rnd]:
                        mean_chars.extend(chars_by[k][rnd])
                if n_eval == 0: continue
                pass_at_n = pass_at_n_count / n_eval
                oracle_win = oracle_win_count / n_eval
                # better-than-greedy mass (spec §11.9): p = 1 - (1-W_N)^(1/N)
                p_better = 1 - (1 - oracle_win) ** (1.0/N) if 0 <= oracle_win <= 1 else float('nan')
                rows.append({
                    "prompt_family":   family,
                    "eval_round":      rnd,
                    "N":               N,
                    "n_cases":         n_eval,
                    "pass_at_N":       pass_at_n,
                    "oracle_win_rate_at_N": oracle_win,
                    "mean_selected_chars":  float(np.mean(mean_chars)) if mean_chars else 0,
                    "better_than_greedy_mass_estimate": p_better,
                })
    df = pd.DataFrame(rows).sort_values(["prompt_family", "eval_round", "N"])
    df.to_csv(args.out, index=False)
    print(f"[10] wrote pass_at_n_curve -> {args.out} ({len(df)} rows)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
