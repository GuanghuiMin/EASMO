"""Stage 07 — build selector decisions table (spec §10 + §11.7).

Selectors:
  greedy / random_sample / shortest_sample / oracle_best_of_n / best_c1 /
  best_ck / pointwise_verifier / pairwise_verifier / continuation_entropy

For each (task, family, eval_round) and selector, find the chosen
candidate and record its true pass / score / length.

Writes:
  outputs/tables/selector_recovery_summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pandas as pd
import numpy as np

from motivation_v11.data import ensure_outputs, read_jsonl, raw_path, table_path  # noqa

LAMBDA_LENGTH = 0.05


def _read_jsonl(p):
    out = []
    if not Path(p).exists(): return out
    for line in open(p):
        line = line.strip()
        if line: out.append(json.loads(line))
    return out


def _true_reward(p: bool, chars: int) -> float:
    return float(bool(p)) - LAMBDA_LENGTH * (chars / 2000.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default=str(raw_path("behavior_runs.jsonl")))
    ap.add_argument("--candidates", default=str(raw_path("candidate_compressions_c1.jsonl")))
    ap.add_argument("--pointwise", default=str(raw_path("pointwise_verifier_scores.jsonl")))
    ap.add_argument("--pairwise", default=str(table_path("pairwise_selector_by_case.csv")))
    ap.add_argument("--entropy", default=str(table_path("continuation_entropy_selector_by_case.csv")))
    ap.add_argument("--out", default=str(table_path("selector_recovery_summary.csv")))
    ap.add_argument("--selector_seed", type=int, default=20260531)
    args = ap.parse_args()
    ensure_outputs()

    behavior = _read_jsonl(args.behavior)
    cands = {c["candidate_id"]: c for c in _read_jsonl(args.candidates)}

    # Build (cid, round) -> (success, score, chars)
    true_pass: Dict[Tuple[str, str], bool] = {}
    true_score: Dict[Tuple[str, str], float] = {}
    chars_round: Dict[Tuple[str, str], int] = {}
    for r in behavior:
        if r.get("error"): continue
        k = (r["candidate_id"], r["eval_round"])
        true_pass[k] = bool(r.get("success"))
        chars_round[k] = r.get("compressed_context_chars", 0)
        true_score[k] = _true_reward(true_pass[k], chars_round[k])

    # Pointwise selector decisions per (task, family, round)
    pw = _read_jsonl(args.pointwise)
    pw_score: Dict[Tuple[str, str], float] = {
        (r["candidate_id"], r["eval_round"]): r.get("selector_score", 0.0)
        for r in pw
    }

    # Pairwise tournament decisions
    pair_winner: Dict[Tuple[str, str, str], str] = {}  # (task, family, round) -> cid
    if Path(args.pairwise).exists():
        for r in pd.read_csv(args.pairwise).to_dict(orient="records"):
            pair_winner[(r["task_id"], r["prompt_family"], r["eval_round"])] = \
                r["winner_candidate_id"]

    # Entropy selector decisions (may be only ACON_UTCO)
    ent_winner: Dict[Tuple[str, str, str], str] = {}
    if Path(args.entropy).exists():
        for r in pd.read_csv(args.entropy).to_dict(orient="records"):
            ent_winner[(r["task_id"], r["prompt_family"], r["eval_round"])] = \
                r["winner_candidate_id"]

    # Group candidates by (task, family)
    by_tf: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for c in cands.values():
        if c.get("c1_text") and not c.get("generation_error"):
            by_tf[(c["task_id"], c["prompt_family"])].append(c)

    rng = random.Random(args.selector_seed)
    rows = []
    for (task_id, family), group in by_tf.items():
        greedy = next((c for c in group if c["candidate_type"] == "greedy"), None)
        samples = [c for c in group if c["candidate_type"] == "sample"]
        if not greedy or not samples: continue
        sample_ids = [s["candidate_id"] for s in samples]

        for eval_round in ("C1", "CK"):
            def _ok(cid):
                return (cid, eval_round) in true_pass

            # greedy
            g_cid = greedy["candidate_id"]
            # random
            rand_cid = rng.choice(sample_ids)
            # shortest
            shorts = sorted(samples,
                             key=lambda s: chars_round.get(
                                 (s["candidate_id"], eval_round), 10**9))
            short_cid = shorts[0]["candidate_id"] if shorts else g_cid
            # oracle_best
            with_score = [(s["candidate_id"],
                            true_score.get((s["candidate_id"], eval_round), -1e9))
                           for s in samples]
            with_score = [(c, sc) for c, sc in with_score if sc != -1e9]
            with_score.sort(key=lambda x: -x[1])
            oracle_cid = with_score[0][0] if with_score else g_cid
            # best_c1
            with_c1 = [(s["candidate_id"],
                         true_score.get((s["candidate_id"], "C1"), -1e9))
                        for s in samples]
            with_c1 = [(c, sc) for c, sc in with_c1 if sc != -1e9]
            with_c1.sort(key=lambda x: -x[1])
            best_c1_cid = with_c1[0][0] if with_c1 else g_cid
            # best_ck
            with_ck = [(s["candidate_id"],
                         true_score.get((s["candidate_id"], "CK"), -1e9))
                        for s in samples]
            with_ck = [(c, sc) for c, sc in with_ck if sc != -1e9]
            with_ck.sort(key=lambda x: -x[1])
            best_ck_cid = with_ck[0][0] if with_ck else g_cid
            # pointwise
            with_pw = [(s["candidate_id"], pw_score.get((s["candidate_id"], eval_round), -1e9))
                        for s in samples]
            with_pw = [(c, sc) for c, sc in with_pw if sc != -1e9]
            with_pw.sort(key=lambda x: -x[1])
            pw_cid = with_pw[0][0] if with_pw else g_cid
            # pairwise (winner of tournament)
            pair_cid = pair_winner.get((task_id, family, eval_round), g_cid)
            # entropy (only present for ACON_UTCO by default)
            ent_cid = ent_winner.get((task_id, family, eval_round), None)

            sel_choices = {
                "greedy":              g_cid,
                "random_sample":       rand_cid,
                "shortest_sample":     short_cid,
                "oracle_best_of_n":    oracle_cid,
                "best_c1":             best_c1_cid,
                "best_ck":             best_ck_cid,
                "pointwise_verifier":  pw_cid,
                "pairwise_verifier":   pair_cid,
            }
            if ent_cid is not None:
                sel_choices["continuation_entropy"] = ent_cid

            for sel_name, cid in sel_choices.items():
                rows.append({
                    "task_id":         task_id,
                    "prompt_family":   family,
                    "selector":        sel_name,
                    "eval_round":      eval_round,
                    "selected_candidate_id": cid,
                    "selected_pass":   int(true_pass.get((cid, eval_round), False)),
                    "selected_score":  true_score.get((cid, eval_round), 0.0),
                    "selected_chars":  chars_round.get((cid, eval_round), 0),
                })

    df = pd.DataFrame(rows)
    # Aggregate per (family, selector, round)
    summary = []
    for (family, sel, rnd), g in df.groupby(["prompt_family", "selector", "eval_round"]):
        greedy_in_group = df[(df["prompt_family"]==family) & (df["selector"]=="greedy")
                              & (df["eval_round"]==rnd)]
        oracle_in_group = df[(df["prompt_family"]==family) & (df["selector"]=="oracle_best_of_n")
                              & (df["eval_round"]==rnd)]
        greedy_pass = greedy_in_group["selected_pass"].mean() if len(greedy_in_group) else 0
        oracle_pass = oracle_in_group["selected_pass"].mean() if len(oracle_in_group) else 0
        sel_pass = g["selected_pass"].mean()
        sel_chars = g["selected_chars"].mean()
        sel_gain = sel_pass - greedy_pass
        oracle_gain = oracle_pass - greedy_pass
        recovery = sel_gain / oracle_gain if abs(oracle_gain) > 1e-6 else 0.0
        summary.append({
            "prompt_family":      family,
            "selector":           sel,
            "eval_round":         rnd,
            "n_cases":            len(g),
            "selected_pass_rate": sel_pass,
            "selected_mean_chars": sel_chars,
            "greedy_pass_rate":   greedy_pass,
            "oracle_pass_rate":   oracle_pass,
            "oracle_gain_pp":     100 * oracle_gain,
            "selector_gain_pp":   100 * sel_gain,
            "oracle_recovery":    recovery,
        })
    df_sum = pd.DataFrame(summary).sort_values(
        ["prompt_family", "eval_round", "selector"])
    df_sum.to_csv(args.out, index=False)
    print(f"[07] wrote selector_recovery_summary -> {args.out} ({len(df_sum)} rows)")
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
