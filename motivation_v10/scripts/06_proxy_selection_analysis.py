"""Stage 06 — proxy selection analysis (spec §12).

For each (case, eval_round) compute:
  * proxy-selected candidate (max verifier composite)
  * oracle-best candidate    (max true reward = pass - λ·length_norm)
  * random sample baseline
  * majority-vote pairwise-selected candidate (sample vs greedy under CK)

Metrics:
  * pass rate of each selector
  * oracle recovery fraction
  * AUROC for pass prediction (per eval_round)
  * Spearman rank corr (proxy vs true reward)
  * length of selected
  * pass per token (selector throughput)

Outputs:
  outputs/tables/proxy_selection_summary.csv
  outputs/tables/proxy_by_case.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pandas as pd
import numpy as np

from motivation_v10.data import (  # noqa
    ensure_outputs, read_jsonl, raw_path, table_path,
)


LAMBDA_LENGTH = 0.05


def _read_jsonl(p):
    out = []
    if not Path(p).exists():
        return out
    for line in open(p):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _length_norm(chars: int) -> float:
    return chars / 2000.0


def _true_reward(pass_bool: bool, chars: int) -> float:
    return float(bool(pass_bool)) - LAMBDA_LENGTH * _length_norm(chars)


def _spearman(xs, ys):
    if len(xs) < 3:
        return float("nan")
    rx = pd.Series(xs).rank(method="average")
    ry = pd.Series(ys).rank(method="average")
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(rx.corr(ry, method="pearson"))


def _auroc(scores, labels):
    """Mann-Whitney U AUC computed via rank."""
    if len(set(labels)) < 2:
        return float("nan")
    s = pd.Series(scores)
    ranks = s.rank(method="average")
    pos = sum(1 for x in labels if x)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    sum_pos = sum(r for r, l in zip(ranks, labels) if l)
    return float((sum_pos - pos * (pos + 1) / 2) / (pos * neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior",
                    default=str(raw_path("behavior_runs_candidates.jsonl")))
    ap.add_argument("--verifier",
                    default=str(raw_path("proxy_verifier_scores.jsonl")))
    ap.add_argument("--pairwise",
                    default=str(raw_path("proxy_pairwise_scores.jsonl")))
    ap.add_argument("--candidates",
                    default=str(raw_path("minimax_candidates.jsonl")))
    ap.add_argument("--summary_out",
                    default=str(table_path("proxy_selection_summary.csv")))
    ap.add_argument("--by_case_out",
                    default=str(table_path("proxy_by_case.csv")))
    args = ap.parse_args()
    ensure_outputs()

    candidates = {c["candidate_id"]: c for c in _read_jsonl(args.candidates)}
    behavior = _read_jsonl(args.behavior)
    verifier = _read_jsonl(args.verifier)
    pairwise = _read_jsonl(args.pairwise)

    # ------------- per-(candidate, eval_round) true reward -----------
    true_pass: Dict[Tuple[str, str], bool] = {}
    true_score: Dict[Tuple[str, str], float] = {}
    chars_at: Dict[Tuple[str, str], int] = {}
    for r in behavior:
        key = (r["candidate_id"], r["eval_round"])
        true_pass[key]  = bool(r.get("success"))
        true_score[key] = _true_reward(r.get("success"), r.get("compressed_chars", 0))
        chars_at[key]   = r.get("compressed_chars", 0)

    # ------------- verifier composite per (candidate, eval_round) ----
    verifier_score: Dict[Tuple[str, str], float] = {}
    for r in verifier:
        verifier_score[(r["candidate_id"], r["eval_round"])] = r.get("composite", 0.0)

    # ------------- pairwise winner counts per (case, sample) ---------
    # pairwise rows compare greedy (A) vs sample (B) — pick majority winner.
    # We treat (winner == "B") as a vote for the sample candidate.
    sample_votes: Dict[str, int] = Counter()
    for r in pairwise:
        b = r["candidate_b_id"]
        if r.get("winner") == "B":
            sample_votes[b] += 1
        elif r.get("winner") == "A":
            sample_votes[b] -= 1
        # ties contribute 0

    # ------------- group candidates by (case, eval_round) ------------
    by_case: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for cid, c in candidates.items():
        for eval_round in ("C1", "CK"):
            by_case[(c["case_id"], eval_round)].append(cid)

    # ------------- analysis per case --------------------------------
    case_rows: List[dict] = []
    for (case_id, eval_round), cids in by_case.items():
        if not cids:
            continue
        case_split = candidates[cids[0]].get("split", "unknown")
        cids_with_pass = [cid for cid in cids if (cid, eval_round) in true_pass]
        if not cids_with_pass:
            continue
        greedy = next((cid for cid in cids
                       if candidates[cid]["generation_type"] == "greedy"), None)
        if greedy is None:
            continue
        samples = [cid for cid in cids if candidates[cid]["generation_type"] == "sample"]

        oracle_cid = max(cids_with_pass,
                         key=lambda c: true_score.get((c, eval_round), float("-inf")))
        # Proxy selector picks the max verifier composite
        cids_with_verifier = [cid for cid in cids
                              if (cid, eval_round) in verifier_score]
        proxy_cid = max(cids_with_verifier,
                        key=lambda c: verifier_score.get((c, eval_round), float("-inf"))) \
                        if cids_with_verifier else greedy
        # Random sample baseline = deterministic seed for reproducibility
        rng = np.random.RandomState(hash((case_id, eval_round)) % (2**31))
        random_cid = rng.choice(samples) if samples else greedy
        # Pairwise selector — sample with the highest sample_votes among ties
        if samples and any(sample_votes.get(s, 0) > 0 for s in samples):
            pw_cid = max(samples, key=lambda s: sample_votes.get(s, 0))
        else:
            pw_cid = greedy

        def _ps(cid):
            return float(true_pass.get((cid, eval_round), False))

        case_rows.append({
            "case_id":           case_id,
            "split":             case_split,
            "eval_round":        eval_round,
            "n_candidates":      len(cids),
            "greedy_pass":       _ps(greedy),
            "oracle_pass":       _ps(oracle_cid),
            "proxy_pass":        _ps(proxy_cid),
            "pairwise_pass":     _ps(pw_cid),
            "random_pass":       _ps(random_cid),
            "greedy_score":      true_score.get((greedy,   eval_round), 0.0),
            "oracle_score":      true_score.get((oracle_cid, eval_round), 0.0),
            "proxy_score":       true_score.get((proxy_cid,  eval_round), 0.0),
            "pairwise_score":    true_score.get((pw_cid,     eval_round), 0.0),
            "greedy_chars":      chars_at.get((greedy,   eval_round), 0),
            "oracle_chars":      chars_at.get((oracle_cid, eval_round), 0),
            "proxy_chars":       chars_at.get((proxy_cid,  eval_round), 0),
            "pairwise_chars":    chars_at.get((pw_cid,     eval_round), 0),
            "proxy_cid":         proxy_cid,
            "oracle_cid":        oracle_cid,
            "pairwise_cid":      pw_cid,
        })

    df_case = pd.DataFrame(case_rows)
    df_case.to_csv(args.by_case_out, index=False)
    print(f"[06] per-case table -> {args.by_case_out} ({len(df_case)} rows)")

    # ------------- summary per eval_round ---------------------------
    summary_rows = []
    for eval_round, grp in df_case.groupby("eval_round"):
        greedy_pass = grp["greedy_pass"].mean()
        oracle_pass = grp["oracle_pass"].mean()
        proxy_pass  = grp["proxy_pass"].mean()
        pw_pass     = grp["pairwise_pass"].mean()
        random_pass = grp["random_pass"].mean()
        eps = 1e-9
        recovered_proxy = ((proxy_pass - greedy_pass)
                           / (oracle_pass - greedy_pass + eps))
        recovered_pw = ((pw_pass - greedy_pass)
                        / (oracle_pass - greedy_pass + eps))

        # Pass prediction AUROC on (candidate, eval_round) — predict pass from
        # verifier composite. Random selection has no probabilistic score.
        scores = []; labels = []
        for r in verifier:
            if r["eval_round"] != eval_round:
                continue
            k = (r["candidate_id"], r["eval_round"])
            if k in true_pass:
                scores.append(r.get("composite", 0.0))
                labels.append(true_pass[k])
        auroc = _auroc(scores, labels)
        spearman = _spearman(scores,
                             [true_score[(r["candidate_id"], r["eval_round"])]
                              for r in verifier if r["eval_round"] == eval_round
                              and (r["candidate_id"], r["eval_round"]) in true_score])

        summary_rows.append({
            "eval_round":       eval_round,
            "n_cases":          len(grp),
            "greedy_pass":      greedy_pass,
            "random_sample_pass": random_pass,
            "proxy_pass":       proxy_pass,
            "pairwise_pass":    pw_pass,
            "oracle_pass":      oracle_pass,
            "proxy_gain_pp":    100.0 * (proxy_pass - greedy_pass),
            "pairwise_gain_pp": 100.0 * (pw_pass - greedy_pass),
            "oracle_gain_pp":   100.0 * (oracle_pass - greedy_pass),
            "recovered_gain_proxy":   recovered_proxy,
            "recovered_gain_pairwise": recovered_pw,
            "verifier_AUROC":         auroc,
            "verifier_spearman_score": spearman,
            "mean_chars_greedy":      grp["greedy_chars"].mean(),
            "mean_chars_oracle":      grp["oracle_chars"].mean(),
            "mean_chars_proxy":       grp["proxy_chars"].mean(),
            "mean_chars_pairwise":    grp["pairwise_chars"].mean(),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(args.summary_out, index=False)
    print(f"[06] summary -> {args.summary_out}")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
