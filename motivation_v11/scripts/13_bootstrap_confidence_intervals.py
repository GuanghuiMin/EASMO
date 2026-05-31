"""Stage 13 — bootstrap CIs for 8 paired comparisons (spec §13.7).

Paired bootstrap over task_ids, 2000 resamples by default. Each
comparison takes two selector-condition pass-rates (paired at task
level), computes mean diff + 2.5/97.5 percentile CI + bootstrap-p
(two-sided).

Spec §13.7 required comparisons:
1. ACON_UTCO bestN CK − ACON_UTCO greedy CK
2. ACON_UT bestN CK − ACON_UT greedy CK
3. ACON_UTCO bestN CK − general_task_aware bestN CK
4. ACON_UTCO greedy CK − general_task_aware greedy CK
5. ACON_UTCO bestN net_gain − ACON_UTCO greedy net_gain
6. pairwise selector CK − greedy CK
7. pairwise selector CK − oracle bestN CK
8. Best-CK evaluated CK − Best-C1 evaluated CK

Output: outputs/tables/bootstrap_confidence_intervals.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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


def _paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int = 2000,
                       seed: int = 42) -> dict:
    """Paired bootstrap on aligned (a, b) vectors of equal length."""
    if len(a) != len(b) or len(a) == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_bootstrap_two_sided": float("nan"),
                "n_pairs": len(a)}
    rng = np.random.RandomState(seed)
    n = len(a)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, n)
        diffs.append(np.mean(a[idx]) - np.mean(b[idx]))
    diffs = np.array(diffs)
    mean_diff = float(np.mean(a) - np.mean(b))
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    # Two-sided p = 2 * min(P(diff>=0), P(diff<=0)) when null is diff=0
    p_pos = (diffs >= 0).mean()
    p_neg = (diffs <= 0).mean()
    p_two = 2 * min(p_pos, p_neg)
    return {"mean_diff": mean_diff, "ci_low": ci_low, "ci_high": ci_high,
            "p_bootstrap_two_sided": float(p_two), "n_pairs": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(raw_path("full_context_runs.jsonl")))
    ap.add_argument("--behavior", default=str(raw_path("behavior_runs.jsonl")))
    ap.add_argument("--candidates", default=str(raw_path("candidate_compressions_c1.jsonl")))
    ap.add_argument("--transition", default=str(table_path("transition_matrix_by_prompt_selector_round.csv")))
    ap.add_argument("--out", default=str(table_path("bootstrap_confidence_intervals.csv")))
    ap.add_argument("--n_resamples", type=int, default=2000)
    args = ap.parse_args()
    ensure_outputs()

    baseline = _read_jsonl(args.baseline)
    full_pass = {r["task_id"]: bool(r.get("full_success")) for r in baseline}
    task_split = {r["task_id"]: r["split"] for r in baseline}

    cands = {c["candidate_id"]: c for c in _read_jsonl(args.candidates)}
    behavior = _read_jsonl(args.behavior)
    comp_pass: Dict[Tuple[str, str], bool] = {}
    for r in behavior:
        if r.get("error"): continue
        comp_pass[(r["candidate_id"], r["eval_round"])] = bool(r.get("success"))

    # Build per-(task, family, eval_round) helpers
    by_tf = defaultdict(list)
    for c in cands.values():
        if c.get("c1_text") and not c.get("generation_error"):
            by_tf[(c["task_id"], c["prompt_family"])].append(c)

    def _selector_pass_vec(family: str, selector: str, eval_round: str,
                            split_filter: str = "combined") -> Tuple[np.ndarray, list]:
        """Return (pass_vector, task_ids) for the given (family, selector, round)."""
        tasks = []
        vals = []
        for (task_id, fam), group in by_tf.items():
            if fam != family: continue
            if split_filter != "combined" and task_split.get(task_id) != split_filter:
                continue
            greedy = next((c for c in group if c["candidate_type"]=="greedy"), None)
            samples = [c for c in group if c["candidate_type"]=="sample"]
            if not greedy: continue
            if selector == "greedy":
                p = comp_pass.get((greedy["candidate_id"], eval_round))
            elif selector == "oracle_best_of_N":
                ps = [(c["candidate_id"], comp_pass.get((c["candidate_id"], eval_round)))
                       for c in samples]
                ps = [(cid, pp) for cid, pp in ps if pp is not None]
                if not ps: continue
                ps.sort(key=lambda x: -int(x[1]))
                p = int(ps[0][1])
            elif selector == "best_c1":
                ps = [(c["candidate_id"], comp_pass.get((c["candidate_id"], "C1")))
                       for c in samples]
                ps = [(cid, pp) for cid, pp in ps if pp is not None]
                if not ps: continue
                ps.sort(key=lambda x: -int(x[1]))
                bc = ps[0][0]
                p = comp_pass.get((bc, eval_round))
            elif selector == "best_ck":
                ps = [(c["candidate_id"], comp_pass.get((c["candidate_id"], "CK")))
                       for c in samples]
                ps = [(cid, pp) for cid, pp in ps if pp is not None]
                if not ps: continue
                ps.sort(key=lambda x: -int(x[1]))
                bc = ps[0][0]
                p = comp_pass.get((bc, eval_round))
            else:
                p = None
            if p is None: continue
            tasks.append(task_id); vals.append(int(p))
        return np.array(vals, dtype=float), tasks

    def _aligned(va, ta, vb, tb):
        """Align two vectors on common task IDs."""
        ta_set = {t: i for i, t in enumerate(ta)}
        common_ti = [(i, ta_set[t]) for i, t in enumerate(tb) if t in ta_set]
        if not common_ti: return np.array([]), np.array([])
        return va[[i for _, i in common_ti]], vb[[i for i, _ in common_ti]]

    # Required comparisons
    comparisons = []
    # Per split breakdown
    for split_label in ("train", "dev", "combined"):
        # 1
        va, ta = _selector_pass_vec("ACON_UTCO", "oracle_best_of_N", "CK", split_label)
        vb, tb = _selector_pass_vec("ACON_UTCO", "greedy", "CK", split_label)
        a_al, b_al = _aligned(va, ta, vb, tb)
        comparisons.append(("ACON_UTCO bestN_CK vs greedy_CK", split_label, a_al, b_al))
        # 2
        va, ta = _selector_pass_vec("ACON_UT", "oracle_best_of_N", "CK", split_label)
        vb, tb = _selector_pass_vec("ACON_UT", "greedy", "CK", split_label)
        a_al, b_al = _aligned(va, ta, vb, tb)
        comparisons.append(("ACON_UT bestN_CK vs greedy_CK", split_label, a_al, b_al))
        # 3
        va, ta = _selector_pass_vec("ACON_UTCO", "oracle_best_of_N", "CK", split_label)
        vb, tb = _selector_pass_vec("general_task_aware", "oracle_best_of_N", "CK", split_label)
        a_al, b_al = _aligned(va, ta, vb, tb)
        comparisons.append(("ACON_UTCO bestN_CK vs general_task_aware bestN_CK",
                             split_label, a_al, b_al))
        # 4
        va, ta = _selector_pass_vec("ACON_UTCO", "greedy", "CK", split_label)
        vb, tb = _selector_pass_vec("general_task_aware", "greedy", "CK", split_label)
        a_al, b_al = _aligned(va, ta, vb, tb)
        comparisons.append(("ACON_UTCO greedy_CK vs general_task_aware greedy_CK",
                             split_label, a_al, b_al))
        # 8: best_ck CK vs best_c1 CK
        va, ta = _selector_pass_vec("ACON_UTCO", "best_ck", "CK", split_label)
        vb, tb = _selector_pass_vec("ACON_UTCO", "best_c1", "CK", split_label)
        a_al, b_al = _aligned(va, ta, vb, tb)
        comparisons.append(("ACON_UTCO best_ck CK vs best_c1 CK", split_label, a_al, b_al))
    # Note: comparisons 5/6/7 require selector_recovery_summary.csv (pairwise) and
    # net_gain which is per-table not per-task; we leave those as later additions
    # once the selector_recovery_summary.csv is populated by stage 09.

    rows = []
    for (name, split_label, a, b) in comparisons:
        stats = _paired_bootstrap(a, b, n_resamples=args.n_resamples)
        rows.append({
            "split":                  split_label,
            "comparison_name":        name,
            "metric":                 "pass_rate",
            "mean_diff":              stats["mean_diff"],
            "ci_low":                 stats["ci_low"],
            "ci_high":                stats["ci_high"],
            "p_bootstrap_two_sided":  stats["p_bootstrap_two_sided"],
            "n_pairs":                stats["n_pairs"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[13] wrote {len(df)} bootstrap rows -> {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
