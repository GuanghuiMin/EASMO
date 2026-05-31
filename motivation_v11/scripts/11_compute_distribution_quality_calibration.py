"""Stage 11 — distribution quality + calibration gap (spec §2.3, §11.1-§11.3, §13.2).

For each (split ∈ {train, dev, combined}, prompt_family, eval_round):

  Q_dist_all          = best-of-N pass rate on all tasks
  Q_dist_preserve     = P(BestN=1 | F=1)
  Q_dist_rescue       = P(BestN=1 | F=0)
  G_calib_all         = Q_dist_all − greedy_pass_all
  G_calib_preserve    = Q_dist_preserve − greedy_pass | F=1
  G_calib_rescue      = Q_dist_rescue   − greedy_pass | F=0
  greedy_harm_rate    = P(F=1, greedy=0) / N
  greedy_rescue_rate  = P(F=0, greedy=1) / N
  bestn_harm_rate     = P(F=1, bestN=0) / N
  bestn_rescue_rate   = P(F=0, bestN=1) / N

Writes:
  outputs/tables/distribution_quality_calibration_gap.csv      (spec §13.2)
  outputs/tables/prompt_family_behavior_summary.csv            (per family×selector×round)
  outputs/tables/ut_vs_utco_headroom.csv                       (spec §13.4)
  outputs/tables/pass_at_n_curve.csv                           (spec §13.5)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

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
    ap.add_argument("--baseline", default=str(raw_path("full_context_runs.jsonl")))
    ap.add_argument("--behavior", default=str(raw_path("behavior_runs.jsonl")))
    ap.add_argument("--candidates", default=str(raw_path("candidate_compressions_c1.jsonl")))
    ap.add_argument("--dqcg_out", default=str(table_path("distribution_quality_calibration_gap.csv")))
    ap.add_argument("--summary_out", default=str(table_path("prompt_family_behavior_summary.csv")))
    ap.add_argument("--utvs_out", default=str(table_path("ut_vs_utco_headroom.csv")))
    ap.add_argument("--passn_out", default=str(table_path("pass_at_n_curve.csv")))
    args = ap.parse_args()
    ensure_outputs()

    baseline = _read_jsonl(args.baseline)
    full_pass = {r["task_id"]: bool(r.get("full_success")) for r in baseline}
    task_split = {r["task_id"]: r["split"] for r in baseline}

    cands = {c["candidate_id"]: c for c in _read_jsonl(args.candidates)}
    behavior = _read_jsonl(args.behavior)

    # (task, family) -> {greedy_C1/CK pass, samples_C1/CK passes (ordered by sample_id),
    #                     greedy/sample chars per round}
    pass_by: Dict[Tuple[str, str], dict] = defaultdict(lambda: {
        "greedy_C1": None, "greedy_CK": None,
        "samples_C1": {}, "samples_CK": {},
        "greedy_chars_C1": 0, "greedy_chars_CK": 0,
        "samples_chars_C1": {}, "samples_chars_CK": {},
    })
    for r in behavior:
        if r.get("error"): continue
        c = cands.get(r["candidate_id"])
        if not c: continue
        key = (c["task_id"], c["prompt_family"]); rnd = r["eval_round"]
        chars = r.get("compressed_chars", r.get("compressed_context_chars", 0))
        if c["candidate_type"] == "greedy":
            pass_by[key][f"greedy_{rnd}"] = bool(r.get("success"))
            pass_by[key][f"greedy_chars_{rnd}"] = chars
        else:
            pass_by[key][f"samples_{rnd}"][c["sample_id"]] = bool(r.get("success"))
            pass_by[key][f"samples_chars_{rnd}"][c["sample_id"]] = chars

    # Now compute per (split, family, round) all metrics
    families = sorted({c["prompt_family"] for c in cands.values()})

    def _split_filter(task_id, split_label):
        if split_label == "combined": return True
        return task_split.get(task_id) == split_label

    dqcg_rows = []
    summary_rows = []
    utvs_rows = []
    passn_rows = []

    for split_label in ("train", "dev", "combined"):
        for family in families:
            relevant = [(t, f) for (t, f) in pass_by.keys()
                        if f == family and _split_filter(t, split_label)]
            if not relevant: continue
            for rnd in ("C1", "CK"):
                greedy_pass_list = []
                bestn_pass_list = []
                greedy_chars_list = []
                bestn_chars_list = []
                # for preserve/rescue conditional
                greedy_F1_pass = []; greedy_F0_pass = []
                bestn_F1_pass  = []; bestn_F0_pass  = []
                # per-sample pass lists for pass@N
                passN_per_case = []  # list of [pass_sample_0, ..., pass_sample_7]
                for (task_id, _) in relevant:
                    d = pass_by[(task_id, family)]
                    g = d[f"greedy_{rnd}"]
                    if g is None: continue
                    f = full_pass.get(task_id)
                    if f is None: continue
                    g_chars = d[f"greedy_chars_{rnd}"]
                    greedy_pass_list.append(int(g))
                    greedy_chars_list.append(g_chars)
                    if f: greedy_F1_pass.append(int(g))
                    else: greedy_F0_pass.append(int(g))
                    # samples
                    s_dict = d[f"samples_{rnd}"]
                    s_chars_dict = d[f"samples_chars_{rnd}"]
                    # bestN: any sample passes
                    sample_ids = sorted(s_dict.keys())
                    if not sample_ids: continue
                    sample_passes = [int(s_dict[i]) for i in sample_ids]
                    bestn = max(sample_passes) if sample_passes else 0
                    bestn_pass_list.append(bestn)
                    # bestN selected chars: shortest passing sample, else greedy
                    passing = [i for i in sample_ids if s_dict[i]]
                    if passing:
                        sel = min(passing, key=lambda i: s_chars_dict.get(i, 10**9))
                        bestn_chars_list.append(s_chars_dict.get(sel, g_chars))
                    else:
                        bestn_chars_list.append(g_chars)
                    if f: bestn_F1_pass.append(bestn)
                    else: bestn_F0_pass.append(bestn)
                    # pass@N: pad to length 8
                    pass_vec = sample_passes + [0] * (8 - len(sample_passes))
                    passN_per_case.append(pass_vec[:8])

                if not greedy_pass_list: continue

                n = len(greedy_pass_list)
                greedy_pass_all = float(np.mean(greedy_pass_list))
                bestn_pass_all  = float(np.mean(bestn_pass_list)) if bestn_pass_list else 0
                Q_dist_all      = bestn_pass_all
                Q_dist_preserve = float(np.mean(bestn_F1_pass)) if bestn_F1_pass else 0
                Q_dist_rescue   = float(np.mean(bestn_F0_pass)) if bestn_F0_pass else 0
                greedy_preserve = float(np.mean(greedy_F1_pass)) if greedy_F1_pass else 0
                greedy_rescue   = float(np.mean(greedy_F0_pass)) if greedy_F0_pass else 0
                G_calib_all      = Q_dist_all - greedy_pass_all
                G_calib_preserve = Q_dist_preserve - greedy_preserve
                G_calib_rescue   = Q_dist_rescue   - greedy_rescue
                greedy_harm_rate    = sum(1 for f, g in zip(
                    [full_pass[t] for (t, _) in relevant if pass_by[(t,family)][f"greedy_{rnd}"] is not None],
                    greedy_pass_list) if f and not g) / max(n, 1)
                greedy_rescue_rate  = sum(1 for f, g in zip(
                    [full_pass[t] for (t, _) in relevant if pass_by[(t,family)][f"greedy_{rnd}"] is not None],
                    greedy_pass_list) if not f and g) / max(n, 1)
                bestn_harm_rate     = sum(1 for f, b in zip(
                    [full_pass[t] for (t, _) in relevant if pass_by[(t,family)][f"greedy_{rnd}"] is not None],
                    bestn_pass_list) if f and not b) / max(n, 1) if bestn_pass_list else 0
                bestn_rescue_rate   = sum(1 for f, b in zip(
                    [full_pass[t] for (t, _) in relevant if pass_by[(t,family)][f"greedy_{rnd}"] is not None],
                    bestn_pass_list) if not f and b) / max(n, 1) if bestn_pass_list else 0

                dqcg_rows.append({
                    "split":                          split_label,
                    "prompt_family":                  family,
                    "eval_round":                     rnd,
                    "n_tasks":                        n,
                    "greedy_pass_all":                greedy_pass_all,
                    "bestn_pass_all":                 bestn_pass_all,
                    "q_dist_all":                     Q_dist_all,
                    "gap_all_pp":                     100 * G_calib_all,
                    "greedy_pass_given_full_success": greedy_preserve,
                    "bestn_pass_given_full_success":  Q_dist_preserve,
                    "q_dist_preserve":                Q_dist_preserve,
                    "gap_preserve_pp":                100 * G_calib_preserve,
                    "greedy_pass_given_full_fail":    greedy_rescue,
                    "bestn_pass_given_full_fail":     Q_dist_rescue,
                    "q_dist_rescue":                  Q_dist_rescue,
                    "gap_rescue_pp":                  100 * G_calib_rescue,
                    "greedy_harm_rate":               greedy_harm_rate,
                    "bestn_harm_rate":                bestn_harm_rate,
                    "greedy_rescue_rate":             greedy_rescue_rate,
                    "bestn_rescue_rate":              bestn_rescue_rate,
                    "greedy_mean_chars":              float(np.mean(greedy_chars_list)) if greedy_chars_list else 0,
                    "bestn_mean_chars":               float(np.mean(bestn_chars_list)) if bestn_chars_list else 0,
                })

                # summary rows for each (family, selector ∈ {greedy, oracle_best_of_n}, round)
                for sel, ps, cs in (("greedy", greedy_pass_list, greedy_chars_list),
                                       ("oracle_best_of_n", bestn_pass_list, bestn_chars_list)):
                    if not ps: continue
                    pr = float(np.mean(ps)); mc = float(np.mean(cs)) if cs else 0
                    summary_rows.append({
                        "split":              split_label,
                        "prompt_family":      family,
                        "selector":           sel,
                        "eval_round":         rnd,
                        "n_tasks":            len(ps),
                        "pass_rate":          pr,
                        "mean_chars":         mc,
                        "median_chars":       int(np.median(cs)) if cs else 0,
                        "pass_per_1k_chars":  pr / max(mc/1000, 1e-9),
                    })

                # pass@N curve (spec §11.8)
                if passN_per_case:
                    for N in (1, 2, 4, 8):
                        if N > 8: continue
                        pass_at_N_count = 0
                        oracle_win_count = 0
                        for case_passes, greedy_pass in zip(passN_per_case, greedy_pass_list):
                            first_N = case_passes[:N]
                            if any(first_N):
                                pass_at_N_count += 1
                            if any(first_N) and not greedy_pass:
                                oracle_win_count += 1
                        pass_at_N = pass_at_N_count / len(passN_per_case)
                        oracle_win = oracle_win_count / len(passN_per_case)
                        p_better = 1 - (1 - oracle_win) ** (1.0/N) if 0 <= oracle_win <= 1 else 0
                        passn_rows.append({
                            "split":                            split_label,
                            "prompt_family":                    family,
                            "eval_round":                       rnd,
                            "N":                                N,
                            "pass_at_N":                        pass_at_N,
                            "oracle_win_rate_vs_greedy":        oracle_win,
                            "estimated_better_than_greedy_mass": p_better,
                            "mean_selected_chars":              float(np.mean(bestn_chars_list)) if bestn_chars_list else 0,
                            "median_selected_chars":            int(np.median(bestn_chars_list)) if bestn_chars_list else 0,
                        })

    pd.DataFrame(dqcg_rows).to_csv(args.dqcg_out, index=False)
    print(f"[11] wrote dqcg -> {args.dqcg_out} ({len(dqcg_rows)} rows)")
    pd.DataFrame(summary_rows).to_csv(args.summary_out, index=False)
    print(f"[11] wrote prompt_family_behavior_summary ({len(summary_rows)} rows)")
    pd.DataFrame(passn_rows).to_csv(args.passn_out, index=False)
    print(f"[11] wrote pass_at_n_curve ({len(passn_rows)} rows)")

    # ut_vs_utco_headroom.csv
    utvs = []
    df_dqcg = pd.DataFrame(dqcg_rows)
    for split_label in ("train", "dev", "combined"):
        for rnd in ("C1", "CK"):
            row = {"split": split_label, "eval_round": rnd, "metric": "pass_rate"}
            for family in ("ACON_UT", "ACON_UTCO"):
                sub = df_dqcg[(df_dqcg["split"]==split_label) &
                               (df_dqcg["prompt_family"]==family) &
                               (df_dqcg["eval_round"]==rnd)]
                if sub.empty: continue
                gp = float(sub["greedy_pass_all"].iloc[0])
                bp = float(sub["bestn_pass_all"].iloc[0])
                row[f"{family}_greedy"]  = gp
                row[f"{family}_bestN"]   = bp
                row[f"{family}_headroom_pp"] = 100 * (bp - gp)
            row["UTCO_minus_UT_greedy_pp"] = 100 * (
                row.get("ACON_UTCO_greedy", 0) - row.get("ACON_UT_greedy", 0))
            row["UTCO_minus_UT_bestN_pp"] = 100 * (
                row.get("ACON_UTCO_bestN", 0) - row.get("ACON_UT_bestN", 0))
            utvs.append(row)
    pd.DataFrame(utvs).to_csv(args.utvs_out, index=False)
    print(f"[11] wrote ut_vs_utco_headroom -> {args.utvs_out}")


if __name__ == "__main__":
    main()
