"""Stage 08 — distribution quality (Q_dist) + calibration gap (G_calib) (spec §11.1-§11.3).

Writes outputs/tables/distribution_quality_calibration_gap.csv
and outputs/tables/prompt_family_behavior_summary.csv (§12.2)
and outputs/tables/ut_vs_utco_headroom.csv (§12.4).
"""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--dqcg_out", default=str(table_path("distribution_quality_calibration_gap.csv")))
    ap.add_argument("--summary_out", default=str(table_path("prompt_family_behavior_summary.csv")))
    ap.add_argument("--utvs_out", default=str(table_path("ut_vs_utco_headroom.csv")))
    args = ap.parse_args()
    ensure_outputs()

    behavior = _read_jsonl(args.behavior)
    cands = {c["candidate_id"]: c for c in _read_jsonl(args.candidates)}

    # Group: per (task, family) compute greedy + best-of-N pass per round
    pass_by = defaultdict(lambda: {"greedy_C1": [], "greedy_CK": [],
                                     "samples_C1": [], "samples_CK": [],
                                     "greedy_chars_C1": [], "greedy_chars_CK": [],
                                     "samples_chars_C1": [], "samples_chars_CK": []})
    for r in behavior:
        if r.get("error"): continue
        c = cands.get(r["candidate_id"])
        if not c: continue
        key = (c["task_id"], c["prompt_family"])
        is_greedy = c["candidate_type"] == "greedy"
        rnd = r["eval_round"]
        ks = f"{'greedy' if is_greedy else 'samples'}_{rnd}"
        pass_by[key][ks].append(bool(r.get("success")))
        pass_by[key][f"{'greedy' if is_greedy else 'samples'}_chars_{rnd}"].append(
            r.get("compressed_context_chars", 0))

    # Per-family aggregate
    family_stats = defaultdict(lambda: {
        "greedy_pass_C1": [], "greedy_pass_CK": [],
        "oracle_pass_C1": [], "oracle_pass_CK": [],
        "greedy_chars_C1": [], "greedy_chars_CK": [],
        "oracle_chars_C1": [], "oracle_chars_CK": [],
    })
    for (task, family), d in pass_by.items():
        # greedy
        if d["greedy_C1"]:
            family_stats[family]["greedy_pass_C1"].append(int(d["greedy_C1"][0]))
            family_stats[family]["greedy_chars_C1"].append(d["greedy_chars_C1"][0])
        if d["greedy_CK"]:
            family_stats[family]["greedy_pass_CK"].append(int(d["greedy_CK"][0]))
            family_stats[family]["greedy_chars_CK"].append(d["greedy_chars_CK"][0])
        # oracle = max across samples
        if d["samples_C1"]:
            best_idx_c1 = max(range(len(d["samples_C1"])),
                               key=lambda i: int(d["samples_C1"][i]))
            family_stats[family]["oracle_pass_C1"].append(int(d["samples_C1"][best_idx_c1]))
            family_stats[family]["oracle_chars_C1"].append(d["samples_chars_C1"][best_idx_c1])
        if d["samples_CK"]:
            best_idx_ck = max(range(len(d["samples_CK"])),
                               key=lambda i: int(d["samples_CK"][i]))
            family_stats[family]["oracle_pass_CK"].append(int(d["samples_CK"][best_idx_ck]))
            family_stats[family]["oracle_chars_CK"].append(d["samples_chars_CK"][best_idx_ck])

    # distribution_quality_calibration_gap.csv
    dqcg_rows = []
    for family, s in family_stats.items():
        def _m(k): return np.mean(s[k]) if s[k] else 0.0
        Q_C1 = _m("oracle_pass_C1"); Q_CK = _m("oracle_pass_CK")
        g_C1 = _m("greedy_pass_C1"); g_CK = _m("greedy_pass_CK")
        dqcg_rows.append({
            "prompt_family":     family,
            "n_cases":           len(s["greedy_pass_CK"]),
            "Q_dist_C1":         Q_C1,
            "Q_dist_CK":         Q_CK,
            "greedy_pass_C1":    g_C1,
            "greedy_pass_CK":    g_CK,
            "G_calib_C1":        Q_C1 - g_C1,
            "G_calib_CK":        Q_CK - g_CK,
            "calibration_ratio_CK": g_CK / Q_CK if Q_CK > 0 else float('nan'),
            "oracle_len_CK":     _m("oracle_chars_CK"),
            "greedy_len_CK":     _m("greedy_chars_CK"),
            "length_ratio_oracle_over_greedy_CK":
                (_m("oracle_chars_CK") / _m("greedy_chars_CK"))
                if _m("greedy_chars_CK") > 0 else float('nan'),
        })
    df_dqcg = pd.DataFrame(dqcg_rows).sort_values("prompt_family")
    df_dqcg.to_csv(args.dqcg_out, index=False)
    print(f"[08] wrote dqcg -> {args.dqcg_out}")
    print(df_dqcg.to_string(index=False))

    # prompt_family_behavior_summary.csv (per family × selector × round)
    summary_rows = []
    for family, s in family_stats.items():
        for sel, kp in (("greedy", "greedy"), ("oracle_best_of_n", "oracle")):
            for rnd in ("C1", "CK"):
                ps = s[f"{kp}_pass_{rnd}"]
                cs = s[f"{kp}_chars_{rnd}"]
                if not ps: continue
                pr = np.mean(ps); mc = np.mean(cs)
                summary_rows.append({
                    "prompt_family":   family,
                    "selector":        sel,
                    "eval_round":      rnd,
                    "n_cases":         len(ps),
                    "pass_rate":       pr,
                    "mean_chars":      mc,
                    "median_chars":    int(np.median(cs)),
                    "pass_per_1k_chars": pr / max(mc/1000, 1e-9),
                })
    pd.DataFrame(summary_rows).to_csv(args.summary_out, index=False)
    print(f"[08] wrote prompt_family_behavior_summary -> {args.summary_out}")

    # ut_vs_utco_headroom.csv
    utvs = []
    for family in ("ACON_UT", "ACON_UTCO"):
        s = family_stats.get(family)
        if not s: continue
        def _m(k): return float(np.mean(s[k])) if s[k] else 0.0
        utvs.append({
            "prompt_family":  family,
            "greedy_C1":      _m("greedy_pass_C1"),
            "oracle_C1":      _m("oracle_pass_C1"),
            "headroom_C1":    _m("oracle_pass_C1") - _m("greedy_pass_C1"),
            "greedy_CK":      _m("greedy_pass_CK"),
            "oracle_CK":      _m("oracle_pass_CK"),
            "headroom_CK":    _m("oracle_pass_CK") - _m("greedy_pass_CK"),
            "greedy_len_CK":  _m("greedy_chars_CK"),
            "oracle_len_CK":  _m("oracle_chars_CK"),
        })
    if utvs:
        pd.DataFrame(utvs).to_csv(args.utvs_out, index=False)
        print(f"[08] wrote ut_vs_utco_headroom -> {args.utvs_out}")


if __name__ == "__main__":
    main()
