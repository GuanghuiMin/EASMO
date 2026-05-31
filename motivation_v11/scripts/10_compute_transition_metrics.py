"""Stage 10 — full-vs-compressed transition matrix (★ spec §2.1 + §13.1).

For each (split ∈ {train, dev, combined}, prompt_family, selector,
eval_round), join the full-context baseline pass with the compressed-
context pass over the same task_ids, classify each (task) row into
one of the 4 transition cells, and aggregate.

| F (full) | C (compressed) | name              |
|----------|----------------|-------------------|
|   pass   |     pass       | preserve_success  |
|   pass   |     fail       | harm              |
|   fail   |     pass       | rescue            |
|   fail   |     fail       | both_fail         |

Core identity:
    overall_gain_pp = 100 * (compressed_pass_rate - full_pass_rate)
                    = 100 * (rescue_rate - harm_rate)

Writes:
  outputs/tables/transition_matrix_by_prompt_selector_round.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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


def _classify(f: bool, c: bool) -> str:
    if f and c:         return "preserve_success"
    if f and not c:     return "harm"
    if not f and c:     return "rescue"
    return "both_fail"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(raw_path("full_context_runs.jsonl")))
    ap.add_argument("--behavior", default=str(raw_path("behavior_runs.jsonl")))
    ap.add_argument("--candidates", default=str(raw_path("candidate_compressions_c1.jsonl")))
    ap.add_argument("--selector_decisions",
                    default=str(table_path("selector_recovery_summary.csv")),
                    help="Per (task, family, selector, round) selected candidate_id.")
    ap.add_argument("--out", default=str(table_path("transition_matrix_by_prompt_selector_round.csv")))
    args = ap.parse_args()
    ensure_outputs()

    # Full-context pass per (task, split)
    baseline = _read_jsonl(args.baseline)
    full_pass = {r["task_id"]: bool(r.get("full_success")) for r in baseline}
    task_split = {r["task_id"]: r["split"] for r in baseline}

    # Candidate metadata
    cands = {c["candidate_id"]: c for c in _read_jsonl(args.candidates)}

    # Compressed pass per (candidate_id, round)
    behavior = _read_jsonl(args.behavior)
    comp_pass: Dict = {}
    comp_chars: Dict = {}
    for r in behavior:
        if r.get("error"): continue
        k = (r["candidate_id"], r["eval_round"])
        comp_pass[k] = bool(r.get("success"))
        comp_chars[k] = r.get("compressed_chars", r.get("compressed_context_chars", 0))

    # Selector decisions: read selector_recovery_summary.csv which has per-row
    # (task, family, selector, round, selected_candidate_id, selected_pass, ...)
    try:
        sel_df = pd.read_csv(args.selector_decisions)
    except Exception:
        sel_df = pd.DataFrame()
    # If the per-case selector table is empty, fall back to computing two
    # "implicit" selectors directly from behavior data:
    #   - greedy:        candidate_type == "greedy"
    #   - oracle_best_of_n_CK: per (task, family) the sample with max CK pass
    # This lets us still produce the transition matrix even before stage 09 runs.

    rows = []  # one row per (split, family, selector, round, task)
    if sel_df.empty:
        print("[10] WARNING: no selector_recovery_summary.csv yet. Computing greedy + oracle from raw.")
        # Greedy + oracle from raw
        by_tf = defaultdict(list)
        for cid, c in cands.items():
            if c.get("c1_text") and not c.get("generation_error"):
                by_tf[(c["task_id"], c["prompt_family"])].append(c)
        for (task_id, family), group in by_tf.items():
            greedy = next((c for c in group if c["candidate_type"]=="greedy"), None)
            samples = [c for c in group if c["candidate_type"]=="sample"]
            if not greedy: continue
            for eval_round in ("C1", "CK"):
                g_pass = comp_pass.get((greedy["candidate_id"], eval_round))
                if g_pass is None: continue
                rows.append({
                    "task_id": task_id, "split": task_split.get(task_id, "?"),
                    "prompt_family": family, "selector": "greedy",
                    "eval_round": eval_round,
                    "selected_candidate_id": greedy["candidate_id"],
                    "selected_pass": int(g_pass),
                    "selected_chars": comp_chars.get((greedy["candidate_id"], eval_round), 0),
                })
                # oracle
                with_pass = [(s["candidate_id"], comp_pass.get((s["candidate_id"], eval_round)))
                              for s in samples]
                with_pass = [(c, p) for c, p in with_pass if p is not None]
                if not with_pass: continue
                # max pass, then shortest, then smaller sample_id
                with_pass.sort(key=lambda x: (-int(x[1]),
                                                comp_chars.get((x[0], eval_round), 10**9)))
                ocid = with_pass[0][0]
                rows.append({
                    "task_id": task_id, "split": task_split.get(task_id, "?"),
                    "prompt_family": family,
                    "selector": "oracle_best_of_N_" + eval_round,
                    "eval_round": eval_round,
                    "selected_candidate_id": ocid,
                    "selected_pass": int(with_pass[0][1]),
                    "selected_chars": comp_chars.get((ocid, eval_round), 0),
                })
    else:
        # Use selector_decisions as authoritative
        sel_df["task_id"] = sel_df.get("task_id")
        sel_df["split"]   = sel_df.get("split", "?")
        sel_df["selected_pass"] = sel_df.get("selected_pass", 0).astype(int)
        for _, r in sel_df.iterrows():
            rows.append({
                "task_id":               r["task_id"],
                "split":                 r.get("split", task_split.get(r["task_id"], "?")),
                "prompt_family":         r["prompt_family"],
                "selector":              r["selector"],
                "eval_round":            r["eval_round"],
                "selected_candidate_id": r.get("selected_candidate_id", ""),
                "selected_pass":         int(r["selected_pass"]),
                "selected_chars":        int(r.get("selected_chars", 0)),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[10] no rows assembled (no baseline + candidates + behavior intersection). Wrote empty.")
        Path(args.out).write_text("split,prompt_family,selector,eval_round,n_tasks,"
                                    "full_pass_rate,compressed_pass_rate,overall_delta_pp,"
                                    "preserve_success_count,harm_count,rescue_count,both_fail_count,"
                                    "preserve_success_rate,harm_rate,rescue_rate,net_gain_pp,"
                                    "mean_chars,median_chars,pass_per_1k_chars\n")
        return

    # Add full-context label
    df["full_pass"] = df["task_id"].map(full_pass).fillna(False).astype(bool)
    df["selected_pass_bool"] = df["selected_pass"].astype(bool)
    df["transition"] = df.apply(
        lambda r: _classify(r["full_pass"], r["selected_pass_bool"]), axis=1)

    out_rows = []
    # Aggregate per (split ∈ {train, dev, combined}, family, selector, round)
    for split_label in ("train", "dev", "combined"):
        if split_label == "combined":
            sub_all = df
        else:
            sub_all = df[df["split"] == split_label]
        for (family, sel, rnd), g in sub_all.groupby(["prompt_family", "selector", "eval_round"]):
            n = len(g)
            if n == 0: continue
            full_pass_rate = g["full_pass"].mean()
            comp_pass_rate = g["selected_pass_bool"].mean()
            counts = g["transition"].value_counts().to_dict()
            preserve = counts.get("preserve_success", 0)
            harm     = counts.get("harm", 0)
            rescue   = counts.get("rescue", 0)
            both_f   = counts.get("both_fail", 0)
            mean_chars = g["selected_chars"].mean()
            median_chars = int(g["selected_chars"].median())
            net_gain_pp = 100 * (rescue - harm) / max(n, 1)
            overall_delta_pp = 100 * (comp_pass_rate - full_pass_rate)
            out_rows.append({
                "split":                 split_label,
                "prompt_family":         family,
                "selector":              sel,
                "eval_round":            rnd,
                "n_tasks":               n,
                "full_pass_rate":        full_pass_rate,
                "compressed_pass_rate":  comp_pass_rate,
                "overall_delta_pp":      overall_delta_pp,
                "preserve_success_count": preserve,
                "harm_count":            harm,
                "rescue_count":          rescue,
                "both_fail_count":       both_f,
                "preserve_success_rate": preserve / max(n, 1),
                "harm_rate":             harm / max(n, 1),
                "rescue_rate":           rescue / max(n, 1),
                "net_gain_pp":           net_gain_pp,
                "mean_chars":            mean_chars,
                "median_chars":          median_chars,
                "pass_per_1k_chars":     comp_pass_rate / max(mean_chars / 1000, 1e-9),
            })

    df_out = pd.DataFrame(out_rows).sort_values(
        ["split", "prompt_family", "selector", "eval_round"])
    df_out.to_csv(args.out, index=False)
    print(f"[10] wrote transition matrix -> {args.out} ({len(df_out)} rows)")
    print(df_out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
