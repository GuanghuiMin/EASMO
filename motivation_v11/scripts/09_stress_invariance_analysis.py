"""Stage 09 — serial recompression robustness analysis (spec §11.5, §11.6, §12.6, §12.7)."""

from __future__ import annotations

import argparse
import difflib
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
    ap.add_argument("--stress", default=str(raw_path("stress_chains.jsonl")))
    ap.add_argument("--out", default=str(table_path("stress_invariance_by_prompt_selector.csv")))
    ap.add_argument("--cross_out", default=str(table_path("best_c1_vs_best_ck_cross_eval.csv")))
    args = ap.parse_args()
    ensure_outputs()

    behavior = _read_jsonl(args.behavior)
    cands = {c["candidate_id"]: c for c in _read_jsonl(args.candidates)}
    stress = _read_jsonl(args.stress)

    # C1 + CK text per candidate
    c1_text = {c["candidate_id"]: c["c1_text"] for c in cands.values()}
    ck_text = {}
    final_round = {}
    for r in stress:
        cid = r["candidate_id"]
        if r["round"] >= final_round.get(cid, -1):
            final_round[cid] = r["round"]
            ck_text[cid] = r["context_text"]

    # pass / chars per (cid, round)
    pass_round = {}
    chars_round = {}
    for r in behavior:
        if r.get("error"): continue
        k = (r["candidate_id"], r["eval_round"])
        pass_round[k] = bool(r.get("success"))
        chars_round[k] = r.get("compressed_context_chars", 0)

    # ==== §11.5 fragility per (family, generation_type) and per selector ====
    rows = []
    for family in sorted({c["prompt_family"] for c in cands.values()}):
        for gen in ("greedy", "sample"):
            tags = [c["candidate_id"] for c in cands.values()
                    if c["prompt_family"] == family and c["candidate_type"] == gen]
            tags_c1 = [cid for cid in tags if (cid, "C1") in pass_round]
            tags_ck = [cid for cid in tags if (cid, "CK") in pass_round]
            if not tags_c1 and not tags_ck: continue
            n_robust_pass = sum(1 for cid in tags
                                 if pass_round.get((cid,"C1"))
                                 and pass_round.get((cid,"CK")))
            n_fragile = sum(1 for cid in tags
                             if pass_round.get((cid,"C1"))
                             and (cid,"CK") in pass_round
                             and not pass_round.get((cid,"CK")))
            n_stress_improved = sum(1 for cid in tags
                                     if (cid,"C1") in pass_round
                                     and not pass_round.get((cid,"C1"))
                                     and pass_round.get((cid,"CK")))
            pass_C1 = sum(pass_round.get((cid,"C1"), False) for cid in tags_c1) / max(len(tags_c1),1)
            pass_CK = sum(pass_round.get((cid,"CK"), False) for cid in tags_ck) / max(len(tags_ck),1)
            chars_c1 = [chars_round.get((cid,"C1"), 0) for cid in tags_c1]
            chars_ck = [chars_round.get((cid,"CK"), 0) for cid in tags_ck]
            mean_c1 = float(np.mean(chars_c1)) if chars_c1 else 0
            mean_ck = float(np.mean(chars_ck)) if chars_ck else 0
            # length drift, text similarity, exact fixed point — averages per family/gen
            sims = []; fps = 0; nfp = 0
            for cid in tags:
                if cid in c1_text and cid in ck_text:
                    a = c1_text[cid]; b = ck_text[cid]
                    sims.append(difflib.SequenceMatcher(None, a, b).ratio())
                    if a == b: fps += 1
                    nfp += 1
            rows.append({
                "prompt_family":          family,
                "selector":               gen,
                "n_cases":                len(tags),
                "pass_C1":                pass_C1,
                "pass_CK":                pass_CK,
                "delta_pass_C1_to_CK_pp": 100 * (pass_CK - pass_C1),
                "fragility_rate":         n_fragile / max(n_robust_pass + n_fragile, 1),
                "stress_improved_rate":   n_stress_improved / max(len(tags), 1),
                "mean_chars_C1":          mean_c1,
                "mean_chars_CK":          mean_ck,
                "length_drift_pct":       (mean_ck - mean_c1) / max(mean_c1, 1),
                "text_similarity_C1_CK":  float(np.mean(sims)) if sims else 0,
                "exact_fixed_point_rate": fps / max(nfp, 1),
                "pass_per_1k_chars_CK":   pass_CK / max(mean_ck/1000, 1e-9),
            })
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[09] wrote stress_invariance -> {args.out} ({len(df)} rows)")
    print(df.to_string(index=False))

    # ==== §12.7 best_c1 vs best_ck cross eval ====
    by_tf = defaultdict(list)
    for c in cands.values():
        if c["candidate_type"] == "sample":
            by_tf[(c["task_id"], c["prompt_family"])].append(c)

    LAM = 0.05
    def _r(p, ch): return float(bool(p)) - LAM * (ch/2000.0)

    cross_rows = []
    for (task, family), samples in by_tf.items():
        # best_c1 = max C1 reward
        c1s = [(s["candidate_id"], _r(pass_round.get((s["candidate_id"],"C1"), 0),
                                        chars_round.get((s["candidate_id"],"C1"), 0)))
               for s in samples
               if (s["candidate_id"], "C1") in pass_round]
        ck_s = [(s["candidate_id"], _r(pass_round.get((s["candidate_id"],"CK"), 0),
                                         chars_round.get((s["candidate_id"],"CK"), 0)))
                for s in samples
                if (s["candidate_id"], "CK") in pass_round]
        if not c1s or not ck_s: continue
        c1s.sort(key=lambda x: -x[1]); ck_s.sort(key=lambda x: -x[1])
        bc1 = c1s[0][0]; bck = ck_s[0][0]
        cross_rows.append({
            "task_id":       task,
            "prompt_family": family,
            "best_c1_cid":   bc1,
            "best_c1_pass_C1": int(pass_round.get((bc1,"C1"), 0)),
            "best_c1_pass_CK": int(pass_round.get((bc1,"CK"), 0)),
            "best_c1_chars_CK": chars_round.get((bc1,"CK"), 0),
            "best_ck_cid":   bck,
            "best_ck_pass_C1": int(pass_round.get((bck,"C1"), 0)),
            "best_ck_pass_CK": int(pass_round.get((bck,"CK"), 0)),
            "best_ck_chars_CK": chars_round.get((bck,"CK"), 0),
            "stress_selection_gain_pp": 100 * (
                int(pass_round.get((bck,"CK"), 0)) - int(pass_round.get((bc1,"CK"), 0))
            ),
        })
    df_cross = pd.DataFrame(cross_rows)
    df_cross.to_csv(args.cross_out, index=False)
    print(f"[09] wrote best_c1_vs_best_ck cross -> {args.cross_out} ({len(df_cross)} rows)")


if __name__ == "__main__":
    main()
