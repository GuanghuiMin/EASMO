"""Stage 13b — emit the full candidate bank per spec §12.2.

One row per (candidate_id) joining:
  outputs/raw/candidate_compressions_c1.jsonl   (C1 text + provenance)
  outputs/raw/stress_chains.jsonl               (per-round text + sha256)
  outputs/raw/behavior_runs.jsonl               (C1/CK pass + score)
  outputs/raw/full_context_runs.jsonl           (per-task full_pass)
  outputs/tables/selector_recovery_summary.csv  (optional — selector_tags)

Output: outputs/data/full_train_dev_compression_candidate_bank.jsonl

Schema (spec §12.2):
{
  "task_id":              "...",
  "split":                "train|dev",
  "prompt_family":        "ACON_UTCO",
  "candidate_id":         "...",
  "candidate_type":       "greedy|sample",
  "sample_id":            3,
  "temperature":          0.7,
  "seed":                 1003,
  "task_instruction":     "...",
  "evaluation_protocol":  "checkpoint_continuation",
  "full_success":         false,
  "full_score":           0.0,
  "c1_text":              "...",
  "ck_text":              "...",
  "stress_chain_hashes":  ["...", "...", "..."],
  "pass_c1":              true,
  "score_c1":             1.0,
  "pass_ck":              true,
  "score_ck":             1.0,
  "chars_c1":             1220,
  "chars_ck":             1185,
  "length_drift_pct":     -0.029,
  "selector_tags":        ["oracle_best_ck", "shortest_passing_ck"]
}

Resumable: writes fresh each run (cheap, ~5,220 rows max for plan β).
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

from motivation_v11.data import ensure_outputs, raw_path, table_path, data_out_path  # noqa


def _read_jsonl(p: Path) -> List[dict]:
    out: List[dict] = []
    p = Path(p)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",
                    default=str(raw_path("candidate_compressions_c1.jsonl")))
    ap.add_argument("--stress",
                    default=str(raw_path("stress_chains.jsonl")))
    ap.add_argument("--behavior",
                    default=str(raw_path("behavior_runs.jsonl")))
    ap.add_argument("--baseline",
                    default=str(raw_path("full_context_runs.jsonl")))
    ap.add_argument("--selectors",
                    default=str(table_path("selector_recovery_summary.csv")),
                    help="Optional — if present, populate selector_tags.")
    ap.add_argument("--out",
                    default=str(data_out_path("full_train_dev_compression_candidate_bank.jsonl")))
    ap.add_argument("--evaluation_protocol",
                    default="checkpoint_continuation",
                    help="v11 uses checkpoint_continuation (spec §3.4). Override if needed.")
    args = ap.parse_args()
    ensure_outputs()

    cands     = _read_jsonl(Path(args.candidates))
    stress    = _read_jsonl(Path(args.stress))
    behavior  = _read_jsonl(Path(args.behavior))
    baseline  = _read_jsonl(Path(args.baseline))

    print(f"[13b] inputs: candidates={len(cands)} stress={len(stress)} "
          f"behavior={len(behavior)} baseline={len(baseline)}")

    full_pass:  Dict[str, bool]  = {}
    full_score: Dict[str, float] = {}
    for r in baseline:
        full_pass[r["task_id"]]  = bool(r.get("full_success"))
        full_score[r["task_id"]] = float(r.get("full_score", 0.0) or 0.0)

    chain_hashes: Dict[str, List[str]] = defaultdict(list)
    ck_text_by_cid: Dict[str, str] = {}
    ck_chars_by_cid: Dict[str, int] = {}
    final_round: Dict[str, int] = {}
    for r in stress:
        cid = r["candidate_id"]
        chain_hashes[cid].append((r["round"], r.get("text_sha256", "")))
        if r["round"] >= final_round.get(cid, -1):
            final_round[cid] = r["round"]
            ck_text_by_cid[cid]  = r.get("context_text", "")
            ck_chars_by_cid[cid] = int(r.get("chars", len(r.get("context_text", ""))))

    behav_c1: Dict[str, dict] = {}
    behav_ck: Dict[str, dict] = {}
    for r in behavior:
        cid = r["candidate_id"]
        if r.get("eval_round") == "C1":
            behav_c1[cid] = r
        elif r.get("eval_round") == "CK":
            behav_ck[cid] = r

    selector_tags: Dict[str, List[str]] = defaultdict(list)
    sel_path = Path(args.selectors)
    if sel_path.exists():
        try:
            import csv
            with open(sel_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("selected_candidate_id") or row.get("candidate_id")
                    sel = row.get("selector")
                    if cid and sel:
                        if sel not in selector_tags[cid]:
                            selector_tags[cid].append(sel)
            print(f"[13b] selector_tags loaded from {sel_path.name}")
        except Exception as e:
            print(f"[13b] WARNING: failed to read selectors CSV: {e}")
    else:
        print(f"[13b] no selectors CSV at {sel_path} — selector_tags will be empty lists")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped_no_c1 = 0
    n_skipped_gen_err = 0
    with open(out_path, "w", encoding="utf-8") as f_out:
        for cand in cands:
            cid = cand["candidate_id"]
            if cand.get("generation_error"):
                n_skipped_gen_err += 1
                continue
            if not cand.get("c1_text"):
                n_skipped_no_c1 += 1
                continue

            tid = cand["task_id"]
            split = cand.get("split", "")
            family = cand["prompt_family"]
            ctype = cand.get("candidate_type", "")
            sid   = cand.get("sample_id", -1)
            temp  = cand.get("temperature", None)
            seed  = cand.get("seed", None)
            instr = cand.get("task_instruction", "")

            c1_text  = cand.get("c1_text", "") or ""
            c1_chars = int(cand.get("c1_chars", len(c1_text)))
            ck_text  = ck_text_by_cid.get(cid, "")
            ck_chars = ck_chars_by_cid.get(cid, len(ck_text))

            length_drift = (
                (ck_chars - c1_chars) / c1_chars if c1_chars > 0 else 0.0
            )

            chain = sorted(chain_hashes.get(cid, []), key=lambda x: x[0])
            chain_h = [h for (_r, h) in chain]

            bc1 = behav_c1.get(cid, {})
            bck = behav_ck.get(cid, {})

            row = {
                "task_id":              tid,
                "split":                split,
                "prompt_family":        family,
                "candidate_id":         cid,
                "candidate_type":       ctype,
                "sample_id":            sid,
                "temperature":          temp,
                "seed":                 seed,
                "task_instruction":     instr,
                "evaluation_protocol":  args.evaluation_protocol,
                "full_success":         full_pass.get(tid, None),
                "full_score":           full_score.get(tid, None),
                "c1_text":              c1_text,
                "ck_text":              ck_text,
                "stress_chain_hashes":  chain_h,
                "pass_c1":              bool(bc1.get("success")) if bc1 else None,
                "score_c1":             bc1.get("score") if bc1 else None,
                "pass_ck":              bool(bck.get("success")) if bck else None,
                "score_ck":             bck.get("score") if bck else None,
                "chars_c1":             c1_chars,
                "chars_ck":             ck_chars,
                "length_drift_pct":     length_drift,
                "selector_tags":        sorted(selector_tags.get(cid, [])),
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[13b] wrote {n_written} candidate-bank rows -> {out_path}")
    if n_skipped_gen_err:
        print(f"[13b]   skipped {n_skipped_gen_err} rows with generation_error")
    if n_skipped_no_c1:
        print(f"[13b]   skipped {n_skipped_no_c1} rows with no c1_text")
    by_split = defaultdict(int)
    by_family = defaultdict(int)
    n_with_ck = 0
    n_with_pass_ck = 0
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            by_split[r["split"]] += 1
            by_family[r["prompt_family"]] += 1
            if r.get("ck_text"):
                n_with_ck += 1
            if r.get("pass_ck") is not None:
                n_with_pass_ck += 1
    print(f"[13b] coverage: split={dict(by_split)} family={dict(by_family)}")
    print(f"[13b]           with_ck_text={n_with_ck}  with_pass_ck={n_with_pass_ck}")


if __name__ == "__main__":
    main()
