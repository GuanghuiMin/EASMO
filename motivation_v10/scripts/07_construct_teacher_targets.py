"""Stage 07 — construct SFT teacher targets (spec §13).

For each (case, eval_round in {C1, CK}):

  teacher = argmax over {greedy + 8 samples} of:
    (true Pass(round), -length, -num_steps)

Then build two SFT JSONL files:

  outputs/data/sft_targets_c1.jsonl  — teacher selected by C1 pass
  outputs/data/sft_targets_ck.jsonl  — teacher selected by CK pass

Each row:
{
  "case_id":             "...",
  "split":               "teacher_train|legacy_v9|...",
  "input_text":          "<fully rendered ACON UTCO user prompt>",
  "target_text":         "<teacher's compressed_text, no <think>>",
  "target_type":         "C1|CK",
  "teacher_candidate_id":"...__sample_03",
  "teacher_pass_C1":     true,
  "teacher_pass_CK":     true,
  "target_chars":        987,
  "target_quality":      "strong|weak",
  "selection_reason":    "stress_pass_shortest|c1_pass_shortest|best_proxy_no_pass"
}

Only `target_quality=strong` rows are used for SFT by default (spec §13.2).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v10.data import (  # noqa
    ensure_outputs, read_jsonl, raw_path, sft_data_path,
)
from motivation_v10.acon_prompt_loader import load_utco_bundle, render_prompt  # noqa


def _read_jsonl(p):
    out = []
    if not Path(p).exists():
        return out
    for line in open(p):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _select_teacher_for_round(
    *,
    eval_round: str,
    candidates_for_case: List[dict],
    true_pass: Dict[Tuple[str, str], bool],
    chars_for_round: Dict[Tuple[str, str], int],
    iterations_for_round: Dict[Tuple[str, str], int],
    proxy_composite: Dict[Tuple[str, str], float],
) -> Tuple[Optional[dict], str, str]:
    """Pick teacher candidate for this eval_round.

    Returns (candidate_row, target_quality, selection_reason).
    """
    passing = [
        c for c in candidates_for_case
        if true_pass.get((c["candidate_id"], eval_round), False)
    ]
    if passing:
        passing.sort(key=lambda c: (
            chars_for_round.get((c["candidate_id"], eval_round), 10**9),
            iterations_for_round.get((c["candidate_id"], eval_round), 10**9),
        ))
        return passing[0], "strong", f"{eval_round}_pass_shortest"

    # Fallback: highest proxy composite at eval_round (weak target)
    scored = [(c, proxy_composite.get((c["candidate_id"], eval_round), float("-inf")))
              for c in candidates_for_case]
    scored = [x for x in scored if x[1] != float("-inf")]
    if scored:
        scored.sort(key=lambda x: -x[1])
        return scored[0][0], "weak", "best_proxy_no_pass"
    return None, "weak", "no_candidate"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "v10_cases.jsonl"))
    ap.add_argument("--candidates",
                    default=str(raw_path("minimax_candidates.jsonl")))
    ap.add_argument("--behavior",
                    default=str(raw_path("behavior_runs_candidates.jsonl")))
    ap.add_argument("--stress",
                    default=str(raw_path("stress_chains.jsonl")))
    ap.add_argument("--verifier",
                    default=str(raw_path("proxy_verifier_scores.jsonl")))
    ap.add_argument("--c1_out",
                    default=str(sft_data_path("sft_targets_c1.jsonl")))
    ap.add_argument("--ck_out",
                    default=str(sft_data_path("sft_targets_ck.jsonl")))
    ap.add_argument("--include_weak", action="store_true", default=False,
                    help="Include target_quality=weak rows. Default: strong only.")
    ap.add_argument("--target_max_chars", type=int, default=1500)
    ap.add_argument("--splits", default="teacher_train,legacy_v9",
                    help="Which v10 splits to include as SFT cases. Default = "
                         "teacher_train + legacy_v9 (dev_proxy and test_behavior excluded).")
    args = ap.parse_args()
    ensure_outputs()

    cases_by_id = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    wanted_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    cases_in_scope = {cid: c for cid, c in cases_by_id.items()
                      if c.get("split") in wanted_splits}
    print(f"[07] {len(cases_in_scope)} cases in scope (splits={sorted(wanted_splits)})")

    candidates = _read_jsonl(args.candidates)
    candidates = [c for c in candidates
                  if c.get("compressed_text") and not c.get("error")
                  and c.get("case_id") in cases_in_scope]
    by_case: Dict[str, List[dict]] = defaultdict(list)
    for c in candidates:
        by_case[c["case_id"]].append(c)

    # Behavior table — (candidate_id, eval_round) -> pass + chars + iter
    behavior = _read_jsonl(args.behavior)
    true_pass: Dict[Tuple[str, str], bool] = {}
    chars_round: Dict[Tuple[str, str], int] = {}
    iter_round: Dict[Tuple[str, str], int] = {}
    for r in behavior:
        k = (r["candidate_id"], r["eval_round"])
        true_pass[k]   = bool(r.get("success"))
        chars_round[k] = r.get("compressed_chars", 0)
        iter_round[k]  = r.get("iterations", 0)

    # Verifier composite — proxy fallback for weak targets
    verifier = _read_jsonl(args.verifier)
    proxy_composite: Dict[Tuple[str, str], float] = {}
    for r in verifier:
        proxy_composite[(r["candidate_id"], r["eval_round"])] = r.get("composite", 0.0)

    # Stress — needed to identify the actual context the agent saw for CK
    # (we still use the candidate's own raw compressed_text as SFT target,
    # since the student learns to *produce* one-step compressions; the
    # CK selection criterion is just whether the *recompressed* output
    # passes, not the recompressed text itself.)
    bundle = load_utco_bundle()

    n_c1 = {"strong": 0, "weak": 0, "no_candidate": 0}
    n_ck = {"strong": 0, "weak": 0, "no_candidate": 0}
    rows_c1: List[dict] = []
    rows_ck: List[dict] = []

    for case_id, case in cases_in_scope.items():
        cands = by_case.get(case_id, [])
        if not cands:
            continue
        for eval_round, target_list, counter in (("C1", rows_c1, n_c1),
                                                  ("CK", rows_ck, n_ck)):
            cand, quality, reason = _select_teacher_for_round(
                eval_round=eval_round,
                candidates_for_case=cands,
                true_pass=true_pass,
                chars_for_round=chars_round,
                iterations_for_round=iter_round,
                proxy_composite=proxy_composite,
            )
            counter[quality if cand else "no_candidate"] += 1
            if cand is None:
                continue
            if quality == "weak" and not args.include_weak:
                continue
            input_text = render_prompt(
                bundle,
                task=case["user_instruction"],
                history=case["full_trajectory_text"],
                prev_summary="",
                max_chars=args.target_max_chars,
            )
            target_list.append({
                "case_id":              case_id,
                "split":                case.get("split", "unknown"),
                "input_text":           input_text,
                "target_text":          cand["compressed_text"],
                "target_type":          eval_round,
                "teacher_candidate_id": cand["candidate_id"],
                "teacher_sample_id":    cand["sample_id"],
                "teacher_pass_C1":      true_pass.get((cand["candidate_id"], "C1"), False),
                "teacher_pass_CK":      true_pass.get((cand["candidate_id"], "CK"), False),
                "target_chars":         len(cand["compressed_text"]),
                "target_quality":       quality,
                "selection_reason":     reason,
                "acon_history_prompt_sha256": bundle.sha256,
            })

    with open(args.c1_out, "w", encoding="utf-8") as f:
        for r in rows_c1:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.ck_out, "w", encoding="utf-8") as f:
        for r in rows_ck:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[07] wrote {len(rows_c1)} C1 SFT targets -> {args.c1_out}")
    print(f"     C1 quality breakdown: {n_c1}")
    print(f"[07] wrote {len(rows_ck)} CK SFT targets -> {args.ck_out}")
    print(f"     CK quality breakdown: {n_ck}")
    print(f"[07] include_weak={args.include_weak}; "
          f"only strong rows are emitted by default.")


if __name__ == "__main__":
    main()
