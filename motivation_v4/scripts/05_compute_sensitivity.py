"""Stage 05 — compute final span sensitivity scores.

Combines rule-based distance (from stage 03 ablated states vs stage 02
reference state) with the LLM-judge score (stage 04) per spec §6.1+§6.2:

  final_sensitivity = 0.5 * rule_based_score + 0.5 * llm_judge_score

Outputs:
  outputs/raw/span_sensitivity_scores.jsonl   one row per span
  outputs/tables/span_sensitivity_scores.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def main():
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    from motivation_v4.data import (
        ensure_outputs, raw_path, table_path, read_jsonl, write_jsonl,
    )
    from motivation_v4.distance import rule_based_distance

    ensure_outputs()
    refs = read_jsonl(raw_path("reference_decision_states.jsonl"))
    abls = read_jsonl(raw_path("span_ablation_probes.jsonl"))
    judges = read_jsonl(raw_path("span_judge_distances.jsonl"))

    ref_by_task = {r["task_id"]: r["decision_state"] for r in refs}
    judge_by_key = {(r["task_id"], r["span_id"]): r["judge"] for r in judges}

    out_records: List[dict] = []
    for a in abls:
        if not a.get("parse_ok"):
            continue
        tid, span_id = a["task_id"], a["span_id"]
        ref = ref_by_task.get(tid)
        if ref is None:
            continue
        rule = rule_based_distance(ref, a["decision_state"])
        rule_norm = rule.normalised_sensitivity
        judge = judge_by_key.get((tid, span_id), {})
        judge_score = float(judge.get("score", 0.0))
        final = 0.5 * rule_norm + 0.5 * judge_score
        out_records.append({
            "task_id": tid,
            "span_id": span_id,
            "rule_based": rule.to_dict(),
            "judge_score": round(judge_score, 4),
            "judge_severity": judge.get("severity", "none"),
            "judge_changed_fields": judge.get("changed_fields", []),
            "rule_norm": round(rule_norm, 4),
            "final_sensitivity": round(final, 4),
        })

    out_path = raw_path("span_sensitivity_scores.jsonl")
    write_jsonl(out_path, out_records)
    print(f"[05] wrote {len(out_records)} sensitivity rows -> {out_path}")

    # Flat CSV for easy inspection.
    csv_path = table_path("span_sensitivity_scores.csv")
    flat = []
    for r in out_records:
        flat.append({
            "task_id": r["task_id"],
            "span_id": r["span_id"],
            "rule_norm": r["rule_norm"],
            "judge_severity": r["judge_severity"],
            "judge_score": r["judge_score"],
            "final_sensitivity": r["final_sensitivity"],
            "next_action_type_changed": r["rule_based"]["next_action_type_changed"],
            "next_action_arguments_f1_loss": r["rule_based"]["next_action_arguments_f1_loss"],
            "candidate_objects_f1_loss": r["rule_based"]["candidate_objects_f1_loss"],
            "active_constraints_f1_loss": r["rule_based"]["active_constraints_f1_loss"],
        })
    with open(csv_path, "w", newline="") as f:
        if flat:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader()
            for r in flat:
                w.writerow(r)
    print(f"[05] wrote {len(flat)} rows -> {csv_path}")

    if out_records:
        sensitivities = [r["final_sensitivity"] for r in out_records]
        print()
        print(f"=== Sensitivity stats ===")
        print(f"  total spans scored: {len(sensitivities)}")
        print(f"  mean: {sum(sensitivities)/len(sensitivities):.3f}")
        print(f"  pct > 0.3: {100*sum(1 for x in sensitivities if x>0.3)/len(sensitivities):.1f}%")
        print(f"  pct > 0.6: {100*sum(1 for x in sensitivities if x>0.6)/len(sensitivities):.1f}%")


if __name__ == "__main__":
    main()
