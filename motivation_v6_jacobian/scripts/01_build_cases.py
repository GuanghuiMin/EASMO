"""Stage 01 — build per-task cases by joining v4 spans, reference
decision states, sensitivity scores, and v3 task instructions.

Output: outputs/raw/cases.jsonl  (one record per task)
Schema:
{
  "task_id": "...",
  "task_instruction": "...",
  "context_text": "...",
  "target_text": "...",          # canonical compact JSON
  "spans": [
     {
       "span_id": "step_001",
       "step_id": 1,
       "span_text": "...",
       "v4_token_count": 237,
       "v4_final_sensitivity": 0.82,
       "v4_rule_norm": 0.10,
       "v4_judge_score": 0.25
     }, ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v6_jacobian.data import (  # noqa: E402
    ensure_outputs, raw_path, write_jsonl,
    load_v4_spans_by_task, load_v4_reference_states,
    load_v4_span_sensitivities, load_task_instructions,
    render_full_context,
)
from motivation_v6_jacobian.prompts import canonicalise_target  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_cases", type=int, default=None,
                    help="Limit number of tasks; useful for smoke tests.")
    ap.add_argument("--out", type=str,
                    default=str(raw_path("cases.jsonl")))
    args = ap.parse_args()

    ensure_outputs()
    print("[01] loading v4 spans, decision states, sensitivities…")
    spans_by_task = load_v4_spans_by_task()
    ref_states = load_v4_reference_states()
    sensitivities = load_v4_span_sensitivities()
    print(f"      spans for {len(spans_by_task)} tasks, "
          f"reference states for {len(ref_states)}, "
          f"sensitivities for {len(sensitivities)}")

    print("[01] loading v3 task instructions (via v2 loader)…")
    instructions = load_task_instructions()
    print(f"      {len(instructions)} instructions")

    cases = []
    skipped = 0
    for task_id, spans in spans_by_task.items():
        instr = instructions.get(task_id, "")
        ref = ref_states.get(task_id)
        if ref is None:
            skipped += 1
            print(f"  - skip {task_id}: no reference decision state")
            continue
        sens = sensitivities.get(task_id, {})
        rich_spans = []
        for s in spans:
            sd = sens.get(s["span_id"], {})
            rich_spans.append({
                "span_id": s["span_id"],
                "step_id": s["step_id"],
                "span_text": s["span_text"],
                "v4_token_count": s.get("token_count"),
                "v4_final_sensitivity": float(sd.get("final_sensitivity", float("nan"))),
                "v4_rule_norm": float(sd.get("rule_norm", float("nan"))),
                "v4_judge_score": float(sd.get("judge_score", float("nan"))),
            })
        context_text = render_full_context(spans)
        target_text = canonicalise_target(ref)
        cases.append({
            "task_id": task_id,
            "task_instruction": instr,
            "context_text": context_text,
            "target_text": target_text,
            "n_spans": len(rich_spans),
            "context_chars": len(context_text),
            "target_chars": len(target_text),
            "spans": rich_spans,
        })

    cases.sort(key=lambda c: c["task_id"])
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    n = write_jsonl(Path(args.out), cases)
    print(f"[01] wrote {n} cases -> {args.out} (skipped {skipped})")


if __name__ == "__main__":
    main()
