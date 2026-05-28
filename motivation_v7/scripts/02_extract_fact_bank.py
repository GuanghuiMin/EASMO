"""Stage 02 — extract the fact bank (spec §7).

Per case:
  1. run deterministic regex candidates (cheap);
  2. call the LLM fact inventory once with FACT_INVENTORY_PROMPT;
  3. normalise + substring-ground each LLM-extracted fact;
  4. apply per-case caps (3 narrative + 3 task-state + 6 executable + 2 control + 1 other).

Outputs:
  data/fact_candidates_deterministic.jsonl   — all deterministic candidates
  data/fact_bank_raw.jsonl                   — all LLM facts after normalisation
  data/fact_bank_filtered.jsonl              — capped facts (primary)
  outputs/tables/fact_bank_grounding.csv     — grounding statistics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v7.data import (  # noqa: E402
    ensure_outputs, read_jsonl, write_jsonl, table_path,
)
from motivation_v7.clients import (  # noqa: E402
    chat, parallel_chat, ChatResult, parse_json_object,
)
from motivation_v7.fact_extract import (  # noqa: E402
    TAXONOMY, COARSE_GROUP,
    deterministic_candidates,
    normalise_fact_record,
    apply_per_case_caps,
    estimate_tokens,
)


_FACT_INVENTORY_SYSTEM = (
    "You are a careful AppWorld trajectory information auditor.\n"
    "Return only valid JSON. Do not include prose outside JSON.\n"
    "Do not invent facts. Every fact must have a short verbatim quote from the trajectory."
)

_FACT_INVENTORY_TEMPLATE = """\
You will be given an AppWorld task and a successful full trajectory.
Extract atomic facts that could plausibly affect future compression or future tool-use.

You must classify each fact into exactly one fact_type from this list:
NARRATIVE_GOAL, NARRATIVE_PROGRESS, HIGH_LEVEL_REASONING,
PENDING_SUBTASK, COMPLETED_SUBTASK, RUNTIME_VARIABLE,
AUTH_OR_ACCESS_TOKEN, EXACT_IDENTIFIER, FILE_PATH_OR_RESOURCE_LOCATOR,
API_SCHEMA_OR_PARAMETER, ACTION_OUTCOME, ENVIRONMENT_STATE,
NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT, STALE_OR_OVERWRITTEN_STATE,
NUMERIC_OR_DATE_LITERAL, OTHER_CONCRETE_DETAIL.

Rules:
1. Extract atomic facts, not long paragraphs.
2. Every fact must include a source_quote copied verbatim from the trajectory.
3. Do not invent IDs, tokens, paths, API names, dates, amounts, or action outcomes.
4. Prefer concrete execution facts when present: API parameters, IDs, tokens, paths, returned objects, failures, state changes.
5. Also extract a small number of narrative/progress facts for comparison.
6. Mark is_exact_literal=true for facts that must be copied exactly to remain useful.
7. If unsure, omit the fact.
8. Return between 12 and 20 facts when possible; never more than 25.

Return JSON only:
{{
  "case_id": "{case_id}",
  "facts": [
    {{
      "canonical_fact": "short normalized fact",
      "fact_type": "one label",
      "source_step_ids": [0],
      "source_quote": "verbatim quote from trajectory",
      "verbatim_surface": "short surface form likely to appear in summaries",
      "is_exact_literal": true,
      "literal_values": ["optional exact literals"],
      "why_it_might_matter": "one sentence"
    }}
  ]
}}

Task:
{user_instruction}

Trajectory:
{full_trajectory_text}
"""


def _build_prompt(case: dict) -> str:
    return _FACT_INVENTORY_TEMPLATE.format(
        case_id=case["case_id"],
        user_instruction=case["user_instruction"][:600],
        full_trajectory_text=case["full_trajectory_text"][:14000],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "case_pool.jsonl"))
    ap.add_argument("--out_raw", default=str(_REPO / "data" / "fact_bank_raw.jsonl"))
    ap.add_argument("--out_filtered",
                    default=str(_REPO / "data" / "fact_bank_filtered.jsonl"))
    ap.add_argument("--out_candidates",
                    default=str(_REPO / "data" / "fact_candidates_deterministic.jsonl"))
    ap.add_argument("--grounding_table",
                    default=str(table_path("fact_bank_grounding.csv")))
    ap.add_argument("--extractor", default="minimax",
                    help="Which LLM to use for the inventory pass.")
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ensure_outputs()
    cases = read_jsonl(Path(args.cases))
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    print(f"[02] {len(cases)} cases, extractor={args.extractor}, workers={args.workers}")

    # ---- deterministic candidates ----
    det_rows: List[dict] = []
    for c in cases:
        cands = deterministic_candidates(c["trajectory_steps"])
        for i, cand in enumerate(cands):
            det_rows.append(cand.to_dict(c["case_id"], f"{c['case_id']}.det{i:04d}"))
    write_jsonl(Path(args.out_candidates), det_rows)
    print(f"[02] wrote {len(det_rows)} deterministic candidates "
          f"-> {args.out_candidates}")

    # ---- LLM fact inventory (parallel) ----
    prompts = [(c["case_id"], _build_prompt(c)) for c in cases]
    t0 = time.time()
    results = parallel_chat(
        name=args.extractor,
        prompts=prompts,
        system=_FACT_INVENTORY_SYSTEM,
        max_workers=args.workers,
        max_tokens=4096,
        json_mode=True,
        seed=42,
    )
    print(f"[02] LLM extraction elapsed: {(time.time()-t0)/60:.1f} min")

    raw_rows: List[dict] = []
    filtered_rows: List[dict] = []
    grounding_rows: List[dict] = []
    for case in cases:
        res: ChatResult = results.get(case["case_id"])
        if res is None or not res.is_ok:
            grounding_rows.append({
                "case_id": case["case_id"],
                "extractor": args.extractor,
                "n_facts_raw": 0,
                "n_facts_grounded": 0,
                "n_facts_kept": 0,
                "elapsed_s": res.elapsed_s if res else 0.0,
                "error": (res.error if res else "no_result")[:160],
            })
            continue
        obj = parse_json_object(res.text)
        facts_in = (obj or {}).get("facts", []) if isinstance(obj, dict) else []
        normalised: List[dict] = []
        for i, f in enumerate(facts_in):
            fact_id = f"{case['case_id']}.fact{i:04d}"
            nf = normalise_fact_record(
                case_id=case["case_id"],
                fact_id=fact_id,
                fact_record=f,
                trajectory_text=case["full_trajectory_text"],
            )
            if nf is None:
                continue
            normalised.append(nf)
        raw_rows.extend(normalised)
        kept, miss = apply_per_case_caps(normalised)
        filtered_rows.extend(kept)
        grounded = sum(1 for f in normalised if f["grounded_by_substring"])
        grounding_rows.append({
            "case_id": case["case_id"],
            "extractor": args.extractor,
            "n_facts_raw": len(facts_in),
            "n_facts_normalised": len(normalised),
            "n_facts_grounded": grounded,
            "n_facts_kept": len(kept),
            "missing_NARRATIVE": miss.get("NARRATIVE", 0),
            "missing_TASK_STATE": miss.get("TASK_STATE", 0),
            "missing_EXECUTABLE": miss.get("EXECUTABLE", 0),
            "missing_CONTROL": miss.get("CONTROL", 0),
            "elapsed_s": res.elapsed_s,
            "error": "",
        })
        print(f"  + {case['case_id']:>12s}  raw={len(facts_in):>2d} "
              f"norm={len(normalised):>2d} grounded={grounded:>2d} "
              f"kept={len(kept):>2d}  dt={res.elapsed_s:.1f}s",
              flush=True)

    write_jsonl(Path(args.out_raw), raw_rows)
    write_jsonl(Path(args.out_filtered), filtered_rows)
    pd.DataFrame(grounding_rows).to_csv(args.grounding_table, index=False)
    print(f"[02] wrote raw={len(raw_rows)} filtered={len(filtered_rows)}")
    print(f"[02] grounding table -> {args.grounding_table}")


if __name__ == "__main__":
    main()
