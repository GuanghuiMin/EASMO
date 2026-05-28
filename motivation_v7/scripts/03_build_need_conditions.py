"""Stage 03 — build matched needed/unneeded conditions per fact (spec §8).

For each filtered fact:
  1. render NEED_CONDITION_PROMPT and call generator (MiniMax by default);
  2. extract needed + unneeded condition tasks;
  3. run rule-based quality checks (length match, no direct mention);
  4. optionally run LLM validator (CONDITION_VALIDATOR_PROMPT);
  5. write one row per condition (needed + unneeded) to need_conditions.jsonl
     and a quality CSV.

Primary analysis uses only condition pairs that pass quality checks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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
from motivation_v7.need_conditions import (  # noqa: E402
    render_need_prompt, render_validator_prompt, quality_check_pair,
)


_GEN_SYSTEM = (
    "You create controlled counterfactual task conditions for an "
    "AppWorld compression experiment.\n"
    "Return only valid JSON. Do not include prose outside JSON."
)

_VAL_SYSTEM = (
    "You validate controlled counterfactual conditions.\n"
    "Return only valid JSON."
)


def _local_context(case: dict, source_step_ids: List[int]) -> str:
    """Return a short excerpt around the source step(s) — at most ~600 chars."""
    if not source_step_ids:
        return ""
    by_id = {int(s["step_id"]): s for s in case["trajectory_steps"]}
    parts = []
    for sid in source_step_ids[:3]:
        s = by_id.get(int(sid))
        if not s:
            continue
        a = (s.get("action") or "").strip()[:200]
        o = (s.get("observation") or "").strip()[:200]
        parts.append(f"step {sid} action: {a}\nstep {sid} output: {o}")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "case_pool.jsonl"))
    ap.add_argument("--facts",
                    default=str(_REPO / "data" / "fact_bank_filtered.jsonl"))
    ap.add_argument("--out", default=str(_REPO / "data" / "need_conditions.jsonl"))
    ap.add_argument("--quality_out",
                    default=str(table_path("need_condition_quality.csv")))
    ap.add_argument("--generator", default="minimax")
    ap.add_argument("--validator", default="minimax")
    ap.add_argument("--validate", action="store_true",
                    help="Also run the LLM validator pass.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_facts_total", type=int, default=None,
                    help="Cap on total facts (for smoke).")
    args = ap.parse_args()

    ensure_outputs()
    cases = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    facts = read_jsonl(Path(args.facts))
    if args.max_facts_total is not None:
        facts = facts[: args.max_facts_total]
    print(f"[03] {len(facts)} facts across {len(set(f['case_id'] for f in facts))} cases")

    # ---- generator pass ----
    prompts = []
    for f in facts:
        case = cases.get(f["case_id"])
        if not case:
            continue
        prompt = render_need_prompt(
            case_id=f["case_id"],
            fact_id=f["fact_id"],
            user_instruction=case["user_instruction"],
            canonical_fact=f["canonical_fact"],
            fact_type=f["fact_type"],
            source_quote=f["source_quote"],
            local_context=_local_context(case, f.get("source_step_ids", [])),
        )
        prompts.append((f["fact_id"], prompt))

    t0 = time.time()
    gen_results: Dict[str, ChatResult] = parallel_chat(
        name=args.generator,
        prompts=prompts,
        system=_GEN_SYSTEM,
        max_workers=args.workers,
        max_tokens=1024,
        json_mode=True,
        seed=42,
    )
    print(f"[03] generator pass: {(time.time()-t0)/60:.1f} min")

    out_rows: List[dict] = []
    quality_rows: List[dict] = []
    fact_by_id = {f["fact_id"]: f for f in facts}

    for fact_id, res in gen_results.items():
        f = fact_by_id.get(fact_id)
        if not f:
            continue
        obj = parse_json_object(res.text) if res.is_ok else None
        needed_obj = (obj or {}).get("needed_condition") or {}
        unneeded_obj = (obj or {}).get("unneeded_condition") or {}
        needed_task   = (needed_obj.get("condition_task") or "").strip()
        unneeded_task = (unneeded_obj.get("condition_task") or "").strip()
        needed_why    = (needed_obj.get("why_fact_is_needed") or "").strip()
        unneeded_why  = (unneeded_obj.get("why_fact_is_not_needed") or "").strip()
        if not needed_task or not unneeded_task:
            quality_rows.append({
                "fact_id": fact_id, "case_id": f["case_id"],
                "fact_type": f["fact_type"], "passed": False,
                "reason": "empty_condition_task",
                "needed_chars": 0, "unneeded_chars": 0,
                "delta_len_pct": 0,
                "needed_condition_mentions_fact": False,
                "unneeded_condition_mentions_fact": False,
                "length_match_ok": False,
                "elapsed_s": res.elapsed_s, "error": res.error or "",
            })
            continue
        qc = quality_check_pair(
            canonical_fact=f["canonical_fact"],
            needed_task=needed_task,
            unneeded_task=unneeded_task,
            literal_values=f.get("literal_values") or [],
        )
        passed = (
            not qc["needed_condition_mentions_fact"] and
            not qc["unneeded_condition_mentions_fact"] and
            qc["length_match_ok"]
        )
        quality_rows.append({
            "fact_id": fact_id, "case_id": f["case_id"],
            "fact_type": f["fact_type"], "passed": passed,
            **qc, "elapsed_s": res.elapsed_s, "error": res.error or "",
        })
        # always write the conditions; downstream stages filter by passed
        out_rows.append({
            "case_id": f["case_id"], "fact_id": fact_id,
            "condition_id": f"{fact_id}.needed", "need_label": 1,
            "condition_task": needed_task,
            "condition_rationale": needed_why,
            "matched_condition_id": f"{fact_id}.unneeded",
            "quality_passed": bool(passed),
        })
        out_rows.append({
            "case_id": f["case_id"], "fact_id": fact_id,
            "condition_id": f"{fact_id}.unneeded", "need_label": 0,
            "condition_task": unneeded_task,
            "condition_rationale": unneeded_why,
            "matched_condition_id": f"{fact_id}.needed",
            "quality_passed": bool(passed),
        })

    write_jsonl(Path(args.out), out_rows)
    pd.DataFrame(quality_rows).to_csv(args.quality_out, index=False)
    n_pass = sum(1 for q in quality_rows if q["passed"])
    print(f"[03] wrote {len(out_rows)} conditions "
          f"({n_pass}/{len(quality_rows)} pairs passed) -> {args.out}")
    print(f"[03] quality table -> {args.quality_out}")


if __name__ == "__main__":
    main()
