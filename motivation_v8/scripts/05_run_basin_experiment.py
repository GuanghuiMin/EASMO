"""Stage 05 — basin-of-attraction experiment (spec §11).

For N_BASIN_CASES cases:
  For each model:
    For each init_type in {RAW_FULL, DETAIL_HEAVY, NARRATIVE_HEAVY, FACT_TABLE_ONLY}:
      Run P1 (needed) iterative chain × ROUNDS

Appends to outputs/raw/iterative_chains.jsonl with init_type tagged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v8.data import (  # noqa: E402
    ensure_outputs, read_jsonl, raw_path,
)
from motivation_v8.clients import make_client  # noqa: E402
from motivation_v8.prompts import get_bundle, P2  # noqa: E402
from motivation_v8.compress import compress_once, iterate_compression  # noqa: E402
from motivation_v8.iterate import (  # noqa: E402
    build_raw_full, build_detail_heavy,
    build_narrative_heavy, build_fact_table_only,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "cases.jsonl"))
    ap.add_argument("--facts", default=str(_REPO / "data" / "fact_bank_filtered.jsonl"))
    ap.add_argument("--conditions",
                    default=str(_REPO / "data" / "need_conditions_validated.jsonl"))
    ap.add_argument("--targets", default=str(_REPO / "data" / "selected_iterative_targets.jsonl"))
    ap.add_argument("--out", default=str(raw_path("iterative_chains.jsonl")))
    ap.add_argument("--models", default="qwen,minimax")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--budget_chars", type=int, default=1500)
    ap.add_argument("--n_basin_cases", type=int, default=12)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    ensure_outputs()
    cases = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    facts = read_jsonl(Path(args.facts))
    conds = read_jsonl(Path(args.conditions))
    targets = read_jsonl(Path(args.targets))[: args.n_basin_cases]
    facts_by_case: Dict[str, List[dict]] = defaultdict(list)
    for f in facts:
        facts_by_case[f["case_id"]].append(f)
    cond_index: Dict[Tuple[str, int], dict] = {}
    for c in conds:
        cond_index[(c["fact_id"], int(c["need_label"]))] = c

    bundle_p1 = get_bundle("P1")
    bundle_p2 = get_bundle("P2")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    clients = {m: make_client(m) for m in models}

    # Step 1: precompute generic summary (P2) per (case, model) — needed for NARRATIVE_HEAVY init.
    print(f"[05] precomputing P2 summaries for NARRATIVE_HEAVY init "
          f"({len(targets)*len(models)} calls)")
    p2_summary: Dict[Tuple[str, str], str] = {}
    def _p2_summary(args_t):
        case_id, model = args_t
        case = cases[case_id]
        res = compress_once(
            client=clients[model], model_name=model, bundle=bundle_p2,
            context=case["full_trajectory_text"], condition_task=None,
            max_chars=args.budget_chars, round_idx=1,
        )
        return (case_id, model), res.compressed_context or "(empty summary)"
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_p2_summary, (t["case_id"], m)): (t["case_id"], m)
                for t in targets for m in models}
        for fut in as_completed(futs):
            key, summary = fut.result()
            p2_summary[key] = summary
    print(f"[05] P2 summaries: {(time.time()-t0)/60:.1f} min")

    # Step 2: build chains per (case, model, init_type)
    work: List[Tuple] = []
    init_types = ("RAW_FULL", "DETAIL_HEAVY", "NARRATIVE_HEAVY", "FACT_TABLE_ONLY")
    for t in targets:
        case = cases[t["case_id"]]
        case_facts = facts_by_case[t["case_id"]]
        cond = cond_index.get((t["target_fact_id"], 1))  # needed condition
        if not cond:
            continue
        for model in models:
            for init_type in init_types:
                work.append((t, case, case_facts, cond, model, init_type))

    print(f"[05] {len(work)} basin chains × ({args.rounds}+1 rounds) = "
          f"{len(work)*(args.rounds+1)} rows total")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _build_init(init_type: str, case: dict, facts_in_case: List[dict],
                    p2_text: str) -> str:
        if init_type == "RAW_FULL":
            return build_raw_full(full_trajectory_text=case["full_trajectory_text"])
        if init_type == "DETAIL_HEAVY":
            return build_detail_heavy(
                full_trajectory_text=case["full_trajectory_text"],
                facts=facts_in_case,
            )
        if init_type == "NARRATIVE_HEAVY":
            return build_narrative_heavy(
                full_trajectory_text=case["full_trajectory_text"],
                generic_summary=p2_text,
            )
        if init_type == "FACT_TABLE_ONLY":
            return build_fact_table_only(facts=facts_in_case)
        raise ValueError(f"unknown init_type: {init_type}")

    def _run_chain(item):
        t, case, facts_in_case, cond, model, init_type = item
        client = clients[model]
        p2_text = p2_summary.get((case["case_id"], model), "(empty summary)")
        x0 = _build_init(init_type, case, facts_in_case, p2_text)
        chain_id = (
            f"{case['case_id']}__{t['target_fact_id']}__needed__"
            f"{bundle_p1.family}__{model.split('-')[0]}__{init_type.lower()}"
        )
        rows = [{
            "chain_id": chain_id,
            "case_id": case["case_id"],
            "target_fact_id": t["target_fact_id"],
            "target_fact_type": t["target_fact_type"],
            "condition_type": "needed",
            "condition_task": cond["condition_task"],
            "prompt_family": bundle_p1.family,
            "model": model,
            "budget_chars": args.budget_chars,
            "init_type": init_type,
            "round": 0,
            "context_text": x0,
            "context_chars": len(x0),
            "budget_violation": False,
            "error": None,
        }]
        results = iterate_compression(
            client=client, model_name=model, bundle=bundle_p1,
            x0=x0, condition_task=cond["condition_task"],
            rounds=args.rounds, max_chars=args.budget_chars,
        )
        for res in results:
            rows.append({
                "chain_id": chain_id,
                "case_id": case["case_id"],
                "target_fact_id": t["target_fact_id"],
                "target_fact_type": t["target_fact_type"],
                "condition_type": "needed",
                "condition_task": cond["condition_task"],
                "prompt_family": bundle_p1.family,
                "model": model,
                "budget_chars": args.budget_chars,
                "init_type": init_type,
                "round": res.round,
                "context_text": res.compressed_context,
                "context_chars": res.output_chars,
                "budget_violation": res.budget_violation,
                "error": res.error,
            })
        return rows

    # append (not overwrite) so stage 04's chains stay
    t0 = time.time()
    n_chains = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_chain, w): w for w in work}
            for fut in as_completed(futs):
                rows = fut.result()
                for r in rows:
                    f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                f_out.flush()
                n_chains += 1
                eta = (time.time() - t0) / n_chains * (len(work) - n_chains)
                print(f"  [{n_chains:>3d}/{len(work)}] eta={eta:.0f}s", flush=True)
    print(f"[05] appended {n_chains} basin chains -> {out_path}")
    print(f"[05] total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
