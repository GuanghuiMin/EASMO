"""Stage 03 — single-round general-prompt compression (spec §9).

Run matrix:  (case, fact, condition, prompt_family ∈ {P1, P2}, model ∈ {Qwen, MiniMax}).

For P2 (general_task_agnostic), the prompt ignores ``condition_task`` —
but we still emit one row per condition_label so the same fact gets
two rows and downstream metrics see a matched pair. The compressed
output for P2 should be identical between needed/unneeded rows
because the same context is fed in (we run P2 with no condition once
per case and replicate the result for the matched pair).

Output: outputs/raw/single_round_compressions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v8.data import (  # noqa: E402
    ensure_outputs, read_jsonl, raw_path,
)
from motivation_v8.clients import make_client  # noqa: E402
from motivation_v8.prompts import get_bundle, P1, P2  # noqa: E402
from motivation_v8.compress import compress_once  # noqa: E402


# Concrete-fact priority order for trimming per case (spec §4)
PRIORITY_TYPES = (
    "AUTH_OR_ACCESS_TOKEN",
    "API_SCHEMA_OR_PARAMETER",
    "RUNTIME_VARIABLE",
    "FILE_PATH_OR_RESOURCE_LOCATOR",
    "EXACT_IDENTIFIER",
    "ACTION_OUTCOME",
    "NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT",
    "ENVIRONMENT_STATE",
    "NARRATIVE_GOAL",
    "NARRATIVE_PROGRESS",
)


def _select_facts_per_case(
    facts: List[dict], conds: List[dict], max_per_case: int,
) -> List[str]:
    """Pick at most ``max_per_case`` fact_ids per case in priority order.
    Only facts that have BOTH a needed and unneeded passing condition
    are eligible."""
    by_fact_conds: Dict[str, set] = defaultdict(set)
    for c in conds:
        by_fact_conds[c["fact_id"]].add(int(c["need_label"]))
    eligible = {fid for fid, labs in by_fact_conds.items() if labs == {0, 1}}
    by_case: Dict[str, List[dict]] = defaultdict(list)
    for f in facts:
        if f["fact_id"] in eligible:
            by_case[f["case_id"]].append(f)
    selected: List[str] = []
    for case_id, fs in by_case.items():
        fs.sort(key=lambda r: (
            PRIORITY_TYPES.index(r["fact_type"]) if r["fact_type"] in PRIORITY_TYPES else 999,
            not bool(r.get("is_exact_literal")),
            r.get("length_tokens", 0),
            r["fact_id"],
        ))
        for f in fs[:max_per_case]:
            selected.append(f["fact_id"])
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "cases.jsonl"))
    ap.add_argument("--facts", default=str(_REPO / "data" / "fact_bank_filtered.jsonl"))
    ap.add_argument("--conditions",
                    default=str(_REPO / "data" / "need_conditions_validated.jsonl"))
    ap.add_argument("--out", default=str(raw_path("single_round_compressions.jsonl")))
    ap.add_argument("--models", default="qwen,minimax")
    ap.add_argument("--prompt_families", default="P1,P2")
    ap.add_argument("--budget_chars", type=int, default=1500)
    ap.add_argument("--max_facts_per_case", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    ensure_outputs()
    cases = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    facts = read_jsonl(Path(args.facts))
    fact_by_id = {f["fact_id"]: f for f in facts}
    conds = read_jsonl(Path(args.conditions))
    selected_fact_ids = set(_select_facts_per_case(facts, conds, args.max_facts_per_case))
    print(f"[03] selected {len(selected_fact_ids)} fact_ids across "
          f"{len({fact_by_id[fid]['case_id'] for fid in selected_fact_ids})} cases")

    sel_conds = [c for c in conds if c["fact_id"] in selected_fact_ids]
    by_fact: Dict[str, dict] = defaultdict(dict)
    for c in sel_conds:
        by_fact[c["fact_id"]][int(c["need_label"])] = c

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    families = [f.strip() for f in args.prompt_families.split(",") if f.strip()]
    bundles = {fam: get_bundle(fam) for fam in families}

    print(f"[03] compressing {len(selected_fact_ids)} facts × 2 conditions × "
          f"{len(models)} models × {len(families)} prompt families")

    # Pre-compute P2 outputs per (case, model): same context, no condition.
    # We use this to fill BOTH need=0 and need=1 rows for P2 (matched).
    p2_cache: Dict[Tuple[str, str], dict] = {}
    if "P2" in families or "general_task_agnostic" in families:
        cases_to_compress = sorted({fact_by_id[fid]["case_id"] for fid in selected_fact_ids})
        clients = {m: make_client(m) for m in models}
        bundle_p2 = bundles.get("P2") or bundles.get("general_task_agnostic")
        t0 = time.time()
        print(f"[03] precomputing P2 per (case × model): {len(cases_to_compress)*len(models)} calls")
        def _p2(args_t):
            case_id, model = args_t
            case = cases[case_id]
            res = compress_once(
                client=clients[model], model_name=model, bundle=bundle_p2,
                context=case["full_trajectory_text"], condition_task=None,
                max_chars=args.budget_chars, round_idx=1,
            )
            return (case_id, model), res
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_p2, (cid, m)): (cid, m)
                    for cid in cases_to_compress for m in models}
            for fut in as_completed(futs):
                key, res = fut.result()
                p2_cache[key] = {
                    "compressed_context": res.compressed_context,
                    "compressed_chars": res.output_chars,
                    "input_context_chars": res.input_chars,
                    "elapsed_s": res.elapsed_s,
                    "budget_violation": res.budget_violation,
                    "error": res.error,
                }
        print(f"[03] P2 precompute done in {(time.time()-t0)/60:.1f} min")

    # Now run P1 (and rebroadcast P2) per (case, fact, condition, model, family)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")
    clients = {m: make_client(m) for m in models}

    work: List[Tuple] = []
    for fact_id in selected_fact_ids:
        f = fact_by_id[fact_id]
        case = cases.get(f["case_id"])
        if not case:
            continue
        for cond_label, cond_row in by_fact[fact_id].items():
            for model in models:
                for fam in families:
                    work.append((case, f, cond_row, model, fam))
    print(f"[03] {len(work)} cells to emit (P1 forwarded; P2 from cache)")

    def _run_one(item):
        case, f, cond, model, family = item
        bundle = bundles[family]
        condition_task = cond["condition_task"]
        if bundle.family == "general_task_agnostic":
            cached = p2_cache.get((case["case_id"], model))
            if cached is None:
                return None
            return {
                "case_id": case["case_id"],
                "fact_id": f["fact_id"],
                "fact_type": f["fact_type"],
                "coarse_group": f["coarse_group"],
                "need_label": int(cond["need_label"]),
                "condition_task": condition_task,
                "prompt_family": bundle.family,
                "model": model,
                "budget_chars": args.budget_chars,
                "input_context_chars": cached["input_context_chars"],
                "compressed_context": cached["compressed_context"],
                "compressed_chars": cached["compressed_chars"],
                "budget_violation": cached["budget_violation"],
                "elapsed_s": cached["elapsed_s"],
                "error": cached["error"],
            }
        res = compress_once(
            client=clients[model], model_name=model, bundle=bundle,
            context=case["full_trajectory_text"], condition_task=condition_task,
            max_chars=args.budget_chars, round_idx=1,
        )
        return {
            "case_id": case["case_id"],
            "fact_id": f["fact_id"],
            "fact_type": f["fact_type"],
            "coarse_group": f["coarse_group"],
            "need_label": int(cond["need_label"]),
            "condition_task": condition_task,
            "prompt_family": bundle.family,
            "model": model,
            "budget_chars": args.budget_chars,
            "input_context_chars": res.input_chars,
            "compressed_context": res.compressed_context,
            "compressed_chars": res.output_chars,
            "budget_violation": res.budget_violation,
            "elapsed_s": res.elapsed_s,
            "error": res.error,
        }

    t0 = time.time()
    n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, w): w for w in work}
            for fut in as_completed(futs):
                rec = fut.result()
                if rec is None:
                    continue
                if rec.get("error"):
                    n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                n_done += 1
                if n_done % 50 == 0 or n_done <= 5:
                    eta = (time.time() - t0) / n_done * (len(work) - n_done)
                    print(f"  [{n_done:>4d}/{len(work)}] err={n_err:<3d} eta={eta:.0f}s",
                          flush=True)
    print(f"[03] wrote {n_done} compressions ({n_err} errors) -> {out_path}")
    print(f"[03] total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
