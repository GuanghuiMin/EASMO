"""Stage 04 — iterative fixed-point compression (spec §10).

For each selected case:
  P1 (general_task_aware):  needed chain + unneeded chain
  P2 (general_task_agnostic): single chain (no task condition)
× 2 models × ROUNDS rounds, RAW_FULL initialisation.

Target fact per case: matches v7 if v7 ran iterative on this case;
otherwise picks the first EXECUTABLE fact in priority order.

Output: outputs/raw/iterative_chains.jsonl
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
    ensure_outputs, read_jsonl, raw_path, V7_DATA,
)
from motivation_v8.clients import make_client  # noqa: E402
from motivation_v8.prompts import get_bundle  # noqa: E402
from motivation_v8.compress import iterate_compression  # noqa: E402
from motivation_v8.iterate import build_raw_full  # noqa: E402


PRIORITY_TYPES = (
    "AUTH_OR_ACCESS_TOKEN",
    "API_SCHEMA_OR_PARAMETER",
    "RUNTIME_VARIABLE",
    "FILE_PATH_OR_RESOURCE_LOCATOR",
    "EXACT_IDENTIFIER",
    "ACTION_OUTCOME",
    "NUMERIC_OR_DATE_LITERAL",
    "NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT",
)


def _v7_target_fact_by_case() -> Dict[str, str]:
    """Return {case_id: fact_id} from v7's iterative_compressions.jsonl
    (which used the same selection rule)."""
    p = V7_DATA.parent / "outputs" / "raw" / "iterative_compressions.jsonl"
    if not p.exists():
        return {}
    out: Dict[str, str] = {}
    for r in read_jsonl(p):
        out.setdefault(r["case_id"], r["fact_id"])
    return out


def _pick_target(case_id: str, facts: List[dict],
                 conds: List[dict]) -> Optional[dict]:
    eligible: Dict[str, set] = defaultdict(set)
    for c in conds:
        if c["case_id"] == case_id:
            eligible[c["fact_id"]].add(int(c["need_label"]))
    cands = [f for f in facts if f["case_id"] == case_id
             and eligible.get(f["fact_id"]) == {0, 1}
             and f["fact_type"] in PRIORITY_TYPES]
    if not cands:
        cands = [f for f in facts if f["case_id"] == case_id
                 and eligible.get(f["fact_id"]) == {0, 1}]
    if not cands:
        return None
    cands.sort(key=lambda r: (
        PRIORITY_TYPES.index(r["fact_type"]) if r["fact_type"] in PRIORITY_TYPES else 999,
        not bool(r.get("is_exact_literal")),
        r.get("length_tokens", 0),
        r["fact_id"],
    ))
    return cands[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "cases.jsonl"))
    ap.add_argument("--facts", default=str(_REPO / "data" / "fact_bank_filtered.jsonl"))
    ap.add_argument("--conditions",
                    default=str(_REPO / "data" / "need_conditions_validated.jsonl"))
    ap.add_argument("--out", default=str(raw_path("iterative_chains.jsonl")))
    ap.add_argument("--targets_out",
                    default=str(_REPO / "data" / "selected_iterative_targets.jsonl"))
    ap.add_argument("--models", default="qwen,minimax")
    ap.add_argument("--prompt_families", default="P1,P2")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--budget_chars", type=int, default=1500)
    ap.add_argument("--n_iter_cases", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reuse_v7_target", action="store_true", default=True)
    args = ap.parse_args()

    ensure_outputs()
    cases = read_jsonl(Path(args.cases))
    facts = read_jsonl(Path(args.facts))
    conds = read_jsonl(Path(args.conditions))
    cond_index: Dict[Tuple[str, int], dict] = {}
    for c in conds:
        cond_index[(c["fact_id"], int(c["need_label"]))] = c

    v7_targets = _v7_target_fact_by_case() if args.reuse_v7_target else {}

    # Select N cases (alphabetical by case_id for reproducibility)
    selected_cases = sorted({c["case_id"] for c in cases})[: args.n_iter_cases]
    fact_by_id = {f["fact_id"]: f for f in facts}

    targets: List[dict] = []
    for cid in selected_cases:
        chosen_id = v7_targets.get(cid)
        chosen = fact_by_id.get(chosen_id) if chosen_id else None
        if chosen is None:
            chosen = _pick_target(cid, facts, conds)
        if chosen is None:
            print(f"  ! no target fact for {cid}; skipping")
            continue
        targets.append({
            "case_id": cid,
            "target_fact_id": chosen["fact_id"],
            "target_fact_type": chosen["fact_type"],
            "coarse_group": chosen["coarse_group"],
            "v7_match": chosen["fact_id"] == v7_targets.get(cid),
        })

    from motivation_v8.data import write_jsonl
    write_jsonl(Path(args.targets_out), targets)
    print(f"[04] selected {len(targets)} target facts "
          f"({sum(1 for t in targets if t['v7_match'])} match v7)")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    families = [f.strip() for f in args.prompt_families.split(",") if f.strip()]
    bundles = {fam: get_bundle(fam) for fam in families}
    clients = {m: make_client(m) for m in models}

    # Build chain list: for P1 we have 2 condition_types per case; for P2 single
    work: List[Tuple] = []
    case_by_id = {c["case_id"]: c for c in cases}
    for t in targets:
        case = case_by_id[t["case_id"]]
        for fam in families:
            bundle = bundles[fam]
            if bundle.family == "general_task_aware":
                for need_label in (1, 0):
                    cond = cond_index.get((t["target_fact_id"], need_label))
                    if not cond:
                        continue
                    for model in models:
                        work.append((t, case, fam, cond, "needed" if need_label else "unneeded", model))
            else:  # task-agnostic
                for model in models:
                    work.append((t, case, fam, None, "task_agnostic", model))
    print(f"[04] {len(work)} chains × {args.rounds} rounds = "
          f"{len(work) * args.rounds} iterative compressions")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    def _run_chain(item):
        t, case, fam, cond, cond_type, model = item
        bundle = bundles[fam]
        client = clients[model]
        x0 = build_raw_full(full_trajectory_text=case["full_trajectory_text"])
        chain_id = (
            f"{case['case_id']}__{t['target_fact_id']}__{cond_type}__"
            f"{bundle.family}__{model.split('-')[0]}__raw_full"
        )
        # Round 0 row first
        rows = [{
            "chain_id": chain_id,
            "case_id": case["case_id"],
            "target_fact_id": t["target_fact_id"],
            "target_fact_type": t["target_fact_type"],
            "condition_type": cond_type,
            "condition_task": cond["condition_task"] if cond else None,
            "prompt_family": bundle.family,
            "model": model,
            "budget_chars": args.budget_chars,
            "init_type": "RAW_FULL",
            "round": 0,
            "context_text": x0,
            "context_chars": len(x0),
            "budget_violation": False,
            "error": None,
        }]
        results = iterate_compression(
            client=client, model_name=model, bundle=bundle,
            x0=x0, condition_task=(cond["condition_task"] if cond else None),
            rounds=args.rounds, max_chars=args.budget_chars,
        )
        for res in results:
            rows.append({
                "chain_id": chain_id,
                "case_id": case["case_id"],
                "target_fact_id": t["target_fact_id"],
                "target_fact_type": t["target_fact_type"],
                "condition_type": cond_type,
                "condition_task": cond["condition_task"] if cond else None,
                "prompt_family": bundle.family,
                "model": model,
                "budget_chars": args.budget_chars,
                "init_type": "RAW_FULL",
                "round": res.round,
                "context_text": res.compressed_context,
                "context_chars": res.output_chars,
                "budget_violation": res.budget_violation,
                "error": res.error,
            })
        return rows

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
    print(f"[04] wrote {n_chains} chains × ({args.rounds}+1 rounds) -> {out_path}")
    print(f"[04] total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
