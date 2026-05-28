"""Stage 06 — iterative compression for ``ROUNDS`` rounds (spec §13).

For each (case, condition_task, model, prompt_variant, budget):

    x_0 = initial trajectory text
    x_{r+1} = ACON(task=condition_task, history=x_r, prev_summary="", max_chars=B)

Per Plan B scope, we run **2 chains per case**: needed + unneeded for
ONE representative fact (the first EXECUTABLE fact in the case, falling
back to the first fact of any group). This dramatically reduces cost
vs spec's "every condition_task" wording but preserves the per-case
contrast.

Output: outputs/raw/iterative_compressions.jsonl
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

from motivation_v7.data import (  # noqa: E402
    ensure_outputs, read_jsonl, write_jsonl, raw_path,
)
from motivation_v7.clients import make_client  # noqa: E402
from motivation_v7.acon_prompt_loader import load_bundle  # noqa: E402
from motivation_v7.compress import iterate_compression  # noqa: E402


def _pick_target_fact_id(case_id: str, facts: List[dict]) -> Optional[str]:
    """Pick a representative fact per case for the iterative chain.

    Preference order: EXECUTABLE (priority on is_exact_literal) >
    CONTROL > TASK_STATE > NARRATIVE > OTHER. Ties broken by
    shortest length_tokens.
    """
    pri = {"EXECUTABLE": 0, "CONTROL": 1, "TASK_STATE": 2,
           "NARRATIVE": 3, "OTHER": 4}
    grp = [f for f in facts if f["case_id"] == case_id]
    if not grp:
        return None
    grp.sort(key=lambda f: (
        pri.get(f["coarse_group"], 9),
        not bool(f.get("is_exact_literal")),
        f.get("length_tokens", 0),
        f["fact_id"],
    ))
    return grp[0]["fact_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "case_pool.jsonl"))
    ap.add_argument("--facts",
                    default=str(_REPO / "data" / "fact_bank_filtered.jsonl"))
    ap.add_argument("--conditions",
                    default=str(_REPO / "data" / "need_conditions.jsonl"))
    ap.add_argument("--models", default="qwen,minimax")
    ap.add_argument("--prompt_variants", default="UTCO")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--budget_chars", type=int, default=1500)
    ap.add_argument("--out", default=str(raw_path("iterative_compressions.jsonl")))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only_passed_quality", action="store_true",
                    default=True)
    ap.add_argument("--no_only_passed_quality", action="store_false",
                    dest="only_passed_quality")
    args = ap.parse_args()

    ensure_outputs()
    cases = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    facts = read_jsonl(Path(args.facts))
    conds = read_jsonl(Path(args.conditions))
    if args.only_passed_quality:
        conds = [c for c in conds if c.get("quality_passed")]

    # Pick one fact per case
    case_ids = sorted({c["case_id"] for c in conds})
    target_fact_by_case = {
        cid: _pick_target_fact_id(cid, facts) for cid in case_ids
    }
    fact_by_id = {f["fact_id"]: f for f in facts}

    # Filter conditions to only those for chosen target facts
    chosen = []
    for cond in conds:
        if cond["fact_id"] == target_fact_by_case.get(cond["case_id"]):
            chosen.append(cond)
    print(f"[06] {len(chosen)} chains "
          f"(2/case × {len({c['case_id'] for c in chosen})} cases) "
          f"× {len(args.models.split(','))} models")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    variants = [v.strip() for v in args.prompt_variants.split(",") if v.strip()]
    bundles = {v: load_bundle(v) for v in variants}
    clients = {m: make_client(m) for m in models}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    def _run_chain(item):
        cond, model, variant = item
        bundle = bundles[variant]
        client = clients[model]
        case = cases.get(cond["case_id"])
        if not case:
            return []
        results = iterate_compression(
            client=client, model_name=model, bundle=bundle,
            task=cond["condition_task"],
            x0=case["full_trajectory_text"],
            rounds=args.rounds, max_chars=args.budget_chars,
        )
        out: List[dict] = []
        for r in results:
            out.append({
                "case_id":         cond["case_id"],
                "fact_id":         cond["fact_id"],
                "condition_id":    cond["condition_id"],
                "need_label":      int(cond["need_label"]),
                "compressor_model": model,
                "prompt_variant":  variant,
                "budget_chars":    args.budget_chars,
                "round":           r.round,
                "input_chars":     r.input_chars,
                "output_chars":    r.output_chars,
                "compressed_text": r.compressed_text,
                "text_sha256":     r.text_sha256,
                "elapsed_s":       r.elapsed_s,
                "prompt_tokens":   r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "invalid_output":  r.invalid_output,
                "error":           r.error,
            })
        return out

    work = [(c, m, v) for c in chosen for m in models for v in variants]
    print(f"[06] {len(work)} chains total; {len(work) * args.rounds} compressions")

    t0 = time.time()
    n_chains = 0
    n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_run_chain, w): w for w in work}
            for fut in as_completed(futures):
                cond, model, variant = futures[fut]
                rows = fut.result()
                for r in rows:
                    if r.get("error"):
                        n_err += 1
                    f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                f_out.flush()
                n_chains += 1
                eta = (time.time() - t0) / n_chains * (len(work) - n_chains)
                print(f"  [{n_chains:>4d}/{len(work)}] "
                      f"{cond['case_id']:>12s} model={model:<8s} "
                      f"cond={'need' if cond['need_label'] else 'unneed'}  "
                      f"rounds={len(rows)} errs_total={n_err} "
                      f"eta={eta:.0f}s", flush=True)
    print(f"[06] {n_chains} chains × {args.rounds} rounds  err={n_err} "
          f"-> {out_path}")
    print(f"[06] total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
