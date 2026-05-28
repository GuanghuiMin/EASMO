"""Stage 04 — single-round need-conditioned compression (spec §12).

For every passing (case, fact, condition, model, prompt_variant, budget):
  1. render the original ACON prompt;
  2. call the compressor with temp 0;
  3. save compressed text.

Output: outputs/raw/single_round_compressions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v7.data import (  # noqa: E402
    ensure_outputs, read_jsonl, write_jsonl, raw_path, append_jsonl,
)
from motivation_v7.clients import make_client  # noqa: E402
from motivation_v7.acon_prompt_loader import load_bundle  # noqa: E402
from motivation_v7.compress import compress_once  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "case_pool.jsonl"))
    ap.add_argument("--facts",
                    default=str(_REPO / "data" / "fact_bank_filtered.jsonl"))
    ap.add_argument("--conditions",
                    default=str(_REPO / "data" / "need_conditions.jsonl"))
    ap.add_argument("--models", default="qwen,minimax",
                    help="Comma list of compressor names.")
    ap.add_argument("--prompt_variants", default="UTCO",
                    help="Comma list (UT, UTCO).")
    ap.add_argument("--budget_chars", type=int, default=1500)
    ap.add_argument("--out", default=str(raw_path("single_round_compressions.jsonl")))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_facts_per_case", type=int, default=None)
    ap.add_argument("--only_passed_quality", action="store_true",
                    default=True)
    ap.add_argument("--no_only_passed_quality", action="store_false",
                    dest="only_passed_quality")
    args = ap.parse_args()

    ensure_outputs()
    cases = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    fact_by_id = {f["fact_id"]: f for f in read_jsonl(Path(args.facts))}
    conds = read_jsonl(Path(args.conditions))
    if args.only_passed_quality:
        conds = [c for c in conds if c.get("quality_passed")]

    if args.max_facts_per_case:
        # group conds by case and trim
        from collections import defaultdict
        by_case: Dict[str, List[dict]] = defaultdict(list)
        for c in conds:
            by_case[c["case_id"]].append(c)
        trimmed: List[dict] = []
        for case_id, lst in by_case.items():
            # keep at most 2 × max_facts_per_case rows (needed + unneeded)
            seen_facts = []
            for c in lst:
                if c["fact_id"] not in seen_facts:
                    seen_facts.append(c["fact_id"])
                if len(seen_facts) > args.max_facts_per_case:
                    break
                trimmed.append(c)
            # include all matched conditions for retained facts
        conds = [c for c in conds if c["fact_id"] in {x["fact_id"] for x in trimmed}]

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    variants = [v.strip() for v in args.prompt_variants.split(",") if v.strip()]
    bundles = {v: load_bundle(v) for v in variants}

    print(f"[04] {len(conds)} conditions × {len(models)} models × {len(variants)} variants "
          f"= {len(conds) * len(models) * len(variants)} compression cells")

    # Build full work list.
    work: List[Tuple] = []
    for cond in conds:
        case = cases.get(cond["case_id"])
        fact = fact_by_id.get(cond["fact_id"])
        if not case or not fact:
            continue
        for model in models:
            for variant in variants:
                work.append((cond, case, fact, model, variant))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Truncate file at the start so reruns don't duplicate.
    out_path.write_text("")

    # Per-model clients (one each, reused across threads).
    clients = {m: make_client(m) for m in models}

    def _run_one(item):
        cond, case, fact, model, variant = item
        bundle = bundles[variant]
        client = clients[model]
        res = compress_once(
            client=client, model_name=model, bundle=bundle,
            task=cond["condition_task"],
            history=case["full_trajectory_text"],
            max_chars=args.budget_chars, round_idx=1,
        )
        return {
            "case_id":         case["case_id"],
            "fact_id":         fact["fact_id"],
            "condition_id":    cond["condition_id"],
            "need_label":      int(cond["need_label"]),
            "fact_type":       fact["fact_type"],
            "coarse_group":    fact["coarse_group"],
            "compressor_model": model,
            "prompt_variant":  variant,
            "budget_chars":    args.budget_chars,
            "round":           1,
            "input_chars":     res.input_chars,
            "output_chars":    res.output_chars,
            "output_tokens_est": res.completion_tokens,
            "compressed_text": res.compressed_text,
            "text_sha256":     res.text_sha256,
            "elapsed_s":       res.elapsed_s,
            "prompt_tokens":   res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "invalid_output":  res.invalid_output,
            "error":           res.error,
        }

    t0 = time.time()
    n_done = 0
    n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_run_one, w): w for w in work}
            for fut in as_completed(futures):
                rec = fut.result()
                if rec.get("error"):
                    n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                n_done += 1
                eta = (time.time() - t0) / n_done * (len(work) - n_done)
                if n_done % 20 == 0 or n_done <= 5:
                    print(f"  [{n_done:>4d}/{len(work)}] "
                          f"err={n_err:<3d} "
                          f"eta={eta:.0f}s",
                          flush=True)
    print(f"[04] wrote {n_done} compressions ({n_err} errors) -> {out_path}")
    print(f"[04] total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
