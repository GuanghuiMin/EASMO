"""Stage 07 — score retention of every case-level fact against every
iterative compressed text (spec §16).

For each iterative compression row × every fact in the same case:
  - deterministic substring match;
  - LLM semantic scorer via the cross-model evaluator.

We score *all case-level facts* (not just the target fact of the
condition) so we can build per-fact-type survival curves.

Output: outputs/raw/fact_retention_scores_iterative.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v7.data import (  # noqa: E402
    ensure_outputs, read_jsonl, write_jsonl, raw_path,
)
from motivation_v7.clients import (  # noqa: E402
    chat, make_client, parse_json_object,
)
from motivation_v7.retention import (  # noqa: E402
    render_retention_prompt, deterministic_exact_retained,
    parse_retention_response, combine_scores,
)


_SCORER_SYSTEM = (
    "You are a strict retention evaluator.\n"
    "Return only valid JSON. Do not include prose outside JSON."
)


_CROSS_EVAL = {
    "qwen": "minimax",
    "minimax": "qwen",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts",
                    default=str(_REPO / "data" / "fact_bank_filtered.jsonl"))
    ap.add_argument("--iterative",
                    default=str(raw_path("iterative_compressions.jsonl")))
    ap.add_argument("--out",
                    default=str(raw_path("fact_retention_scores_iterative.jsonl")))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--scorer", default="auto",
                    help="auto | qwen | minimax (auto = cross-eval)")
    args = ap.parse_args()

    ensure_outputs()
    facts_by_case: Dict[str, List[dict]] = defaultdict(list)
    for f in read_jsonl(Path(args.facts)):
        facts_by_case[f["case_id"]].append(f)

    iter_rows = read_jsonl(Path(args.iterative))
    print(f"[07] {len(iter_rows)} iterative compressions; "
          f"{sum(len(v) for v in facts_by_case.values())} facts total")

    clients = {n: make_client(n) for n in ("qwen", "minimax")}

    work = []
    for row in iter_rows:
        case_facts = facts_by_case.get(row["case_id"], [])
        for fact in case_facts:
            work.append((row, fact))
    print(f"[07] {len(work)} retention scoring calls")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    def _score_one(item):
        row, fact = item
        compressor = row["compressor_model"].split("-")[0].lower()
        scorer = (args.scorer if args.scorer != "auto"
                  else _CROSS_EVAL.get(compressor, "minimax"))
        prompt = render_retention_prompt(
            fact_id=fact["fact_id"],
            canonical_fact=fact["canonical_fact"],
            fact_type=fact["fact_type"],
            source_quote=fact["source_quote"],
            literal_values=fact.get("literal_values") or [],
            compressed_text=row.get("compressed_text") or "",
        )
        exact = deterministic_exact_retained(
            compressed_text=row.get("compressed_text") or "",
            source_quote=fact["source_quote"],
            verbatim_surface=fact.get("verbatim_surface") or "",
            literal_values=fact.get("literal_values") or [],
        )
        if exact:
            # Skip the LLM call entirely if we already have exact match —
            # large savings since many concrete facts are verbatim-preserved.
            llm_score = parse_retention_response(fact["fact_id"], {
                "retention_label": "exact",
                "retention_score": 1.0,
                "evidence_in_compressed_text": fact["source_quote"][:200],
                "is_distorted": False, "confidence": 1.0,
                "short_reason": "deterministic exact substring match",
            })
            scorer_elapsed = 0.0
            scorer_error = None
        else:
            try:
                res = chat(
                    name=scorer, user=prompt,
                    system=_SCORER_SYSTEM,
                    client=clients[scorer],
                    temperature=0.0, max_tokens=384,
                    json_mode=True, seed=42,
                )
                obj = parse_json_object(res.text)
                llm_score = parse_retention_response(fact["fact_id"], obj)
                scorer_elapsed = res.elapsed_s
                scorer_error = res.error
            except Exception as e:
                llm_score = parse_retention_response(fact["fact_id"], None)
                scorer_elapsed = 0.0
                scorer_error = str(e)

        combined = combine_scores(exact=exact, llm=llm_score)
        return {
            "case_id":         row["case_id"],
            "fact_id":         fact["fact_id"],
            "condition_id":    row["condition_id"],
            "need_label":      int(row["need_label"]),
            "fact_type":       fact["fact_type"],
            "coarse_group":    fact["coarse_group"],
            "compressor_model": row["compressor_model"],
            "prompt_variant":  row["prompt_variant"],
            "budget_chars":    row["budget_chars"],
            "round":           int(row["round"]),
            **combined,
            "evidence_in_compressed_text": llm_score.evidence,
            "is_distorted":    llm_score.is_distorted,
            "confidence":      llm_score.confidence,
            "scorer_model":    scorer,
            "scorer_elapsed_s": scorer_elapsed,
            "scorer_error":    scorer_error,
        }

    t0 = time.time()
    n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_score_one, w) for w in work]
            for fut in as_completed(futures):
                rec = fut.result()
                if rec.get("scorer_error"):
                    n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if n_done % 200 == 0:
                    f_out.flush()
                n_done += 1
                if n_done % 200 == 0 or n_done <= 3:
                    eta = (time.time() - t0) / n_done * (len(work) - n_done)
                    print(f"  [{n_done:>5d}/{len(work)}] "
                          f"err={n_err:<3d}  eta={eta:.0f}s", flush=True)

    print(f"[07] wrote {n_done} ({n_err} errors) -> {out_path}")
    print(f"[07] total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
