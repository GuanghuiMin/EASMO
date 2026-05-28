"""Stage 05 — score single-round retention (spec §14).

For each (compressed_text, fact) pair:
  - deterministic exact substring match;
  - LLM semantic scorer (RETENTION_SCORER_PROMPT) via the *other* model
    (Qwen for MiniMax compressions, MiniMax for Qwen compressions).

We only score the fact that the compression was conditioned on
(i.e. the per-row ``fact_id``). Other facts are scored later in
stage 07 for the iterative chains.

Output: outputs/raw/fact_retention_scores_single_round.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


# Cross-evaluator rule: never score with the same model that compressed.
_CROSS_EVAL = {
    "qwen": "minimax",
    "minimax": "qwen",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts",
                    default=str(_REPO / "data" / "fact_bank_filtered.jsonl"))
    ap.add_argument("--compressions",
                    default=str(raw_path("single_round_compressions.jsonl")))
    ap.add_argument("--out",
                    default=str(raw_path("fact_retention_scores_single_round.jsonl")))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ensure_outputs()
    fact_by_id = {f["fact_id"]: f for f in read_jsonl(Path(args.facts))}
    compressions = read_jsonl(Path(args.compressions))
    print(f"[05] {len(compressions)} compressions to score")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    clients = {name: make_client(name) for name in {"qwen", "minimax"}}

    def _score_one(row):
        fact = fact_by_id.get(row["fact_id"])
        if not fact:
            return None
        compressor = row["compressor_model"].split("-")[0].lower()  # "qwen" or "minimax"
        if compressor not in _CROSS_EVAL:
            scorer = "minimax"
        else:
            scorer = _CROSS_EVAL[compressor]

        prompt = render_retention_prompt(
            fact_id=fact["fact_id"],
            canonical_fact=fact["canonical_fact"],
            fact_type=fact["fact_type"],
            source_quote=fact["source_quote"],
            literal_values=fact.get("literal_values") or [],
            compressed_text=row.get("compressed_text") or "",
        )
        # Deterministic exact retention
        exact = deterministic_exact_retained(
            compressed_text=row.get("compressed_text") or "",
            source_quote=fact["source_quote"],
            verbatim_surface=fact.get("verbatim_surface") or "",
            literal_values=fact.get("literal_values") or [],
        )
        # LLM scorer
        try:
            res = chat(
                name=scorer, user=prompt,
                system=_SCORER_SYSTEM,
                client=clients[scorer],
                temperature=0.0, max_tokens=512, json_mode=True, seed=42,
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
            "round":           1,
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
            futures = [ex.submit(_score_one, r) for r in compressions]
            for fut in as_completed(futures):
                rec = fut.result()
                if rec is None:
                    continue
                if rec.get("scorer_error"):
                    n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                n_done += 1
                if n_done % 50 == 0 or n_done <= 3:
                    eta = (time.time() - t0) / n_done * (len(compressions) - n_done)
                    print(f"  [{n_done:>4d}/{len(compressions)}] "
                          f"err={n_err:<3d}  eta={eta:.0f}s", flush=True)

    print(f"[05] wrote {n_done} retention scores ({n_err} errors) -> {out_path}")
    print(f"[05] total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
