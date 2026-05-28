"""Stage 02b — retry the cases that OOM'd in stage 02 with a smaller
max_context_tokens budget. Appends to existing jacobian_span_scores.jsonl,
jacobian_case_summary.jsonl, active_vector_metadata.jsonl, and merges
the active_vectors_layer{L}.npz.

Usage:
  python scripts/02b_retry_oom_cases.py \
     --model_path … --task_ids 57c3486_2 57c3486_3 \
     --max_context_tokens 6000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from motivation_v6_jacobian.data import (  # noqa: E402
    raw_path, read_jsonl, write_jsonl, append_jsonl,
)
from _model_loader import load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--cases", default=str(raw_path("cases.jsonl")))
    ap.add_argument("--task_ids", nargs="+", required=True)
    ap.add_argument("--max_context_tokens", type=int, default=6000)
    ap.add_argument("--layer_index", type=int, default=18)
    args = ap.parse_args()

    cases = [c for c in read_jsonl(Path(args.cases)) if c["task_id"] in args.task_ids]
    print(f"[02b] retrying {len(cases)} cases @ max_ctx={args.max_context_tokens}")

    model, tok = load_model(args.model_path)
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("[02b] grad checkpointing enabled")
    except Exception as e:
        print(f"[02b] grad checkpointing failed: {e}")

    from motivation_v6_jacobian.gradients import compute_jacobian_saliency

    # Load existing artifacts
    span_scores_path = raw_path("jacobian_span_scores.jsonl")
    case_summary_path = raw_path("jacobian_case_summary.jsonl")
    meta_path = raw_path("active_vector_metadata.jsonl")
    npz_path = raw_path(f"active_vectors_layer{args.layer_index}.npz")

    existing_meta = read_jsonl(meta_path)
    existing_npz = np.load(npz_path)
    examples = existing_npz["example"]
    spans = existing_npz["span"]

    span_meta_existing = [m for m in existing_meta if m["matrix"] == "span"]
    example_meta_existing = [m for m in existing_meta if m["matrix"] == "example"]

    new_span_rows = []
    new_case_rows = []
    new_example_vecs = []
    new_span_vecs = []
    new_example_meta = []
    new_span_meta = []

    for case in cases:
        v4_by_id = {s["span_id"]: s for s in case["spans"]}
        t0 = time.time()
        try:
            res = compute_jacobian_saliency(
                model=model, tokenizer=tok,
                task_id=case["task_id"],
                task_instruction=case["task_instruction"],
                spans=case["spans"],
                target_text=case["target_text"],
                max_context_tokens=args.max_context_tokens,
                capture_active=True,
                active_layer_index=args.layer_index,
            )
        except Exception as e:
            print(f"  ! still failed {case['task_id']}: {e}")
            continue

        scored = res.spans
        scored.sort(key=lambda r: -r["span_gxa_sqrtlen"])
        j_rank = {r["span_id"]: i + 1 for i, r in enumerate(scored)}
        scored.sort(key=lambda r: -v4_by_id.get(r["span_id"], {})
                                          .get("v4_final_sensitivity", -1))
        v4_rank = {r["span_id"]: i + 1 for i, r in enumerate(scored)}
        for r in res.spans:
            v4 = v4_by_id.get(r["span_id"], {})
            new_span_rows.append({
                **r,
                "v4_final_sensitivity": v4.get("v4_final_sensitivity"),
                "v4_rule_norm": v4.get("v4_rule_norm"),
                "v4_judge_score": v4.get("v4_judge_score"),
                "v4_token_count": v4.get("v4_token_count"),
                "jacobian_rank": j_rank.get(r["span_id"]),
                "v4_sensitivity_rank": v4_rank.get(r["span_id"]),
            })

        # Active vectors
        cursor = 0
        for r in res.spans:
            n = r["token_count"]
            if n <= 0:
                continue
            lo, hi = cursor, cursor + n
            hidden_slice = res.active_hidden[lo:hi]
            grad_slice = res.active_grad[lo:hi]
            a_span = (hidden_slice * grad_slice).sum(axis=0)
            new_span_vecs.append(a_span.astype(np.float32))
            new_span_meta.append({
                "task_id": case["task_id"],
                "span_id": r["span_id"],
                "step_id": r["step_id"],
                "v4_final_sensitivity": v4_by_id.get(r["span_id"], {})
                                            .get("v4_final_sensitivity"),
                "token_count": n,
            })
            cursor = hi
        a_example = (res.active_hidden * res.active_grad).sum(axis=0)
        new_example_vecs.append(a_example.astype(np.float32))
        new_example_meta.append({
            "task_id": case["task_id"],
            "n_context_tokens": res.n_context_tokens,
            "loss": res.loss,
        })

        new_case_rows.append({
            "task_id": case["task_id"],
            "loss": res.loss,
            "n_context_tokens": res.n_context_tokens,
            "n_target_tokens": res.n_target_tokens,
            "n_spans_used": len(res.spans),
            "retry": True,
            "max_context_tokens": args.max_context_tokens,
        })
        print(f"  + {case['task_id']}  loss={res.loss:.3f} "
              f"n_ctx={res.n_context_tokens} spans={len(res.spans)} "
              f"dt={time.time()-t0:.1f}s",
              flush=True)

    # Merge .npz
    new_examples = np.concatenate([examples] + (
        [np.stack(new_example_vecs)] if new_example_vecs else []
    ), axis=0)
    new_spans = np.concatenate([spans] + (
        [np.stack(new_span_vecs)] if new_span_vecs else []
    ), axis=0)
    np.savez_compressed(npz_path, example=new_examples, span=new_spans)

    # Merge metadata: re-number rows
    final_meta = []
    for i, m in enumerate(example_meta_existing):
        final_meta.append({**m, "row": i})
    for j, m in enumerate(new_example_meta):
        final_meta.append({**m, "matrix": "example",
                           "row": len(example_meta_existing) + j})
    for i, m in enumerate(span_meta_existing):
        final_meta.append({**m, "row": i})
    for j, m in enumerate(new_span_meta):
        final_meta.append({**m, "matrix": "span",
                           "row": len(span_meta_existing) + j})
    write_jsonl(meta_path, final_meta)

    # Append span scores
    with open(span_scores_path, "a", encoding="utf-8") as f:
        for r in new_span_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Append case summary
    with open(case_summary_path, "a", encoding="utf-8") as f:
        for r in new_case_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[02b] appended {len(new_span_rows)} span rows, "
          f"{len(new_case_rows)} cases, "
          f"npz now {new_examples.shape[0]} example + {new_spans.shape[0]} span rows")


if __name__ == "__main__":
    main()
