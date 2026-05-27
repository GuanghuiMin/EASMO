"""Stage 02 — compute per-span Jacobian saliency for every case and
optionally capture mid-layer Jacobian-weighted activations.

Outputs:
  outputs/raw/jacobian_span_scores.jsonl    — one row per (task, span)
  outputs/raw/jacobian_case_summary.jsonl   — one row per task (loss, n_tokens)
  outputs/raw/active_vectors_layer{L}.npz   — example + span active vectors
  outputs/raw/active_vector_metadata.jsonl  — row metadata for the npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v6_jacobian.data import (  # noqa: E402
    ensure_outputs, raw_path, read_jsonl, write_jsonl,
)


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _model_loader import load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default=str(raw_path("jacobian_span_scores.jsonl")))
    ap.add_argument("--case_summary_out",
                    default=str(raw_path("jacobian_case_summary.jsonl")))
    ap.add_argument("--max_context_tokens", type=int, default=12000)
    ap.add_argument("--layer_index", type=int, default=None,
                    help="If set, capture hidden+grad at this layer "
                         "for the active-subspace experiment. -1 = last; "
                         "default omits the index and uses N/2.")
    ap.add_argument("--capture_active", action="store_true",
                    help="If set, save active vectors (H ⊙ G) to npz.")
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--gradient_checkpointing", action="store_true",
                    help="Enable gradient checkpointing for memory.")
    args = ap.parse_args()

    ensure_outputs()
    cases = read_jsonl(Path(args.cases))
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    print(f"[02] {len(cases)} cases")

    model, tok = load_model(args.model_path)
    device = next(model.parameters()).device

    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("[02] gradient checkpointing enabled")
        except Exception as e:
            print(f"[02] WARN: gradient_checkpointing_enable failed ({e}); "
                  "continuing without it")

    # Resolve target layer for active capture.
    from motivation_v6_jacobian.hooks import find_transformer_blocks, resolve_layer_index
    n_layers = len(find_transformer_blocks(model))
    if args.layer_index is None:
        layer_index = n_layers // 2
    else:
        layer_index = resolve_layer_index(model, args.layer_index)
    print(f"[02] model has {n_layers} layers; using layer_index={layer_index}"
          f"  capture_active={args.capture_active}")

    from motivation_v6_jacobian.gradients import compute_jacobian_saliency

    span_rows: List[dict] = []
    case_rows: List[dict] = []
    active_examples: List[np.ndarray] = []
    active_spans: List[np.ndarray] = []
    span_meta: List[dict] = []
    example_meta: List[dict] = []
    t_start = time.time()

    for ci, case in enumerate(cases, 1):
        t0 = time.time()
        try:
            res = compute_jacobian_saliency(
                model=model, tokenizer=tok,
                task_id=case["task_id"],
                task_instruction=case["task_instruction"],
                spans=case["spans"],
                target_text=case["target_text"],
                max_context_tokens=args.max_context_tokens,
                capture_active=args.capture_active,
                active_layer_index=layer_index,
                device=device,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"  ! OOM on {case['task_id']}: {e}; skipping")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  ! ERROR on {case['task_id']}: {type(e).__name__}: {e}")
            continue

        # Pair Jacobian span rows with the v4 sensitivity scores
        # already on the case.spans list.
        v4_by_id = {s["span_id"]: s for s in case["spans"]}
        # Rankings within this case for jacobian_score and v4 sensitivity
        scored = res.spans
        scored.sort(key=lambda r: -r["span_gxa_sqrtlen"])
        j_rank = {r["span_id"]: i + 1 for i, r in enumerate(scored)}
        scored.sort(key=lambda r: -v4_by_id.get(r["span_id"], {})
                                          .get("v4_final_sensitivity", -1))
        v4_rank = {r["span_id"]: i + 1 for i, r in enumerate(scored)}

        for r in res.spans:
            v4 = v4_by_id.get(r["span_id"], {})
            span_rows.append({
                **r,
                "v4_final_sensitivity": v4.get("v4_final_sensitivity"),
                "v4_rule_norm": v4.get("v4_rule_norm"),
                "v4_judge_score": v4.get("v4_judge_score"),
                "v4_token_count": v4.get("v4_token_count"),
                "jacobian_rank": j_rank.get(r["span_id"]),
                "v4_sensitivity_rank": v4_rank.get(r["span_id"]),
            })
            if args.capture_active and res.active_hidden is not None:
                lo = r.get("token_count", 0)
                # We rebuild offsets from the order spans appear; the
                # gradient module already returned span_records in
                # chronological order so we can re-derive the active
                # vector here. For correctness we re-traverse res.spans:
                pass  # see active-vector aggregation below

        # ---- aggregate active vectors per span and per example ----
        if args.capture_active and res.active_hidden is not None:
            # Re-derive span token offsets *inside the context* from
            # the per-span token_count (chronological order, no gaps).
            cursor = 0
            for r in res.spans:
                n = r["token_count"]
                if n <= 0:
                    cursor += 0
                    continue
                lo = cursor
                hi = cursor + n
                hidden_slice = res.active_hidden[lo:hi]
                grad_slice = res.active_grad[lo:hi]
                # Sum H ⊙ G across the span tokens → (hidden_dim,)
                a_span = (hidden_slice * grad_slice).sum(axis=0)
                active_spans.append(a_span.astype(np.float32))
                span_meta.append({
                    "task_id": case["task_id"],
                    "span_id": r["span_id"],
                    "step_id": r["step_id"],
                    "v4_final_sensitivity": v4_by_id.get(r["span_id"], {})
                                                  .get("v4_final_sensitivity"),
                    "token_count": n,
                })
                cursor = hi
            a_example = (res.active_hidden * res.active_grad).sum(axis=0)
            active_examples.append(a_example.astype(np.float32))
            example_meta.append({
                "task_id": case["task_id"],
                "n_context_tokens": res.n_context_tokens,
                "loss": res.loss,
            })

        case_rows.append({
            "task_id": case["task_id"],
            "loss": res.loss,
            "n_context_tokens": res.n_context_tokens,
            "n_target_tokens": res.n_target_tokens,
            "n_spans_used": len(res.spans),
        })
        dt = time.time() - t0
        eta = (time.time() - t_start) / ci * (len(cases) - ci)
        print(f"  [{ci:>2d}/{len(cases)}] {case['task_id']:>12s} "
              f"loss={res.loss:.3f} n_ctx={res.n_context_tokens:>5d} "
              f"spans={len(res.spans):>3d} dt={dt:>5.1f}s "
              f"eta={eta:.0f}s",
              flush=True)

    write_jsonl(Path(args.out), span_rows)
    write_jsonl(Path(args.case_summary_out), case_rows)
    print(f"[02] wrote {len(span_rows)} span rows -> {args.out}")
    print(f"[02] wrote {len(case_rows)} case rows -> {args.case_summary_out}")

    if args.capture_active and active_spans:
        npz_path = raw_path(f"active_vectors_layer{layer_index}.npz")
        np.savez_compressed(
            npz_path,
            example=np.stack(active_examples, axis=0),
            span=np.stack(active_spans, axis=0),
        )
        meta_path = raw_path("active_vector_metadata.jsonl")
        write_jsonl(meta_path, [
            {"matrix": "example", "row": i, **m}
            for i, m in enumerate(example_meta)
        ] + [
            {"matrix": "span", "row": i, **m}
            for i, m in enumerate(span_meta)
        ])
        print(f"[02] wrote active vectors -> {npz_path}")
        print(f"      metadata -> {meta_path}")

    print(f"[02] total elapsed {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
