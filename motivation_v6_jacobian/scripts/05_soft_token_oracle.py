"""Stage 05 — soft-token oracle (Experiment C).

For each case and k ∈ args.ks, optimise k continuous soft tokens to
match the teacher-forced target NLL, and record alongside the
full/no/recent/acon baselines.

Outputs:
  outputs/tables/soft_token_oracle_losses.csv
  outputs/raw/soft_token_histories.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v6_jacobian.data import (  # noqa: E402
    ensure_outputs, raw_path, table_path, read_jsonl, write_jsonl,
    load_v4_compressed_contexts, render_recent_context,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default=str(table_path("soft_token_oracle_losses.csv")))
    ap.add_argument("--histories_out",
                    default=str(raw_path("soft_token_histories.jsonl")))
    ap.add_argument("--ks", default="4,8,16,32,64")
    ap.add_argument("--num_steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--max_context_tokens", type=int, default=12000)
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--skip_baselines", action="store_true")
    args = ap.parse_args()

    ensure_outputs()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    cases = read_jsonl(Path(args.cases))
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    print(f"[05] {len(cases)} cases, ks={ks}, num_steps={args.num_steps}")

    sys.path.insert(0, str(_REPO / "scripts"))
    from _model_loader import load_model  # type: ignore
    model, tok = load_model(args.model_path)
    from motivation_v6_jacobian.soft_tokens import (
        train_soft_tokens, baseline_losses, gap_recovery,
    )

    # Acquire v4 ACON compressed text per task (best-effort).
    compressed = load_v4_compressed_contexts()

    rows: List[dict] = []
    histories: List[dict] = []
    t_start = time.time()
    for ci, case in enumerate(cases, 1):
        case_t0 = time.time()
        # baselines
        bl: Dict[str, float] = {}
        if not args.skip_baselines:
            acon_text: Optional[str] = None
            cmap = compressed.get(case["task_id"], {})
            for key in ("acon", "acon_baseline", "acon_baseline_uncapped"):
                if key in cmap:
                    acon_text = cmap[key].get("compressed_text") or cmap[key].get("text")
                    if acon_text:
                        break
            recent_text = render_recent_context(case["spans"], n_tail=5)
            try:
                bl = baseline_losses(
                    model=model, tokenizer=tok,
                    task_instruction=case["task_instruction"],
                    target_text=case["target_text"],
                    full_context=case["context_text"],
                    recent_context=recent_text,
                    acon_context=acon_text,
                    max_context_tokens=args.max_context_tokens,
                )
            except Exception as e:
                print(f"  ! baseline failed on {case['task_id']}: {e}")
                bl = {"full": float("nan"), "no": float("nan"),
                      "recent": float("nan"), "acon": float("nan")}
        # soft tokens for each k
        row = {
            "task_id": case["task_id"],
            "full_loss": bl.get("full", float("nan")),
            "no_loss": bl.get("no", float("nan")),
            "recent_loss": bl.get("recent", float("nan")),
            "acon_loss": bl.get("acon", float("nan")),
        }
        for k in ks:
            try:
                run = train_soft_tokens(
                    model=model, tokenizer=tok,
                    task_id=case["task_id"],
                    task_instruction=case["task_instruction"],
                    target_text=case["target_text"],
                    k=k,
                    num_steps=args.num_steps,
                    lr=args.lr,
                    patience=args.patience,
                )
                row[f"soft_loss_k{k}"] = run.final_loss
                row[f"soft_steps_k{k}"] = run.n_steps
                histories.append({
                    "task_id": case["task_id"], "k": k,
                    "final_loss": run.final_loss,
                    "n_steps": run.n_steps,
                    "history": run.history,
                })
            except torch.cuda.OutOfMemoryError as e:
                print(f"  ! OOM on {case['task_id']} k={k}: {e}")
                row[f"soft_loss_k{k}"] = float("nan")
                row[f"soft_steps_k{k}"] = 0
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ! err on {case['task_id']} k={k}: {type(e).__name__}: {e}")
                row[f"soft_loss_k{k}"] = float("nan")
                row[f"soft_steps_k{k}"] = 0

        # gap recovery columns
        L_no = row["no_loss"]; L_full = row["full_loss"]
        for k in ks:
            row[f"soft_gap_recovery_k{k}"] = gap_recovery(
                L_no, L_full, row.get(f"soft_loss_k{k}", float("nan"))
            )
        row["recent_gap_recovery"] = gap_recovery(L_no, L_full, row["recent_loss"])
        row["acon_gap_recovery"] = gap_recovery(L_no, L_full, row["acon_loss"])

        rows.append(row)
        dt = time.time() - case_t0
        eta = (time.time() - t_start) / ci * (len(cases) - ci)
        soft_str = " ".join(f"k{k}={row[f'soft_loss_k{k}']:.3f}" for k in ks)
        print(f"  [{ci:>2d}/{len(cases)}] {case['task_id']:>12s}  "
              f"full={row['full_loss']:.3f} no={row['no_loss']:.3f}  "
              f"{soft_str}  dt={dt:.1f}s eta={eta:.0f}s",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    write_jsonl(Path(args.histories_out), histories)
    print(f"[05] wrote {len(rows)} rows -> {args.out}")
    print(f"[05] wrote histories -> {args.histories_out}")
    print(f"[05] total elapsed {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
