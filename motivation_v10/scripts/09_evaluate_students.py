"""Stage 09 — evaluate student compressors (spec §15).

Five variants per test_behavior case:
  * `MiniMax-greedy`           — reuse stage 02 greedy compression
  * `MiniMax-oracle-bestofN`   — reuse stage 04 oracle pick
  * `Raw-Qwen`                 — Qwen3-4B base (no LoRA) with ACON UTCO prompt
  * `Qwen-SFT-C1`              — adapter-merged at outputs/models/qwen_sft_c1/
  * `Qwen-SFT-CK`              — adapter-merged at outputs/models/qwen_sft_ck/

Two-phase script:

  Phase A (EASMO/.venv): generate student compressions for the
                         Qwen variants on test_behavior cases, write
                         outputs/raw/student_compressions.jsonl.
  Phase B (acon/.venv):  run the AppWorld downstream agent on every
                         (variant × case × {C1, CK}) compression and
                         write outputs/raw/student_behavior_runs.jsonl.

Stage 09 expects:
  * vLLM on port 8000 to be STOPPED during phase A (we're loading
    a 4B model via HF transformers, which conflicts with vLLM's
    GPU memory hold).
  * vLLM to be RUNNING (or absent) during phase B — phase B uses
    the remote MiniMax-M2.5 endpoint for the downstream agent.

By default this script runs phase A only. Use `--phase B` for the
agent-eval phase, or `--phase both` to do both in one invocation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _read_jsonl(p):
    out = []
    if not Path(p).exists():
        return out
    for line in open(p):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# ----------------------------------------------------------------------
# Phase A — generate student compressions  (EASMO/.venv)
# ----------------------------------------------------------------------

def phase_a(args) -> None:
    from motivation_v10.data import (
        ensure_outputs, read_jsonl, raw_path, model_path,
    )
    from motivation_v10.acon_prompt_loader import load_utco_bundle, render_prompt
    from motivation_v10.student_inference import StudentCompressor

    ensure_outputs()
    bundle = load_utco_bundle()
    cases = [c for c in read_jsonl(_REPO / "data" / "v10_cases.jsonl")
             if c.get("split") in {s.strip() for s in args.splits.split(",")}]
    print(f"[09a] {len(cases)} cases in scope (splits={args.splits})")

    variants = []
    if args.include_raw:
        variants.append(("Raw-Qwen", None))
    if args.include_sft_c1:
        variants.append(("Qwen-SFT-C1", model_path("qwen_sft_c1")))
    if args.include_sft_ck:
        variants.append(("Qwen-SFT-CK", model_path("qwen_sft_ck")))

    out_path = Path(args.compressions_out)
    done = set()
    if out_path.exists():
        for r in _read_jsonl(out_path):
            done.add((r["variant"], r["case_id"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for variant_name, adapter_dir in variants:
        if adapter_dir is not None and not Path(adapter_dir).exists():
            print(f"[09a] adapter dir missing for {variant_name}: {adapter_dir} — skipping")
            continue
        sc = StudentCompressor(
            adapter_dir=str(adapter_dir) if adapter_dir else None,
            base_model_id=args.base_model_id,
        )
        t0 = time.time(); n_done = 0
        with open(out_path, "a", encoding="utf-8") as f_out:
            for case in cases:
                key = (variant_name, case["case_id"])
                if key in done:
                    continue
                user_text = render_prompt(
                    bundle,
                    task=case["user_instruction"],
                    history=case["full_trajectory_text"],
                    prev_summary="",
                    max_chars=1500,
                )
                t1 = time.time()
                try:
                    gen = sc.generate(
                        system_text=bundle.system_text,
                        user_text=user_text,
                        temperature=0.0, seed=42,
                        max_new_tokens=args.max_new_tokens,
                    )
                    rec = {
                        "variant":          variant_name,
                        "case_id":          case["case_id"],
                        "split":            case.get("split", "unknown"),
                        "adapter_dir":      str(adapter_dir) if adapter_dir else "",
                        "base_model_id":    args.base_model_id,
                        "temperature":      0.0,
                        "seed":             42,
                        "compressed_text":  gen["text"],
                        "compressed_chars": len(gen["text"]),
                        "prompt_tokens":    gen["prompt_tokens"],
                        "completion_tokens": gen["completion_tokens"],
                        "elapsed_s":        time.time() - t1,
                        "error":            None,
                    }
                except Exception as e:
                    rec = {
                        "variant":          variant_name,
                        "case_id":          case["case_id"],
                        "split":            case.get("split", "unknown"),
                        "adapter_dir":      str(adapter_dir) if adapter_dir else "",
                        "compressed_text":  "",
                        "compressed_chars": 0,
                        "elapsed_s":        time.time() - t1,
                        "error":            str(e),
                    }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                n_done += 1
                if n_done % 5 == 0 or n_done <= 3:
                    eta = (time.time() - t0) / n_done * (len(cases) - n_done)
                    print(f"  [{variant_name}] {n_done}/{len(cases)} done eta={eta:.0f}s",
                          flush=True)
        print(f"[09a] {variant_name}: {n_done} new compressions in "
              f"{(time.time()-t0)/60:.1f} min")
        sc.unload()
    print("[09a] phase A done.")


# ----------------------------------------------------------------------
# Phase B — agent runs on student compressions  (acon/.venv)
# ----------------------------------------------------------------------

def _agent_run_one(args_tuple):
    (variant, case_id, eval_round, text, max_steps, tag) = args_tuple
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, "/workspace/EASMO/motivation_v4")
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v4.runner import run_with_compressed_context  # noqa

    res = run_with_compressed_context(
        task_id=case_id,
        method=f"v10_student_{variant}_{eval_round}",
        compressed_context=text,
        max_steps=max_steps,
        tag=tag,
    )
    return {
        "run_id":             f"{variant}__{case_id}__{eval_round}__cap{max_steps}",
        "variant":            variant,
        "case_id":            case_id,
        "eval_round":         eval_round,
        "max_steps":          max_steps,
        "success":            res.success,
        "score":              res.final_reward,
        "iterations":         res.iterations,
        "compressed_chars":   len(text),
        "termination_reason": res.termination_reason,
        "elapsed_s":          res.elapsed_s,
        "output_dir":         res.output_dir,
        "error":              res.error,
    }


def phase_b(args) -> None:
    """Run AppWorld agent on every (student compression × {C1, CK}).

    For Qwen-* variants:
      C1 = student's own one-shot compression (stage 09 phase A output)
      CK = MiniMax-stress-recompressed student output (K=2)

    For MiniMax-greedy and MiniMax-oracle-bestofN: reuse the rows in
    behavior_runs_candidates.jsonl with the appropriate candidate_id.
    """
    student_runs = _read_jsonl(args.compressions_out)
    behavior = _read_jsonl(args.candidate_behavior)
    proxy_summary_path = _REPO / "outputs" / "tables" / "proxy_by_case.csv"

    # Recompose work units --------------------------------------------
    work: List = []

    # MiniMax-greedy + MiniMax-oracle from existing stage 04 rows
    by_case_round = {(r["case_id"], r["eval_round"]): [] for r in behavior}
    for r in behavior:
        by_case_round.setdefault((r["case_id"], r["eval_round"]), []).append(r)
    if proxy_summary_path.exists():
        import csv
        proxy_rows = list(csv.DictReader(open(proxy_summary_path)))
    else:
        proxy_rows = []

    test_case_ids = {c["case_id"] for c in _read_jsonl(_REPO / "data" / "v10_cases.jsonl")
                     if c.get("split") in {"test_behavior", "legacy_v9"}}

    out_path = Path(args.behavior_out)
    done = set()
    if out_path.exists():
        for r in _read_jsonl(out_path):
            done.add(r.get("run_id"))

    # student variants → use phase-A compressions
    for r in student_runs:
        if r["case_id"] not in test_case_ids:
            continue
        if not r.get("compressed_text"):
            continue
        wid = f"{r['variant']}__{r['case_id']}__C1__cap{args.cap_steps}"
        if wid not in done:
            work.append((r["variant"], r["case_id"], "C1",
                         r["compressed_text"], args.cap_steps, args.tag))
        # CK for student variant requires stress on the student output —
        # implemented in a separate stress_students helper script
        # (deferred; placeholder is single-step CK = C1 for now).

    # MiniMax-greedy / oracle baselines reuse existing rows
    proxy_oracle_by_case = {}
    proxy_proxy_by_case = {}
    if proxy_rows:
        for r in proxy_rows:
            pass  # ignored — we just refer back to behavior table

    # For test_behavior cases, run MiniMax-greedy on C1+CK if any greedy/
    # oracle row exists in behavior_runs_candidates.jsonl — append "logical"
    # rows from the existing run.
    minimax_baselines = []
    for (case_id, eval_round), rows in by_case_round.items():
        if case_id not in test_case_ids:
            continue
        greedy_rows = [r for r in rows
                       if r.get("generation_type") == "greedy"]
        if greedy_rows:
            r = greedy_rows[0]
            minimax_baselines.append({
                "run_id":           f"MiniMax-greedy__{case_id}__{eval_round}__cap{args.cap_steps}",
                "variant":          "MiniMax-greedy",
                "case_id":          case_id,
                "eval_round":       eval_round,
                "max_steps":        args.cap_steps,
                "success":          r["success"],
                "score":            r["score"],
                "iterations":       r["iterations"],
                "compressed_chars": r["compressed_chars"],
                "termination_reason": r["termination_reason"],
                "elapsed_s":        r["elapsed_s"],
                "output_dir":       r["output_dir"],
                "error":            None,
                "reused_from":      r["candidate_id"],
            })
        # Oracle-best = max-score row in that group
        if rows:
            r = max(rows, key=lambda x: float(x.get("score") or 0.0))
            minimax_baselines.append({
                "run_id":           f"MiniMax-oracle__{case_id}__{eval_round}__cap{args.cap_steps}",
                "variant":          "MiniMax-oracle-bestofN",
                "case_id":          case_id,
                "eval_round":       eval_round,
                "max_steps":        args.cap_steps,
                "success":          r["success"],
                "score":            r["score"],
                "iterations":       r["iterations"],
                "compressed_chars": r["compressed_chars"],
                "termination_reason": r["termination_reason"],
                "elapsed_s":        r["elapsed_s"],
                "output_dir":       r["output_dir"],
                "error":            None,
                "reused_from":      r["candidate_id"],
            })

    # Write MiniMax baselines (no agent calls — pure data copy)
    with open(out_path, "a", encoding="utf-8") as f_out:
        for rec in minimax_baselines:
            if rec["run_id"] in done:
                continue
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[09b] MiniMax baselines (reused): {len(minimax_baselines)}; "
          f"new student-variant agent runs: {len(work)}")

    t0 = time.time(); n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_agent_run_one, w): w for w in work}
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"error": str(e)}
                if rec.get("error"):
                    n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                n_done += 1
                if n_done % 5 == 0 or n_done <= 3:
                    eta = (time.time() - t0) / n_done * (len(work) - n_done)
                    print(f"  [{n_done}/{len(work)}] err={n_err} eta={eta:.0f}s",
                          flush=True)
    print(f"[09b] phase B done: {n_done} new agent runs ({n_err} errors), "
          f"elapsed {(time.time()-t0)/60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("A", "B", "both"), default="A")
    ap.add_argument("--splits", default="test_behavior,legacy_v9")
    ap.add_argument("--include_raw", action="store_true", default=True)
    ap.add_argument("--include_sft_c1", action="store_true", default=True)
    ap.add_argument("--include_sft_ck", action="store_true", default=True)
    ap.add_argument("--base_model_id", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--compressions_out",
                    default=str(_REPO / "outputs" / "raw" / "student_compressions.jsonl"))
    ap.add_argument("--candidate_behavior",
                    default=str(_REPO / "outputs" / "raw" / "behavior_runs_candidates.jsonl"))
    ap.add_argument("--behavior_out",
                    default=str(_REPO / "outputs" / "raw" / "student_behavior_runs.jsonl"))
    ap.add_argument("--cap_steps", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="mv10_student")
    args = ap.parse_args()

    if args.phase in ("A", "both"):
        phase_a(args)
    if args.phase in ("B", "both"):
        phase_b(args)


if __name__ == "__main__":
    main()
