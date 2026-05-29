"""Stage 10 — GRPO readiness sampling (spec §16).

For each held-out test case and each Qwen variant in
{Raw-Qwen, Qwen-SFT-C1, Qwen-SFT-CK}:

  1. Generate one greedy compression (temp=0.0, seed=42).
  2. Generate N=8 stochastic compressions (temp=0.7, seeds=1000..1007).
  3. Apply stress T^K via MiniMax recompressor.
  4. Score each sample with the MiniMax verifier composite (proxy).
  5. (Optional) On a 25 % subset, also run true AppWorld pass.

Output:
  outputs/raw/grpo_readiness_compressions.jsonl
  outputs/raw/grpo_readiness_stress.jsonl
  outputs/raw/grpo_readiness_proxy.jsonl
  outputs/raw/grpo_readiness_behavior.jsonl   (subset only)
  outputs/tables/grpo_readiness_summary.csv
  outputs/tables/grpo_readiness_by_case.csv

Requires vLLM off for the Qwen compression part (same as stage 09 phase A).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd


def _read_jsonl(p):
    out = []
    if not Path(p).exists():
        return out
    for line in open(p):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def phase_compress(args) -> None:
    from motivation_v10.data import (
        ensure_outputs, read_jsonl, raw_path, model_path,
    )
    from motivation_v10.acon_prompt_loader import load_utco_bundle, render_prompt
    from motivation_v10.student_inference import StudentCompressor

    ensure_outputs()
    bundle = load_utco_bundle()
    cases = [c for c in read_jsonl(_REPO / "data" / "v10_cases.jsonl")
             if c.get("split") in {s.strip() for s in args.splits.split(",")}]
    print(f"[10] {len(cases)} cases (splits={args.splits})")

    variants = []
    if args.include_raw:
        variants.append(("Raw-Qwen", None))
    if args.include_sft_c1:
        variants.append(("Qwen-SFT-C1", model_path("qwen_sft_c1")))
    if args.include_sft_ck:
        variants.append(("Qwen-SFT-CK", model_path("qwen_sft_ck")))

    decodings = [("greedy", 0.0, 42)] + [
        (f"sample_{i:02d}", args.temperature_sample, 1000 + i)
        for i in range(args.n_samples)
    ]

    out_path = Path(args.compressions_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for r in _read_jsonl(out_path):
            done.add((r["variant"], r["case_id"], r["sample_id"]))

    for variant_name, adapter_dir in variants:
        if adapter_dir is not None and not Path(adapter_dir).exists():
            print(f"[10] skipping {variant_name}: adapter dir missing ({adapter_dir})")
            continue
        sc = StudentCompressor(
            adapter_dir=str(adapter_dir) if adapter_dir else None,
            base_model_id=args.base_model_id,
        )
        t0 = time.time(); n_done = 0
        target_n = len(cases) * len(decodings)
        with open(out_path, "a", encoding="utf-8") as f_out:
            for case in cases:
                user_text = render_prompt(
                    bundle,
                    task=case["user_instruction"],
                    history=case["full_trajectory_text"],
                    prev_summary="",
                    max_chars=1500,
                )
                for sample_id, temperature, seed in decodings:
                    key = (variant_name, case["case_id"], sample_id)
                    if key in done:
                        continue
                    t1 = time.time()
                    try:
                        gen = sc.generate(
                            system_text=bundle.system_text,
                            user_text=user_text,
                            temperature=temperature, seed=seed,
                            max_new_tokens=args.max_new_tokens,
                        )
                        rec = {
                            "variant":          variant_name,
                            "case_id":          case["case_id"],
                            "split":            case.get("split", "unknown"),
                            "sample_id":        sample_id,
                            "temperature":      temperature,
                            "seed":             seed,
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
                            "sample_id":        sample_id,
                            "compressed_text":  "",
                            "compressed_chars": 0,
                            "elapsed_s":        time.time() - t1,
                            "error":            str(e),
                        }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                    n_done += 1
                    if n_done % 10 == 0 or n_done <= 3:
                        eta = (time.time() - t0) / n_done * (target_n - n_done)
                        print(f"  [{variant_name}] {n_done}/{target_n} eta={eta:.0f}s",
                              flush=True)
        print(f"[10] {variant_name} compressions done: {n_done} new in {(time.time()-t0)/60:.1f} min")
        sc.unload()


def phase_stress(args) -> None:
    """Apply K=2 MiniMax recompression to every grpo_readiness compression."""
    from motivation_v10.data import ensure_outputs, read_jsonl, raw_path
    from motivation_v10.clients import make_client
    from motivation_v10.acon_prompt_loader import load_utco_bundle
    from motivation_v10.stress import stress_chain

    ensure_outputs()
    bundle = load_utco_bundle()
    client = make_client("minimax")

    cases = {c["case_id"]: c for c in read_jsonl(_REPO / "data" / "v10_cases.jsonl")}
    compressions = _read_jsonl(args.compressions_out)
    compressions = [c for c in compressions
                    if c.get("compressed_text") and not c.get("error")]

    out_path = Path(args.stress_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_cids = set()
    if out_path.exists():
        for r in _read_jsonl(out_path):
            done_cids.add((r["variant"], r["case_id"], r["sample_id"]))

    pending = [c for c in compressions
               if (c["variant"], c["case_id"], c["sample_id"]) not in done_cids]
    print(f"[10] stress phase: {len(pending)} pending out of {len(compressions)}")

    t0 = time.time(); n_done = 0; n_err = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _do(c):
        case = cases.get(c["case_id"])
        if not case:
            return []
        rows = stress_chain(
            candidate_id=f"{c['variant']}__{c['case_id']}__{c['sample_id']}",
            case_id=c["case_id"],
            client=client,
            model_name="minimax",
            bundle=bundle,
            user_instruction=case["user_instruction"],
            c0=c["compressed_text"],
            rounds=args.rounds,
            max_chars=1500,
        )
        return [{
            "variant":          c["variant"],
            "case_id":          c["case_id"],
            "sample_id":        c["sample_id"],
            "round":            r.round,
            "context_text":     r.context_text,
            "chars":            r.chars,
            "tokens_est":       r.tokens_est,
            "text_hash":        r.text_hash,
            "elapsed_s":        r.elapsed_s,
            "error":            r.error,
        } for r in rows]

    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_do, c): c for c in pending}
            for fut in as_completed(futs):
                rows = fut.result()
                for r in rows:
                    if r.get("error"): n_err += 1
                    f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                f_out.flush()
                n_done += 1
                if n_done % 30 == 0 or n_done <= 3:
                    eta = (time.time() - t0) / n_done * (len(pending) - n_done)
                    print(f"  stress [{n_done}/{len(pending)}] err={n_err} eta={eta:.0f}s",
                          flush=True)
    print(f"[10] stress done: {n_done} new chains ({n_err} errors)")


def phase_score(args) -> None:
    """Verifier composite on every (variant, case, sample, round)."""
    from motivation_v10.data import ensure_outputs, read_jsonl
    from motivation_v10.clients import make_client
    from motivation_v10.proxy import verifier_score

    ensure_outputs()
    client = make_client("minimax")
    cases = {c["case_id"]: c for c in read_jsonl(_REPO / "data" / "v10_cases.jsonl")}
    stress = _read_jsonl(args.stress_out)

    # Build (variant, case_id, sample_id) -> {round -> text}
    by_cs = defaultdict(dict)
    for r in stress:
        by_cs[(r["variant"], r["case_id"], r["sample_id"])][r["round"]] = r["context_text"]

    out_path = Path(args.proxy_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for r in _read_jsonl(out_path):
            done.add((r["variant"], r["case_id"], r["sample_id"], r["eval_round"]))

    work = []
    for (variant, case_id, sample_id), rounds_dict in by_cs.items():
        case = cases.get(case_id)
        if not case:
            continue
        c1_text = rounds_dict.get(0)
        ck_text = rounds_dict.get(max(rounds_dict.keys()))
        if c1_text and (variant, case_id, sample_id, "C1") not in done:
            work.append((variant, case_id, sample_id, "C1", c1_text, case["user_instruction"]))
        if ck_text and (variant, case_id, sample_id, "CK") not in done:
            work.append((variant, case_id, sample_id, "CK", ck_text, case["user_instruction"]))

    print(f"[10] verifier proxy: {len(work)} pending")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _do(w):
        variant, case_id, sample_id, eval_round, text, instr = w
        s = verifier_score(
            candidate_id=f"{variant}__{case_id}__{sample_id}",
            eval_round=eval_round,
            user_instruction=instr, compressed_text=text,
            client=client, max_tokens=2048,
        )
        return {
            "variant":       variant,
            "case_id":       case_id,
            "sample_id":     sample_id,
            "eval_round":    eval_round,
            "verifier_model": s.verifier_model,
            "composite":     s.composite(),
            "predicted_success_probability": s.predicted_success_probability,
            "missing_information_risk":      s.missing_information_risk,
            "execution_specificity":         s.execution_specificity,
            "risk_of_repeating_completed_actions": s.risk_of_repeating_completed_actions,
            "risk_of_wrong_api_arguments":   s.risk_of_wrong_api_arguments,
            "short_reason":  s.short_reason,
            "error":         s.error,
        }

    t0 = time.time(); n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_do, w): w for w in work}
            for fut in as_completed(futs):
                rec = fut.result()
                if rec.get("error"): n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                n_done += 1
                if n_done % 30 == 0 or n_done <= 3:
                    eta = (time.time() - t0) / n_done * (len(work) - n_done)
                    print(f"  score [{n_done}/{len(work)}] err={n_err} eta={eta:.0f}s",
                          flush=True)
    print(f"[10] verifier proxy done: {n_done} new ({n_err} errors)")


def phase_summarize(args) -> None:
    """Compute reward spread per (variant, case)."""
    from motivation_v10.data import ensure_outputs, table_path
    ensure_outputs()
    proxy = _read_jsonl(args.proxy_out)
    if not proxy:
        print("[10] no proxy rows yet — skip summary"); return

    df = pd.DataFrame(proxy)
    df = df[df["eval_round"] == "CK"]  # GRPO trains on the stress-evaluated reward
    rows = []
    for (variant, case_id), grp in df.groupby(["variant", "case_id"]):
        scores = grp["composite"].astype(float).tolist()
        greedy_rows = grp[grp["sample_id"] == "greedy"]
        greedy_score = float(greedy_rows["composite"].iloc[0]) if len(greedy_rows) else None
        sample_scores = grp[grp["sample_id"] != "greedy"]["composite"].astype(float).tolist()
        oracle_win = (
            any(s > (greedy_score or 0) for s in sample_scores)
            if greedy_score is not None and sample_scores else False
        )
        all_low = all(s < 0.3 for s in scores)
        all_high = all(s > 0.7 for s in scores)
        rows.append({
            "variant":              variant,
            "case_id":              case_id,
            "n_samples":            len(scores),
            "greedy_score":         greedy_score,
            "mean_score":           float(np.mean(scores)) if scores else 0.0,
            "std_score":            float(np.std(scores))  if scores else 0.0,
            "max_score":            float(np.max(scores))  if scores else 0.0,
            "min_score":            float(np.min(scores))  if scores else 0.0,
            "oracle_win_over_greedy": bool(oracle_win),
            "all_low":              bool(all_low),
            "all_high":             bool(all_high),
        })
    by_case = pd.DataFrame(rows)
    by_case.to_csv(table_path("grpo_readiness_by_case.csv"), index=False)

    summary = []
    for variant, grp in by_case.groupby("variant"):
        summary.append({
            "variant":                       variant,
            "n_cases":                       len(grp),
            "mean_within_case_std":          float(grp["std_score"].mean()),
            "oracle_win_rate_over_greedy":   float(grp["oracle_win_over_greedy"].mean()),
            "all_low_rate":                  float(grp["all_low"].mean()),
            "all_high_rate":                 float(grp["all_high"].mean()),
            "mean_best_of_n_gain":           float((grp["max_score"] - grp["greedy_score"]).mean()),
            "mean_greedy_score":             float(grp["greedy_score"].mean()),
        })
    pd.DataFrame(summary).to_csv(
        table_path("grpo_readiness_summary.csv"), index=False
    )
    print(f"[10] wrote grpo_readiness_summary.csv + grpo_readiness_by_case.csv")
    print(pd.DataFrame(summary).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("compress", "stress", "score",
                                         "summarize", "all"), default="all")
    ap.add_argument("--splits", default="test_behavior,legacy_v9")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature_sample", type=float, default=0.7)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--include_raw", action="store_true", default=True)
    ap.add_argument("--include_sft_c1", action="store_true", default=True)
    ap.add_argument("--include_sft_ck", action="store_true", default=True)
    ap.add_argument("--base_model_id", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--compressions_out",
                    default=str(_REPO / "outputs" / "raw" / "grpo_readiness_compressions.jsonl"))
    ap.add_argument("--stress_out",
                    default=str(_REPO / "outputs" / "raw" / "grpo_readiness_stress.jsonl"))
    ap.add_argument("--proxy_out",
                    default=str(_REPO / "outputs" / "raw" / "grpo_readiness_proxy.jsonl"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if args.phase in ("compress", "all"): phase_compress(args)
    if args.phase in ("stress", "all"):   phase_stress(args)
    if args.phase in ("score", "all"):    phase_score(args)
    if args.phase in ("summarize", "all"): phase_summarize(args)


if __name__ == "__main__":
    main()
