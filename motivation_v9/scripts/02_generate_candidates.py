"""Stage 02 — generate greedy + N stochastic ACON candidates per case."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v9.data import ensure_outputs, read_jsonl, raw_path  # noqa
from motivation_v9.clients import make_client  # noqa
from motivation_v9.acon_prompt_loader import load_utco_bundle  # noqa
from motivation_v9.compress import compress_once  # noqa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_REPO / "data" / "v9_cases.jsonl"))
    ap.add_argument("--out", default=str(raw_path("candidate_compressions.jsonl")))
    ap.add_argument("--models", default="minimax",
                    help="comma list — minimax, qwen, or both")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--target_max_chars", type=int, default=1500)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature_sample", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ensure_outputs()

    cases = read_jsonl(Path(args.cases))
    bundle = load_utco_bundle()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    clients = {m: make_client(m) for m in models}

    # Work units: (case, model, generation_type, sample_id, temperature, seed)
    work: List[Tuple] = []
    for case in cases:
        for model in models:
            work.append((case, model, "greedy", "greedy", 0.0, 42))
            for i in range(args.n_samples):
                work.append((case, model, "sample", f"sample_{i:02d}",
                              args.temperature_sample, 1000 + i))

    print(f"[02] {len(work)} candidate compressions "
          f"(cases={len(cases)}, models={models}, N={args.n_samples})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    def _run_one(item):
        case, model, gen_type, sample_id, temperature, seed = item
        res = compress_once(
            client=clients[model], model_name=model, bundle=bundle,
            user_instruction=case["user_instruction"],
            history=case["full_trajectory_text"],
            max_chars=args.target_max_chars,
            temperature=temperature, seed=seed, sample_id=sample_id,
            max_tokens=args.max_tokens,
        )
        return {
            "candidate_id":      f"{case['case_id']}__{model}__{sample_id}",
            "case_id":           case["case_id"],
            "compressor_model":  model,
            "prompt_variant":    bundle.variant,
            "generation_type":   gen_type,
            "sample_id":         sample_id,
            "temperature":       temperature,
            "seed":              seed,
            "input_context_hash": "",
            "compressed_text":   res.compressed_text,
            "compressed_chars":  res.output_chars,
            "compressed_tokens_est": max(1, res.output_chars // 4),
            "text_sha256":       res.text_sha256,
            "elapsed_s":         res.elapsed_s,
            "prompt_tokens":     res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "error":             res.error,
        }

    t0 = time.time()
    n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, w): w for w in work}
            for fut in as_completed(futs):
                rec = fut.result()
                if rec.get("error"):
                    n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                n_done += 1
                if n_done % 30 == 0 or n_done <= 3:
                    eta = (time.time() - t0) / n_done * (len(work) - n_done)
                    print(f"  [{n_done:>4d}/{len(work)}] err={n_err:<3d} eta={eta:.0f}s",
                          flush=True)
    print(f"[02] wrote {n_done} candidates ({n_err} errors) -> {out_path}")
    print(f"[02] total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
