"""Stage 00 — prepare provenance & run config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v9.data import ensure_outputs, PROVENANCE  # noqa
from motivation_v9.acon_prompt_loader import (  # noqa
    load_utco_bundle, install_provenance,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cases", type=int, default=30)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--stress_rounds_k", type=int, default=2)
    ap.add_argument("--chunk_ablation_max_cases", type=int, default=12)
    ap.add_argument("--chunk_ablation_max_chunks", type=int, default=12)
    args = ap.parse_args()
    ensure_outputs()

    bundle = load_utco_bundle()
    record = install_provenance(
        bundle, prompts_dir=_REPO / "prompts",
        provenance_dir=PROVENANCE,
    )
    print(f"[00] ACON UTCO commit={record['acon_repo_commit']}")
    print(f"[00] history sha256={record['history_prompt_sha256']}")
    print(f"[00] system  sha256={record['system_prompt_sha256']}")

    cfg = {
        "N_CASES": args.n_cases,
        "N_SAMPLES": args.n_samples,
        "STRESS_ROUNDS_K": args.stress_rounds_k,
        "TEMPERATURE_GREEDY": 0.0,
        "TEMPERATURE_SAMPLE": 0.7,
        "BUDGET_MAX_STEPS_PRIMARY": 15,
        "BUDGET_MAX_STEPS_SECONDARY": 8,
        "MAX_TRAJECTORY_CHARS": 18000,
        "MAX_COMPRESS_TOKENS": 2048,
        "TARGET_MAX_CHARS": 1500,
        "CHUNK_ABLATION_MAX_CASES": args.chunk_ablation_max_cases,
        "CHUNK_ABLATION_MAX_CHUNKS_PER_CONTEXT": args.chunk_ablation_max_chunks,
        "LAMBDA_LEN": 0.02,
    }
    (PROVENANCE / "run_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[00] run config written, n_cases={args.n_cases}, n_samples={args.n_samples}, K={args.stress_rounds_k}")

    endpoints = {
        "qwen3_4b_instruct": "http://127.0.0.1:8000/v1",
        "minimax_m2_5":      "http://10.183.22.68:8005/v1",
        "downstream_agent_model": "MiniMaxAI/MiniMax-M2.5",
        "downstream_runner_venv": "/workspace/acon/.venv",
    }
    (PROVENANCE / "model_endpoints.json").write_text(
        json.dumps(endpoints, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
