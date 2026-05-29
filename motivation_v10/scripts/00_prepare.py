"""Stage 00 — record provenance + ensure output directories.

Writes:
  outputs/provenance/acon_commit.txt
  outputs/provenance/acon_utco_prompt_sha256.json
  outputs/provenance/pip_freeze_easmo_venv.txt
  outputs/provenance/v10_config.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v10.data import ensure_outputs, PROVENANCE          # noqa
from motivation_v10.acon_prompt_loader import (                     # noqa
    load_utco_bundle, install_provenance,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--stress_rounds_k", type=int, default=2)
    ap.add_argument("--cap_steps", type=int, default=15)
    args = ap.parse_args()

    ensure_outputs()

    # ACON commit + prompt sha
    bundle = load_utco_bundle()
    rec = install_provenance(bundle, _REPO / "prompts", PROVENANCE)
    (PROVENANCE / "acon_commit.txt").write_text(bundle.commit_hash + "\n")
    (PROVENANCE / "acon_utco_prompt_sha256.json").write_text(
        json.dumps(rec, indent=2)
    )

    # pip freeze of EASMO venv (training stack identity)
    try:
        freeze = subprocess.check_output(
            ["/workspace/EASMO/.venv/bin/pip", "freeze"], text=True
        )
        (PROVENANCE / "pip_freeze_easmo_venv.txt").write_text(freeze)
    except Exception as e:
        (PROVENANCE / "pip_freeze_easmo_venv.txt").write_text(f"ERROR: {e}\n")

    # v10 config (frozen hyperparameters)
    cfg = {
        "n_samples": args.n_samples,
        "stress_rounds_k": args.stress_rounds_k,
        "cap_steps": args.cap_steps,
        "temperature_greedy": 0.0,
        "temperature_sample": 0.7,
        "seed_greedy": 42,
        "seed_sample_base": 1000,
        "max_tokens_compression": 2048,
        "max_tokens_verifier": 2048,
        "max_tokens_pairwise": 1536,
        "max_tokens_chunk_label": 2048,
        "max_chars_compression": 1500,
        "lambda_length": 0.05,
        "acon_commit": bundle.commit_hash,
        "acon_history_prompt_sha256": bundle.sha256,
    }
    (PROVENANCE / "v10_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False)
    )
    print(f"[00] provenance written to {PROVENANCE}")
    print(f"     ACON commit: {bundle.commit_hash}")
    print(f"     history prompt sha256: {bundle.sha256}")


if __name__ == "__main__":
    main()
