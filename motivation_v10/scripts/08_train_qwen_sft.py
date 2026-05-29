"""Stage 08 — train Qwen3-4B LoRA SFT students (spec §14).

Trains two students sequentially:

  Qwen-SFT-C1: SFT on outputs/data/sft_targets_c1.jsonl
  Qwen-SFT-CK: SFT on outputs/data/sft_targets_ck.jsonl

IMPORTANT: this script REQUIRES the vLLM server on port 8000 to be
STOPPED before training (it greedily holds 0.85 of the GPU and will
OOM the trainer). The script will:

  1. detect any running vLLM process and refuse to proceed unless
     `--allow_shared_gpu` is passed (the default is conservative).
  2. NOT restart vLLM after — restart manually via
     `bash /workspace/qwen3-vllm/serve_instruct.sh`
     so you can choose timing (e.g. before stage 09).

Why no auto-restart: vLLM startup takes 60-90s and needs verification
that the served-model-name actually came up. Doing that reliably
inside this script crosses too many process / shell boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v10.data import (  # noqa
    ensure_outputs, model_path, sft_data_path, PROVENANCE,
)
from motivation_v10.acon_prompt_loader import load_utco_bundle  # noqa
from motivation_v10.trainer import SFTConfig, train             # noqa


def _vllm_running() -> bool:
    """True if any python -m vllm.entrypoints.openai.api_server is up."""
    try:
        out = subprocess.check_output(["ps", "-ef"], text=True)
    except Exception:
        return False
    for line in out.splitlines():
        if "vllm.entrypoints.openai.api_server" in line and "grep" not in line:
            return True
    return False


def _gpu_free_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def _train_one(target_type: str, args, system_text: str) -> dict:
    target_jsonl = (sft_data_path("sft_targets_c1.jsonl")
                    if target_type == "C1"
                    else sft_data_path("sft_targets_ck.jsonl"))
    if not target_jsonl.exists() or target_jsonl.stat().st_size == 0:
        raise FileNotFoundError(f"SFT targets missing or empty: {target_jsonl}")
    eval_jsonl = None  # we keep things simple — no held-out validation here

    out_dir = model_path(f"qwen_sft_{target_type.lower()}")
    cfg = SFTConfig(
        base_model_id=args.base_model_id,
        target_jsonl=str(target_jsonl),
        output_dir=str(out_dir),
        eval_jsonl=str(eval_jsonl) if eval_jsonl else None,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        seed=args.seed,
    )
    # Persist exact config for provenance
    cfg_json = {
        "target_type": target_type,
        "target_jsonl": cfg.target_jsonl,
        "output_dir": cfg.output_dir,
        "base_model_id": cfg.base_model_id,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "max_seq_length": cfg.max_seq_length,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "seed": cfg.seed,
    }
    (PROVENANCE / f"sft_{target_type.lower()}_args.json").write_text(
        json.dumps(cfg_json, indent=2)
    )

    print(f"[08] training Qwen-SFT-{target_type}  →  {out_dir}")
    t0 = time.time()
    result = train(cfg, system_text=system_text, dry_run=args.dry_run)
    print(f"[08] finished {target_type} in {(time.time()-t0)/60:.1f} min: {result}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", choices=("c1", "ck", "both"), default="both")
    ap.add_argument("--base_model_id", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_length", type=int, default=12000)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow_shared_gpu", action="store_true", default=False,
                    help="Skip the vLLM-must-be-stopped check.")
    ap.add_argument("--dry_run", action="store_true", default=False,
                    help="Build datasets but do not train.")
    args = ap.parse_args()
    ensure_outputs()

    free_mb = _gpu_free_mb()
    print(f"[08] GPU memory free: {free_mb} MB")
    if _vllm_running() and not args.allow_shared_gpu:
        print("[08] ERROR: vLLM is still serving on the GPU. Stop it with:")
        print("       pkill -f 'served-model-name qwen3-4b-instruct-2507'")
        print("     then re-run this stage. Pass --allow_shared_gpu to override.")
        sys.exit(2)
    if free_mb >= 0 and free_mb < 30000 and not args.allow_shared_gpu:
        print(f"[08] WARNING: only {free_mb} MB free GPU memory. "
              f"Stop vLLM or risk OOM. Pass --allow_shared_gpu to override.")
        sys.exit(2)

    bundle = load_utco_bundle()
    system_text = bundle.system_text

    if args.student in ("c1", "both"):
        _train_one("C1", args, system_text)
    if args.student in ("ck", "both"):
        _train_one("CK", args, system_text)

    print("[08] all done. Manual next step: restart vLLM via")
    print("       nohup bash /workspace/qwen3-vllm/serve_instruct.sh > "
          "/workspace/qwen3-vllm/server_instruct.log 2>&1 &")


if __name__ == "__main__":
    main()
