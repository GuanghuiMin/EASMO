"""Shared HF model loader for v6 scripts.

Uses bf16, sdpa attention, and freezes all weights. ``device_map`` is
set to the default visible CUDA device.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device: str = "cuda"):
    print(f"[load_model] {model_path}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[load_model] done in {time.time() - t0:.1f}s; "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params",
          flush=True)
    return model, tok
