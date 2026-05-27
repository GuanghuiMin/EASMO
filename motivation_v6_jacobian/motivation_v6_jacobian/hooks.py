"""Hook helpers for capturing mid-layer hidden states and gradients.

Used by Experiment B (active subspace spectrum). We need both H_L (the
residual stream at layer L) and G_L = d L / d H_L during the same
backward pass. PyTorch only retains intermediate gradients if we call
``.retain_grad()`` on the captured tensor inside the forward hook.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch
from torch import nn


def find_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """Locate the ModuleList of transformer decoder blocks.

    Works for Qwen3 (``model.model.layers``) and most other HF causal
    LMs. Falls back to scanning ``named_modules()`` for a ModuleList
    whose children look like transformer blocks (have ``self_attn``).
    """
    candidates = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        candidates.append(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        candidates.append(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        candidates.append(model.gpt_neox.layers)
    for c in candidates:
        if isinstance(c, nn.ModuleList) and len(c) > 0:
            return c
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            first = mod[0]
            if any(n for n, _ in first.named_modules() if "attn" in n.lower()):
                return mod
    raise RuntimeError("Could not locate transformer blocks on this model.")


def resolve_layer_index(model: nn.Module, layer_index: int) -> int:
    """Normalise negative / -1 indices and assert in range."""
    blocks = find_transformer_blocks(model)
    n = len(blocks)
    if layer_index < 0:
        layer_index = n + layer_index
    if not (0 <= layer_index < n):
        raise ValueError(f"layer_index {layer_index} out of range (0..{n-1})")
    return layer_index


@contextmanager
def capture_layer_hidden_with_grad(
    model: nn.Module,
    layer_index: int,
) -> Iterator[dict]:
    """Context manager that registers a forward hook on a transformer
    block and yields a dict that will be populated with:

        cache['hidden'] : tensor with retain_grad enabled
                          (gradient available on cache['hidden'].grad
                           after backward())

    Caller is responsible for calling loss.backward() inside the
    context. After the block exits, the hook handle is removed.
    """
    blocks = find_transformer_blocks(model)
    n = len(blocks)
    if layer_index < 0:
        layer_index = n + layer_index
    block = blocks[layer_index]

    cache: dict = {"hidden": None}

    def _hook(_module: nn.Module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        hidden.retain_grad()
        cache["hidden"] = hidden
        return output

    handle = block.register_forward_hook(_hook)
    try:
        yield cache
    finally:
        handle.remove()


__all__ = [
    "find_transformer_blocks",
    "resolve_layer_index",
    "capture_layer_hidden_with_grad",
]
