"""Jacobian saliency over context tokens for the decision-state probe.

For one case, we tokenise

    chat_template(user = build_probe_prompt(instr, ctx)) + target_json

teacher-force the model to emit ``target_json`` and run a single
backward pass. From the embedding-layer gradient we read off
per-token saliencies (||grad||, |g·e|, etc.), aggregate to span
scores, and (optionally during the same pass) capture mid-layer
hidden+grad for the active-subspace spectrum experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


# Sentinel placeholder so prompt templates can be rendered before we
# know what the actual context text will be. We tokenise the prompt
# *with* the sentinel, then replace its token range with the real
# context-token range, which lets us compute exact "context token
# indices" without re-tokenising.
CTX_SENTINEL = "<<<CTX_REPLACE_ME>>>"


def _shift_token_ranges(
    spans: List[dict],
    ctx_token_start_in_full: int,
) -> List[dict]:
    """v4 spans carry character offsets within the rendered context.
    We map them to *token indices in the full input* by re-using the
    fast tokenizer's offset mapping on the context substring (computed
    separately via _align_spans_to_tokens). This helper only adjusts
    the in-context token ranges by the absolute prompt prefix length.
    """
    out = []
    for s in spans:
        d = dict(s)
        d["abs_token_start"] = ctx_token_start_in_full + d["ctx_token_start"]
        d["abs_token_end"] = ctx_token_start_in_full + d["ctx_token_end"]
        out.append(d)
    return out


def _align_spans_to_tokens(
    tokenizer,
    context_text: str,
    spans: List[dict],
) -> Tuple[List[dict], List[int]]:
    """Return (spans_with_ctx_token_ranges, context_token_ids).

    Strategy
    --------
    1. Concatenate spans with the same separator used by render_full_context
       and remember the cumulative character offset of each span's start
       and end in the concatenated string.
    2. Tokenise the concatenated string with offset_mapping=True.
    3. For each span, collect token indices whose offsets fall inside
       the span's char range.
    """
    sep = "\n\n"
    pieces: List[str] = []
    char_ranges: List[Tuple[int, int]] = []
    cursor = 0
    for i, s in enumerate(spans):
        text = s["span_text"]
        if i > 0:
            pieces.append(sep)
            cursor += len(sep)
        start = cursor
        pieces.append(text)
        cursor += len(text)
        char_ranges.append((start, cursor))
    full = "".join(pieces)
    assert full == context_text, "context_text must equal sep.join(span_text)"

    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    ids = enc["input_ids"]

    spans_out: List[dict] = []
    for s, (cs, ce) in zip(spans, char_ranges):
        # Token i belongs to span if its offset midpoint is inside
        # [cs, ce). Use overlap to be safe for tokens that straddle
        # the separator boundary.
        tok_start = None
        tok_end = None
        for ti, (a, b) in enumerate(offsets):
            if b <= cs:
                continue
            if a >= ce:
                break
            if tok_start is None:
                tok_start = ti
            tok_end = ti + 1
        if tok_start is None or tok_end is None:
            tok_start = 0
            tok_end = 0
        spans_out.append({
            **s,
            "ctx_token_start": tok_start,
            "ctx_token_end": tok_end,
            "ctx_token_count": tok_end - tok_start,
        })
    return spans_out, ids


@dataclass
class JacobianResult:
    task_id: str
    loss: float
    n_context_tokens: int
    n_target_tokens: int
    spans: List[dict]                       # one row per span
    per_token_grad_norm: np.ndarray         # shape (n_ctx,) — for debug / extra agg
    per_token_gxa_abs: np.ndarray           # shape (n_ctx,)
    # active-subspace tensors (only populated if --capture_active)
    active_hidden: Optional[np.ndarray] = None   # (n_ctx, hidden_dim)
    active_grad: Optional[np.ndarray] = None     # (n_ctx, hidden_dim)


def _build_input_ids(
    tokenizer,
    task_instruction: str,
    context_text: str,
    target_text: str,
    max_context_tokens: int,
    spans: List[dict],
) -> Tuple[torch.Tensor, List[dict], int, int]:
    """Build the full input_ids tensor and bookkeeping needed for the
    backward pass. Returns:

        input_ids                  : (1, T)
        spans                      : with abs_token_start/abs_token_end
        target_start_idx            : first token of target
        ctx_token_start_in_full     : first token of the context inside the
                                      full input (used by gradient extraction)

    If the rendered context exceeds ``max_context_tokens`` (counted in
    the white-box tokenizer), we drop spans from the FRONT (oldest)
    until it fits. The rendered ``context_text`` and the per-span
    char ranges are kept consistent throughout.
    """
    from .prompts import build_probe_prompt
    spans_used = list(spans)

    while True:
        ctx_text = "\n\n".join(s["span_text"] for s in spans_used)
        # tokenize just the context to count
        ctx_ids = tokenizer(ctx_text, add_special_tokens=False,
                            return_attention_mask=False)["input_ids"]
        if len(ctx_ids) <= max_context_tokens or len(spans_used) <= 1:
            break
        spans_used = spans_used[1:]

    aligned_spans, ctx_ids = _align_spans_to_tokens(
        tokenizer, ctx_text, spans_used
    )

    # Build prompt with sentinel context, then chat-template, then
    # replace the sentinel-token range with the actual context tokens.
    prompt_with_sentinel = build_probe_prompt(task_instruction, CTX_SENTINEL)
    messages = [{"role": "user", "content": prompt_with_sentinel}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # find sentinel in the chat_text by character search and locate
    # the *first* token whose offset range overlaps the sentinel
    sentinel_char_start = chat_text.find(CTX_SENTINEL)
    sentinel_char_end = sentinel_char_start + len(CTX_SENTINEL)
    if sentinel_char_start < 0:
        raise RuntimeError("CTX_SENTINEL missing after chat templating")

    chat_enc = tokenizer(chat_text, add_special_tokens=False,
                         return_offsets_mapping=True,
                         return_attention_mask=False)
    chat_ids: List[int] = chat_enc["input_ids"]
    chat_offsets = chat_enc["offset_mapping"]
    sent_tok_start = None
    sent_tok_end = None
    for ti, (a, b) in enumerate(chat_offsets):
        if b <= sentinel_char_start:
            continue
        if a >= sentinel_char_end:
            break
        if sent_tok_start is None:
            sent_tok_start = ti
        sent_tok_end = ti + 1
    if sent_tok_start is None:
        raise RuntimeError("Could not locate sentinel tokens")

    prompt_prefix_ids = chat_ids[:sent_tok_start]
    prompt_suffix_ids = chat_ids[sent_tok_end:]

    target_ids = tokenizer(target_text, add_special_tokens=False,
                           return_attention_mask=False)["input_ids"]
    # Force EOS at the end so the loss sees a natural stop signal too.
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]

    full_ids = prompt_prefix_ids + ctx_ids + prompt_suffix_ids + target_ids

    ctx_token_start_in_full = len(prompt_prefix_ids)
    target_start_idx = ctx_token_start_in_full + len(ctx_ids) + len(prompt_suffix_ids)

    # Adjust span ranges to absolute positions inside full_ids.
    for s in aligned_spans:
        s["abs_token_start"] = ctx_token_start_in_full + s["ctx_token_start"]
        s["abs_token_end"]   = ctx_token_start_in_full + s["ctx_token_end"]

    input_ids = torch.tensor([full_ids], dtype=torch.long)
    return input_ids, aligned_spans, target_start_idx, ctx_token_start_in_full


def compute_jacobian_saliency(
    *,
    model: nn.Module,
    tokenizer,
    task_id: str,
    task_instruction: str,
    spans: List[dict],
    target_text: str,
    max_context_tokens: int = 12000,
    capture_active: bool = False,
    active_layer_index: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> JacobianResult:
    """Run one backward pass and return span-level Jacobian scores.

    Spans must be sorted chronologically. The returned spans are
    pruned to whatever survived the max_context_tokens truncation.
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids, used_spans, target_start_idx, ctx_token_start = _build_input_ids(
        tokenizer, task_instruction, "\n\n".join(s["span_text"] for s in spans),
        target_text, max_context_tokens, spans,
    )
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    # We optimise memory: only the embeddings need grad; weights are frozen.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach().clone()
    inputs_embeds.requires_grad_(True)

    labels = input_ids.clone()
    labels[:, :target_start_idx] = -100

    # optional capture-hook
    if capture_active:
        from .hooks import capture_layer_hidden_with_grad, resolve_layer_index
        L = resolve_layer_index(model, active_layer_index
                                if active_layer_index is not None else -1)
        ctx = capture_layer_hidden_with_grad(model, L)
    else:
        class _Null:
            def __enter__(self): return {"hidden": None}
            def __exit__(self, *a): return False
        ctx = _Null()

    with ctx as cache:
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = out.loss
        if loss.dim() > 0:
            loss = loss.mean()
        loss.backward()

    # Gather grads — restrict to context tokens to save memory.
    ctx_lo = ctx_token_start
    ctx_hi = ctx_token_start + sum(1 for _ in range(  # number of ctx tokens
        max(0, target_start_idx - ctx_token_start)
    ))
    # The number of ctx tokens is exactly the count of indices strictly
    # before the prompt_suffix; we can compute it from used_spans.
    if used_spans:
        ctx_hi = max(s["abs_token_end"] for s in used_spans)
    else:
        ctx_hi = ctx_lo

    g_all = inputs_embeds.grad.detach()
    e_all = inputs_embeds.detach()
    g = g_all[0, ctx_lo:ctx_hi, :].float()
    e = e_all[0, ctx_lo:ctx_hi, :].float()

    grad_norm = torch.linalg.norm(g, dim=-1)              # (n_ctx,)
    gxa_abs = torch.sum(torch.abs(g * e), dim=-1)         # (n_ctx,)
    # Convenience scalars also recorded below.

    grad_norm_np = grad_norm.cpu().numpy()
    gxa_abs_np = gxa_abs.cpu().numpy()
    g_dot_x_abs_np = torch.abs(torch.sum(g * e, dim=-1)).cpu().numpy()

    span_records: List[dict] = []
    for s in used_spans:
        lo = s["abs_token_start"] - ctx_lo
        hi = s["abs_token_end"] - ctx_lo
        if hi <= lo:
            span_records.append({
                "task_id": task_id,
                "span_id": s["span_id"],
                "step_id": s["step_id"],
                "token_count": 0,
                "span_grad_sum": 0.0,
                "span_grad_mean": 0.0,
                "span_gxa_sum": 0.0,
                "span_gxa_mean": 0.0,
                "span_gxa_sqrtlen": 0.0,
                "span_top10_mean": 0.0,
                "span_g_dot_x_abs_sum": 0.0,
            })
            continue
        gn = grad_norm_np[lo:hi]
        gx = gxa_abs_np[lo:hi]
        gd = g_dot_x_abs_np[lo:hi]
        n = hi - lo
        top_k = min(10, n)
        top10 = float(np.sort(gx)[-top_k:].mean()) if n > 0 else 0.0
        span_records.append({
            "task_id": task_id,
            "span_id": s["span_id"],
            "step_id": s["step_id"],
            "token_count": int(n),
            "span_grad_sum": float(gn.sum()),
            "span_grad_mean": float(gn.mean()),
            "span_gxa_sum": float(gx.sum()),
            "span_gxa_mean": float(gx.mean()),
            "span_gxa_sqrtlen": float(gx.sum() / math.sqrt(max(n, 1))),
            "span_top10_mean": top10,
            "span_g_dot_x_abs_sum": float(gd.sum()),
        })

    active_hidden = None
    active_grad = None
    if capture_active and cache["hidden"] is not None:
        h = cache["hidden"].detach()[0, ctx_lo:ctx_hi, :].float().cpu().numpy()
        gh = cache["hidden"].grad.detach()[0, ctx_lo:ctx_hi, :].float().cpu().numpy()
        active_hidden = h
        active_grad = gh

    # Clean up the grad / cached buffer to avoid OOM across cases.
    inputs_embeds.grad = None
    if capture_active and cache["hidden"] is not None:
        try:
            cache["hidden"].grad = None
        except Exception:
            pass
        cache["hidden"] = None
    del inputs_embeds, e_all, g_all, g, e, grad_norm, gxa_abs
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return JacobianResult(
        task_id=task_id,
        loss=float(loss.detach().cpu()),
        n_context_tokens=int(ctx_hi - ctx_lo),
        n_target_tokens=int(input_ids.shape[1] - target_start_idx),
        spans=span_records,
        per_token_grad_norm=grad_norm_np,
        per_token_gxa_abs=gxa_abs_np,
        active_hidden=active_hidden,
        active_grad=active_grad,
    )


__all__ = [
    "JacobianResult",
    "compute_jacobian_saliency",
    "CTX_SENTINEL",
]
