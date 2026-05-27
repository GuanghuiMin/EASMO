"""Soft-token oracle (Experiment C).

For each case and each k ∈ {4,8,16,32,64} we

  1. Build a prompt-without-context: chat-template(user = build_probe_prompt(instr, "")).
     (Empty context — the soft tokens are replacing the context.)
  2. Insert k trainable embeddings between the prompt-suffix's last
     token and the target tokens, then run AdamW for ``num_steps``
     steps minimising the teacher-forced cross-entropy on the
     target.
  3. Record the final loss alongside the baselines.

The model weights are frozen; only the soft embeddings carry grads.
We keep an fp32 master copy of soft embeddings because bf16
optimisation is fragile (spec §7.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


def _empty_context_input_ids(
    tokenizer,
    task_instruction: str,
    target_text: str,
) -> Tuple[torch.Tensor, int, int]:
    """Returns (full_ids, soft_insert_idx, target_start_idx) for the
    empty-context probe + target arrangement.

    soft_insert_idx is the position at which soft tokens will be
    inserted (right after the prompt suffix, right before the target).
    target_start_idx is the position of the first target token *after*
    inserting k soft tokens; the caller adds k.
    """
    from .prompts import build_probe_prompt
    prompt_text = build_probe_prompt(task_instruction, "")
    messages = [{"role": "user", "content": prompt_text}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chat_ids = tokenizer(chat_text, add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False,
                           return_attention_mask=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]
    soft_insert_idx = len(chat_ids)
    full_ids = chat_ids + target_ids  # caller will splice soft embeds in
    return (
        torch.tensor([full_ids], dtype=torch.long),
        soft_insert_idx,
        soft_insert_idx,  # target starts here before soft-token splice
    )


@dataclass
class SoftTokenRun:
    task_id: str
    k: int
    final_loss: float
    n_steps: int
    converged_at: int
    history: List[float]


def _baseline_loss_with_context(
    *,
    model: nn.Module,
    tokenizer,
    task_instruction: str,
    context_text: str,
    target_text: str,
    max_context_tokens: int,
    device: torch.device,
) -> float:
    """Teacher-forced CE on the target, prompt = chat-templated probe
    with ``context_text`` substituted in. Returns mean CE (nats)."""
    from .prompts import build_probe_prompt
    # truncate context to roughly max_context_tokens (greedy from the back)
    if context_text:
        ctx_ids = tokenizer(context_text, add_special_tokens=False,
                            return_attention_mask=False)["input_ids"]
        if len(ctx_ids) > max_context_tokens:
            # keep the tail; oldest tokens dropped
            keep = ctx_ids[-max_context_tokens:]
            context_text = tokenizer.decode(keep, skip_special_tokens=False)
    prompt_text = build_probe_prompt(task_instruction, context_text)
    messages = [{"role": "user", "content": prompt_text}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chat_ids = tokenizer(chat_text, add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False,
                           return_attention_mask=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]
    full_ids = chat_ids + target_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, :len(chat_ids)] = -100
    with torch.no_grad():
        out = model(input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    labels=labels, use_cache=False)
        loss = out.loss
        if loss.dim() > 0:
            loss = loss.mean()
    return float(loss.detach().cpu())


def baseline_losses(
    *,
    model: nn.Module,
    tokenizer,
    task_instruction: str,
    target_text: str,
    full_context: str,
    recent_context: str,
    acon_context: Optional[str],
    max_context_tokens: int,
) -> Dict[str, float]:
    """Returns dict of baseline losses keyed by 'full', 'no', 'recent', 'acon'."""
    device = next(model.parameters()).device
    out: Dict[str, float] = {}
    out["full"] = _baseline_loss_with_context(
        model=model, tokenizer=tokenizer,
        task_instruction=task_instruction, context_text=full_context,
        target_text=target_text, max_context_tokens=max_context_tokens,
        device=device,
    )
    out["no"] = _baseline_loss_with_context(
        model=model, tokenizer=tokenizer,
        task_instruction=task_instruction, context_text="",
        target_text=target_text, max_context_tokens=max_context_tokens,
        device=device,
    )
    out["recent"] = _baseline_loss_with_context(
        model=model, tokenizer=tokenizer,
        task_instruction=task_instruction, context_text=recent_context,
        target_text=target_text, max_context_tokens=max_context_tokens,
        device=device,
    )
    if acon_context:
        out["acon"] = _baseline_loss_with_context(
            model=model, tokenizer=tokenizer,
            task_instruction=task_instruction, context_text=acon_context,
            target_text=target_text, max_context_tokens=max_context_tokens,
            device=device,
        )
    else:
        out["acon"] = float("nan")
    return out


def train_soft_tokens(
    *,
    model: nn.Module,
    tokenizer,
    task_id: str,
    task_instruction: str,
    target_text: str,
    k: int,
    num_steps: int = 200,
    lr: float = 0.05,
    patience: int = 30,
    min_delta: float = 1e-4,
    init_scale: float = 0.02,
) -> SoftTokenRun:
    """Optimise k soft tokens against the target NLL. Returns final
    loss and the per-step loss history."""
    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()
    hidden_dim = embed_layer.embedding_dim

    full_ids, soft_insert_idx, _ = _empty_context_input_ids(
        tokenizer, task_instruction, target_text
    )
    full_ids = full_ids.to(device)
    prefix_ids = full_ids[:, :soft_insert_idx]
    target_ids = full_ids[:, soft_insert_idx:]

    with torch.no_grad():
        prefix_embeds = embed_layer(prefix_ids).detach()
        target_embeds = embed_layer(target_ids).detach()
        mean_e = embed_layer.weight.detach().float().mean(dim=0)

    # fp32 master copy of soft embeddings (we will cast each step).
    soft_master = torch.nn.Parameter(
        (mean_e.unsqueeze(0).expand(k, hidden_dim).clone() +
         torch.randn(k, hidden_dim, device=device) * init_scale).to(torch.float32),
        requires_grad=True,
    )

    optimizer = torch.optim.AdamW([soft_master], lr=lr, weight_decay=0.0)
    history: List[float] = []
    best = float("inf")
    best_step = 0

    # labels ignore prefix and soft tokens
    n_prefix = prefix_ids.shape[1]
    n_target = target_ids.shape[1]
    labels = torch.full(
        (1, n_prefix + k + n_target), -100, dtype=torch.long, device=device,
    )
    labels[:, n_prefix + k:] = target_ids[0]
    attention_mask = torch.ones_like(labels)

    for step in range(num_steps):
        soft = soft_master.to(prefix_embeds.dtype)
        inputs_embeds = torch.cat([
            prefix_embeds, soft.unsqueeze(0), target_embeds
        ], dim=1)
        out = model(inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels, use_cache=False)
        loss = out.loss
        if loss.dim() > 0:
            loss = loss.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        val = float(loss.detach().cpu())
        history.append(val)
        if val + min_delta < best:
            best = val
            best_step = step
        if step - best_step >= patience:
            break

    torch.cuda.empty_cache()
    return SoftTokenRun(
        task_id=task_id,
        k=k,
        final_loss=float(best),
        n_steps=len(history),
        converged_at=best_step,
        history=history,
    )


def gap_recovery(L_no: float, L_full: float, L_method: float) -> float:
    """(L_no - L_method) / (L_no - L_full); guards against div by 0
    and returns 0 if the no↔full gap is negative (full > no — model
    actually does worse with context)."""
    denom = L_no - L_full
    if denom <= 0:
        return float("nan")
    return (L_no - L_method) / denom


__all__ = [
    "SoftTokenRun",
    "train_soft_tokens",
    "baseline_losses",
    "gap_recovery",
]
