"""Qwen3-4B LoRA SFT trainer for v10 (spec §14, stage 08).

Uses trl.SFTTrainer with peft.LoraConfig. The trainer expects a JSONL
of SFT targets with `input_text` and `target_text` fields built by
stage 07. The input is the *raw history* (no compression yet); the
target is the teacher's compressed output. We frame the SFT example
as a standard chat completion:

    system = ACON UTCO system prompt (loaded verbatim from acon repo)
    user   = ACON UTCO history-compression user prompt (rendered with
             the raw history + task + max_chars=1500)
    assistant = teacher target compressed_text (no <think> blocks)

This means the student learns to *behave like the teacher* under the
exact same prompt template, so deployment is a drop-in replacement.

Important constraints:
* Single GPU (we have one 80 GB H100 / A100 class card).
* vLLM server on port 8000 MUST be stopped before training — peak
  memory with seq_len ~12K and batch_size 2 is ~30-50 GB, and vLLM
  greedily holds 0.85 of the card.
* We use bf16 + gradient_checkpointing to keep memory bounded.

Outputs:
    model_path/adapter_config.json
    model_path/adapter_model.safetensors
    model_path/tokenizer/                 (tokenizer files for serving)
    logs/train_loss.jsonl                 (one row per train step)
    logs/eval_loss.jsonl                  (held-out)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Keep this lazy — heavy imports happen only inside train().
def _import_training_stack():
    import torch
    import transformers
    import peft
    import trl
    import datasets as hf_datasets
    return torch, transformers, peft, trl, hf_datasets


@dataclass
class SFTConfig:
    base_model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    target_jsonl: str = ""           # required
    output_dir: str = ""             # required
    eval_jsonl: Optional[str] = None # optional held-out
    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "auto"     # use peft auto-pick for Qwen3
    # Training
    learning_rate: float = 1e-4
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 12000
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 5
    save_steps: int = 0              # 0 = save once at end
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42


def _build_chat_example(row: dict, system_text: str, user_template: str) -> list:
    """Render one SFT example as a list of chat messages.

    row must contain `input_text` (raw history) and `target_text`
    (teacher compressed output). user_template must be the ACON UTCO
    user template; it is rendered by Jinja in stage 07 already, so
    here we pass the rendered string as `input_text` directly. The
    `system_text` is the ACON system prompt.
    """
    return [
        {"role": "system",    "content": system_text},
        {"role": "user",      "content": row["input_text"]},
        {"role": "assistant", "content": row["target_text"]},
    ]


def train(cfg: SFTConfig, *, system_text: str, dry_run: bool = False) -> dict:
    """Run LoRA SFT and save adapters to cfg.output_dir."""
    torch, transformers, peft, trl, hf_datasets = _import_training_stack()

    if not cfg.target_jsonl or not Path(cfg.target_jsonl).exists():
        raise FileNotFoundError(f"target_jsonl missing: {cfg.target_jsonl}")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build chat-format datasets ---------------------------------------
    def _gen(path: str):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield {"messages": _build_chat_example(row, system_text, "")}

    train_ds = hf_datasets.Dataset.from_generator(_gen,
                                                  gen_kwargs={"path": cfg.target_jsonl})
    eval_ds = None
    if cfg.eval_jsonl and Path(cfg.eval_jsonl).exists():
        eval_ds = hf_datasets.Dataset.from_generator(_gen,
                                                     gen_kwargs={"path": cfg.eval_jsonl})

    print(f"[trainer] train rows: {len(train_ds)}; "
          f"eval rows: {len(eval_ds) if eval_ds is not None else 0}")

    if dry_run:
        return {"dry_run": True, "n_train": len(train_ds),
                "n_eval": len(eval_ds) if eval_ds else 0}

    # Tokenizer + model -----------------------------------------------
    tok = transformers.AutoTokenizer.from_pretrained(
        cfg.base_model_id, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # gradient_checkpointing + use_cache=True is incompatible
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    # PEFT LoRA --------------------------------------------------------
    target_modules = (
        cfg.target_modules
        if cfg.target_modules != "auto"
        else ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]
    )
    lora_cfg = peft.LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = peft.get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # TRL SFTTrainer ---------------------------------------------------
    sft_args = trl.SFTConfig(
        output_dir=str(out_dir),
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_length=cfg.max_seq_length,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_strategy="epoch" if cfg.save_steps == 0 else "steps",
        save_steps=cfg.save_steps if cfg.save_steps else None,
        save_total_limit=2,
        bf16=cfg.bf16,
        report_to="none",
        seed=cfg.seed,
        dataset_text_field=None,            # using messages format
    )

    trainer = trl.SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return {
        "ok": True,
        "n_train": len(train_ds),
        "n_eval": len(eval_ds) if eval_ds else 0,
        "output_dir": str(out_dir),
    }


__all__ = ["SFTConfig", "train"]
