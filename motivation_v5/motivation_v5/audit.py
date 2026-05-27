"""Audit prompt runners.

Loads the four spec prompts from `prompts/*.md` and dispatches them
to the right model (Qwen for case-level / addition / recompression
audits; MiniMax for verifier resolution; deterministic for rule-based
grounding check)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .clients import (
    ChatResult, chat_minimax, chat_qwen, pack_prompt_for_qwen, parse_json,
)


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


CASE_FAILURE_TEMPLATE = _load("01_case_failure_audit.md")
ADDITION_AUDIT_TEMPLATE = _load("02_audit_addition_audit.md")
RECOMPRESSION_AUDIT_TEMPLATE = _load("03_recompression_loss_audit.md")
VERIFIER_RESOLUTION_TEMPLATE = _load("04_verifier_resolution.md")
AGGREGATE_SUMMARY_TEMPLATE = _load("05_aggregate_summary.md")


# ----------------------------------------------------------------------
# Qwen-side audits (primary)
# ----------------------------------------------------------------------


def run_case_failure_audit(case: dict) -> ChatResult:
    fields = {
        "task_id": case["task_id"],
        "task_name": case.get("task_name", ""),
        "user_instruction": case.get("user_instruction", ""),
        "baseline_success": case.get("baseline_success", False),
        "acon_success": case.get("acon_success", False),
        "baseline_env_steps": case.get("baseline_env_steps", 0),
        "acon_env_steps": case.get("acon_env_steps", 0),
        "step_ratio": case.get("step_ratio", 0.0),
        "compression_type": case.get("compression_type", "history"),
        "acon_variant": case.get("acon_variant", "prompting"),
        "baseline_history": case.get("baseline_history", ""),
        "acon_compressed_history": case.get("acon_compressed_history", ""),
        "acon_full_trajectory": case.get("acon_full_trajectory", ""),
        "failure_report": case.get("failure_report", ""),
    }
    prompt = pack_prompt_for_qwen(
        CASE_FAILURE_TEMPLATE,
        fields=fields,
        reserve_output_tokens=2048,
    )
    return chat_qwen(prompt, temperature=0.0, max_tokens=2048, json_mode=True)


def run_addition_audit(case: dict) -> ChatResult:
    fields = {
        "task_id": case["task_id"],
        "user_instruction": case.get("user_instruction", ""),
        "baseline_history": case.get("baseline_history", ""),
        "acon_compressed_history": case.get("acon_compressed_history", ""),
        "audit_augmented_context": case.get("audit_augmented_context", ""),
    }
    prompt = pack_prompt_for_qwen(
        ADDITION_AUDIT_TEMPLATE,
        fields=fields,
        reserve_output_tokens=2048,
    )
    return chat_qwen(prompt, temperature=0.0, max_tokens=2048, json_mode=True)


def run_recompression_audit(case: dict) -> ChatResult:
    fields = {
        "task_id": case["task_id"],
        "user_instruction": case.get("user_instruction", ""),
        "baseline_history": case.get("baseline_history", ""),
        "acon_compressed_history": case.get("acon_compressed_history", ""),
        "audit_augmented_context": case.get("audit_augmented_context", ""),
        "recompressed_context": case.get("recompressed_context", ""),
    }
    prompt = pack_prompt_for_qwen(
        RECOMPRESSION_AUDIT_TEMPLATE,
        fields=fields,
        reserve_output_tokens=2048,
    )
    return chat_qwen(prompt, temperature=0.0, max_tokens=2048, json_mode=True)


# ----------------------------------------------------------------------
# MiniMax-side verifier (resolution)
# ----------------------------------------------------------------------


def run_verifier(case: dict, qwen_audit_json: dict) -> ChatResult:
    fields = {
        "task_id": case["task_id"],
        "user_instruction": case.get("user_instruction", ""),
        "baseline_history": case.get("baseline_history", ""),
        "acon_compressed_history": case.get("acon_compressed_history", ""),
        "audit_augmented_context": case.get("audit_augmented_context", "") or "",
        "recompressed_context": case.get("recompressed_context", "") or "",
        "qwen_audit_json": json.dumps(qwen_audit_json, ensure_ascii=False)[:6000],
    }
    # MiniMax has a much larger context window, no need for aggressive packing.
    # But still pack to ~25K input tokens to be safe with the vLLM endpoint.
    prompt = pack_prompt_for_qwen(  # reuse the packer (works for any model)
        VERIFIER_RESOLUTION_TEMPLATE,
        fields=fields,
        reserve_output_tokens=2048,
    )
    return chat_minimax(prompt, temperature=0.0, max_tokens=2048, json_mode=False)
