"""Audit-augmenter: builds `audit_augmented_context` per spec §1.

The audit model (Qwen3-4B) is shown:
  * task instruction
  * the full baseline trajectory (rendered)
  * the ACON compressed history

…and is asked to add back any actionable facts that were dropped by
ACON. The output is appended to the ACON compressed history under an
``[AUDIT_AUGMENTATION]`` block so the recompressor can see both the
original ACON summary AND the audit-added facts together.

Important design choice: the augmenter does NOT rewrite the ACON
summary. It only appends an extension. This way, downstream stages
can compare ACON-vs-augmented and augmented-vs-recompressed cleanly.
"""

from __future__ import annotations

from typing import Optional

from .clients import chat_qwen, ChatResult, pack_prompt_for_qwen


_AUGMENT_PROMPT = """\
You are an audit assistant for an AppWorld agent context compression.

You are shown:
1. the original user task,
2. the FULL successful baseline trajectory (no compression),
3. the ACON COMPRESSED summary that was injected back into the agent.

Your job is to identify concrete, actionable facts that exist in the
baseline trajectory but are missing or vague in the ACON compressed
summary, and add them back as short symbolic items.

Allowed item categories (use exactly these tags):
- RUNTIME_VARIABLE  (e.g. access_token=ey..., page_index=0, file_path=/notes/work.txt)
- AUTH_CREDENTIAL   (e.g. spotify_username=alice@x, password=*****)
- API_SCHEMA        (e.g. apis.spotify.show_album(album_id))
- ENVIRONMENT_STATE (e.g. liked_songs={1,2,3}, deleted_files=[...])
- ACTION_OUTCOME    (e.g. step 7 succeeded with token=...; step 9 failed: 401)
- PENDING_SUBTASK   (e.g. still need to like songs from artist_id=20)
- NEGATIVE_EVIDENCE (e.g. apis.X.Y returned empty; that path is a dead end)
- GUARDRAIL         (e.g. do not delete inbox messages older than today)
- OTHER

Output ONLY the augmentation block in this exact format (no prose, no JSON):

[AUDIT_AUGMENTATION]
- (CATEGORY) item_text  // brief reason it matters
- (CATEGORY) item_text  // brief reason it matters
...
[/AUDIT_AUGMENTATION]

Hard rules:
1. Every item must be backed by a verbatim string from the baseline trajectory.
2. Do not invent IDs, tokens, file paths, or numerical values.
3. Skip items already present in the ACON summary.
4. Keep items short (one line, <= 200 chars).
5. Order items by criticality (most critical first).
6. Output at most 12 items.
7. If the ACON summary already covers all critical state, output an empty block:
   [AUDIT_AUGMENTATION]
   [/AUDIT_AUGMENTATION]

---
TASK:
{{user_instruction}}

BASELINE_HISTORY_START
{{baseline_history}}
BASELINE_HISTORY_END

ACON_COMPRESSED_HISTORY_START
{{acon_compressed_history}}
ACON_COMPRESSED_HISTORY_END

Respond with the augmentation block only.
"""


def build_augmented_context(
    *,
    user_instruction: str,
    baseline_history: str,
    acon_compressed_history: str,
) -> ChatResult:
    """Returns a ChatResult whose `text` is the rendered augmented context:
    ACON summary + [AUDIT_AUGMENTATION] block."""
    prompt = pack_prompt_for_qwen(
        _AUGMENT_PROMPT,
        fields={
            "user_instruction": user_instruction or "(no instruction)",
            "baseline_history": baseline_history or "(no baseline)",
            "acon_compressed_history": acon_compressed_history or "(no ACON summary)",
        },
        reserve_output_tokens=1024,
    )
    return chat_qwen(
        prompt,
        system=("You are a careful audit assistant. Add back only "
                "grounded, actionable facts. Output the [AUDIT_AUGMENTATION] "
                "block exactly, no prose."),
        temperature=0.0,
        max_tokens=1024,
        json_mode=False,  # output is a bracketed text block, not JSON
    )


def merge_acon_and_augmentation(
    acon_compressed_history: str,
    augmentation_block: str,
) -> str:
    """Stitch the augmentation block onto the ACON summary."""
    s = (augmentation_block or "").strip()
    if "[AUDIT_AUGMENTATION]" not in s:
        # If the model failed to wrap, wrap it ourselves.
        s = f"[AUDIT_AUGMENTATION]\n{s}\n[/AUDIT_AUGMENTATION]"
    return f"{(acon_compressed_history or '').rstrip()}\n\n{s.strip()}\n"
