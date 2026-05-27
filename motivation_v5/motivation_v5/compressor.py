"""ACON-style compressor for v5.

Reuses motivation_v3's `acon_style_summary` prompt verbatim. Used in
two places:

  1. Stage 04 — recompress the audit-augmented context to obtain
     `recompressed_context`.
  2. (Optional smoke / sanity) re-derive the ACON compressed context
     when v3 data is unavailable; in practice we just reuse v3's
     pre-built outputs.

The prompt is sourced from
`motivation_v3.prompts.COMPRESS_ACON_STYLE` so any change to v3's
ACON-style prompt automatically propagates here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_MV3 = Path("/workspace/EASMO/motivation_v3")
sys.path.insert(0, str(_MV3))

from motivation_v3.prompts import COMPRESS_ACON_STYLE  # noqa: E402

from .clients import chat_minimax, ChatResult


def recompress(
    task_instruction: str,
    text_to_compress: str,
    *,
    max_tokens: int = 2048,
) -> ChatResult:
    """Run the ACON-style compressor on the audit-augmented context.

    The compressor sees `text_to_compress` as the 'trajectory' input
    (the ACON prompt is written to accept any prose context). Output
    is the structured-section summary text.
    """
    prompt = COMPRESS_ACON_STYLE.format(
        task_instruction=task_instruction or "(no task instruction)",
        trajectory_text=text_to_compress or "(empty)",
    )
    return chat_minimax(
        prompt,
        system=("You are a careful compressor that follows instructions "
                "exactly. Respond ONLY in the requested output format."),
        temperature=0.2,
        max_tokens=max_tokens,
        json_mode=False,
    )
