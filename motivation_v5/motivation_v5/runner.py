"""Downstream agent runner for Stage 05 (final_after_recompression_success).

Wraps v3's `run_with_compressed_context` with the v5 tag namespace so
trajectory directories don't clash with v3 / v4.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MV3 = Path("/workspace/EASMO/motivation_v3")
sys.path.insert(0, str(_MV3))

from motivation_v3.runner import run_with_compressed_context as _v3_run  # noqa: E402


def run_recompressed_downstream(
    task_id: str,
    *,
    recompressed_context: str,
    max_steps: int,
    split: str = "dev",
    tag: str = "mv5_recomp",
    method: str = "recompressed_context",
):
    """Run the downstream MiniMax agent with the recompressed context."""
    return _v3_run(
        task_id,
        method=method,
        compressed_context=recompressed_context,
        max_steps=max_steps,
        split=split,
        strategy="direct",
        seed=42,
        tag=tag,
    )
