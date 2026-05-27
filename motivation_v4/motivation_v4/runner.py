"""Downstream-agent runner for v4 (mirrors motivation_v3/runner.py).

Same downstream agent prompt as v3 so v3 conditions and v4 conditions
can be merged in the same Table 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Reuse v3's runner directly — only change the tag namespace so cell
# directories don't clash.
_MV3 = Path("/workspace/EASMO/motivation_v3")
sys.path.insert(0, str(_MV3))

from motivation_v3.runner import run_with_compressed_context as _v3_run  # noqa: E402

# Re-export with default tag = mv4_run
def run_with_compressed_context(
    task_id: str,
    *,
    method: str,
    compressed_context: str,
    max_steps: int,
    split: str = "dev",
    strategy: str = "direct",
    model_name: str = "MiniMaxAI/MiniMax-M2.5",
    seed: int = 42,
    tag: str = "mv4_run",
):
    return _v3_run(
        task_id,
        method=method,
        compressed_context=compressed_context,
        max_steps=max_steps,
        split=split,
        strategy=strategy,
        model_name=model_name,
        seed=seed,
        tag=tag,
    )
