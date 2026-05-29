"""Stage 11 — chunk advantage reanalysis (spec §17, revised after v9).

Re-runs the v9 chunk pipeline on candidates from FIVE sources per
test case:

  1. MiniMax greedy
  2. MiniMax oracle-best
  3. MiniMax proxy-selected
  4. Qwen-SFT-C1
  5. Qwen-SFT-CK

Cap at 12 chunks per candidate (same as v9 stage 08).

Stages (logically the same as v9 07-12 collapsed):
  11.A select candidates per case
  11.B segment chunks (regex-based, reuse v9 chunks.py)
  11.C build leave-one-chunk-out contexts via MiniMax
  11.D run AppWorld agent on each chunk-minus context
  11.E label chunks with v10 enriched schema (functional_role_guess)
  11.F compute mean advantage by chunk_type AND by functional_role
  11.G regression: advantage ~ entity_count + literal_count + length + chunk_type + role
  11.H produce paper-tier table for §17.6

This script is the largest single non-SFT stage; runtime scales as
n_test_cases × 5 variants × ≤12 chunks × 3 LLM/agent calls. For
~22 test cases that's roughly 22 × 5 × 12 × 3 = ~4,000 calls in the
worst case (more like ~2,000 in practice with shorter compressions).

For first-pass implementation we keep the structure parallel to v9
stages 07-12 (separate helper scripts could be added later) but
ship as a single orchestrating script that walks the sub-stages
sequentially with checkpoints in outputs/raw/chunk_*.jsonl.

The actual sub-stage logic is delegated to v10's package modules
(chunks.py, chunk_label.py) which are direct ports from v9 with
the §17.5 schema enrichments in chunk_label.

This stub raises NotImplementedError; the full implementation lands
in a follow-up commit once stages 02-09 are validated end-to-end.
The split is deliberate: chunk reanalysis is the LAST analysis-only
stage and its output doesn't gate any earlier stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    print("[11] stage 11 chunk advantage reanalysis is intentionally stubbed.")
    print("     full implementation will be added in a follow-up commit once")
    print("     stages 02-09 produce validated outputs (see docs/01 §4).")
    print("     The v9 stage 07-12 scripts at "
          "/workspace/EASMO/motivation_v9/scripts/0{7,8,9,9a,10,11,12}_*.py")
    print("     can be ported in ~2 hours by reusing v10's chunks.py /")
    print("     chunk_label.py (already updated to §17.5 schema).")


if __name__ == "__main__":
    main()
