"""Stage 02 — sample Tier 1 + Tier 2 cases for the audit pipeline.

Per spec §4.2 we prioritise:
  Tier 1: baseline_success=True AND acon_success=False
  Tier 2: baseline_success=True AND acon_success=True BUT step_ratio >= 1.5

Outputs:
  data/sampled_cases.jsonl   (the cases that will be audited)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_ratio_threshold", type=float, default=1.5)
    parser.add_argument("--include_all", action="store_true",
                        help="Also include cases not in Tier 1+2 (default: T1+T2 only)")
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v5.data import (
        DATA, read_jsonl, write_jsonl, filter_tiers,
    )

    raw_path = DATA / "raw_cases.jsonl"
    if not raw_path.exists():
        sys.exit(f"missing {raw_path}; run scripts/01_build_raw_cases.py first")
    cases = read_jsonl(raw_path)
    tiers = filter_tiers(cases, step_ratio_threshold=args.step_ratio_threshold)
    sampled = tiers["all"] if args.include_all else tiers["sampled"]

    out = DATA / "sampled_cases.jsonl"
    write_jsonl(out, sampled)
    print(f"[02] wrote {len(sampled)} sampled cases -> {out}")
    print(f"     (Tier 1: {len(tiers['tier1'])}, Tier 2: {len(tiers['tier2'])})")


if __name__ == "__main__":
    main()
