"""Stage 01 — build raw_cases.jsonl from v3's existing data.

For each (task, budget) cell in v3 we materialise the spec §4.1
case record with baseline_history (rendered full successful
trajectory), acon_compressed_history (v3's acon_style_summary text),
acon_full_trajectory (rendered env_history from the ACON downstream
run), plus all metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def main():
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v5.data import (
        DATA, ensure_dirs, build_raw_cases, filter_tiers, write_jsonl,
    )

    ensure_dirs()
    cases = build_raw_cases(
        baseline_method="full_context",
        acon_method="acon_style_summary",
        budgets=(15, 8),
    )
    out = DATA / "raw_cases.jsonl"
    write_jsonl(out, cases)
    print(f"[01] wrote {len(cases)} raw cases -> {out}")
    print()

    tiers = filter_tiers(cases)
    print(f"  Tier 1 (baseline=T, acon=F):        {len(tiers['tier1']):>3d} cases")
    print(f"  Tier 2 (both=T, step_ratio>=1.5):   {len(tiers['tier2']):>3d} cases")
    print(f"  Sampled (T1+T2 dedup):              {len(tiers['sampled']):>3d} cases")
    print()
    # Stats
    from collections import Counter
    diff = Counter(c["difficulty"] for c in cases)
    print(f"  difficulty distribution: {dict(diff)}")
    by_cap = Counter(c["budget_max_steps"] for c in cases)
    print(f"  budget distribution: {dict(by_cap)}")


if __name__ == "__main__":
    main()
