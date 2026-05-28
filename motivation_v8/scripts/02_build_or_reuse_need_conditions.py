"""Stage 02 — reuse v7's quality-validated need/unneeded conditions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v8.data import (  # noqa: E402
    ensure_outputs, write_jsonl, load_v7_need_conditions,
    v7_need_conditions_path, PROVENANCE,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=str(_REPO / "data" / "need_conditions_validated.jsonl"))
    args = ap.parse_args()
    ensure_outputs()

    src = v7_need_conditions_path()
    if src is None:
        raise SystemExit("v7 need conditions missing; cannot reuse")
    conds = load_v7_need_conditions(only_passed_quality=True)
    n = write_jsonl(Path(args.out), conds)
    # also derive matched fact_ids
    fact_ids = sorted({c["fact_id"] for c in conds})
    print(f"[02] reused {n} quality-passed conditions across {len(fact_ids)} facts -> {args.out}")

    (PROVENANCE / "need_conditions_provenance.json").write_text(
        json.dumps({
            "source": str(src),
            "n_rows": n,
            "n_fact_pairs": len(fact_ids),
            "only_passed_quality": True,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
