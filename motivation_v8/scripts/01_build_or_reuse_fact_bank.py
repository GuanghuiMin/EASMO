"""Stage 01 — reuse v7's substring-grounded fact bank (spec §16)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v8.data import (  # noqa: E402
    ensure_outputs, write_jsonl, load_v7_facts,
    v7_fact_bank_path, PROVENANCE,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_REPO / "data" / "fact_bank_filtered.jsonl"))
    args = ap.parse_args()
    ensure_outputs()

    src = v7_fact_bank_path()
    if src is None:
        raise SystemExit("v7 fact bank missing; cannot reuse")
    facts = load_v7_facts()
    n = write_jsonl(Path(args.out), facts)
    print(f"[01] reused {n} facts from {src} -> {args.out}")

    summary = {
        "source": str(src), "n_facts": n,
        "by_coarse_group": {},
        "by_fact_type": {},
    }
    from collections import Counter
    summary["by_coarse_group"] = dict(Counter(f["coarse_group"] for f in facts))
    summary["by_fact_type"] = dict(Counter(f["fact_type"] for f in facts))
    print(f"[01] coarse_group distribution: {summary['by_coarse_group']}")

    (PROVENANCE / "fact_bank_provenance.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
