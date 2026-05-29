"""Stage 12 — analyze chunk advantage by chunk type."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v9.data import ensure_outputs, read_jsonl, raw_path, table_path  # noqa
from motivation_v9.metrics import chunk_advantage_by_type  # noqa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--advantage",
                    default=str(table_path("chunk_information_advantage.csv")))
    ap.add_argument("--labels",
                    default=str(raw_path("chunk_type_labels.jsonl")))
    args = ap.parse_args()
    ensure_outputs()

    adv = (pd.read_csv(args.advantage)
           if Path(args.advantage).exists() else pd.DataFrame())
    labels = pd.DataFrame(read_jsonl(Path(args.labels)))
    out = chunk_advantage_by_type(adv, labels)
    out.to_csv(table_path("chunk_advantage_by_type.csv"), index=False)
    print(f"[12] wrote {len(out)} rows -> chunk_advantage_by_type.csv")
    if not out.empty:
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
