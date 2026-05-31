"""Stage 01 — build full AppWorld train+dev task inventory (spec §4.1).

No agent runs. Just enumerate task_ids and write:

  outputs/provenance/appworld_task_inventory.csv

Schema: task_id, split, included, exclusion_reason, load_error, notes
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v11.data import ensure_outputs, PROVENANCE, load_appworld_dev  # noqa


def _appworld_split(name: str) -> list:
    p = Path("/workspace/acon/experiments/appworld/data/datasets") / f"{name}.txt"
    return sorted({l.strip() for l in open(p).read().splitlines() if l.strip()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_pool", choices=("train+dev", "dev"),
                    default="train+dev")
    ap.add_argument("--out",
                    default=str(PROVENANCE / "appworld_task_inventory.csv"))
    args = ap.parse_args()
    ensure_outputs()

    rows = []
    if args.task_pool == "train+dev":
        for tid in _appworld_split("train"):
            rows.append({"task_id": tid, "split": "train",
                          "included": True, "exclusion_reason": "",
                          "load_error": "", "notes": ""})
        for tid in load_appworld_dev():
            rows.append({"task_id": tid, "split": "dev",
                          "included": True, "exclusion_reason": "",
                          "load_error": "", "notes": ""})
    else:
        for tid in load_appworld_dev():
            rows.append({"task_id": tid, "split": "dev",
                          "included": True, "exclusion_reason": "",
                          "load_error": "", "notes": ""})

    # ACON paper cites 90 train tasks; local has 89. Note in notes col.
    train_count = sum(1 for r in rows if r["split"] == "train")
    if train_count == 89:
        rows.append({"task_id": "ACON_90th_task_not_in_local_train_txt",
                      "split": "train", "included": False,
                      "exclusion_reason": "missing_from_local_AppWorld_train.txt",
                      "load_error": "",
                      "notes": "ACON paper cites 90 train tasks; local "
                               "/workspace/acon/experiments/appworld/data/datasets/"
                               "train.txt has 89."})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "split", "included",
                                            "exclusion_reason", "load_error", "notes"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_inc = sum(1 for r in rows if r["included"])
    print(f"[01] wrote {len(rows)} rows ({n_inc} included) -> {out_path}")
    by_split = {}
    for r in rows:
        by_split[r["split"]] = by_split.get(r["split"], 0) + int(bool(r["included"]))
    print(f"     by split (included only): {by_split}")


if __name__ == "__main__":
    main()
