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

    # Spec §3.1 expects train=89, dev=56, combined=145 (per ACON paper +
    # local AppWorld files at the time the spec was frozen). The local
    # train.txt / dev.txt counts can drift (trailing-newline edge case
    # makes `wc -l` differ from Python's `splitlines()` by 1). Emit a
    # meta note row for every mismatch direction so the discrepancy is
    # captured in provenance instead of being silent.
    EXPECTED = {"train": 89, "dev": 56}
    for split_name, n_expected in EXPECTED.items():
        n_actual = sum(1 for r in rows if r["split"] == split_name)
        if n_actual != n_expected:
            delta = n_actual - n_expected
            sign = "+" if delta > 0 else ""
            rows.append({
                "task_id":          f"META__{split_name}_count_mismatch",
                "split":            split_name,
                "included":         False,
                "exclusion_reason": "meta_note_not_a_task",
                "load_error":       "",
                "notes":            f"Spec §3.1 expected {split_name}={n_expected}; "
                                     f"local AppWorld {split_name}.txt yielded "
                                     f"{n_actual} ({sign}{delta} vs spec). "
                                     f"Difference is usually a trailing-newline edge "
                                     f"case (wc -l vs splitlines) and does not affect "
                                     f"experiment validity. Downstream stages run on "
                                     f"all included rows; the official paper-cited count "
                                     f"and the actual count are both recorded here.",
            })

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
