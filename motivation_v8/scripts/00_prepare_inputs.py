"""Stage 00 — prepare cases.jsonl by reusing v7's case pool (spec §16)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v8.data import (  # noqa: E402
    ensure_outputs, write_jsonl, load_v7_cases,
    v7_case_pool_path, PROVENANCE,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_REPO / "data" / "cases.jsonl"))
    ap.add_argument("--max_chars", type=int, default=18000)
    ap.add_argument("--max_cases", type=int, default=None)
    args = ap.parse_args()
    ensure_outputs()

    src = v7_case_pool_path()
    if src is None:
        raise SystemExit("v7 case pool missing; cannot prepare inputs")
    print(f"[00] reusing v7 case pool: {src}")
    cases = load_v7_cases()
    # truncate trajectory if needed
    for c in cases:
        if c.get("full_trajectory_text") and len(c["full_trajectory_text"]) > args.max_chars:
            c["full_trajectory_text"] = c["full_trajectory_text"][: args.max_chars]
    cases.sort(key=lambda c: c["case_id"])
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    n = write_jsonl(Path(args.out), cases)
    print(f"[00] wrote {n} cases (capped @ {args.max_chars} chars) -> {args.out}")

    # provenance
    prov = {
        "source": str(src),
        "n_cases": n,
        "max_chars": args.max_chars,
    }
    (PROVENANCE / "source_artifacts.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[00] provenance -> {PROVENANCE / 'source_artifacts.json'}")


if __name__ == "__main__":
    main()
