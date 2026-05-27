"""Stage 2-bis — re-parse the existing symbolic_evidence raw responses
with the improved regex-fallback parser.

Stage 02 was already run; the raw_response strings are saved in
motivation_compressed_contexts.jsonl. This script re-runs the parser
on those strings, regenerates the rendered [SYMBOLIC_CONTEXT] block
and per-unit JSONL, and rewrites both files in place. No new LLM
calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def main():
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v3.compressors import _parse_symbolic_units, render_symbolic_block, _count_tokens
    from motivation_v3.data import jsonl_path, read_jsonl

    in_path = jsonl_path("motivation_compressed_contexts.jsonl")
    out_path = in_path
    units_path = jsonl_path("motivation_symbolic_units.jsonl")

    rows = read_jsonl(in_path)
    print(f"[02b] loaded {len(rows)} rows from {in_path}")

    n_old_ok = sum(1 for r in rows if r.get("method") == "symbolic_evidence" and r.get("n_units", 0) > 0)
    print(f"[02b] before reparse: {n_old_ok}/{sum(1 for r in rows if r.get('method')=='symbolic_evidence')} symbolic tasks had >=1 unit")

    new_rows = []
    n_new_ok = 0
    for r in rows:
        if r.get("method") != "symbolic_evidence" or r.get("error"):
            new_rows.append(r)
            continue
        raw = r.get("raw_response", "") or ""
        units = _parse_symbolic_units(raw)
        if units:
            text = render_symbolic_block(units)
            r = dict(r)
            r["units"] = units
            r["text"] = text
            r["n_tokens"] = _count_tokens(text)
            r["n_units"] = len(units)
            n_new_ok += 1
        else:
            r = dict(r)
            r["units"] = []
            r["text"] = ""
            r["n_tokens"] = 0
            r["n_units"] = 0
        new_rows.append(r)

    print(f"[02b] after reparse: {n_new_ok}/{sum(1 for r in rows if r.get('method')=='symbolic_evidence')} symbolic tasks have >=1 unit")

    with open(out_path, "w") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[02b] wrote {len(new_rows)} rows -> {out_path}")

    # Rebuild symbolic_units.jsonl
    n_units_total = 0
    with open(units_path, "w") as f:
        for r in new_rows:
            if r.get("method") != "symbolic_evidence" or r.get("error"):
                continue
            for u in (r.get("units") or []):
                f.write(json.dumps({"task_id": r["task_id"], **u},
                                   ensure_ascii=False) + "\n")
                n_units_total += 1
    print(f"[02b] wrote {n_units_total} units -> {units_path}")


if __name__ == "__main__":
    main()
