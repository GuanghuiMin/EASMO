"""Stage 08 — segment selected candidates into natural-language chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v9.data import ensure_outputs, read_jsonl, write_jsonl, raw_path  # noqa
from motivation_v9.chunks import segment_chunks  # noqa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=str(raw_path("chunk_case_selection.jsonl")))
    ap.add_argument("--candidates",
                    default=str(raw_path("candidate_compressions.jsonl")))
    ap.add_argument("--out", default=str(raw_path("chunks.jsonl")))
    ap.add_argument("--max_chunks", type=int, default=12)
    args = ap.parse_args()
    ensure_outputs()

    selection = read_jsonl(Path(args.selection))
    cands = {c["candidate_id"]: c for c in read_jsonl(Path(args.candidates))}

    rows: List[dict] = []
    for s in selection:
        cid = s["selected_candidate_id"]
        cand = cands.get(cid)
        if not cand or not cand.get("compressed_text"):
            continue
        chunks = segment_chunks(
            candidate_id=cid,
            case_id=s["case_id"],
            text=cand["compressed_text"],
            max_chunks=args.max_chunks,
        )
        for ch in chunks:
            rows.append({
                "chunk_id":        ch.chunk_id,
                "candidate_id":    ch.candidate_id,
                "case_id":         ch.case_id,
                "chunk_index":     ch.chunk_index,
                "chunk_text":      ch.chunk_text,
                "chunk_chars":     ch.chunk_chars,
                "chunk_tokens_est":ch.chunk_tokens_est,
                "char_span_start": ch.char_span_start,
                "char_span_end":   ch.char_span_end,
            })

    n = write_jsonl(Path(args.out), rows)
    print(f"[08] wrote {n} chunks across {len(selection)} candidates -> {args.out}")
    if rows:
        from collections import Counter
        per_cand = Counter(r["candidate_id"] for r in rows)
        print(f"     chunks/candidate min={min(per_cand.values())} median={sorted(per_cand.values())[len(per_cand)//2]} max={max(per_cand.values())}")


if __name__ == "__main__":
    main()
