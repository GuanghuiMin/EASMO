"""Stage 11 — chunk type labeling with **MiniMax-M2.5 only** (spec §3.3)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v9.data import ensure_outputs, read_jsonl, raw_path  # noqa
from motivation_v9.clients import make_client  # noqa
from motivation_v9.chunk_label import label_chunk  # noqa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(raw_path("chunks.jsonl")))
    ap.add_argument("--cases", default=str(_REPO / "data" / "v9_cases.jsonl"))
    ap.add_argument("--out", default=str(raw_path("chunk_type_labels.jsonl")))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ensure_outputs()

    chunks = read_jsonl(Path(args.chunks))
    cases = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    client = make_client("minimax")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    def _label(ch):
        case = cases.get(ch["case_id"])
        instr = case["user_instruction"] if case else ""
        return label_chunk(
            chunk_id=ch["chunk_id"],
            chunk_text=ch["chunk_text"],
            user_instruction=instr,
            client=client,
        )

    t0 = time.time()
    n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_label, ch): ch for ch in chunks}
            for fut in as_completed(futs):
                lbl = fut.result()
                if lbl.error:
                    n_err += 1
                f_out.write(json.dumps({
                    "chunk_id":                  lbl.chunk_id,
                    "labeler_model":             lbl.labeler_model,
                    "labeler_role":              "chunk_labeler",
                    "qwen_used_as_auditor":      False,
                    "chunk_type":                lbl.chunk_type,
                    "contains_exact_literals":   lbl.contains_exact_literals,
                    "contains_causal_relation":  lbl.contains_causal_relation,
                    "contains_negative_evidence":lbl.contains_negative_evidence,
                    "one_sentence_rationale":    lbl.one_sentence_rationale,
                    "error":                     lbl.error,
                }) + "\n")
                f_out.flush()
                n_done += 1
                if n_done % 20 == 0 or n_done <= 3:
                    eta = (time.time() - t0) / n_done * (len(chunks) - n_done)
                    print(f"  [{n_done:>4d}/{len(chunks)}] err={n_err:<3d} eta={eta:.0f}s",
                          flush=True)
    print(f"[11] wrote {n_done} labels ({n_err} errors) -> {out_path}")


if __name__ == "__main__":
    main()
