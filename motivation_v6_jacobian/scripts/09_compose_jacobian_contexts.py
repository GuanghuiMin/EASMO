"""Stage 09 — Experiment D part 1: compose gradient-ranked text contexts.

For each task, build three conditions by greedy fill on a token budget:

  jacobian_high_spans          : top-ranked by `span_gxa_sqrtlen / token_count`
  jacobian_high_spans_raw      : top-ranked by `span_gxa_sqrtlen` (no /tok)
  jacobian_low_spans           : bottom-ranked by `span_gxa_sqrtlen / token_count`
  jacobian_recent_hybrid       : ~50% recent tail + ~50% high-Jacobian

Each composed context is wrapped in [SELECTED_HISTORY_SPANS] like v4.

Outputs:
  outputs/raw/jacobian_compressed_contexts.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v6_jacobian.data import (  # noqa: E402
    ensure_outputs, raw_path, read_jsonl, write_jsonl,
)


_HEADER = "[SELECTED_HISTORY_SPANS]"
_FOOTER = "[/SELECTED_HISTORY_SPANS]"


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _wrap(spans_chrono: List[dict]) -> str:
    if not spans_chrono:
        return _HEADER + "\n" + _FOOTER
    body = "\n".join(s["span_text"] for s in spans_chrono)
    return f"{_HEADER}\n{body}\n{_FOOTER}"


def _greedy_fill_by_score(
    candidates: List[Tuple[dict, float]],
    budget: int,
    descending: bool = True,
) -> List[dict]:
    sorted_cands = sorted(
        candidates,
        key=lambda sx: (sx[1], -sx[0]["step_id"]),
        reverse=descending,
    )
    chosen: List[dict] = []
    used = 0
    overhead = _approx_tokens(_HEADER) + _approx_tokens(_FOOTER) + 4
    for span, _score in sorted_cands:
        tok = span.get("v4_token_count") or _approx_tokens(span["span_text"])
        if used + tok + overhead > budget:
            continue
        chosen.append(span)
        used += tok
        if used + overhead >= budget:
            break
    chosen.sort(key=lambda x: x["step_id"])
    return chosen


def _hybrid(
    spans: List[dict],
    scores: Dict[str, float],
    budget: int,
    recent_share: float = 0.5,
) -> List[dict]:
    """Half the budget on the most-recent spans (descending step_id),
    other half on the highest-Jacobian spans from the rest."""
    half_budget = int(budget * recent_share)
    recent_cands = [(s, float(s["step_id"])) for s in spans]
    recent_pick = _greedy_fill_by_score(recent_cands, half_budget)
    picked_ids = {s["span_id"] for s in recent_pick}
    rest = [s for s in spans if s["span_id"] not in picked_ids]
    rest_cands = [(s, scores.get(s["span_id"], 0.0) /
                    max(1, s.get("v4_token_count") or 1))
                  for s in rest]
    rest_pick = _greedy_fill_by_score(
        rest_cands, budget - half_budget, descending=True
    )
    out = recent_pick + rest_pick
    out.sort(key=lambda x: x["step_id"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(raw_path("cases.jsonl")))
    ap.add_argument("--span_scores",
                    default=str(raw_path("jacobian_span_scores.jsonl")))
    ap.add_argument("--out",
                    default=str(raw_path("jacobian_compressed_contexts.jsonl")))
    ap.add_argument("--budgets", nargs="+", type=int, default=[2048, 4096])
    args = ap.parse_args()

    ensure_outputs()
    cases = read_jsonl(Path(args.cases))
    span_rows = read_jsonl(Path(args.span_scores))
    scores_by_task: Dict[str, Dict[str, float]] = {}
    for r in span_rows:
        scores_by_task.setdefault(r["task_id"], {})[r["span_id"]] = \
            float(r.get("span_gxa_sqrtlen") or 0.0)

    out_rows: List[dict] = []
    for case in cases:
        tid = case["task_id"]
        spans = case["spans"]
        scores = scores_by_task.get(tid, {})
        for budget in args.budgets:
            cands_norm = [(s, scores.get(s["span_id"], 0.0) /
                           max(1, s.get("v4_token_count") or 1))
                          for s in spans]
            high = _greedy_fill_by_score(cands_norm, budget, descending=True)
            low  = _greedy_fill_by_score(cands_norm, budget, descending=False)
            cands_raw = [(s, scores.get(s["span_id"], 0.0)) for s in spans]
            high_raw = _greedy_fill_by_score(cands_raw, budget, descending=True)
            hybrid = _hybrid(spans, scores, budget, recent_share=0.5)
            for method, picked in [
                (f"jacobian_high_spans", high),
                (f"jacobian_high_spans_raw", high_raw),
                (f"jacobian_low_spans", low),
                (f"jacobian_recent_hybrid", hybrid),
            ]:
                text = _wrap(picked)
                out_rows.append({
                    "task_id": tid,
                    "method": method,
                    "budget_tokens": budget,
                    "compressed_text": text,
                    "n_spans": len(picked),
                    "kept_span_ids": [s["span_id"] for s in picked],
                })
    write_jsonl(Path(args.out), out_rows)
    print(f"[09] wrote {len(out_rows)} contexts -> {args.out}")


if __name__ == "__main__":
    main()
