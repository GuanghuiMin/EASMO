"""Stage 06 — build per-task compressed contexts for the 6 NEW v4
methods, under a matched token budget.

Budget per spec §7: average tokens of v3's task_aware_summary for that
task. Falls back to 400 tokens if v3 data is unavailable.

Outputs:
  outputs/raw/compressed_contexts.jsonl   one row per (task, method)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_budget", type=int, default=400)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    from motivation_v4.data import (
        ensure_outputs, raw_path, read_jsonl, write_jsonl,
        load_v3_compressed_contexts,
    )
    from motivation_v4.compose import (
        compose_high_sensitivity, compose_low_sensitivity,
        compose_recent, compose_random,
    )
    from motivation_v4.spans import Span

    ensure_outputs()
    spans_rows = read_jsonl(raw_path("history_spans.jsonl"))
    sens_rows = read_jsonl(raw_path("span_sensitivity_scores.jsonl"))

    spans_by_task: Dict[str, List[Span]] = defaultdict(list)
    for r in spans_rows:
        spans_by_task[r["task_id"]].append(Span(**r))
    for tid in spans_by_task:
        spans_by_task[tid].sort(key=lambda s: s.step_id)

    sens_by_task: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in sens_rows:
        sens_by_task[r["task_id"]][r["span_id"]] = float(r["final_sensitivity"])

    # Per-task budget = avg task_aware_summary tokens (from v3).
    v3_compressed = load_v3_compressed_contexts()
    task_aware_tokens: Dict[str, int] = {}
    for r in v3_compressed:
        if r.get("method") == "task_aware_summary" and not r.get("error"):
            task_aware_tokens[r["task_id"]] = int(r.get("n_tokens", 0))

    out_records: List[dict] = []
    for tid, spans in spans_by_task.items():
        budget = task_aware_tokens.get(tid, args.default_budget)
        sens_map = sens_by_task.get(tid, {})

        out_records.append({
            "task_id": tid, "method": "high_sensitivity_spans",
            "budget_tokens": budget,
            "compressed_text": compose_high_sensitivity(spans, sens_map, budget),
        })
        out_records.append({
            "task_id": tid, "method": "low_sensitivity_spans",
            "budget_tokens": budget,
            "compressed_text": compose_low_sensitivity(spans, sens_map, budget),
        })
        out_records.append({
            "task_id": tid, "method": "recent_spans",
            "budget_tokens": budget,
            "compressed_text": compose_recent(spans, budget),
        })
        for seed in args.seeds:
            out_records.append({
                "task_id": tid, "method": f"random_spans_seed{seed}",
                "budget_tokens": budget,
                "compressed_text": compose_random(spans, budget, seed=seed),
            })

    out_path = raw_path("compressed_contexts.jsonl")
    write_jsonl(out_path, out_records)
    print(f"[06] wrote {len(out_records)} rows -> {out_path}")
    methods = sorted({r['method'] for r in out_records})
    print(f"[06] methods: {methods}")


if __name__ == "__main__":
    main()
