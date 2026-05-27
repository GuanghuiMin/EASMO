"""Stage 01 — split each v3-selected dev trajectory into spans.

Outputs:
  outputs/raw/history_spans.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def main():
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    from motivation_v4.data import (
        ensure_outputs, raw_path, write_jsonl,
        load_v3_trajectories, load_trajectory,
    )
    from motivation_v4.spans import trajectory_to_spans

    ensure_outputs()
    sel = load_v3_trajectories()
    print(f"[01] {len(sel)} v3 dev trajectories")

    all_spans = []
    n_skipped = 0
    for r in sel:
        td = Path(r["output_dir"])
        if not td.exists():
            n_skipped += 1
            continue
        try:
            traj = load_trajectory(td)
        except Exception:
            n_skipped += 1
            continue
        spans = trajectory_to_spans(traj)
        for s in spans:
            all_spans.append(s.to_dict())
        print(f"  {traj.task_id:>11s}: {len(spans):>3d} spans, "
              f"avg {sum(s.token_count for s in spans)/max(len(spans),1):.0f} tok")

    out_path = raw_path("history_spans.jsonl")
    n = write_jsonl(out_path, all_spans)
    print(f"[01] wrote {n} spans -> {out_path}")
    if n_skipped:
        print(f"[01] skipped {n_skipped} tasks (missing trajectory dir / load failure)")
    if all_spans:
        per_task = {}
        for s in all_spans:
            per_task.setdefault(s["task_id"], 0)
            per_task[s["task_id"]] += 1
        spans_per = list(per_task.values())
        toks = [s["token_count"] for s in all_spans]
        print(f"[01] spans/task: min={min(spans_per)}, mean={sum(spans_per)/len(spans_per):.1f}, max={max(spans_per)}")
        print(f"[01] tok/span:   min={min(toks)}, mean={sum(toks)/len(toks):.0f}, max={max(toks)}")


if __name__ == "__main__":
    main()
