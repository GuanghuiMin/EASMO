"""Stage 01 — build the case_pool.jsonl from v3-selected successful
AppWorld dev trajectories (spec §5).

Each row contains the case_id, task_id, instruction, rendered
trajectory text, structured per-step records, and an ``apps_used``
list. We log length-stratification stats but accept v3's 30 cases
as-is (v3 has no <15-step cases).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v7.data import (  # noqa: E402
    ensure_outputs, write_jsonl, load_v3_trajectories,
    render_full_trajectory_text,
)


def _apps_used(traj) -> List[str]:
    """Pull app names by scanning `apis.<app>.<api>(` patterns in actions."""
    import re
    apps = set()
    for s in traj.steps:
        for m in re.finditer(r"\bapis\.([a-zA-Z0-9_]+)\.", s.action or ""):
            apps.add(m.group(1))
    return sorted(apps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_REPO / "data" / "case_pool.jsonl"))
    ap.add_argument("--max_chars", type=int, default=18000)
    ap.add_argument("--max_cases", type=int, default=None,
                    help="Cap on # cases (for smoke testing)")
    args = ap.parse_args()

    ensure_outputs()
    # v2's loader returns Trajectory objects with steps + instruction.
    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v2.data import load_trajectory  # noqa

    sel = load_v3_trajectories()
    cases: List[dict] = []
    skip = 0
    for r in sel:
        td = Path(r["output_dir"])
        if not td.exists():
            skip += 1
            continue
        try:
            traj = load_trajectory(td)
        except Exception:
            skip += 1
            continue
        if not traj.steps:
            skip += 1
            continue
        steps = []
        for s in traj.steps:
            steps.append({
                "step_id":     int(s.step),
                "thought":     None,
                "action":      s.action or "",
                "observation": s.output or "",
            })
        text = render_full_trajectory_text(traj, max_total_chars=args.max_chars)
        cases.append({
            "case_id":              traj.task_id,
            "task_id":              traj.task_id,
            "task_name":            None,
            "user_instruction":     traj.instruction or "",
            "full_trajectory_text": text,
            "trajectory_steps":     steps,
            "success":              True,
            "num_steps":            len(steps),
            "apps_used":            _apps_used(traj),
        })
    cases.sort(key=lambda c: c["case_id"])
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    n = write_jsonl(Path(args.out), cases)
    print(f"[01] wrote {n} cases -> {args.out}  (skipped {skip})")
    if n:
        lens = [c["num_steps"] for c in cases]
        chars = [len(c["full_trajectory_text"]) for c in cases]
        print(f"      num_steps: min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)}")
        print(f"      chars:     min={min(chars)} median={sorted(chars)[len(chars)//2]} max={max(chars)}")
        n_short = sum(1 for x in lens if x <= 14)
        n_med   = sum(1 for x in lens if 15 <= x <= 24)
        n_long  = sum(1 for x in lens if x >= 25)
        print(f"      stratification:  short={n_short}  medium={n_med}  long={n_long}")


if __name__ == "__main__":
    main()
