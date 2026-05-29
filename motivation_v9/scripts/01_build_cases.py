"""Stage 01 — build v9 case pool from v3 successful AppWorld dev."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v9.data import (  # noqa
    ensure_outputs, write_jsonl, load_v3_trajectories,
    render_full_trajectory_text,
)


def _apps_used(traj) -> list:
    apps = set()
    for s in traj.steps:
        for m in re.finditer(r"\bapis\.([a-zA-Z0-9_]+)\.", s.action or ""):
            apps.add(m.group(1))
    return sorted(apps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_REPO / "data" / "v9_cases.jsonl"))
    ap.add_argument("--max_chars", type=int, default=18000)
    ap.add_argument("--max_cases", type=int, default=None)
    args = ap.parse_args()
    ensure_outputs()

    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v2.data import load_trajectory  # noqa

    sel = load_v3_trajectories()
    cases = []; skip = 0
    for r in sel:
        td = Path(r["output_dir"])
        if not td.exists():
            skip += 1; continue
        try:
            traj = load_trajectory(td)
        except Exception:
            skip += 1; continue
        if not traj.steps:
            skip += 1; continue
        steps = []
        for s in traj.steps:
            steps.append({
                "step_id": int(s.step),
                "thought": None,
                "action": s.action or "",
                "observation": s.output or "",
            })
        text = render_full_trajectory_text(traj, max_total_chars=args.max_chars)
        apps = _apps_used(traj)
        cases.append({
            "case_id":              traj.task_id,
            "task_id":              traj.task_id,
            "user_instruction":     traj.instruction or "",
            "full_trajectory_text": text,
            "trajectory_steps":     steps,
            "baseline_success":     True,
            "baseline_iterations":  len(steps),
            "apps_used":            apps,
            "n_apps":               len(apps),
            "case_priority":        "hard_success" if len(steps) >= 20 else "long_multitool",
            "notes":                "",
        })
    cases.sort(key=lambda c: c["case_id"])
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    n = write_jsonl(Path(args.out), cases)
    print(f"[01] wrote {n} cases -> {args.out}  (skipped {skip})")
    lens = [c["baseline_iterations"] for c in cases]
    chars = [len(c["full_trajectory_text"]) for c in cases]
    print(f"     steps: min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)}")
    print(f"     chars: min={min(chars)} median={sorted(chars)[len(chars)//2]} max={max(chars)}")


if __name__ == "__main__":
    main()
