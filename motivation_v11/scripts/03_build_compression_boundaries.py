"""Stage 03 — build compression boundaries (spec §7).

For each task in the inventory:
* load the full-context baseline trajectory from acon's output dir,
* render it as plain text,
* expose it as the `history` text that all 4 prompt families will
  compress in stage 05.

Spec §7.1 prefers the "online checkpoint continuation" protocol
(`T_hist=4096` threshold, then env restore to step t for evaluation).
The local productive_agents runner does NOT support env restoration,
so we fall back to **spec §7.2 trajectory-derived** protocol — every
row gets `evaluation_protocol="trajectory_derived"`.

Output: outputs/raw/compression_boundaries.jsonl

Schema per spec §7:
  task_id, split, task_instruction, evaluation_protocol,
  full_success, full_score, full_steps,
  compression_triggered, boundary_step,
  history_chars, history_tokens, history_text,
  latest_observation_text (optional), notes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _read_jsonl_plain(p):
    out = []
    if not Path(p).exists(): return out
    for line in open(p):
        line = line.strip()
        if line: out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_runs",
                    default=str(_REPO / "outputs" / "raw" / "full_context_runs.jsonl"))
    ap.add_argument("--out",
                    default=str(_REPO / "outputs" / "raw" / "compression_boundaries.jsonl"))
    ap.add_argument("--max_history_chars", type=int, default=18000,
                    help="Cap on rendered trajectory text (matches v9/v10).")
    args = ap.parse_args()

    runs = _read_jsonl_plain(args.baseline_runs)
    print(f"[03] {len(runs)} baseline runs to convert into compression inputs")

    sys.path.insert(0, "/workspace/EASMO/motivation_v3")
    sys.path.insert(0, "/workspace/EASMO/motivation_v2")
    from motivation_v2.data import load_trajectory  # noqa
    from motivation_v3.data import render_trajectory  # noqa

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0; n_empty = 0; n_err = 0
    with open(out_path, "w") as f_out:
        for r in runs:
            tid = r["task_id"]
            split = r["split"]
            out_dir = Path(r["output_dir"])
            history_text = ""
            instruction = ""
            try:
                if out_dir.exists():
                    traj = load_trajectory(out_dir)
                    instruction = traj.instruction or ""
                    history_text = render_trajectory(traj,
                                                      max_total_chars=args.max_history_chars)
            except Exception as e:
                n_err += 1
                history_text = f"[load_error: {e}]"
            if not history_text.strip():
                n_empty += 1
            rec = {
                "task_id":              tid,
                "split":                split,
                "task_instruction":     instruction,
                "evaluation_protocol":  "trajectory_derived",
                "full_success":         bool(r.get("full_success")),
                "full_score":           float(r.get("full_score") or 0.0),
                "full_steps":           int(r.get("full_steps") or 0),
                "compression_triggered": True,  # always under trajectory_derived
                "boundary_step":        int(r.get("full_steps") or 0),
                "history_chars":        len(history_text),
                "history_tokens":       max(1, len(history_text) // 4),
                "history_text":         history_text,
                "latest_observation_text": "",
                "notes":                ("fallback trajectory_derived protocol because "
                                          "local productive_agents has no env-restore support; "
                                          "history is full baseline trajectory rendered up to "
                                          f"{args.max_history_chars} chars"),
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
    print(f"[03] wrote {n_done} boundaries ({n_empty} empty histories, {n_err} load errors)")
    print(f"     protocol: trajectory_derived (spec §7.2 fallback)")


if __name__ == "__main__":
    main()
