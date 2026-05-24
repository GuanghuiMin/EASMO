"""Run M1 -> M2 -> M3 -> M4 in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent


def _run(name, cfg, no_wandb):
    cmd = [sys.executable, "-m", f"scripts.{name}", "--config", cfg]
    if no_wandb:
        cmd.append("--no-wandb")
    print(f"\n=== {name} ===")
    subprocess.run(cmd, cwd=str(_REPO), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--skip", default="", help="comma-separated list of stages to skip")
    args = parser.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    for stage in ["run_m1", "run_m2", "run_m3", "run_m3_judge", "run_m4", "run_m5"]:
        if stage in skip:
            print(f"Skipping {stage}")
            continue
        _run(stage, args.config, args.no_wandb)


if __name__ == "__main__":
    main()
