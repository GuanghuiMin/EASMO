"""End-to-end smoke test: M1 (tiny) -> M2 -> M3 -> M4.

Run this first after any change to the package or the MiniMax endpoint.
With the default ``configs/smoke.yaml`` it should finish in ~5-15 minutes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent


def _step(label, name, cfg, no_wandb):
    cmd = [sys.executable, "-m", f"scripts.{name}", "--config", cfg]
    if no_wandb:
        cmd.append("--no-wandb")
    print(f"\n>>> [{label}] {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(_REPO), check=True)
    print(f"<<< [{label}] done in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--only", default="",
                        help="comma-separated subset of M1,M2,M3,M4 to run")
    args = parser.parse_args()

    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}

    stages = [
        ("M1", "run_m1"),
        ("M2", "run_m2"),
        ("M3", "run_m3"),
        ("M4", "run_m4"),
    ]
    t_all = time.time()
    for label, script in stages:
        if only and label not in only:
            continue
        _step(label, script, args.config, args.no_wandb)

    print(f"\nSmoke test complete in {(time.time() - t_all)/60:.1f} min.")


if __name__ == "__main__":
    main()
