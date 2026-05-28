"""Stage 00 — sync the official ACON history-compression prompts (spec §10).

We do not clone the repo here because /workspace/acon already contains
the official microsoft/acon repo. We do verify the remote URL, record
the commit hash, copy the UT + UTCO + system prompt files into
``prompts/`` under the v7 repo, and write provenance to
``outputs/provenance/``.

Fails loudly if any prompt is missing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v7.acon_prompt_loader import (  # noqa: E402
    ACON_ROOT, UT_SRC, UTCO_SRC, SYSTEM_SRC, install_into_repo,
)
from motivation_v7.data import ensure_outputs  # noqa: E402


def _check_remote() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ACON_ROOT), "remote", "get-url", "origin"],
            text=True,
        ).strip()
    except Exception:
        out = "UNKNOWN"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(_REPO / "prompts"))
    ap.add_argument("--provenance_dir",
                    default=str(_REPO / "outputs/provenance"))
    args = ap.parse_args()

    ensure_outputs()
    remote = _check_remote()
    if "microsoft/acon" not in remote.lower():
        print(f"[00] WARNING: /workspace/acon remote is '{remote}'  "
              "(expected microsoft/acon). Continuing because the prompt "
              "files exist; this is your responsibility to verify.")
    for must_exist in (UT_SRC, UTCO_SRC, SYSTEM_SRC):
        if not must_exist.exists():
            raise FileNotFoundError(f"required ACON prompt missing: {must_exist}")

    record = install_into_repo(Path(args.out_dir), Path(args.provenance_dir))

    print(f"[00] ACON repo: {ACON_ROOT}  remote={remote}")
    print(f"[00] commit:    {record['acon_commit_hash']}")
    for variant in ("UT", "UTCO"):
        r = record[variant]
        print(f"      {variant:>5s}: source={r['source_path']}")
        print(f"             sha256={r['sha256']}")
        print(f"             has_max_chars_variable={r['has_max_chars_variable']}")
    print(f"[00] system prompt sha256={record['system']['sha256']}")
    print(f"[00] provenance written to {args.provenance_dir}")


if __name__ == "__main__":
    main()
