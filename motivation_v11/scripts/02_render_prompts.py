"""Stage 02 — render 3 example prompts per family for provenance (spec §6, §16)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v11.data import ensure_outputs, PROVENANCE, read_jsonl  # noqa
from motivation_v11 import prompt_families as pf                         # noqa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary_cases",
                    default=str(_REPO / "data" / "v11_primary_cases.jsonl"))
    ap.add_argument("--n_examples", type=int, default=3)
    ap.add_argument("--max_chars", type=int, default=2000)
    args = ap.parse_args()
    ensure_outputs()

    out_root = PROVENANCE / "rendered_prompt_examples"
    out_root.mkdir(parents=True, exist_ok=True)

    cases = read_jsonl(Path(args.primary_cases))
    cases = cases[: args.n_examples]
    print(f"[02] rendering {args.n_examples} example prompts × "
          f"{len(list(pf.all_families()))} families")
    for fam in pf.all_families():
        out_dir = out_root / fam
        out_dir.mkdir(parents=True, exist_ok=True)
        b = pf.get_bundle(fam)
        for c in cases:
            rendered = pf.render(b,
                                  task=c["user_instruction"],
                                  history=c["full_trajectory_text"],
                                  max_chars=args.max_chars)
            (out_dir / f"{c['task_id']}.txt").write_text(
                f"# family: {fam}\n# task_id: {c['task_id']}\n\n"
                f"## SYSTEM\n{b.system_text}\n\n## USER\n{rendered}"
            )
    print(f"[02] wrote {len(list(pf.all_families())) * len(cases)} rendered examples "
          f"-> {out_root}")


if __name__ == "__main__":
    main()
