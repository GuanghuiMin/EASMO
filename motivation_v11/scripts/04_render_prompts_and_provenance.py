"""Stage 04 — render N example prompts per family for provenance (spec §6, §16).

Reads boundaries from `outputs/raw/compression_boundaries.jsonl`
(written by stage 03) and writes 3 sample rendered prompts per family.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v11.data import ensure_outputs, PROVENANCE, read_jsonl  # noqa
from motivation_v11 import prompt_families as pf                         # noqa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundaries",
                    default=str(_REPO / "outputs" / "raw" / "compression_boundaries.jsonl"))
    ap.add_argument("--n_examples", type=int, default=3)
    ap.add_argument("--max_chars", type=int, default=2000)
    args = ap.parse_args()
    ensure_outputs()

    out_root = PROVENANCE / "rendered_prompt_examples"
    out_root.mkdir(parents=True, exist_ok=True)

    boundaries = read_jsonl(Path(args.boundaries))
    boundaries = [b for b in boundaries if b.get("history_text", "").strip()]
    boundaries = boundaries[: args.n_examples]
    print(f"[04] rendering {len(boundaries)} example prompts × "
          f"{len(list(pf.all_families()))} families")
    for fam in pf.all_families():
        b = pf.get_bundle(fam)
        for case in boundaries:
            rendered = pf.render(b,
                                  task=case["task_instruction"],
                                  history=case["history_text"],
                                  max_chars=args.max_chars)
            fname = f"{fam}_{case['task_id']}.txt"
            (out_root / fname).write_text(
                f"# family: {fam}\n# task_id: {case['task_id']}\n# split: {case['split']}\n\n"
                f"## SYSTEM\n{b.system_text}\n\n## USER\n{rendered}"
            )
    n_written = len(list(pf.all_families())) * len(boundaries)
    print(f"[04] wrote {n_written} rendered examples -> {out_root}")


if __name__ == "__main__":
    main()
