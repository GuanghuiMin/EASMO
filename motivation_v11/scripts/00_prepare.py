"""Stage 00 — provenance + config snapshot (spec §16, §6).

Writes:
  outputs/provenance/acon_repo_commit.txt
  outputs/provenance/acon_ut_prompt.txt
  outputs/provenance/acon_utco_prompt.txt
  outputs/provenance/acon_system_prompt.txt
  outputs/provenance/prompt_sha256.json
  outputs/provenance/general_prompt_templates.md
  outputs/provenance/pip_freeze_easmo_venv.txt
  outputs/config_v11.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v11.data import ensure_outputs, OUTPUTS, PROVENANCE      # noqa
from motivation_v11 import prompt_families as pf                          # noqa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--stress_rounds_k", type=int, default=2)
    ap.add_argument("--cap_steps", type=int, default=15)
    ap.add_argument("--max_chars", type=int, default=2000)
    ap.add_argument("--task_pool", default="train+dev")
    args = ap.parse_args()
    ensure_outputs()

    # ACON commit
    try:
        acon_commit = subprocess.check_output(
            ["git", "-C", str(pf.ACON_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as e:
        acon_commit = f"UNKNOWN ({e})"
    (PROVENANCE / "acon_repo_commit.txt").write_text(acon_commit + "\n")

    # Prompt provenance per family
    sha_record = pf.provenance()
    (PROVENANCE / "prompt_sha256.json").write_text(
        json.dumps(sha_record, indent=2, ensure_ascii=False)
    )

    # Save raw prompt texts per family
    for fam in pf.all_families():
        b = pf.get_bundle(fam)
        (PROVENANCE / f"{fam}_system.txt").write_text(b.system_text)
        (PROVENANCE / f"{fam}_user_template.txt").write_text(b.user_template)

    # General prompt templates as markdown for paper appendix
    md = ["# motivation_v11 general prompt templates\n"]
    for fam in ("general_task_agnostic", "general_task_aware"):
        b = pf.get_bundle(fam)
        md.append(f"\n## {fam}\n\n### System\n\n```text\n{b.system_text}\n```\n")
        md.append(f"\n### User template\n\n```text\n{b.user_template}\n```\n")
    md.append("\n## ACON_UT and ACON_UTCO\n\n")
    md.append("Loaded verbatim from the microsoft/acon repository — see "
              "`acon_ut_prompt.txt` and `acon_utco_prompt.txt` for full text "
              "and `prompt_sha256.json` for hashes / commit.\n")
    (PROVENANCE / "general_prompt_templates.md").write_text("\n".join(md))

    # Save full ACON prompts too
    (PROVENANCE / "acon_ut_prompt.txt").write_text(pf.get_bundle("ACON_UT").user_template)
    (PROVENANCE / "acon_utco_prompt.txt").write_text(pf.get_bundle("ACON_UTCO").user_template)
    (PROVENANCE / "acon_system_prompt.txt").write_text(pf.get_bundle("ACON_UT").system_text)

    # pip freeze
    try:
        freeze = subprocess.check_output(
            ["/workspace/EASMO/.venv/bin/pip", "freeze"], text=True
        )
        (PROVENANCE / "pip_freeze_easmo_venv.txt").write_text(freeze)
    except Exception as e:
        (PROVENANCE / "pip_freeze_easmo_venv.txt").write_text(f"ERROR: {e}\n")

    # Config snapshot
    cfg = {
        "spec_path": "user_feedback/motivation_v11_final_full_dev_behavior_prompt_family_experiment.md",
        "task_pool": args.task_pool,
        "n_samples": args.n_samples,
        "stress_rounds_k": args.stress_rounds_k,
        "cap_steps": args.cap_steps,
        "max_chars": args.max_chars,
        "temperature_greedy": 0.0,
        "temperature_sample": 0.7,
        "seed_greedy": 42,
        "seed_sample_base": 1000,
        "max_tokens_compression": 2048,
        "max_tokens_verifier": 1536,
        "max_tokens_pairwise": 1024,
        "max_tokens_entropy": 1024,
        "entropy_M": 5,
        "entropy_scope_initial": ["ACON_UTCO"],
        "entropy_scope_extensible_to": ["ACON_UT", "general_task_aware", "general_task_agnostic"],
        "lambda_length": 0.05,
        "selector_random_seed": 20260531,
        "acon_commit": acon_commit,
        "downstream_agent_model": "MiniMaxAI/MiniMax-M2.5",
        "compressor_model": "MiniMaxAI/MiniMax-M2.5",
        "verifier_model": "MiniMaxAI/MiniMax-M2.5",
    }
    (OUTPUTS / "config_v11.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False)
    )

    print(f"[00] provenance written under {PROVENANCE}")
    print(f"     ACON commit: {acon_commit}")
    for fam in pf.all_families():
        s = sha_record[fam]
        print(f"     {fam}: user_sha256={s['sha256_user'][:16]}...")


if __name__ == "__main__":
    main()
