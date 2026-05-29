"""ACON UTCO history-compression prompt loader (spec §4).

Reuses the same official microsoft/acon prompts as v7 (commit
d63f9ae18959dc7215ff62899c94c5e8c56847ae) — DO NOT edit them.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import jinja2

ACON_ROOT = Path("/workspace/acon")
UTCO_SRC = (
    ACON_ROOT
    / "experiments/prompt_optimizer/outputs_appworld/stage1_minimax/optimized_prompts/improved_history_prompt_samples_4.jinja"
)
SYSTEM_SRC = ACON_ROOT / "experiments/appworld/prompts/context_opt/system_prompt.jinja"


@dataclass
class AconPromptBundle:
    variant: str
    template_text: str
    system_text: str
    source_path: str
    sha256: str
    commit_hash: str


def _sha256(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def acon_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ACON_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def load_utco_bundle() -> AconPromptBundle:
    if not UTCO_SRC.exists():
        raise FileNotFoundError(f"ACON UTCO prompt not at {UTCO_SRC}")
    if not SYSTEM_SRC.exists():
        raise FileNotFoundError(f"ACON system prompt not at {SYSTEM_SRC}")
    template_text = UTCO_SRC.read_text(encoding="utf-8")
    system_text = SYSTEM_SRC.read_text(encoding="utf-8").strip()
    return AconPromptBundle(
        variant="ACON_UTCO_official",
        template_text=template_text,
        system_text=system_text,
        source_path=str(UTCO_SRC.relative_to(ACON_ROOT)),
        sha256=_sha256(template_text),
        commit_hash=acon_commit_hash(),
    )


def render_prompt(
    bundle: AconPromptBundle,
    *,
    task: str,
    history: str,
    prev_summary: str = "",
    max_chars: int = 1500,
) -> str:
    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        autoescape=False,
        keep_trailing_newline=True,
    )
    tmpl = env.from_string(bundle.template_text)
    return tmpl.render(
        task=task,
        prev_summary=prev_summary,
        history=history,
        max_chars=max_chars,
    )


def install_provenance(
    bundle: AconPromptBundle,
    prompts_dir: Path,
    provenance_dir: Path,
) -> Dict[str, str]:
    prompts_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "acon_utco_official.md").write_text(
        bundle.template_text, encoding="utf-8"
    )
    (prompts_dir / "acon_system_prompt.md").write_text(
        bundle.system_text, encoding="utf-8"
    )
    record = {
        "acon_repo_commit": bundle.commit_hash,
        "prompt_variant": bundle.variant,
        "source_path": bundle.source_path,
        "history_prompt_sha256": bundle.sha256,
        "system_prompt_sha256": _sha256(bundle.system_text),
    }
    (provenance_dir / "prompt_sha256.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return record


__all__ = [
    "ACON_ROOT", "UTCO_SRC", "SYSTEM_SRC",
    "AconPromptBundle",
    "acon_commit_hash",
    "load_utco_bundle",
    "render_prompt",
    "install_provenance",
]
