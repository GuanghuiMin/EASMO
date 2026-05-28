"""ACON history-compression prompt loader (spec §10).

We copy the **original** UT and UTCO prompts from the official
microsoft/acon repository at ``/workspace/acon`` and record provenance
(commit hash + SHA256 of each prompt text). We never hand-rewrite the
prompt template — see spec §2 and §10.

  * UT  = baseline AppWorld history compression
          ``experiments/appworld/prompts/context_opt/prompt_history_v2.jinja``
  * UTCO = the deepest prompt-optimizer-produced variant
          ``experiments/prompt_optimizer/outputs_appworld/stage1_minimax/
              optimized_prompts/improved_history_prompt_samples_4.jinja``

The ACON pipeline only consumes the *user* template; the
system_prompt.jinja used in production is loaded too (one-liner saying
"You are an agent tasked with extracting and refining a concise and
optimized version of the context..."). We use the same system text
for both variants.

Rendering convention follows spec §10:

```python
rendered_prompt = render(template, task=..., prev_summary="", history=..., max_chars=1500)
```

The UTCO template does not carry an explicit ``max_chars`` variable —
the original ACON ran does length control via a separate cap step.
We document this in the provenance file and pass ``max_chars`` only
when the template references it.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import jinja2

ACON_ROOT = Path("/workspace/acon")
UT_SRC = ACON_ROOT / "experiments/appworld/prompts/context_opt/prompt_history_v2.jinja"
UTCO_SRC = (
    ACON_ROOT
    / "experiments/prompt_optimizer/outputs_appworld/stage1_minimax/optimized_prompts/improved_history_prompt_samples_4.jinja"
)
SYSTEM_SRC = ACON_ROOT / "experiments/appworld/prompts/context_opt/system_prompt.jinja"


@dataclass
class AconPromptBundle:
    variant: str                 # 'UT' or 'UTCO'
    template_text: str
    system_text: str
    source_path: str
    sha256: str
    commit_hash: str             # ACON repo commit hash
    has_max_chars_variable: bool # whether the template uses {{ max_chars }}


def _sha256(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def acon_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ACON_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        out = "UNKNOWN"
    return out


def load_bundle(variant: str) -> AconPromptBundle:
    """Load the UT or UTCO original ACON prompt bundle.

    Fails loudly if the source file does not exist (spec §10).
    """
    variant = variant.upper()
    if variant == "UT":
        src = UT_SRC
    elif variant == "UTCO":
        src = UTCO_SRC
    else:
        raise ValueError(f"unknown variant: {variant}")
    if not src.exists():
        raise FileNotFoundError(
            f"ACON prompt {variant} not found at {src} — official "
            "microsoft/acon repo must be available at /workspace/acon"
        )
    template_text = src.read_text(encoding="utf-8")
    if not SYSTEM_SRC.exists():
        raise FileNotFoundError(f"ACON system prompt not found at {SYSTEM_SRC}")
    system_text = SYSTEM_SRC.read_text(encoding="utf-8").strip()
    return AconPromptBundle(
        variant=variant,
        template_text=template_text,
        system_text=system_text,
        source_path=str(src.relative_to(ACON_ROOT)),
        sha256=_sha256(template_text),
        commit_hash=acon_commit_hash(),
        has_max_chars_variable="max_chars" in template_text or "{{ max_chars" in template_text,
    )


def render_prompt(
    bundle: AconPromptBundle,
    *,
    task: str,
    history: str,
    prev_summary: str = "",
    max_chars: Optional[int] = None,
) -> str:
    """Render the ACON template with the official variable names.

    Per spec §10.4, the canonical variables are
    ``task``, ``prev_summary``, ``history``, ``max_chars``.
    If the template does not reference ``max_chars`` we still pass it
    so the template author can opt in.
    """
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
        max_chars=max_chars if max_chars is not None else "",
    )


def install_into_repo(out_dir: Path, provenance_dir: Path) -> Dict[str, dict]:
    """Copy UT + UTCO + system prompts into the v7 repo at
    ``prompts/`` and write provenance under ``outputs/provenance/``.
    Returns the provenance dict for inspection.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    record: Dict[str, dict] = {}
    commit = acon_commit_hash()
    (provenance_dir / "acon_commit.txt").write_text(commit + "\n", encoding="utf-8")

    for variant in ("UT", "UTCO"):
        b = load_bundle(variant)
        local = out_dir / f"acon_history_{variant.lower()}_original.md"
        local.write_text(b.template_text, encoding="utf-8")
        record[variant] = {
            "variant": variant,
            "source_path": b.source_path,
            "local_copy": str(local),
            "sha256": b.sha256,
            "has_max_chars_variable": b.has_max_chars_variable,
        }
    # also persist the system prompt
    (out_dir / "acon_system_prompt.md").write_text(
        SYSTEM_SRC.read_text(encoding="utf-8"), encoding="utf-8",
    )
    record["system"] = {
        "source_path": str(SYSTEM_SRC.relative_to(ACON_ROOT)),
        "local_copy": str(out_dir / "acon_system_prompt.md"),
        "sha256": _sha256(SYSTEM_SRC.read_text(encoding="utf-8")),
    }
    record["acon_commit_hash"] = commit

    (provenance_dir / "acon_prompt_sha256.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return record


__all__ = [
    "ACON_ROOT", "UT_SRC", "UTCO_SRC", "SYSTEM_SRC",
    "AconPromptBundle",
    "acon_commit_hash",
    "load_bundle",
    "render_prompt",
    "install_into_repo",
]
