"""Four prompt families for v11 (spec §5).

* `general_task_agnostic` — generic summarization control (§5.1).
* `general_task_aware`    — generic with task instruction injected (§5.2).
* `ACON_UT`               — official ACON utility-optimized history prompt
                            from `experiments/appworld/prompts/context_opt/
                            prompt_history_v2.jinja` (§5.3).
* `ACON_UTCO`             — official ACON utility + compression optimized
                            prompt (§5.4) — same as v7-v10.

`render(family, task, history, max_chars)` returns the rendered user
prompt; `system_for(family)` returns the system prompt; `provenance()`
returns dict of SHA256 + paths.

Both ACON families load their templates verbatim from the
microsoft/acon repository commit `d63f9ae18959dc7215ff62899c94c5e8c56847ae`
(matches v7/v8/v9/v10).
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import jinja2


ACON_ROOT = Path("/workspace/acon")

ACON_UT_SRC = (
    ACON_ROOT
    / "experiments/appworld/prompts/context_opt/prompt_history_v2.jinja"
)
ACON_UTCO_SRC = (
    ACON_ROOT
    / "experiments/prompt_optimizer/outputs_appworld/stage1_minimax"
    "/optimized_prompts/improved_history_prompt_samples_4.jinja"
)
ACON_SYSTEM_SRC = (
    ACON_ROOT / "experiments/appworld/prompts/context_opt/system_prompt.jinja"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _acon_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ACON_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


# ----------------------------------------------------------------------
# General prompts (verbatim from spec §5.1 and §5.2)
# ----------------------------------------------------------------------

_GEN_TA_AG_SYSTEM = (
    "You are a careful context compression module.\n"
    "Return only the compressed context. Do not include explanations "
    "about your compression process."
)

_GEN_TA_AG_USER = """\
Compress the following interaction history into a shorter version.

Hard budget:
- The compressed context should be no more than {max_chars} characters.

Compression goals:
- Preserve important information.
- Remove redundant, obsolete, or irrelevant details.
- Keep exact values only if they appear important in the history.
- Do not invent facts.
- Do not output the original input.
- Return plain text only. You may use bullets, but do not use a fixed schema.

Interaction history:
{context}

Compressed context:
"""


_GEN_TA_AW_SYSTEM = (
    "You are a careful context compression module for a tool-use agent.\n"
    "Return only the compressed context. Do not include explanations "
    "about your compression process."
)

_GEN_TA_AW_USER = """\
Compress the previous interaction history into a shorter context for a downstream tool-use agent.

The downstream agent will continue the following task:
{task_instruction}

Hard budget:
- The compressed context should be no more than {max_chars} characters.

Compression goals:
- Preserve information that may help the downstream agent continue the task correctly.
- Preserve exact identifiers, API names, parameter names, file paths, dates, amounts, auth values, object IDs, and state-changing action outcomes when they may matter.
- Preserve failed attempts or negative evidence only if they may prevent repeated mistakes.
- Remove redundant, obsolete, or irrelevant details.
- Do not invent facts.
- Do not solve the task.
- Do not output the original input.
- Return plain text only. You may use bullets, but do not use a fixed schema.

Previous interaction history:
{context}

Compressed context:
"""


# ----------------------------------------------------------------------
# Bundle
# ----------------------------------------------------------------------

@dataclass
class PromptBundle:
    family: str
    system_text: str
    user_template: str
    template_kind: str           # "python_format" | "jinja_acon"
    source_path: str             # "in-repo" or acon-relative path
    sha256_system: str
    sha256_user: str
    acon_commit: Optional[str] = None


def load_acon_template(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(f"ACON template missing: {p}")
    return p.read_text(encoding="utf-8")


def get_bundle(family: str) -> PromptBundle:
    if family == "general_task_agnostic":
        return PromptBundle(
            family=family,
            system_text=_GEN_TA_AG_SYSTEM,
            user_template=_GEN_TA_AG_USER,
            template_kind="python_format",
            source_path="in-repo motivation_v11/prompt_families.py",
            sha256_system=_sha256(_GEN_TA_AG_SYSTEM),
            sha256_user=_sha256(_GEN_TA_AG_USER),
            acon_commit=None,
        )
    if family == "general_task_aware":
        return PromptBundle(
            family=family,
            system_text=_GEN_TA_AW_SYSTEM,
            user_template=_GEN_TA_AW_USER,
            template_kind="python_format",
            source_path="in-repo motivation_v11/prompt_families.py",
            sha256_system=_sha256(_GEN_TA_AW_SYSTEM),
            sha256_user=_sha256(_GEN_TA_AW_USER),
            acon_commit=None,
        )
    if family == "ACON_UT":
        tmpl = load_acon_template(ACON_UT_SRC)
        sys_tmpl = load_acon_template(ACON_SYSTEM_SRC).strip()
        return PromptBundle(
            family=family,
            system_text=sys_tmpl,
            user_template=tmpl,
            template_kind="jinja_acon",
            source_path=str(ACON_UT_SRC.relative_to(ACON_ROOT)),
            sha256_system=_sha256(sys_tmpl),
            sha256_user=_sha256(tmpl),
            acon_commit=_acon_commit(),
        )
    if family == "ACON_UTCO":
        tmpl = load_acon_template(ACON_UTCO_SRC)
        sys_tmpl = load_acon_template(ACON_SYSTEM_SRC).strip()
        return PromptBundle(
            family=family,
            system_text=sys_tmpl,
            user_template=tmpl,
            template_kind="jinja_acon",
            source_path=str(ACON_UTCO_SRC.relative_to(ACON_ROOT)),
            sha256_system=_sha256(sys_tmpl),
            sha256_user=_sha256(tmpl),
            acon_commit=_acon_commit(),
        )
    raise ValueError(f"unknown prompt family: {family}")


_JENV = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    autoescape=False,
    keep_trailing_newline=True,
)


def render(bundle: PromptBundle, *, task: str, history: str,
            max_chars: int = 2000) -> str:
    if bundle.template_kind == "python_format":
        # ag template has no {task_instruction}, aware does
        if bundle.family == "general_task_agnostic":
            return bundle.user_template.format(
                max_chars=max_chars, context=history,
            )
        return bundle.user_template.format(
            task_instruction=task, max_chars=max_chars, context=history,
        )
    # jinja_acon
    tmpl = _JENV.from_string(bundle.user_template)
    return tmpl.render(task=task, prev_summary="", history=history,
                        max_chars=max_chars)


def all_families() -> tuple:
    return ("general_task_agnostic", "general_task_aware", "ACON_UT", "ACON_UTCO")


def provenance() -> Dict[str, dict]:
    """Return {family: {sha256_system, sha256_user, source_path, acon_commit}}."""
    out = {}
    for f in all_families():
        b = get_bundle(f)
        out[f] = {
            "sha256_system": b.sha256_system,
            "sha256_user":   b.sha256_user,
            "source_path":   b.source_path,
            "template_kind": b.template_kind,
            "acon_commit":   b.acon_commit,
        }
    return out


__all__ = [
    "ACON_ROOT", "ACON_UT_SRC", "ACON_UTCO_SRC", "ACON_SYSTEM_SRC",
    "PromptBundle", "get_bundle", "render", "all_families", "provenance",
]
