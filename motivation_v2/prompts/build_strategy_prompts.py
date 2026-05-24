"""Materialise the three strategy-specific (jinja, json) prompt pairs.

Reads the canonical acon prompt template, splices in a strategy block
just before the "Now generate code to solve the actual task" turn,
and writes the result under
``acon/experiments/appworld/prompts/_motivation_v2/<strategy>/``.

Run this once after editing STRATEGY_DESIGN.md or the canonical
strategy texts below. The launcher (``scripts/run_appworld_strategy.py``)
calls this implicitly so manual invocation is rare.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

_ACON_APPWORLD = Path("/workspace/acon/experiments/appworld")
_ACON_TEMPLATE_JSON = _ACON_APPWORLD / "prompts" / "prompts_v1.json"
_ACON_TEMPLATE_JINJA = _ACON_APPWORLD / "prompts" / "prompt_v1.jinja"

# Strategy prompts will be written under acon's tree so relative
# paths in the prompts JSON resolve. We DO NOT modify the canonical
# acon files.
_OUT_ROOT = _ACON_APPWORLD / "prompts" / "_motivation_v2"


# --------------------------------------------------------------------------
# Canonical strategy texts (mirrors STRATEGY_DESIGN.md §"The three strategy
# texts"). If you edit one, edit both; CI / tests can diff them.
# --------------------------------------------------------------------------

STRATEGY_TEXTS: Dict[str, str] = {
    "direct": (
        "**STRATEGY: DIRECT (minimum-API, answer-first)**\n"
        "\n"
        "You MUST solve the task with the minimum number of API calls. As "
        "soon as you have enough information to answer with reasonable "
        "confidence, immediately call "
        "`apis.supervisor.complete_task(answer=<answer>)`. DO NOT verify "
        "the answer through a second source. DO NOT enumerate apps via "
        "`apis.api_docs.show_app_descriptions()` if you can already infer "
        "which app to use from the task. DO NOT list extra data that is "
        "not strictly required to compute the answer. Brevity is the "
        "policy.\n"
    ),
    "verify": (
        "**STRATEGY: VERIFY (mandatory cross-validation)**\n"
        "\n"
        "Before returning your final answer, you MUST cross-validate it "
        "through at least one independent code path. For example, if you "
        "computed the answer by aggregating from one library (songs), "
        "also verify by aggregating from a different library (albums or "
        "playlists) when feasible, OR re-fetch the same record by a "
        "different identifier (e.g. song-id vs title), OR call a related "
        "list endpoint and confirm the result is consistent. The "
        "cross-validation step is mandatory; do not skip it. Only call "
        "`apis.supervisor.complete_task` after the second path has "
        "confirmed the answer.\n"
    ),
    "explore": (
        "**STRATEGY: EXPLORE (survey-then-answer)**\n"
        "\n"
        "Before computing the answer, build a comprehensive understanding "
        "of the available data. Step 1: ALWAYS list the available apps "
        "with `apis.api_docs.show_app_descriptions()`. Step 2: enumerate "
        "the key APIs of the most relevant app via "
        "`apis.api_docs.show_api_descriptions(app_name=<app>)`. Step 3: "
        "survey the user state via list-style endpoints (e.g. libraries, "
        "recent items, profiles). Only AFTER you have a broad picture of "
        "the user's state should you compute and return the answer. Use "
        "this exploration phase even if you think you already know the "
        "answer.\n"
    ),
}


# --------------------------------------------------------------------------
# Splice point — line containing this exact substring in the canonical
# jinja marks where we insert the strategy block. The strategy block is
# inserted as a NEW USER turn between the 17 disclaimers and the
# "Using these APIs, now generate code to solve the actual task:" turn.
# --------------------------------------------------------------------------

_SPLICE_MARKER = "Using these APIs, now generate code to solve the actual task:"


def _build_strategy_jinja(canonical_jinja: str, strategy_text: str) -> str:
    if _SPLICE_MARKER not in canonical_jinja:
        raise RuntimeError(
            f"Splice marker {_SPLICE_MARKER!r} not found in canonical jinja "
            f"({_ACON_TEMPLATE_JINJA}). Has acon updated its prompt format?"
        )
    block = (
        "USER:\n"
        f"{strategy_text.rstrip()}\n"
        "\n"
        "USER:\n"
    )
    # Insert the block immediately BEFORE the splice marker line. We
    # split on the start of that line, which is preceded by `USER:\n`.
    head_marker = f"USER:\n{_SPLICE_MARKER}"
    if head_marker not in canonical_jinja:
        # The marker is preceded by a USER: line; this should be true
        # for the official acon template. Defensive fallback:
        head, tail = canonical_jinja.split(_SPLICE_MARKER, 1)
        return head + strategy_text + "\n\nUSER:\n" + _SPLICE_MARKER + tail
    head, tail = canonical_jinja.split(head_marker, 1)
    return head + block + head_marker[len("USER:\n"):] + tail


def main() -> None:
    if not _ACON_TEMPLATE_JSON.exists():
        sys.exit(f"Missing canonical acon prompt JSON: {_ACON_TEMPLATE_JSON}")
    if not _ACON_TEMPLATE_JINJA.exists():
        sys.exit(f"Missing canonical acon jinja: {_ACON_TEMPLATE_JINJA}")

    canonical_json = json.loads(_ACON_TEMPLATE_JSON.read_text(encoding="utf-8"))
    canonical_jinja = _ACON_TEMPLATE_JINJA.read_text(encoding="utf-8")

    _OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for name, strat_text in STRATEGY_TEXTS.items():
        out_dir = _OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_jinja = _build_strategy_jinja(canonical_jinja, strat_text)
        out_jinja_path = out_dir / f"prompt_{name}.jinja"
        out_jinja_path.write_text(out_jinja, encoding="utf-8")

        # Build the prompts JSON pointing at the new jinja. We keep
        # `system_message` identical and only swap `main_prompt_template`.
        new_json = dict(canonical_json)
        # The agent treats `main_prompt_template` as a path relative to
        # the experiments/appworld working dir.
        rel = out_jinja_path.relative_to(_ACON_APPWORLD)
        new_json["main_prompt_template"] = f"./{rel}"
        out_json_path = out_dir / f"prompts_{name}.json"
        out_json_path.write_text(json.dumps(new_json, indent=2), encoding="utf-8")

        print(f"  built {name:<8s}  jinja: {out_jinja_path.relative_to(_ACON_APPWORLD)}  json: {out_json_path.relative_to(_ACON_APPWORLD)}")

    print(f"\nMaterialised {len(STRATEGY_TEXTS)} strategies under {_OUT_ROOT}")
    print("Use any of them via:")
    print("  python scripts/run_appworld_strategy.py --strategy direct --task_id <id>")


if __name__ == "__main__":
    main()
