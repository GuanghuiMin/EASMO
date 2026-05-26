"""Code-role abstraction diagnostic (Sprint 3 / Exp D §8.8).

Spec quote:
    For code role, additionally report:
        % prompted memory units that contain API facts
        % prompted memory units that contain control-flow patterns
        recall against code-pattern diagnostic memory
    The key expected failure mode is:
        The LLM keeps API facts even when asked to compress for a
        coding agent, instead of selecting reusable control-flow
        abstractions.

This script reads prompted-memory JSONL(s) and the ``m_code`` oracle,
then for each prompted-code memory cell classifies each line as:

  * api_fact            — contains apis.<app>.<fn>(...) with concrete args, OR
                          contains an arg-style key=value, OR contains an ID
                          / token / app endpoint string with no code keyword
  * control_flow        — contains for/while/if/elif/else/try/except/def/list-
                          comprehension or aggregate fn (max/min/sorted/...)
                          and at least one structural Python token
  * other               — neither (rare)

Outputs:

  outputs/motivation/code_abstraction_per_line.jsonl
  outputs/motivation/code_abstraction_summary.csv

Plus appends per-condition rows to ``prompted_compression_summary.csv``
with metric ``code_abstraction_share``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v2.canonical_io import (
    DistribStats,
    OUTPUTS_DIR,
    FIGURES_DIR,
    ensure_dirs,
    log_run_meta,
    write_csv,
    write_jsonl,
)


_CODE_KW_RE = re.compile(
    r"\b(?:for|while|if|elif|else|try|except|finally|def|return|yield|"
    r"break|continue|raise|with|in|not|and|or|is|lambda)\b"
)
_CODE_AGGREGATE_RE = re.compile(
    r"\b(?:max|min|sorted|filter|map|sum|len|any|all|enumerate|zip|"
    r"reversed|range)\s*\("
)
_LIST_COMP_RE = re.compile(r"\[\s*[A-Za-z_].*?\s+for\s+\w+\s+in\s+.*?\]")
_API_CALL_FULL_RE = re.compile(
    r"apis\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\s*\(.*?\)", re.DOTALL
)
_API_CALL_SHORT_RE = re.compile(
    r"apis\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+"
)
_KW_ARG_RE = re.compile(
    r"\b[a-zA-Z_]\w*\s*=\s*['\"][^'\"]+['\"]"
)
_DIGIT_RUN_RE = re.compile(r"\b\d{2,}\b")
_PATH_RE = re.compile(r"/[a-zA-Z0-9_]+(?:/[a-zA-Z0-9_{}]+)+")


def classify_line(line: str) -> str:
    """Returns 'api_fact' / 'control_flow' / 'other'."""
    s = line.strip()
    if not s:
        return "other"

    has_code_kw = bool(_CODE_KW_RE.search(s))
    has_code_aggr = bool(_CODE_AGGREGATE_RE.search(s)) or bool(_LIST_COMP_RE.search(s))
    has_full_api = bool(_API_CALL_FULL_RE.search(s))
    has_short_api = bool(_API_CALL_SHORT_RE.search(s))
    has_kw_args = bool(_KW_ARG_RE.search(s))
    has_digits = bool(_DIGIT_RUN_RE.search(s))
    has_path = bool(_PATH_RE.search(s))

    # Primary signals for control-flow patterns:
    #   - python keyword-style structure (for/if/etc.)
    #   - or aggregate / list comprehension
    # If the line ALSO has a concrete API call with args, we count it
    # as control_flow only if the API call uses a `<args>` placeholder
    # OR the line is a 'for' / 'if' wrapping the call (i.e. structure
    # dominates).
    if has_code_aggr or _LIST_COMP_RE.search(s):
        return "control_flow"
    if has_code_kw and not (has_kw_args and has_digits):
        # 'for x in api_fn(...)' is structure dominant; 'login(username="...", password="...")'
        # is fact dominant. Use kw_args + digits as fact signal.
        return "control_flow"

    # API fact signals:
    #   - full apis.<app>.<fn>(...) with concrete args (kw_args / digits)
    #   - bare endpoint paths (/spotify/library/...)
    #   - kw-arg style strings without surrounding code keywords
    if has_full_api or has_short_api or has_path or has_kw_args or has_digits:
        return "api_fact"

    return "other"


# ----------------------------------------------------------------------
# Aggregator
# ----------------------------------------------------------------------


def _classify_records(rows: List[dict]) -> List[dict]:
    """For each prompted-code row (policy_role == 'code') classify each
    non-empty line and emit per-line records."""
    out: List[dict] = []
    for r in rows:
        if r.get("policy_role") != "code":
            continue
        if "memory_text" not in r:
            continue
        cond = r["compressor"]
        for ln in (r.get("memory_text") or "").splitlines():
            if not ln.strip():
                continue
            out.append({
                "task_id": r["task_id"],
                "compressor": cond,
                "budget_tokens": r["budget_tokens"],
                "line_text": ln,
                "line_class": classify_line(ln),
            })
    return out


def _summary_share(per_line: List[dict]) -> List[dict]:
    """For each (compressor, budget) compute share of api_fact /
    control_flow / other lines."""
    bucket: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for r in per_line:
        bucket[(r["compressor"], r["budget_tokens"])].append(r["line_class"])
    out: List[dict] = []
    for (cond, B), classes in sorted(bucket.items()):
        n = len(classes)
        api = sum(1 for c in classes if c == "api_fact")
        cf = sum(1 for c in classes if c == "control_flow")
        ot = sum(1 for c in classes if c == "other")
        out.append({
            "experiment": "D_prompted_code_abstraction",
            "compressor": cond,
            "budget": B,
            "n_lines": n,
            "share_api_fact":     round(api / n, 4) if n else 0.0,
            "share_control_flow": round(cf / n, 4) if n else 0.0,
            "share_other":        round(ot / n, 4) if n else 0.0,
            "n_api_fact": api,
            "n_control_flow": cf,
            "n_other": ot,
        })
    return out


def _plot_abstraction_bar(summary: List[dict], out_path: Path, budget: int = 512):
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in summary if r["budget"] == budget]
    if not rows:
        print(f"[plot] no abstraction summary; skipping {out_path}")
        return
    rows.sort(key=lambda r: r["compressor"])
    conds = [r["compressor"] for r in rows]
    api_vals = [r["share_api_fact"] for r in rows]
    cf_vals  = [r["share_control_flow"] for r in rows]
    ot_vals  = [r["share_other"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(conds))
    ax.bar(x, api_vals, label="API fact",      color="#d62728", edgecolor="black")
    ax.bar(x, cf_vals,  bottom=api_vals,
           label="Control-flow pattern", color="#2ca02c", edgecolor="black")
    ax.bar(x, ot_vals,  bottom=[a + c for a, c in zip(api_vals, cf_vals)],
           label="Other",          color="#7f7f7f", edgecolor="black")
    for i, (a, c) in enumerate(zip(api_vals, cf_vals)):
        ax.text(i, a / 2, f"{a*100:.0f}%", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
        ax.text(i, a + c / 2, f"{c*100:.0f}%", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Share of lines (code-role memory)")
    ax.set_title(f"Code-role memory composition at B={budget}")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def _load_prompted_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            out.append(r)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompted_jsonl",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_pilot/prompted_memories.jsonl",
                        help="prompted_task_role memories")
    parser.add_argument("--prompted_jsonl_extra", action="append", default=[],
                        help="<condition>:<path> additional prompted JSONLs")
    parser.add_argument("--budget_for_plot", type=int, default=512)
    args = parser.parse_args()

    ensure_dirs()
    log_run_meta("D_code_abstraction")

    all_rows: List[dict] = []
    base = _load_prompted_jsonl(Path(args.prompted_jsonl))
    # Tag the base file as prompted_task_role.
    for r in base:
        if "compressor" in r and not r["compressor"].startswith("prompted_"):
            r["compressor"] = "prompted_task_role"
        elif "compressor" not in r or r["compressor"].startswith("prompted_") and r["compressor"] != "prompted_task_role":
            # legacy file uses "prompted_<role>" — re-tag uniformly.
            r["compressor"] = "prompted_task_role"
    all_rows.extend(base)
    print(f"[code_abs] loaded {len(base)} rows from {args.prompted_jsonl} (tagged prompted_task_role)")

    for spec in args.prompted_jsonl_extra:
        cond, path = spec.split(":", 1)
        rows = _load_prompted_jsonl(Path(path))
        for r in rows:
            r["compressor"] = cond
        all_rows.extend(rows)
        print(f"[code_abs] loaded {len(rows)} rows from {path} as {cond}")

    per_line = _classify_records(all_rows)
    print(f"[code_abs] classified {len(per_line)} code-role lines")

    pl_path = OUTPUTS_DIR / "code_abstraction_per_line.jsonl"
    write_jsonl(pl_path, per_line)
    print(f"[code_abs] wrote {len(per_line)} rows -> {pl_path}")

    summary = _summary_share(per_line)
    sum_path = OUTPUTS_DIR / "code_abstraction_summary.csv"
    write_csv(sum_path, summary)
    print(f"[code_abs] wrote {len(summary)} summary rows -> {sum_path}")

    print()
    print("=== Code-role memory composition ===")
    print(f"  {'compressor':>22} | {'B':>4} | {'lines':>5} | {'api%':>5} | {'ctrlflow%':>9} | {'other%':>6}")
    print("-" * 70)
    for r in summary:
        print(f"  {r['compressor']:>22} | {r['budget']:>4} | {r['n_lines']:>5} | "
              f"{r['share_api_fact']*100:>4.0f}% | "
              f"{r['share_control_flow']*100:>8.0f}% | "
              f"{r['share_other']*100:>5.0f}%")

    _plot_abstraction_bar(summary, FIGURES_DIR / "code_abstraction_share.pdf",
                          budget=args.budget_for_plot)

    print()
    print("[code_abs] Done.")


if __name__ == "__main__":
    main()
