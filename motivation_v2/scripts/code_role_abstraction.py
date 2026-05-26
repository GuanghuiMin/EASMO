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
    r"break|continue|raise|with|not|and|or|is|lambda)\b"
)
_CODE_AGGREGATE_RE = re.compile(
    r"\b(?:max|min|sorted|filter|map|sum|len|any|all|enumerate|zip|"
    r"reversed|range)\s*\("
)
_LIST_COMP_RE = re.compile(r"\[\s*[A-Za-z_].*?\s+for\s+\w+\s+in\s+.*?\]")
_VAR_ASSIGN_RE = re.compile(r"^[a-zA-Z_][\w\.]*\s*=\s*[^=]")
_PRINT_OR_COMMENT_RE = re.compile(r"^\s*(?:print\s*\(|#)")
_API_CALL_FULL_RE = re.compile(
    r"apis\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\s*\((.*?)\)", re.DOTALL
)
_API_CALL_SHORT_RE = re.compile(
    r"apis\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+"
)
_KW_ARG_RE = re.compile(
    r"\b[a-zA-Z_]\w*\s*=\s*['\"][^'\"]+['\"]"
)
_BIG_DIGIT_RUN_RE = re.compile(r"\b\d{4,}\b")  # IDs / timestamps
_PATH_RE = re.compile(r"/[a-zA-Z0-9_]+(?:/[a-zA-Z0-9_{}]+)+")
_TOKEN_LIKE_RE = re.compile(r"\b[A-Za-z0-9]{20,}\b")  # access_token-ish
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w-]+")
_QUOTE_LITERAL_RE = re.compile(r"['\"][^'\"]{4,}['\"]")  # any string literal


def _is_concrete_api_fact(s: str) -> bool:
    """Line contains a concrete API call/endpoint AND concrete arg
    values (kw="value", digits-as-ID, access_token-like string,
    email, or path)."""
    has_api_ref = bool(_API_CALL_FULL_RE.search(s)) or bool(_PATH_RE.search(s))
    if not has_api_ref:
        return False
    # Look for concrete arg values inside the API call or the surrounding
    # line.
    has_kw_args = bool(_KW_ARG_RE.search(s))
    has_id      = bool(_BIG_DIGIT_RUN_RE.search(s))
    has_token   = bool(_TOKEN_LIKE_RE.search(s))
    has_email   = bool(_EMAIL_RE.search(s))
    return has_kw_args or has_id or has_token or has_email


def _is_concrete_data_fact(s: str) -> bool:
    """Variable assignment with a concrete literal (access_token=..,
    page_index=0) — not API call shaped, but still a fact leak."""
    if _VAR_ASSIGN_RE.match(s) is None:
        return False
    has_token = bool(_TOKEN_LIKE_RE.search(s))
    has_email = bool(_EMAIL_RE.search(s))
    has_id    = bool(_BIG_DIGIT_RUN_RE.search(s))
    has_quote = bool(_QUOTE_LITERAL_RE.search(s))
    return has_token or has_email or has_id or has_quote


def _is_code_pattern(s: str) -> bool:
    """Has explicit Python control structure or coding idiom."""
    if _CODE_KW_RE.search(s) or _CODE_AGGREGATE_RE.search(s) or _LIST_COMP_RE.search(s):
        return True
    if _PRINT_OR_COMMENT_RE.search(s):
        return True
    if _VAR_ASSIGN_RE.match(s) is not None:
        return True
    return False


def classify_line(line: str) -> str:
    """Returns 'api_fact' / 'code_pattern' / 'other'.

    Priority: api_fact (concrete API call with args / data leak) >
              code_pattern (control flow / aggregate / variable assignment
              / print / comment) >
              other.
    """
    s = line.strip()
    if not s:
        return "other"
    if _is_concrete_api_fact(s) or _is_concrete_data_fact(s):
        return "api_fact"
    if _is_code_pattern(s):
        return "code_pattern"
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
    code_pattern / other lines."""
    bucket: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for r in per_line:
        bucket[(r["compressor"], r["budget_tokens"])].append(r["line_class"])
    out: List[dict] = []
    for (cond, B), classes in sorted(bucket.items()):
        n = len(classes)
        api = sum(1 for c in classes if c == "api_fact")
        cp = sum(1 for c in classes if c == "code_pattern")
        ot = sum(1 for c in classes if c == "other")
        out.append({
            "experiment": "D_prompted_code_abstraction",
            "compressor": cond,
            "budget": B,
            "n_lines": n,
            "share_api_fact":     round(api / n, 4) if n else 0.0,
            "share_code_pattern": round(cp / n, 4) if n else 0.0,
            "share_other":        round(ot / n, 4) if n else 0.0,
            "n_api_fact": api,
            "n_code_pattern": cp,
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
    cp_vals  = [r["share_code_pattern"] for r in rows]
    ot_vals  = [r["share_other"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(conds))
    ax.bar(x, api_vals, label="API fact (concrete args/IDs/tokens)",
           color="#d62728", edgecolor="black")
    ax.bar(x, cp_vals,  bottom=api_vals,
           label="Code pattern (control-flow / assignment / print / comment)",
           color="#2ca02c", edgecolor="black")
    ax.bar(x, ot_vals,  bottom=[a + c for a, c in zip(api_vals, cp_vals)],
           label="Other (prose / unrecognized)", color="#7f7f7f", edgecolor="black")
    for i, (a, c) in enumerate(zip(api_vals, cp_vals)):
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


def _build_oracle_code_records(strategy: str, tag: str, budgets: List[int]) -> List[dict]:
    """Build oracle (m_code) memory records matching the prompted JSONL
    schema, so we can run the same line-classifier on them and contrast
    'oracle keeps control-flow' vs 'prompted keeps API facts'."""
    sys.path.insert(0, str(_REPO))
    from motivation_v2.data import successful_trajectories
    from motivation_v2.role_memory import m_code

    glob = (
        f"/workspace/acon/experiments/appworld/outputs/"
        f"MiniMaxAI_MiniMax-M2.5_{tag}_{strategy}/train/task_*"
    )
    trajs = successful_trajectories(experiments_glob=glob)
    out: List[dict] = []
    for t in trajs:
        for B in budgets:
            em = m_code(t, B)
            text = "\n".join(u.text for u in em.units)
            out.append({
                "task_id": t.task_id,
                "policy_role": "code",
                "compressor": "oracle_m_code",
                "budget_tokens": B,
                "memory_text": text,
                "n_units": em.n_units,
                "n_tokens": em.n_tokens,
            })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompted_jsonl",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_pilot/prompted_memories.jsonl",
                        help="prompted_task_role memories")
    parser.add_argument("--prompted_jsonl_extra", action="append", default=[],
                        help="<condition>:<path> additional prompted JSONLs")
    parser.add_argument("--include_oracle", action="store_true", default=True)
    parser.add_argument("--strategy", default="direct")
    parser.add_argument("--tag", default="mv2_pilot")
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--budget_for_plot", type=int, default=512)
    args = parser.parse_args()

    ensure_dirs()
    log_run_meta("D_code_abstraction")

    all_rows: List[dict] = []
    base = _load_prompted_jsonl(Path(args.prompted_jsonl))
    for r in base:
        # Legacy file uses compressor='prompted_<role>'; re-tag uniformly.
        if "compressor" in r and r["compressor"].startswith("prompted_") \
                and r["compressor"] != "prompted_task_role":
            r["compressor"] = "prompted_task_role"
        elif "compressor" not in r:
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

    if args.include_oracle:
        oracle_rows = _build_oracle_code_records(args.strategy, args.tag, args.budgets)
        all_rows.extend(oracle_rows)
        print(f"[code_abs] built {len(oracle_rows)} oracle (m_code) rows for comparison")

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
    print(f"  {'compressor':>22} | {'B':>4} | {'lines':>5} | {'api%':>5} | {'code%':>5} | {'other%':>6}")
    print("-" * 70)
    for r in summary:
        print(f"  {r['compressor']:>22} | {r['budget']:>4} | {r['n_lines']:>5} | "
              f"{r['share_api_fact']*100:>4.0f}% | "
              f"{r['share_code_pattern']*100:>4.0f}% | "
              f"{r['share_other']*100:>5.0f}%")

    _plot_abstraction_bar(summary, FIGURES_DIR / "code_abstraction_share.pdf",
                          budget=args.budget_for_plot)

    print()
    print("[code_abs] Done.")


if __name__ == "__main__":
    main()
