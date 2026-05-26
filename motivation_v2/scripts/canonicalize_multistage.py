"""Experiment B — Multi-stage real role-agent artifact control (canonical output).

Reads the multi-stage pipeline artefacts produced by run_multi_stage_role.py
and writes:

    outputs/motivation/multistage_role_raw.jsonl    one row per (task, role-pair)
    outputs/motivation/multistage_role_summary.csv  per-pair aggregate stats
    figures/motivation/multistage_role_heatmap.pdf  4x4 cross-role overlap heatmap

Spec reference: experiment_modification.md §6 (Experiment B).

The two pure-agent outputs (planner plan, verifier evidence) are
loaded directly from the agent's JSON output. Tool/code role memories
are derived from the executor's trajectory using the same logic as
analyze_multi_stage_overlap.py.

This experiment closes the projection-vs-agent critique:
    "Cross-role overlap is low only because deterministic extractors
     emit disjoint slices."
By including planner and verifier outputs (independent agent calls),
we show the orthogonality persists when memories come from real
agent runs, not just slices of the same trajectory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v2.canonical_io import (
    DistribStats,
    OUTPUTS_DIR,
    FIGURES_DIR,
    ensure_dirs,
    entity_tokens,
    jaccard,
    log_failure,
    log_run_meta,
    write_csv,
    write_jsonl,
)


_EXEC_DEFAULT = "MiniMaxAI/MiniMax-M2.5"
_ROLE_NAMES = ["plan", "tool", "code", "verify"]
_API_RE = re.compile(r"apis\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*\((.*?)\)", re.DOTALL)
_CODE_PATTERN_RE = re.compile(
    r"(?:^|\n)\s*(for\s+\w+\s+in\s+|while\s+|if\s+|try:|except\s+|def\s+|"
    r"\[\s*\w.*?for\s+\w+|max\(|min\(|sorted\(|filter\(|map\()",
)


# ----------------------------------------------------------------------
# Per-role text extractors
# ----------------------------------------------------------------------

def _load_plan_text(plan_dir: Path) -> str:
    p = plan_dir / "plan.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    sub_goals = d.get("sub_goals", [])
    return "\n".join(sub_goals) if sub_goals else d.get("plan_text", "")


def _load_verifier_text(plan_dir: Path) -> str:
    p = plan_dir / "verifier.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    evidence = d.get("evidence", [])
    return "\n".join(evidence)


def _load_executor_apis(plan_dir: Path) -> str:
    p = plan_dir / "appworld_trajectory.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    parts: List[str] = []
    for s in d.get("trajectory", []):
        for m in _API_RE.finditer(s.get("action") or ""):
            parts.append(f"{m.group(1)}.{m.group(2)}({m.group(3)[:100]})")
        out = (s.get("output") or "").strip()
        if out:
            parts.append(out[:200])
    return "\n".join(parts)


def _load_executor_code(plan_dir: Path) -> str:
    p = plan_dir / "appworld_trajectory.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    parts: List[str] = []
    for s in d.get("trajectory", []):
        action = s.get("action") or ""
        if _CODE_PATTERN_RE.search(action):
            parts.append(action[:300])
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _plot_heatmap(per_pair_stats: Dict[str, dict], out_path: Path, n_tasks: int):
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.full((4, 4), np.nan)
    for i, r1 in enumerate(_ROLE_NAMES):
        for j, r2 in enumerate(_ROLE_NAMES):
            if i == j:
                matrix[i, j] = 1.0
            else:
                key = f"{r1}_{r2}" if f"{r1}_{r2}" in per_pair_stats else f"{r2}_{r1}"
                if key in per_pair_stats:
                    matrix[i, j] = per_pair_stats[key]["mean"]

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(_ROLE_NAMES)
    ax.set_yticklabels(_ROLE_NAMES)
    for i in range(4):
        for j in range(4):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            text = "—" if i == j else f"{v:.3f}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if v < 0.5 else "black",
                    fontsize=11)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Entity-token Jaccard")
    ax.set_title(
        f"Multi-stage cross-role memory overlap "
        f"(n={n_tasks} tasks; planner / verifier are independent agent outputs)"
    )
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="mv2_multi_stage_pilot")
    parser.add_argument("--executor", default=_EXEC_DEFAULT)
    args = parser.parse_args()

    ensure_dirs()
    log_run_meta("B_multistage", executor=args.executor, seed=42)

    base = Path(
        f"/workspace/acon/experiments/appworld/outputs/"
        f"MiniMaxAI_MiniMax-M2.5_{args.tag}/train"
    )
    if not base.exists():
        sys.exit(f"No outputs at {base}")

    task_dirs = sorted(base.glob("task_*"))
    print(f"[B] {len(task_dirs)} task dirs under {base}")

    role_tokens: Dict[str, Dict[str, Set[str]]] = {r: {} for r in _ROLE_NAMES}
    valid_tasks: List[str] = []
    n_failures = 0

    for d in task_dirs:
        tid = d.name.replace("task_", "")
        plan_t = _load_plan_text(d)
        verify_t = _load_verifier_text(d)
        tool_t = _load_executor_apis(d)
        code_t = _load_executor_code(d)
        missing = [
            name for name, t in
            [("plan", plan_t), ("verify", verify_t), ("tool", tool_t)]
            if not t
        ]
        if missing:
            log_failure(
                "B_multistage",
                task_id=tid,
                failure_type="missing_output",
                error_message=f"missing artefacts: {missing}",
                executor=args.executor,
            )
            n_failures += 1
            continue
        valid_tasks.append(tid)
        # Multi-stage memories don't have role brackets, so we don't
        # need to strip prefixes — but using the same entity_tokens
        # keeps the metric definition consistent with Exp A.
        role_tokens["plan"][tid]   = entity_tokens(plan_t,   strip_bracket_prefix=False)
        role_tokens["verify"][tid] = entity_tokens(verify_t, strip_bracket_prefix=False)
        role_tokens["tool"][tid]   = entity_tokens(tool_t,   strip_bracket_prefix=False)
        role_tokens["code"][tid]   = entity_tokens(code_t,   strip_bracket_prefix=False)

    n_tasks = len(valid_tasks)
    print(f"[B] {n_tasks} valid tasks ({n_failures} failed/missing)")

    rows: List[dict] = []
    for r1, r2 in combinations(_ROLE_NAMES, 2):
        pair_label = f"{r1}_{r2}"
        for tid in valid_tasks:
            a = role_tokens[r1][tid]
            b = role_tokens[r2][tid]
            rows.append({
                "experiment": "B_multistage",
                "executor": args.executor,
                "task_id": tid,
                "role_pair": pair_label,
                "role_a": r1,
                "role_b": r2,
                "jaccard_token": jaccard(a, b),
                "n_a_tokens": len(a),
                "n_b_tokens": len(b),
            })

    # Token-set sizes per role (sanity)
    sizes_summary: Dict[str, dict] = {}
    for role in _ROLE_NAMES:
        sizes = [len(role_tokens[role][tid]) for tid in valid_tasks]
        s = DistribStats.from_values(sizes)
        sizes_summary[role] = {
            "experiment": "B_multistage",
            "executor": args.executor,
            "metric": "n_entity_tokens",
            "role": role,
            "n_tasks": n_tasks,
            "mean": round(s.mean, 2),
            "std": round(s.std, 2),
            "median": round(s.median, 1),
            "min": round(s.min_, 1),
            "max": round(s.max_, 1),
        }

    # Per-pair aggregate
    per_pair: Dict[str, dict] = {}
    for r1, r2 in combinations(_ROLE_NAMES, 2):
        pair_label = f"{r1}_{r2}"
        xs = [r["jaccard_token"] for r in rows if r["role_pair"] == pair_label]
        s = DistribStats.from_values(xs)
        per_pair[pair_label] = {
            "experiment": "B_multistage",
            "executor": args.executor,
            "metric": "cross_role_jaccard_token",
            "role_pair": pair_label,
            "n_tasks": s.n,
            "mean":   round(s.mean,   4),
            "std":    round(s.std,    4),
            "median": round(s.median, 4),
            "min":    round(s.min_,   4),
            "max":    round(s.max_,   4),
        }

    # Overall mean
    all_xs = [r["jaccard_token"] for r in rows]
    s_all = DistribStats.from_values(all_xs)
    overall_row = {
        "experiment": "B_multistage",
        "executor": args.executor,
        "metric": "cross_role_jaccard_token",
        "role_pair": "OVERALL",
        "n_tasks": n_tasks,
        "n_pair_obs": s_all.n,
        "mean":   round(s_all.mean,   4),
        "std":    round(s_all.std,    4),
        "median": round(s_all.median, 4),
        "min":    round(s_all.min_,   4),
        "max":    round(s_all.max_,   4),
    }

    raw_path = OUTPUTS_DIR / "multistage_role_raw.jsonl"
    n_raw = write_jsonl(raw_path, rows)
    print(f"[B] wrote {n_raw} raw pair rows -> {raw_path}")

    summary_rows = [overall_row] + list(per_pair.values()) + list(sizes_summary.values())
    sum_path = OUTPUTS_DIR / "multistage_role_summary.csv"
    n_sum = write_csv(sum_path, summary_rows)
    print(f"[B] wrote {n_sum} summary rows -> {sum_path}")

    # Pretty print
    print()
    print(f"=== Cross-role Jaccard (real agent outputs, n={n_tasks} tasks) ===")
    print(f"  {'pair':>14} | {'mean':>6} | {'median':>6} | {'min':>6} | {'max':>6}")
    print("-" * 60)
    for pair_label, p in per_pair.items():
        print(f"  {pair_label:>14} | {p['mean']:>6.3f} | {p['median']:>6.3f} | "
              f"{p['min']:>6.3f} | {p['max']:>6.3f}")
    print(f"  {'OVERALL':>14} | {overall_row['mean']:>6.3f} | {overall_row['median']:>6.3f} | "
          f"{overall_row['min']:>6.3f} | {overall_row['max']:>6.3f}")

    print()
    print("=== Token-set size per role ===")
    for role in _ROLE_NAMES:
        s = sizes_summary[role]
        print(f"  {role:>10}: mean={s['mean']:.1f}, median={s['median']:.0f}, n={s['n_tasks']}")

    plan_verify = per_pair.get("plan_verify", {})
    print()
    print(f"=== Headline (plan ↔ verify, both INDEPENDENT agent outputs) ===")
    if plan_verify:
        print(f"  Jaccard mean  : {plan_verify['mean']:.3f}")
        print(f"  Jaccard median: {plan_verify['median']:.3f}")
        print(f"  range         : [{plan_verify['min']:.3f}, {plan_verify['max']:.3f}]")

    _plot_heatmap(per_pair, FIGURES_DIR / "multistage_role_heatmap.pdf", n_tasks)

    print()
    print("[B] Done.")


if __name__ == "__main__":
    main()
