"""Experiment D — Prompted compression gap (canonical output).

Reads prompted-memory cells (built by build_prompted_memories*.py) and
the role-projected oracle (built on the fly from successful
trajectories). Writes:

    outputs/motivation/prompted_compression_raw.jsonl     one row per cell
    outputs/motivation/prompted_compression_summary.csv   per-(role, budget, condition) stats
    figures/motivation/prompted_vs_reference_heatmap.pdf  cross-role heatmaps
    figures/motivation/prompted_role_recall.pdf           same-role recall bars

Spec reference: experiment_modification.md §8 (Experiment D).

Sprint 1 only canonicalises the existing prompted_task_role variant.
Sprint 3 will add prompted_generic / prompted_task / prompted_role
ablation variants and the code-role abstraction diagnostic.

The canonical condition labels (spec §8.3):
    prompted_generic     : compress generally (no task, no role)
    prompted_task        : compress for the downstream task (no role)
    prompted_role        : compress for the target role (no task)
    prompted_task_role   : compress for both task AND role
    prompted_extractive  : ask LLM to select unit IDs (skipped this round)

Existing prompts (motivation_v2/prompted_memory.py:_PROMPT_TEMPLATES)
include both task and role description, so the existing
prompted_<role> cells map to ``prompted_task_role``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v2.canonical_io import (
    DistribStats,
    OUTPUTS_DIR,
    FIGURES_DIR,
    ensure_dirs,
    entity_tokens,
    jaccard,
    log_run_meta,
    write_csv,
    write_jsonl,
)
from motivation_v2.data import successful_trajectories
from motivation_v2.role_memory import ROLE_BUILDERS


_EXEC_DEFAULT = "MiniMaxAI/MiniMax-M2.5"
_BUDGETS_DEFAULT = (128, 256, 512, 1024)
_ROLE_NAMES = list(ROLE_BUILDERS.keys())


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

def _load_prompted_jsonl(
    path: Path, condition: str = "prompted_task_role",
) -> Dict[Tuple[str, str, int], Set[str]]:
    """Returns {(task_id, role, budget) -> token_set}.

    Uses entity_tokens with bracket-stripping disabled (prompted memory
    has no role-bracket scaffolding).
    """
    out: Dict[Tuple[str, str, int], Set[str]] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            key = (r["task_id"], r["policy_role"], r["budget_tokens"])
            out[key] = entity_tokens(r.get("memory_text", ""), strip_bracket_prefix=False)
    return out


def _build_oracle(
    strategy: str, tag: str, budgets: Tuple[int, ...],
) -> Tuple[
    Dict[Tuple[str, str, int], Set[str]],
    List[str],
]:
    """Returns (token_sets, all_tids), keyed by (task_id, role, budget)."""
    glob = (
        f"/workspace/acon/experiments/appworld/outputs/"
        f"MiniMaxAI_MiniMax-M2.5_{tag}_{strategy}/train/task_*"
    )
    trajs = successful_trajectories(experiments_glob=glob)
    print(f"[D] oracle trajectories: n={len(trajs)}")
    out: Dict[Tuple[str, str, int], Set[str]] = {}
    for t in trajs:
        for role, builder in ROLE_BUILDERS.items():
            for B in budgets:
                em = builder(t, B)
                joined = "\n".join(u.text for u in em.units)
                out[(t.task_id, role, B)] = entity_tokens(joined)  # bracket-stripped
    return out, [t.task_id for t in trajs]


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _plot_cross_role_heatmaps(
    rows: List[dict], out_path: Path, budget: int = 512,
):
    """Two-panel heatmap: cross-role Jaccard for prompted vs oracle."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Mean per (kind, role_pair, budget)
    bucket_p: Dict[str, List[float]] = defaultdict(list)
    bucket_o: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["budget"] != budget or r["metric"] != "cross_role_jaccard":
            continue
        if r["source"] == "prompted_task_role":
            bucket_p[r["role_pair"]].append(r["jaccard"])
        elif r["source"] == "oracle":
            bucket_o[r["role_pair"]].append(r["jaccard"])

    def _mean_matrix(b: Dict[str, List[float]]):
        m = np.full((4, 4), np.nan)
        for i, r1 in enumerate(_ROLE_NAMES):
            m[i, i] = 1.0
            for j, r2 in enumerate(_ROLE_NAMES):
                if i >= j:
                    continue
                key = f"{r1}_{r2}"
                if key in b and b[key]:
                    v = sum(b[key]) / len(b[key])
                    m[i, j] = v
                    m[j, i] = v
        return m

    M_p = _mean_matrix(bucket_p)
    M_o = _mean_matrix(bucket_o)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    for ax, M, title in [
        (axes[0], M_o, f"Oracle (role-projected) cross-role at B={budget}"),
        (axes[1], M_p, f"Prompted (task+role) cross-role at B={budget}"),
    ]:
        im = ax.imshow(M, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(_ROLE_NAMES)
        ax.set_yticklabels(_ROLE_NAMES)
        for i in range(4):
            for j in range(4):
                v = M[i, j]
                if np.isnan(v):
                    continue
                text = "—" if i == j else f"{v:.2f}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=10)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _plot_role_recall(rows: List[dict], out_path: Path, budget: int = 512):
    """Bar chart: per-role recall vs oracle, with code highlighted."""
    import matplotlib.pyplot as plt
    import numpy as np

    bucket: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["budget"] != budget or r["metric"] != "recall_vs_oracle":
            continue
        bucket[r["source"]][r["role"]].append(r["jaccard"])

    sources = sorted(bucket.keys())
    if not sources:
        print(f"[plot] no recall data; skipping {out_path}")
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(_ROLE_NAMES))
    width = max(0.16, 0.8 / max(len(sources), 1))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, src in enumerate(sources):
        ys = []
        ns = []
        for role in _ROLE_NAMES:
            xs = bucket[src][role]
            ys.append(sum(xs) / max(len(xs), 1) if xs else 0.0)
            ns.append(len(xs))
        offset = (i - (len(sources) - 1) / 2) * width
        bars = ax.bar(x + offset, ys, width=width, color=palette[i % len(palette)],
                      label=src, edgecolor="black")
        for j, v in enumerate(ys):
            ax.text(j + offset, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(_ROLE_NAMES)
    ax.set_ylabel("Same-role recall (Jaccard vs oracle)")
    ax.set_title(f"Prompted compression recall vs role-projected oracle at B={budget}")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0.0, max(0.5, ax.get_ylim()[1]))
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _emit_cross_role_rows(
    sets_: Dict[Tuple[str, str, int], Set[str]],
    source: str,
    budgets: Tuple[int, ...],
    tids: List[str],
) -> List[dict]:
    """Pairwise cross-role Jaccard rows."""
    rows: List[dict] = []
    for B in budgets:
        for r1, r2 in combinations(_ROLE_NAMES, 2):
            for tid in tids:
                a = sets_.get((tid, r1, B))
                b = sets_.get((tid, r2, B))
                if a is None or b is None:
                    continue
                rows.append({
                    "experiment": "D_prompted",
                    "executor": _EXEC_DEFAULT,
                    "metric": "cross_role_jaccard",
                    "source": source,
                    "task_id": tid,
                    "role_pair": f"{r1}_{r2}",
                    "budget": B,
                    "jaccard": jaccard(a, b),
                    "n_a": len(a),
                    "n_b": len(b),
                })
    return rows


def _emit_recall_rows(
    prompted: Dict[Tuple[str, str, int], Set[str]],
    oracle: Dict[Tuple[str, str, int], Set[str]],
    source: str,
    budgets: Tuple[int, ...],
    tids: List[str],
) -> List[dict]:
    """Same-role Jaccard between prompted and oracle (recall)."""
    rows: List[dict] = []
    for B in budgets:
        for role in _ROLE_NAMES:
            for tid in tids:
                a = prompted.get((tid, role, B))
                b = oracle.get((tid, role, B))
                if a is None or b is None:
                    continue
                rows.append({
                    "experiment": "D_prompted",
                    "executor": _EXEC_DEFAULT,
                    "metric": "recall_vs_oracle",
                    "source": source,
                    "task_id": tid,
                    "role": role,
                    "budget": B,
                    "jaccard": jaccard(a, b),
                    "n_a": len(a),
                    "n_b": len(b),
                })
    return rows


def _aggregate_cross_role(rows: List[dict]) -> List[dict]:
    bucket: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    for r in rows:
        if r["metric"] != "cross_role_jaccard":
            continue
        bucket[(r["source"], r["role_pair"], r["budget"])].append(r["jaccard"])
    out: List[dict] = []
    for (src, pair, B), xs in sorted(bucket.items()):
        s = DistribStats.from_values(xs)
        out.append({
            "experiment": "D_prompted",
            "executor": _EXEC_DEFAULT,
            "metric": "cross_role_jaccard",
            "source": src,
            "role_pair": pair,
            "budget": B,
            "n_pairs": s.n,
            "mean":   round(s.mean,   4),
            "std":    round(s.std,    4),
            "median": round(s.median, 4),
            "min":    round(s.min_,   4),
            "max":    round(s.max_,   4),
        })
    return out


def _aggregate_recall(rows: List[dict]) -> List[dict]:
    bucket: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    for r in rows:
        if r["metric"] != "recall_vs_oracle":
            continue
        bucket[(r["source"], r["role"], r["budget"])].append(r["jaccard"])
    out: List[dict] = []
    for (src, role, B), xs in sorted(bucket.items()):
        s = DistribStats.from_values(xs)
        out.append({
            "experiment": "D_prompted",
            "executor": _EXEC_DEFAULT,
            "metric": "recall_vs_oracle",
            "source": src,
            "role": role,
            "budget": B,
            "n_pairs": s.n,
            "mean":   round(s.mean,   4),
            "std":    round(s.std,    4),
            "median": round(s.median, 4),
            "min":    round(s.min_,   4),
            "max":    round(s.max_,   4),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompted_jsonl",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_pilot/prompted_memories.jsonl",
                        help="Existing prompted_task_role memories (T2 baseline).")
    parser.add_argument("--prompted_jsonl_extra", action="append", default=[],
                        help="Additional condition:path entries; e.g. "
                             "--prompted_jsonl_extra prompted_generic:/path/file.jsonl")
    parser.add_argument("--strategy", default="direct")
    parser.add_argument("--tag", default="mv2_pilot")
    parser.add_argument("--budgets", nargs="+", type=int, default=list(_BUDGETS_DEFAULT))
    args = parser.parse_args()

    ensure_dirs()
    log_run_meta("D_prompted", executor=_EXEC_DEFAULT, seed=42)

    print(f"[D] loading prompted_task_role from {args.prompted_jsonl}")
    prompted_by_cond: Dict[str, Dict[Tuple[str, str, int], Set[str]]] = {}
    prompted_by_cond["prompted_task_role"] = _load_prompted_jsonl(
        Path(args.prompted_jsonl), condition="prompted_task_role",
    )
    print(f"[D] prompted_task_role: {len(prompted_by_cond['prompted_task_role'])} cells")
    for spec in args.prompted_jsonl_extra:
        cond, path = spec.split(":", 1)
        prompted_by_cond[cond] = _load_prompted_jsonl(Path(path), condition=cond)
        print(f"[D] {cond}: {len(prompted_by_cond[cond])} cells")

    print(f"[D] building oracle (strategy={args.strategy}, tag={args.tag})")
    oracle, all_tids = _build_oracle(args.strategy, args.tag, tuple(args.budgets))

    rows: List[dict] = []
    rows += _emit_cross_role_rows(oracle, "oracle", tuple(args.budgets), all_tids)
    for cond, sets_ in prompted_by_cond.items():
        # Use task IDs that have all 4 role variants at all budgets
        cond_tids = sorted({tid for (tid, _r, _B) in sets_.keys()})
        rows += _emit_cross_role_rows(sets_, cond, tuple(args.budgets), cond_tids)
        rows += _emit_recall_rows(sets_, oracle, cond, tuple(args.budgets), cond_tids)

    raw_path = OUTPUTS_DIR / "prompted_compression_raw.jsonl"
    n_raw = write_jsonl(raw_path, rows)
    print(f"[D] wrote {n_raw} raw rows -> {raw_path}")

    summary_xrole = _aggregate_cross_role(rows)
    summary_recall = _aggregate_recall(rows)
    summary = summary_xrole + summary_recall
    sum_path = OUTPUTS_DIR / "prompted_compression_summary.csv"
    n_sum = write_csv(sum_path, summary)
    print(f"[D] wrote {n_sum} summary rows -> {sum_path}")

    # ---- Pretty-printing ----
    print()
    print("=== Cross-role Jaccard (B=512) ===")
    print(f"{'pair':>14} | {'oracle':>8} | {'prompted_t+r':>13} | {'ratio':>5}")
    print("-" * 50)
    by_key = {(r["source"], r["role_pair"], r["budget"]): r for r in summary_xrole}
    for r1, r2 in combinations(_ROLE_NAMES, 2):
        pair = f"{r1}_{r2}"
        o = by_key.get(("oracle", pair, 512))
        p = by_key.get(("prompted_task_role", pair, 512))
        o_v = o["mean"] if o else 0.0
        p_v = p["mean"] if p else 0.0
        ratio = (p_v / o_v) if o_v > 0 else float("inf")
        ratio_s = "∞" if ratio == float("inf") else f"{ratio:.1f}×"
        print(f"{pair:>14} | {o_v:>8.3f} | {p_v:>13.3f} | {ratio_s:>5}")

    # Mean of pairs
    o_pairs = [r["mean"] for r in summary_xrole
               if r["source"] == "oracle" and r["budget"] == 512]
    p_pairs = [r["mean"] for r in summary_xrole
               if r["source"] == "prompted_task_role" and r["budget"] == 512]
    if o_pairs and p_pairs:
        o_mean = sum(o_pairs) / len(o_pairs)
        p_mean = sum(p_pairs) / len(p_pairs)
        print()
        print(f"  oracle mean (across pairs) at B=512:    {o_mean:.3f}")
        print(f"  prompted mean (across pairs) at B=512:  {p_mean:.3f}")
        print(f"  ratio prompted/oracle:                  {p_mean/max(o_mean, 1e-9):.1f}×")

    print()
    print("=== Same-role recall (Jaccard prompted-vs-oracle, B=512) ===")
    print(f"{'role':>10} | " + "  ".join(f"{src:>20}" for src in sorted(prompted_by_cond.keys())))
    by_recall = {(r["source"], r["role"], r["budget"]): r for r in summary_recall}
    for role in _ROLE_NAMES:
        cells = []
        for src in sorted(prompted_by_cond.keys()):
            r = by_recall.get((src, role, 512))
            cells.append(f"{r['mean']:.3f} (n={r['n_pairs']})" if r else "n/a")
        print(f"{role:>10} | " + "  ".join(f"{c:>20}" for c in cells))

    _plot_cross_role_heatmaps(rows, FIGURES_DIR / "prompted_vs_reference_heatmap.pdf",
                               budget=512)
    _plot_role_recall(rows, FIGURES_DIR / "prompted_role_recall.pdf", budget=512)

    print()
    print("[D] Done.")


if __name__ == "__main__":
    main()
