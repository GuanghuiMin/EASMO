"""T2 analysis: cross-role Jaccard for prompted memory + per-role
recall against the projected oracle.

Two questions:

1. **Cross-role Jaccard for prompted memory** — does the LLM
   produce surface-DIFFERENT memory when conditioned on different
   role descriptions? Predicted: NO (Jaccard high, ≥ 0.5),
   contradicting the m_role oracle's 0.04. This would close T2.

2. **Per-role recall** — for each role X, does the prompted
   memory `m_prompted_X` contain the same evidence units as the
   role-projected oracle `m_X`? Predicted: LOW (Jaccard ≤ 0.2).
   The prompted compressor doesn't recover the right facts.

Both numbers are reported per-budget.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v2.data import successful_trajectories
from motivation_v2.role_memory import ROLE_BUILDERS


import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{4,}")
_STOPWORDS = {
    "this", "that", "with", "from", "have", "will", "been",
    "should", "would", "could", "their", "there", "which",
    "where", "when", "what", "while", "more", "most", "some",
    "into", "about", "after", "before", "needs", "need", "memory",
    "agent", "agents", "context", "trace", "step", "steps",
    "useful", "below", "based", "your", "ours", "your", "task",
    "tasks", "selecting", "selected", "select", "compressed",
    "compress", "calls", "call", "data", "results", "result",
    "returns", "return", "each", "give", "given", "either",
    "first", "second", "third", "true", "false", "these",
    "those", "agent", "agents", "lines", "line", "items",
    "item", "memory", "the", "and", "for", "are", "was", "were",
    "you", "but", "not", "all", "any", "any",
}


def _entity_tokens(text: str) -> Set[str]:
    """Bag-of-significant-tokens. Lowercased alphanumeric tokens of
    length ≥ 4, with frequent stopwords filtered. Robust to paraphrase
    differences between LLM-generated prose and code-formatted memory.
    """
    return {
        t.lower() for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOPWORDS
    }


def _line_keys(memory_text: str) -> Set[str]:
    """Entity-token bag for the whole memory_text."""
    return _entity_tokens(memory_text)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompted_jsonl",
                        default="/workspace/EASMO/motivation_v2/outputs/mv2_pilot/prompted_memories.jsonl")
    parser.add_argument("--strategy", default="direct")
    parser.add_argument("--tag", default="mv2_pilot")
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=[128, 256, 512, 1024])
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    # Load prompted memories.
    prompted: Dict[tuple, Set[str]] = {}
    with open(args.prompted_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            key = (r["task_id"], r["policy_role"], r["budget_tokens"])
            prompted[key] = _line_keys(r.get("memory_text", ""))
    print(f"[prompted_overlap] loaded {len(prompted)} prompted-memory cells")

    # Load projected (oracle) memories from trajectories.
    glob = (
        f"/workspace/acon/experiments/appworld/outputs/"
        f"MiniMaxAI_MiniMax-M2.5_{args.tag}_{args.strategy}/train/task_*"
    )
    trajs = successful_trajectories(experiments_glob=glob)
    print(f"[prompted_overlap] {len(trajs)} oracle trajectories")

    projected: Dict[tuple, Set[str]] = {}
    for t in trajs:
        for role, builder in ROLE_BUILDERS.items():
            for B in args.budgets:
                em = builder(t, B)
                # Aggregate ALL units' text into one entity-token set
                # so this is comparable to the prompted-memory bag.
                joined = "\n".join(u.text for u in em.units)
                projected[(t.task_id, role, B)] = _entity_tokens(joined)

    summary = {"cross_role_prompted": {}, "recall_prompted_vs_oracle": {}}

    # ---- 1. Cross-role Jaccard for prompted memory ----
    print()
    print(f"=== Cross-role Jaccard (PROMPTED memory) ===")
    print(f"  hypothesis: prompts don't differentiate ⇒ Jaccard HIGH (≥ 0.5)")
    print(f"  baseline reference: oracle cross-role Jaccard = 0.036 at B=512")
    print()
    role_names = list(ROLE_BUILDERS.keys())
    print(f"{'B':>6} | " + "  ".join(f"{r1}-{r2:<6s}" for r1, r2 in combinations(role_names, 2)))
    print("-" * 110)

    for B in args.budgets:
        cells = []
        line_data = {}
        for r1, r2 in combinations(role_names, 2):
            xs: List[float] = []
            for t in trajs:
                a = prompted.get((t.task_id, r1, B))
                b = prompted.get((t.task_id, r2, B))
                if a is None or b is None:
                    continue
                xs.append(_jaccard(a, b))
            mean = statistics.mean(xs) if xs else 0.0
            cells.append(f"{mean:>9.3f}")
            line_data[f"{r1}_{r2}_mean"] = mean
            line_data[f"{r1}_{r2}_n"] = len(xs)
        print(f"{B:>6} | " + "  ".join(cells))
        summary["cross_role_prompted"][B] = line_data

    # ---- 2. Per-role recall: prompted vs oracle ----
    print()
    print(f"=== Recall: prompted memory vs role-projected oracle (per role) ===")
    print(f"  hypothesis: LLM picks wrong stuff ⇒ Jaccard LOW (≤ 0.2)")
    print()
    print(f"{'B':>6} | " + "  ".join(f"{r:>10s}" for r in role_names))
    print("-" * 80)

    for B in args.budgets:
        cells = []
        line_data = {}
        for role in role_names:
            xs: List[float] = []
            for t in trajs:
                a = prompted.get((t.task_id, role, B))
                b = projected.get((t.task_id, role, B))
                if a is None or b is None:
                    continue
                xs.append(_jaccard(a, b))
            mean = statistics.mean(xs) if xs else 0.0
            cells.append(f"{mean:>10.3f}")
            line_data[f"{role}_mean"] = mean
            line_data[f"{role}_n"] = len(xs)
        print(f"{B:>6} | " + "  ".join(cells))
        summary["recall_prompted_vs_oracle"][B] = line_data

    # ---- 3. Closure ratio: how much of the role-conditional gap does prompting close? ----
    # Reference numbers from the deterministic role-overlap analysis.
    ORACLE_CROSS_ROLE_512 = 0.036  # see 03_role_memory_extractors.md §3
    print()
    print(f"=== T2 closure ratio (B=512) ===")
    if 512 in args.budgets:
        prompted_xrole_512 = statistics.mean([
            v for k, v in summary["cross_role_prompted"][512].items() if k.endswith("_mean")
        ])
        delta = prompted_xrole_512 - ORACLE_CROSS_ROLE_512
        print(f"  oracle cross-role Jaccard:    {ORACLE_CROSS_ROLE_512:.3f}  (orthogonal)")
        print(f"  prompted cross-role Jaccard:  {prompted_xrole_512:.3f}  (LLM output)")
        print(f"  Δ = prompted − oracle:        {delta:+.3f}")
        print()
        if prompted_xrole_512 >= 0.5:
            print(f"  ✓ STRONG T2 — prompted memory is surface-uniform across roles")
            print(f"    (Jaccard {prompted_xrole_512:.2f} >> oracle 0.04 ⇒ LLM doesn't differentiate)")
        elif prompted_xrole_512 >= 0.25:
            print(f"  ⚠ WEAK T2 — prompting partially differentiates roles")
        else:
            print(f"  ✗ T2 fails — prompting captures most of the role distinction")

        recall_512 = statistics.mean([
            v for k, v in summary["recall_prompted_vs_oracle"][512].items() if k.endswith("_mean")
        ])
        print(f"\n  prompted-vs-oracle recall:    {recall_512:.3f}")
        if recall_512 <= 0.2:
            print(f"  ✓ Prompting recovers ≤ {100*recall_512:.0f}% of oracle units (expected)")
        else:
            print(f"  ⚠ Prompting recovers {100*recall_512:.0f}% of oracle units")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary to {args.output_json}")


if __name__ == "__main__":
    main()
