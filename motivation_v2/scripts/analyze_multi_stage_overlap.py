"""Cross-role Jaccard on REAL multi-agent role outputs.

Loads the artefacts produced by `run_multi_stage_role.py`:

  plan.json:          planner's agent output (sub-goals)
  appworld_trajectory.json: executor's actual run
  verifier.json:      verifier's agent output (evidence list)

For each task, computes:

  m_plan*   : entity tokens in planner's sub_goals (independent agent output)
  m_tool*   : entity tokens in executor's API call list (m_exec_trajectory style)
  m_code*   : entity tokens in executor's Python control flow (m_code style)
  m_verify* : entity tokens in verifier's evidence list (independent agent output)

Then:
  1. Cross-role Jaccard within-task (the headline number)
  2. Per-pair detail (especially plan-vs-verify, two independent agent outputs)
  3. Comparison to the projection-based Jaccard (0.04 from
     analyze_role_overlap.py at B=512)

The cleanest comparison is **plan ↔ verify**: both are LLM agent
outputs, no slicing at all. Their Jaccard is direct evidence of
"different role agents, same task, same upstream context →
disjoint memory needs".
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set


# Same entity-token extractor used elsewhere — keeps comparisons consistent.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{4,}")
_STOPWORDS = {
    "this", "that", "with", "from", "have", "will", "been", "should",
    "would", "could", "their", "there", "which", "where", "when",
    "what", "while", "more", "most", "some", "into", "about", "after",
    "before", "needs", "need", "memory", "agent", "agents", "context",
    "trace", "step", "steps", "useful", "below", "based", "your",
    "ours", "task", "tasks", "selecting", "selected", "select",
    "compressed", "compress", "calls", "call", "data", "results",
    "result", "returns", "return", "each", "give", "given", "either",
    "first", "second", "third", "true", "false", "these", "those",
    "lines", "line", "items", "item", "the", "and", "for", "are",
    "was", "were", "you", "but", "not", "all", "any", "song", "songs",
    "library", "spotify",
    # We mostly want NON-domain entity tokens to count, since domain
    # nouns will trivially co-occur. But the projections also share
    # domain terms, so leaving them in is fine for direct comparison
    # to the projection numbers.
}


def _entity_tokens(text: str) -> Set[str]:
    return {
        t.lower() for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOPWORDS
    }


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


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
    """All `apis.<app>.<fn>` calls + their immediate output snippets."""
    p = plan_dir / "appworld_trajectory.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    parts: List[str] = []
    api_re = re.compile(r"apis\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*\((.*?)\)", re.DOTALL)
    for s in d.get("trajectory", []):
        for m in api_re.finditer(s.get("action") or ""):
            parts.append(f"{m.group(1)}.{m.group(2)}({m.group(3)[:100]})")
        out = (s.get("output") or "").strip()
        if out:
            parts.append(out[:200])
    return "\n".join(parts)


def _load_executor_code(plan_dir: Path) -> str:
    """Python control-flow snippets from action code."""
    p = plan_dir / "appworld_trajectory.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    parts: List[str] = []
    pat = re.compile(
        r"(?:^|\n)\s*(for\s+\w+\s+in\s+|while\s+|if\s+|try:|except\s+|def\s+|"
        r"\[\s*\w.*?for\s+\w+|max\(|min\(|sorted\(|filter\(|map\()",
    )
    for s in d.get("trajectory", []):
        action = s.get("action") or ""
        if pat.search(action):
            parts.append(action[:300])
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="mv2_multi_stage_pilot")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    base = Path(
        f"/workspace/acon/experiments/appworld/outputs/"
        f"MiniMaxAI_MiniMax-M2.5_{args.tag}/train"
    )
    if not base.exists():
        sys.exit(f"No outputs at {base}; did the pipeline run?")

    task_dirs = sorted(base.glob("task_*"))
    print(f"[multi_stage_overlap] {len(task_dirs)} task dirs under {base}")

    if not task_dirs:
        sys.exit("no task dirs found")

    role_tokens: Dict[str, Dict[str, Set[str]]] = defaultdict(dict)
    valid_tasks: List[str] = []

    for d in task_dirs:
        tid = d.name.replace("task_", "")
        plan_t = _load_plan_text(d)
        verify_t = _load_verifier_text(d)
        tool_t = _load_executor_apis(d)
        code_t = _load_executor_code(d)
        if not plan_t or not verify_t or not tool_t:
            print(f"  skipping {tid}: missing artefact")
            continue
        valid_tasks.append(tid)
        role_tokens["plan"][tid]   = _entity_tokens(plan_t)
        role_tokens["verify"][tid] = _entity_tokens(verify_t)
        role_tokens["tool"][tid]   = _entity_tokens(tool_t)
        role_tokens["code"][tid]   = _entity_tokens(code_t)

    n = len(valid_tasks)
    print(f"  {n} valid tasks with all 4 role outputs")

    # Cross-role Jaccard within-task (mean across tasks)
    print()
    print("=== Cross-role Jaccard (within-task, real agent outputs, n=%d tasks) ===" % n)
    print(f"  {'pair':>14} | {'mean':>6} | {'median':>6} | {'min':>6} | {'max':>6}")
    print("-" * 60)
    summary = {"per_pair": {}, "per_task_size": {}}
    role_names = ["plan", "tool", "code", "verify"]
    for r1, r2 in combinations(role_names, 2):
        xs: List[float] = []
        for tid in valid_tasks:
            a = role_tokens[r1][tid]
            b = role_tokens[r2][tid]
            xs.append(_jaccard(a, b))
        if not xs:
            continue
        mean = statistics.mean(xs)
        median = statistics.median(xs)
        print(f"  {r1+'-'+r2:>14} | {mean:>6.3f} | {median:>6.3f} | "
              f"{min(xs):>6.3f} | {max(xs):>6.3f}")
        summary["per_pair"][f"{r1}_{r2}"] = {
            "mean": mean, "median": median,
            "min": min(xs), "max": max(xs), "n": len(xs),
        }

    # Token-set sizes per role (sanity: are the roles emitting comparable amounts?)
    print()
    print(f"=== Token-set size per role (mean across {n} tasks) ===")
    for role in role_names:
        sizes = [len(role_tokens[role][tid]) for tid in valid_tasks]
        mean_s = statistics.mean(sizes) if sizes else 0
        median_s = statistics.median(sizes) if sizes else 0
        print(f"  {role:>10} : mean={mean_s:.1f}, median={median_s:.0f}")
        summary["per_task_size"][role] = {"mean": mean_s, "median": median_s}

    # Headline comparison vs projection
    PROJECTION_BASELINE = 0.036
    print()
    print("=== Headline comparison ===")
    plan_vs_verify = summary["per_pair"].get("plan_verify", {}).get("mean", None)
    overall_mean = statistics.mean([
        v["mean"] for v in summary["per_pair"].values()
    ])
    print(f"  Multi-stage cross-role Jaccard (real agents): "
          f"{overall_mean:.3f} mean across all 6 role pairs")
    print(f"  Projection-based baseline (single-trajectory slicing): "
          f"{PROJECTION_BASELINE:.3f}")
    if plan_vs_verify is not None:
        print(f"  plan ↔ verify (both INDEPENDENT agent outputs): "
              f"{plan_vs_verify:.3f}")
    print()
    if overall_mean <= 0.10:
        print(f"  ✓ STRONG — multi-stage cross-role Jaccard ≤ 0.10. "
              f"Role-orthogonality holds when roles run as real agents, not just "
              f"as projections of one trajectory.")
    elif overall_mean <= 0.20:
        print(f"  ⚠ WEAK — multi-stage cross-role Jaccard ≤ 0.20. "
              f"Some orthogonality preserved but weaker than projection result.")
    else:
        print(f"  ✗ Multi-stage cross-role Jaccard {overall_mean:.2f} > 0.20. "
              f"Real agent runs share substantial memory across roles. "
              f"Reconsider the role definitions or the projection result.")

    summary["overall_mean"] = overall_mean
    summary["projection_baseline_at_B512"] = PROJECTION_BASELINE
    summary["n_tasks"] = n

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2))
        print(f"\nWrote summary to {args.output_json}")


if __name__ == "__main__":
    main()
