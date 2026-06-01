"""Stage 15b — micro-diff case studies (supplementary to stage 15).

Targets the v11 surprise finding:

  Stage 06 shows ACON families have the LOWEST textual variance among
  8 stochastic samples (CV 0.118; 1/4 reach byte-identical fixed points).
  Yet stage 09 oracle pass-rate is +36 pp over greedy on ACON_UTCO CK,
  comparable to less-constrained families.

  -> textual diversity and behavioral diversity are decoupled. A few
  characters of micro-perturbation can flip the agent's outcome even
  when the 95% surrounding text is identical.

This script finds "micro-diff high-variance" cases — where the 8
sample compressions are textually nearly identical but the
downstream agent's pass/fail outcomes are split. Per case it emits
a markdown report with text-level diff to support manual paper-figure
selection.

Selection criteria (configurable):
  * (task, family) has all 9 candidates × 2 rounds done
  * mean pairwise difflib similarity among the 8 samples > 0.85
  * sample pass count on the focal round ∈ {1..7}
    (i.e., not all-pass and not all-fail — there's selection room)

Outputs:
  outputs/reports/case_studies_micro_diff/{family}__{task_id}__{rnd}.md
  outputs/tables/micro_diff_case_index.csv          (browsable index)

This stage is NOT in run_all.sh — invoke manually after stages 07 +
(optionally) 09 have produced data:

  /workspace/EASMO/.venv/bin/python motivation_v11/scripts/15b_micro_diff_case_studies.py

Run once for each round of interest:
  --round CK   (default — paper-relevant stress-tested round)
  --round C1   (one-shot compression, no stress)
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v11.data import ensure_outputs, raw_path, table_path, REPORTS  # noqa


def _read_jsonl(p):
    out = []
    if not Path(p).exists(): return out
    for line in open(p):
        line = line.strip()
        if line: out.append(json.loads(line))
    return out


def _ratio(a: str, b: str) -> float:
    """Quickratio is much faster on long texts and accurate enough for
    a >0.85 / <0.85 thresholding gate."""
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a, b).quick_ratio()


def _exact_ratio(a: str, b: str) -> float:
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / max(len(xs), 1)


def _make_diff(a: str, b: str, label_a: str, label_b: str, n_context: int = 2) -> str:
    """Unified diff trimmed to local edit regions only."""
    lines_a = a.splitlines(keepends=False) or [""]
    lines_b = b.splitlines(keepends=False) or [""]
    diff = list(difflib.unified_diff(
        lines_a, lines_b, fromfile=label_a, tofile=label_b, n=n_context, lineterm=""))
    if not diff:
        # No line-level diff — show char-level inline hint
        sm = difflib.SequenceMatcher(None, a, b)
        opcodes = [op for op in sm.get_opcodes() if op[0] != "equal"]
        if not opcodes:
            return f"(texts are byte-identical)"
        out_lines = [f"--- {label_a}", f"+++ {label_b}",
                     "(line-level diff empty — char-level edits:)"]
        for tag, i1, i2, j1, j2 in opcodes[:10]:
            if tag == "delete":
                out_lines.append(f"  - char[{i1}:{i2}] del: {a[i1:i2]!r}")
            elif tag == "insert":
                out_lines.append(f"  + char[{j1}:{j2}] ins: {b[j1:j2]!r}")
            elif tag == "replace":
                out_lines.append(f"  ~ char[{i1}:{i2}] -> [{j1}:{j2}]: "
                                 f"{a[i1:i2]!r} -> {b[j1:j2]!r}")
        return "\n".join(out_lines)
    return "\n".join(diff)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(raw_path("candidate_compressions_c1.jsonl")))
    ap.add_argument("--stress",     default=str(raw_path("stress_chains.jsonl")))
    ap.add_argument("--behavior",   default=str(raw_path("behavior_runs.jsonl")))
    ap.add_argument("--baseline",   default=str(raw_path("full_context_runs.jsonl")))
    ap.add_argument("--boundaries", default=str(raw_path("compression_boundaries.jsonl")))
    ap.add_argument("--out_dir",    default=str(REPORTS / "case_studies_micro_diff"))
    ap.add_argument("--out_index",  default=str(table_path("micro_diff_case_index.csv")))
    ap.add_argument("--round",      default="CK", choices=("C1","CK"),
                    help="Which eval round to analyze; CK is paper-relevant.")
    ap.add_argument("--min_sim",    type=float, default=0.85,
                    help="Minimum mean pairwise similarity among 8 samples.")
    ap.add_argument("--families",   default="general_task_agnostic,general_task_aware,ACON_UT,ACON_UTCO")
    ap.add_argument("--max_per_family", type=int, default=20,
                    help="Cap qualifying cases per family (sorted by best textual sim, "
                         "to surface the hardest 'looks-same-but-isn't' cases).")
    args = ap.parse_args()
    ensure_outputs()

    print(f"[15b] loading inputs ...")
    cands     = {c["candidate_id"]: c for c in _read_jsonl(args.candidates)}
    behavior  = _read_jsonl(args.behavior)
    baseline  = {r["task_id"]: r for r in _read_jsonl(args.baseline)}
    boundaries = {r["task_id"]: r for r in _read_jsonl(args.boundaries)}

    # CK text per candidate (from stress chains, final round)
    ck_text: Dict[str, str] = {}
    final_round: Dict[str, int] = {}
    for r in _read_jsonl(args.stress):
        cid = r["candidate_id"]
        if r["round"] >= final_round.get(cid, -1):
            final_round[cid] = r["round"]
            ck_text[cid] = r.get("context_text", "")

    # Pass / score per (candidate, round)
    run_pass: Dict[Tuple[str,str], Optional[bool]] = {}
    run_score: Dict[Tuple[str,str], Optional[float]] = {}
    run_iters: Dict[Tuple[str,str], int] = {}
    for r in behavior:
        if r.get("error"): continue
        k = (r["candidate_id"], r["eval_round"])
        run_pass[k]  = bool(r.get("success"))
        run_score[k] = r.get("score")
        run_iters[k] = r.get("iterations", 0)

    # Group candidates per (task, family) — separate greedy + samples
    by_tf: Dict[Tuple[str,str], Dict[str, list]] = defaultdict(
        lambda: {"greedy": None, "samples": []})
    for cid, c in cands.items():
        key = (c["task_id"], c["prompt_family"])
        if c.get("candidate_type") == "greedy":
            by_tf[key]["greedy"] = cid
        else:
            by_tf[key]["samples"].append(cid)

    requested_families = [f.strip() for f in args.families.split(",") if f.strip()]
    qualifying: List[dict] = []

    print(f"[15b] scanning {len(by_tf)} (task, family) cells ...")
    n_skipped_incomplete = 0
    n_skipped_low_sim = 0
    n_skipped_all_or_none_pass = 0

    for (task, fam), v in by_tf.items():
        if fam not in requested_families: continue
        if v["greedy"] is None or len(v["samples"]) < 8:
            n_skipped_incomplete += 1; continue
        candidate_ids = [v["greedy"]] + v["samples"][:8]
        # Need all 9 candidates × current round done
        if not all((cid, args.round) in run_pass for cid in candidate_ids):
            n_skipped_incomplete += 1; continue

        # Pull texts (round-aware)
        def _text(cid: str) -> str:
            if args.round == "C1":
                return cands[cid].get("c1_text", "")
            return ck_text.get(cid, "")

        sample_texts = [_text(cid) for cid in v["samples"][:8]]
        greedy_text = _text(v["greedy"])
        # Sample pass pattern (8 samples on the focal round)
        sample_passes = [run_pass[(cid, args.round)] for cid in v["samples"][:8]]
        n_pass = sum(1 for p in sample_passes if p)
        if n_pass == 0 or n_pass == 8:
            n_skipped_all_or_none_pass += 1; continue

        # Mean pairwise similarity among 8 samples (28 pairs)
        sims = []
        for i in range(8):
            for j in range(i+1, 8):
                sims.append(_ratio(sample_texts[i], sample_texts[j]))
        mean_sim = sum(sims)/len(sims)
        if mean_sim < args.min_sim:
            n_skipped_low_sim += 1; continue

        # Pick representative passing + failing samples
        first_pass_idx = next(i for i,p in enumerate(sample_passes) if p)
        first_fail_idx = next(i for i,p in enumerate(sample_passes) if not p)

        # Char-length stats
        chars = [len(t) for t in sample_texts]
        char_range = max(chars) - min(chars)
        char_p_vs_f = len(sample_texts[first_pass_idx]) - len(sample_texts[first_fail_idx])
        # Exact pairwise sim between the chosen pass-fail pair (use slow accurate ratio)
        pair_sim_pf = _exact_ratio(sample_texts[first_pass_idx], sample_texts[first_fail_idx])
        # Pair sim between pass and greedy
        pair_sim_pg = _exact_ratio(sample_texts[first_pass_idx], greedy_text)

        qualifying.append({
            "task_id": task, "family": fam, "round": args.round,
            "n_sample_pass": n_pass,
            "mean_sample_sim": round(mean_sim, 4),
            "pair_sim_pass_fail": round(pair_sim_pf, 4),
            "pair_sim_pass_greedy": round(pair_sim_pg, 4),
            "char_range": char_range,
            "char_pass_minus_fail": char_p_vs_f,
            "greedy_pass": run_pass[(v["greedy"], args.round)],
            "full_success": baseline.get(task, {}).get("full_success"),
            "_pass_cid":  v["samples"][first_pass_idx],
            "_fail_cid":  v["samples"][first_fail_idx],
            "_greedy_cid": v["greedy"],
            "_pass_text": sample_texts[first_pass_idx],
            "_fail_text": sample_texts[first_fail_idx],
            "_greedy_text": greedy_text,
            "_sample_pass_pattern": [int(bool(p)) for p in sample_passes],
        })

    print(f"[15b] scan done.")
    print(f"      {len(qualifying)} qualifying micro-diff cases")
    print(f"      {n_skipped_incomplete} skipped (incomplete data)")
    print(f"      {n_skipped_low_sim} skipped (mean_sim < {args.min_sim})")
    print(f"      {n_skipped_all_or_none_pass} skipped (all-pass or all-fail)")

    # Sort by mean_sim desc (most surprising = highest similarity but variance in pass)
    qualifying.sort(key=lambda r: (-r["mean_sample_sim"], -abs(r["n_sample_pass"] - 4)))

    # Apply per-family cap
    by_fam_count: Dict[str, int] = defaultdict(int)
    capped = []
    for r in qualifying:
        if by_fam_count[r["family"]] >= args.max_per_family: continue
        capped.append(r)
        by_fam_count[r["family"]] += 1

    print(f"[15b] keeping top {len(capped)} (cap {args.max_per_family} per family)")
    for f in requested_families:
        print(f"      {f}: {by_fam_count[f]}")

    # Write CSV index
    idx_path = Path(args.out_index)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(idx_path, "w", newline="") as f:
        cols = ["task_id","family","round","n_sample_pass","mean_sample_sim",
                "pair_sim_pass_fail","pair_sim_pass_greedy","char_range",
                "char_pass_minus_fail","greedy_pass","full_success"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in capped:
            w.writerow({k: r[k] for k in cols})
    print(f"[15b] wrote index -> {idx_path}")

    # Write per-case markdown
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for r in capped:
        bnd = boundaries.get(r["task_id"], {})
        path = out_root / f"{r['family']}__{r['task_id']}__{r['round']}.md"
        lines = [
            f"# Micro-diff case study — `{r['task_id']}` / `{r['family']}` / round `{r['round']}`\n",
            f"## Headline numbers\n",
            f"| metric | value |",
            f"|---|---|",
            f"| mean pairwise textual similarity (8 samples) | **{r['mean_sample_sim']:.4f}** |",
            f"| sample pass count (of 8) | **{r['n_sample_pass']} / 8** |",
            f"| sample pass pattern (in submission order) | `{r['_sample_pass_pattern']}` |",
            f"| char-length range across samples | {r['char_range']} |",
            f"| char-length (passing) − (failing) representative | {r['char_pass_minus_fail']:+d} |",
            f"| pair sim (passing rep vs failing rep) | **{r['pair_sim_pass_fail']:.4f}** |",
            f"| pair sim (passing rep vs greedy) | {r['pair_sim_pass_greedy']:.4f} |",
            f"| full-context baseline | {r['full_success']} |",
            f"| greedy compressed | {r['greedy_pass']} |",
            f"\n## Task instruction\n",
            f"> {bnd.get('task_instruction','(unavailable)')}\n",
            f"\n## Why this case is surprising\n",
            f"{r['n_sample_pass']} of 8 samples pass the downstream agent, "
            f"yet pairwise textual similarity averages "
            f"{r['mean_sample_sim']:.3f} (i.e., {(1-r['mean_sample_sim'])*100:.1f}% "
            f"of the text differs on average). The passing sample is "
            f"{r['pair_sim_pass_fail']*100:.1f}% similar to a representative "
            f"failing sample, yet flips the outcome. This is the v11 "
            f"\"micro-perturbation behavioral entropy\" claim made concrete.\n",
            f"\n## Greedy compressed text — {len(r['_greedy_text'])} chars\n",
            "```",
            r['_greedy_text'][:4000] + ("\n... [truncated]" if len(r['_greedy_text']) > 4000 else ""),
            "```\n",
            f"\n## Passing sample — {len(r['_pass_text'])} chars (candidate `{r['_pass_cid']}`)\n",
            "```",
            r['_pass_text'][:4000] + ("\n... [truncated]" if len(r['_pass_text']) > 4000 else ""),
            "```\n",
            f"\n## Failing sample (representative) — {len(r['_fail_text'])} chars (candidate `{r['_fail_cid']}`)\n",
            "```",
            r['_fail_text'][:4000] + ("\n... [truncated]" if len(r['_fail_text']) > 4000 else ""),
            "```\n",
            f"\n## Diff: failing → passing\n",
            "```diff",
            _make_diff(r['_fail_text'], r['_pass_text'], "failing", "passing", n_context=2)[:6000],
            "```\n",
            f"\n## Diff: greedy → passing\n",
            "```diff",
            _make_diff(r['_greedy_text'], r['_pass_text'], "greedy", "passing", n_context=2)[:6000],
            "```\n",
        ]
        path.write_text("\n".join(lines))
        n_written += 1

    print(f"[15b] wrote {n_written} markdown reports -> {out_root}")
    if n_written == 0 and (n_skipped_incomplete > 0 or n_skipped_low_sim > 0):
        print(f"      (zero cases qualified at min_sim={args.min_sim}; "
              f"try lowering to 0.80 or wait for more stage 07 data)")


if __name__ == "__main__":
    main()
