"""Stage 15c — form-prior drop case studies (supplementary to stages 15 + 15b).

Targets the v5 / v7 form-prior hypothesis at v11 sample-distribution level:

  Per v5: ACON-dropped items get re-dropped on recompression (93%).
  Per v7: compressor retention conditions on fact_type (form), not need_label (function).

  v11 stochastic-sample variant: across 8 ACON_UTCO samples on the same
  task, the LLM's form prior is applied with stochastic variance — some
  samples preserve a critical entity (JWT / email / phone / kv-pair),
  some drop it. **Preservation correlates with downstream pass.**

This script finds such cases automatically: critical tokens that exist
in the original trajectory, are dropped by ≥3/8 sample CK compressions,
and where preservation predicts pass-rate.

Selection criteria:
  * (task, family) has all 9 candidates × CK behavior rows
  * regex matches at least one critical token in original `history_text`
  * the token is preserved in 1-7 of 8 sample CKs (not all, not none)
  * pass_rate(samples with token) > pass_rate(samples without token)

Outputs:
  outputs/reports/case_studies_form_prior_drop/{family}__{task}__{cat}.md
  outputs/tables/form_prior_drop_case_index.csv

Standalone — NOT in run_all.sh (avoid live-edit gotcha). Invoke after
stage 07 completes:

  /workspace/EASMO/.venv/bin/python motivation_v11/scripts/15c_form_prior_drop_case_studies.py

This stage is complementary to 15b: 15b finds high-textual-similarity
cases (which include hallucination-driven failure modes); 15c specifically
finds critical-token-drop cases (which match v5/v7 form-prior failure).
Paper appendix can include both; main motivation figure typically draws
from 15c (cleaner v5/v7 replication).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v11.data import ensure_outputs, raw_path, table_path, REPORTS  # noqa


# === Critical-token regex patterns ===
#
# Empirically derived from exploring all 147 v11 AppWorld trajectories
# (see notes in the 15c exploration; uuid/long_hex_id/url_id were dropped
# because they had near-zero hit rate on AppWorld trajectories).
PATTERNS = {
    "jwt_token":  r'\beyJ[A-Za-z0-9._\-]{15,}',
    "kv_pair":    r'(?:access_token|target_id|api_key|user_id|account_number|order_id|playlist_id|song_id|message_id|file_id)\s*[:=]\s*[\'"]?([^\'"\s,}]+)',
    "email":      r'[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}',
    "phone":      r'\+?\d[\d\s\-()]{8,}\d',
}

# Pattern weights for sorting paper-quality cases (higher = more likely
# to be agent-critical functional content).
PATTERN_WEIGHT = {
    "jwt_token":  1.0,   # auth → almost always needed
    "kv_pair":    1.0,   # named entity → almost always needed
    "email":      0.8,   # often needed for user lookup
    "phone":      0.7,   # often needed for account lookup
}


def _read_jsonl(p):
    out = []
    if not Path(p).exists(): return out
    for line in open(p):
        line = line.strip()
        if line: out.append(json.loads(line))
    return out


def extract_critical_tokens(text: str) -> List[Tuple[str, str]]:
    """Return [(category, token_str), ...] with dedup."""
    seen = set()
    out: List[Tuple[str, str]] = []
    for cat, pat in PATTERNS.items():
        for m in re.finditer(pat, text):
            s = m.group(0) if not m.groups() else m.group(1)
            s = s.strip('\'"`,;.:').strip()
            if len(s) < 5: continue
            key = (cat, s)
            if key in seen: continue
            seen.add(key)
            out.append(key)
    return out


def _snippet_around(text: str, token: str, n_context: int = 80) -> str:
    """Return a snippet showing the token highlighted in its surrounding text."""
    i = text.find(token)
    if i < 0:
        return f"(token not found in text — search miss)"
    lo = max(0, i - n_context)
    hi = min(len(text), i + len(token) + n_context)
    pre = text[lo:i].replace("\n", " ")
    mid = token
    post = text[i + len(token):hi].replace("\n", " ")
    return f"...{pre}**{mid}**{post}..."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(raw_path("candidate_compressions_c1.jsonl")))
    ap.add_argument("--stress",     default=str(raw_path("stress_chains.jsonl")))
    ap.add_argument("--behavior",   default=str(raw_path("behavior_runs.jsonl")))
    ap.add_argument("--baseline",   default=str(raw_path("full_context_runs.jsonl")))
    ap.add_argument("--boundaries", default=str(raw_path("compression_boundaries.jsonl")))
    ap.add_argument("--out_dir",    default=str(REPORTS / "case_studies_form_prior_drop"))
    ap.add_argument("--out_index",  default=str(table_path("form_prior_drop_case_index.csv")))
    ap.add_argument("--round",      default="CK", choices=("C1","CK"),
                    help="Which eval round to analyze; CK shows the v5 'redrop' effect cleanest.")
    ap.add_argument("--min_preserved", type=int, default=1,
                    help="Minimum sample count that must preserve the token (otherwise nothing to compare).")
    ap.add_argument("--max_preserved", type=int, default=7,
                    help="Maximum sample count (otherwise nothing dropped).")
    ap.add_argument("--min_pass_gap", type=float, default=0.15,
                    help="Minimum pass-rate gap (with - without) to count as a paper-grade case.")
    ap.add_argument("--families", default="general_task_agnostic,general_task_aware,ACON_UT,ACON_UTCO")
    ap.add_argument("--max_per_family", type=int, default=15,
                    help="Cap per family in markdown output (sorted by paper-impact score).")
    args = ap.parse_args()
    ensure_outputs()

    print(f"[15c] loading inputs ...")
    cands     = {c["candidate_id"]: c for c in _read_jsonl(args.candidates)}
    boundaries = {r["task_id"]: r for r in _read_jsonl(args.boundaries)}
    behavior  = _read_jsonl(args.behavior)
    baseline  = {r["task_id"]: r for r in _read_jsonl(args.baseline)}

    # CK / C1 text per candidate
    text_by_cid: Dict[str, str] = {}
    if args.round == "C1":
        for cid, c in cands.items():
            text_by_cid[cid] = c.get("c1_text", "") or ""
    else:
        final_round: Dict[str, int] = {}
        for r in _read_jsonl(args.stress):
            cid = r["candidate_id"]
            if r["round"] >= final_round.get(cid, -1):
                final_round[cid] = r["round"]
                text_by_cid[cid] = r.get("context_text", "") or ""

    # Pass per (cid, round)
    run_pass: Dict[Tuple[str,str], bool] = {}
    for r in behavior:
        if r.get("error"): continue
        run_pass[(r["candidate_id"], r["eval_round"])] = bool(r.get("success"))

    # Group by (task, family)
    by_tf: Dict[Tuple[str,str], Dict[str, list]] = defaultdict(
        lambda: {"greedy": None, "samples": []})
    for cid, c in cands.items():
        key = (c["task_id"], c["prompt_family"])
        if c.get("candidate_type") == "greedy":
            by_tf[key]["greedy"] = cid
        else:
            by_tf[key]["samples"].append(cid)

    requested = [f.strip() for f in args.families.split(",") if f.strip()]
    qualifying: List[dict] = []

    n_cells_seen = 0
    n_cells_complete = 0
    n_tokens_seen = 0
    n_dropcase_seen = 0
    n_quals = 0

    print(f"[15c] scanning {len(by_tf)} (task, family) cells ...")

    for (task, fam), v in by_tf.items():
        if fam not in requested: continue
        n_cells_seen += 1
        if v["greedy"] is None or len(v["samples"]) < 8: continue
        candidate_ids = [v["greedy"]] + v["samples"][:8]
        if not all((cid, args.round) in run_pass for cid in candidate_ids):
            continue
        n_cells_complete += 1

        # Extract tokens from ORIGINAL trajectory
        history = boundaries.get(task, {}).get("history_text", "") or ""
        if not history: continue
        tokens = extract_critical_tokens(history)
        if not tokens: continue

        greedy_text = text_by_cid.get(v["greedy"], "")
        sample_texts = [text_by_cid.get(cid, "") for cid in v["samples"][:8]]
        sample_passes = [run_pass.get((cid, args.round)) for cid in v["samples"][:8]]
        greedy_pass = run_pass.get((v["greedy"], args.round))

        for cat, tok in tokens:
            n_tokens_seen += 1
            preserved = [tok in t for t in sample_texts]
            n_pres = sum(preserved)
            if not (args.min_preserved <= n_pres <= args.max_preserved):
                continue
            n_dropcase_seen += 1
            pass_with = sum(1 for i,p in enumerate(sample_passes) if preserved[i] and p)
            pass_without = sum(1 for i,p in enumerate(sample_passes) if not preserved[i] and p)
            rate_with = pass_with / max(n_pres, 1)
            rate_without = pass_without / max(8 - n_pres, 1)
            gap = rate_with - rate_without
            if gap < args.min_pass_gap: continue

            n_quals += 1
            # Pick representative passing+preserving sample, and failing+dropping sample
            pass_idx = next((i for i,p in enumerate(sample_passes) if preserved[i] and p), None)
            fail_idx = next((i for i,p in enumerate(sample_passes) if (not preserved[i]) and (not p)), None)
            if pass_idx is None: pass_idx = next(i for i,p in enumerate(preserved) if p)
            if fail_idx is None: fail_idx = next((i for i,p in enumerate(preserved) if not p), None)

            score = gap * min(n_pres, 8 - n_pres) * PATTERN_WEIGHT.get(cat, 0.5)

            qualifying.append({
                "task_id": task, "family": fam, "round": args.round,
                "token_category": cat,
                "token_snippet": tok[:60] + ("…" if len(tok) > 60 else ""),
                "n_preserved": n_pres,
                "rate_with_token": round(rate_with, 4),
                "rate_without_token": round(rate_without, 4),
                "pass_gap": round(gap, 4),
                "greedy_preserved": tok in greedy_text,
                "greedy_pass": greedy_pass,
                "full_success": baseline.get(task, {}).get("full_success"),
                "score": round(score, 4),
                "_token_full": tok,
                "_pass_cid":  v["samples"][pass_idx] if pass_idx is not None else None,
                "_fail_cid":  v["samples"][fail_idx] if fail_idx is not None else None,
                "_greedy_cid": v["greedy"],
                "_history": history,
                "_pass_text": sample_texts[pass_idx] if pass_idx is not None else "",
                "_fail_text": sample_texts[fail_idx] if fail_idx is not None else "",
                "_greedy_text": greedy_text,
                "_preserve_pattern": [int(p) for p in preserved],
                "_pass_pattern": [int(bool(p)) for p in sample_passes],
            })

    print(f"[15c] scan done.")
    print(f"      {n_cells_seen} cells inspected; {n_cells_complete} complete (all 9 × {args.round})")
    print(f"      {n_tokens_seen} critical-token candidates in originals")
    print(f"      {n_dropcase_seen} sample-distribution split cases (1-7 / 8 preserved)")
    print(f"      {n_quals} pass-gap-qualified cases (gap >= {args.min_pass_gap})")

    # Sort by paper-impact score desc
    qualifying.sort(key=lambda r: -r["score"])

    by_fam_count: Dict[str, int] = defaultdict(int)
    capped = []
    for r in qualifying:
        if by_fam_count[r["family"]] >= args.max_per_family: continue
        capped.append(r)
        by_fam_count[r["family"]] += 1

    print(f"[15c] keeping top {len(capped)} (cap {args.max_per_family} per family)")
    for f in requested:
        print(f"      {f}: {by_fam_count[f]}")

    # CSV index
    idx_path = Path(args.out_index)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(idx_path, "w", newline="") as f:
        cols = ["task_id","family","round","token_category","token_snippet",
                "n_preserved","rate_with_token","rate_without_token","pass_gap",
                "greedy_preserved","greedy_pass","full_success","score"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in capped:
            w.writerow({k: r[k] for k in cols})
    print(f"[15c] wrote index -> {idx_path}")

    # Per-case markdown
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for r in capped:
        bnd = boundaries.get(r["task_id"], {})
        # Safe filename: replace special chars in token snippet
        safe_token = re.sub(r'[^A-Za-z0-9_\-]', '_', r["token_snippet"])[:30]
        path = out_root / f"{r['family']}__{r['task_id']}__{r['token_category']}__{safe_token}.md"
        tok = r["_token_full"]

        lines = [
            f"# Form-prior drop case — `{r['task_id']}` / `{r['family']}` / round `{r['round']}`\n",
            f"## Token under analysis\n",
            f"- **category**: `{r['token_category']}`",
            f"- **value (truncated)**: `{r['token_snippet']}`",
            f"- **original-trajectory context**:",
            f"  > {_snippet_around(r['_history'], tok)}\n",
            f"## Headline numbers\n",
            f"| metric | value |",
            f"|---|---|",
            f"| sample preservation count | **{r['n_preserved']} / 8** |",
            f"| pass rate WITH token preserved  | **{r['rate_with_token']*100:.0f}%** |",
            f"| pass rate WITHOUT token (dropped) | **{r['rate_without_token']*100:.0f}%** |",
            f"| **pass-rate gap (with − without)** | **{r['pass_gap']*100:+.0f} pp** |",
            f"| greedy preserved this token? | {r['greedy_preserved']} |",
            f"| greedy pass | {r['greedy_pass']} |",
            f"| full-context baseline | {r['full_success']} |",
            f"| sample preserve pattern (submission order) | `{r['_preserve_pattern']}` |",
            f"| sample pass pattern | `{r['_pass_pattern']}` |",
            f"\n## Task instruction\n",
            f"> {bnd.get('task_instruction','(unavailable)')}\n",
            f"\n## v5 / v7 narrative for this case\n",
            f"The token `{r['token_snippet']}` is present in the original "
            f"AppWorld trajectory; under K=2 recompression in the "
            f"`{r['family']}` family, **{8-r['n_preserved']} of 8 stochastic "
            f"samples drop it**, while {r['n_preserved']} preserve it. "
            f"Samples that preserve the token pass downstream at "
            f"{r['rate_with_token']*100:.0f}%; samples that drop it pass at "
            f"{r['rate_without_token']*100:.0f}% ({r['pass_gap']*100:+.0f} pp gap). "
            f"This is the v5 'recovered-then-dropped' phenomenon and the v7 "
            f"'form-prior conditions on fact_type, not need_label' phenomenon "
            f"manifesting at sample-distribution level: the same compressor "
            f"applied to the same input drops critical functional content "
            f"with non-trivial probability, and best-of-N selection exploits "
            f"this variance.\n",
        ]
        # Show greedy + passing + failing texts with token highlighted
        for label, text, cid in [
            ("Greedy compressed", r["_greedy_text"], r["_greedy_cid"]),
            ("Passing sample (token preserved)", r["_pass_text"], r["_pass_cid"]),
            ("Failing sample (token dropped, representative)", r["_fail_text"], r["_fail_cid"]),
        ]:
            preserved_here = tok in text
            lines.append(f"\n## {label} — {len(text)} chars (candidate `{cid}`)\n")
            lines.append(f"- token `{r['token_category']}` preserved: **{preserved_here}**")
            if preserved_here:
                lines.append(f"- snippet around token: > {_snippet_around(text, tok)}")
            lines.append("\n```")
            lines.append(text[:3500] + ("\n... [truncated]" if len(text) > 3500 else ""))
            lines.append("```")
        path.write_text("\n".join(lines))
        n_written += 1

    print(f"[15c] wrote {n_written} markdown reports -> {out_root}")


if __name__ == "__main__":
    main()
