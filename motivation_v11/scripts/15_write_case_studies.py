"""Stage 15 — write 6 representative case studies (spec §18).

Selects one example for each of:
  1. full pass → greedy compressed fail → bestN compressed pass    (best-of-N reduces harm)
  2. full fail → greedy compressed pass                            (compression rescue)
  3. full fail → greedy compressed fail → bestN compressed pass    (hidden rescue in distribution)
  4. C1 pass → CK fail                                             (serial fragility)
  5. C1 fail → CK pass                                             (recompression cleanup)
  6. verifier (pointwise) selects fail, oracle selects pass        (verbal proxy failure)

For each, writes outputs/reports/case_studies/{task_id}.md with the
task instruction, full-context outcome, compressed outcomes, greedy
text, best-of-N text, CK text, and a short explanation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v11.data import ensure_outputs, raw_path, REPORTS  # noqa


def _read_jsonl(p):
    out = []
    if not Path(p).exists(): return out
    for line in open(p):
        line = line.strip()
        if line: out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(raw_path("full_context_runs.jsonl")))
    ap.add_argument("--boundaries", default=str(raw_path("compression_boundaries.jsonl")))
    ap.add_argument("--candidates", default=str(raw_path("candidate_compressions_c1.jsonl")))
    ap.add_argument("--stress", default=str(raw_path("stress_chains.jsonl")))
    ap.add_argument("--behavior", default=str(raw_path("behavior_runs.jsonl")))
    ap.add_argument("--pointwise", default=str(raw_path("pointwise_verifier_scores.jsonl")))
    ap.add_argument("--family", default="ACON_UTCO",
                    help="Restrict case-study search to this prompt family.")
    ap.add_argument("--out_dir", default=str(REPORTS / "case_studies"))
    args = ap.parse_args()
    ensure_outputs()

    full_pass = {r["task_id"]: bool(r.get("full_success"))
                 for r in _read_jsonl(args.baseline)}
    boundaries = {r["task_id"]: r for r in _read_jsonl(args.boundaries)}
    cands = {c["candidate_id"]: c for c in _read_jsonl(args.candidates)
              if c["prompt_family"] == args.family}
    # CK text per candidate
    stress = _read_jsonl(args.stress)
    ck_text: Dict[str, str] = {}
    final_round: Dict[str, int] = {}
    for r in stress:
        cid = r["candidate_id"]
        if cid not in cands: continue
        if r["round"] >= final_round.get(cid, -1):
            final_round[cid] = r["round"]
            ck_text[cid] = r.get("text") or r.get("context_text", "")

    behavior = _read_jsonl(args.behavior)
    comp_pass: Dict[Tuple[str, str], bool] = {}
    for r in behavior:
        if r.get("error"): continue
        cid = r.get("candidate_id")
        if cid is None or cid not in cands: continue
        comp_pass[(cid, r["eval_round"])] = bool(r.get("success"))

    pw = _read_jsonl(args.pointwise)
    pw_sel: Dict[Tuple[str, str], float] = {
        (r["candidate_id"], r["eval_round"]): r.get("selector_score", 0.0)
        for r in pw
    }

    # Group candidates per task
    by_task: Dict[str, dict] = {}
    for cid, c in cands.items():
        by_task.setdefault(c["task_id"], {"greedy": None, "samples": []})
        if c["candidate_type"] == "greedy":
            by_task[c["task_id"]]["greedy"] = c
        else:
            by_task[c["task_id"]]["samples"].append(c)

    pick = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
    for tid, group in by_task.items():
        g = group["greedy"]
        if not g: continue
        gcid = g["candidate_id"]
        g_c1 = comp_pass.get((gcid, "C1"))
        g_ck = comp_pass.get((gcid, "CK"))
        if g_c1 is None or g_ck is None: continue
        f = full_pass.get(tid)
        if f is None: continue
        samples = group["samples"]
        sample_passes_c1 = [comp_pass.get((s["candidate_id"], "C1")) for s in samples]
        sample_passes_ck = [comp_pass.get((s["candidate_id"], "CK")) for s in samples]
        any_sample_c1 = any(p for p in sample_passes_c1 if p)
        any_sample_ck = any(p for p in sample_passes_ck if p)

        # 1) F=1, greedy_CK=0, any-sample_CK=1
        if pick[1] is None and f and not g_ck and any_sample_ck:
            pick[1] = (tid, g, samples)
        # 2) F=0, greedy_CK=1
        if pick[2] is None and not f and g_ck:
            pick[2] = (tid, g, samples)
        # 3) F=0, greedy_CK=0, any-sample_CK=1
        if pick[3] is None and not f and not g_ck and any_sample_ck:
            pick[3] = (tid, g, samples)
        # 4) greedy C1=1, CK=0
        if pick[4] is None and g_c1 and not g_ck:
            pick[4] = (tid, g, samples)
        # 5) greedy C1=0, CK=1
        if pick[5] is None and not g_c1 and g_ck:
            pick[5] = (tid, g, samples)
        # 6) pointwise verifier picks fail, oracle picks pass (on CK)
        if pick[6] is None and samples:
            with_pw = [(s["candidate_id"],
                         pw_sel.get((s["candidate_id"], "CK"), -1e9))
                        for s in samples]
            with_pw = [(c, s) for c, s in with_pw if s != -1e9]
            if with_pw:
                with_pw.sort(key=lambda x: -x[1])
                pw_pick = with_pw[0][0]
                pw_pass = comp_pass.get((pw_pick, "CK"))
                if pw_pass is False and any_sample_ck:
                    pick[6] = (tid, g, samples)

    # Write case studies
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    n_written = 0
    descriptions = {
        1: "FULL pass → GREEDY compressed fail → BEST-OF-N compressed pass — "
           "best-of-N reduces harm.",
        2: "FULL fail → GREEDY compressed pass — compression rescue with greedy.",
        3: "FULL fail → GREEDY compressed fail → BEST-OF-N compressed pass — "
           "hidden rescue inside the sampling distribution.",
        4: "C1 pass → CK fail (greedy) — serial recompression fragility.",
        5: "C1 fail → CK pass (greedy) — recompression cleans up a compression.",
        6: "Pointwise verbal verifier picks a fail; oracle picks a pass — "
           "verbal proxy failure.",
    }
    for idx, item in pick.items():
        if not item: continue
        tid, g, samples = item
        gcid = g["candidate_id"]
        oracle_sample = None
        for s in samples:
            if comp_pass.get((s["candidate_id"], "CK")):
                oracle_sample = s; break
        if not oracle_sample and samples:
            oracle_sample = samples[0]
        bound = boundaries.get(tid, {})
        lines = [
            f"# Case study {idx} — task `{tid}`\n",
            f"**Pattern**: {descriptions[idx]}\n",
            f"**Split**: `{bound.get('split','?')}`  ",
            f"**Prompt family**: `{args.family}`\n",
            f"\n## Task instruction\n\n> {bound.get('task_instruction','(unknown)')}\n",
            f"\n## Outcomes\n",
            f"| condition | pass |",
            f"|---|:---:|",
            f"| Full context | **{full_pass.get(tid)}** |",
            f"| Greedy compressed C1 | {comp_pass.get((gcid,'C1'))} |",
            f"| Greedy compressed CK | {comp_pass.get((gcid,'CK'))} |",
        ]
        if oracle_sample:
            ocid = oracle_sample["candidate_id"]
            lines.append(f"| Best-of-N sample C1 | {comp_pass.get((ocid,'C1'))} |")
            lines.append(f"| Best-of-N sample CK | {comp_pass.get((ocid,'CK'))} |")
        lines.append(f"\n## Greedy compressed (C1) text — {g['c1_chars']} chars\n")
        lines.append("```\n" + g["c1_text"][:2000] + ("\n... [truncated]" if len(g["c1_text"]) > 2000 else "") + "\n```\n")
        if oracle_sample:
            ocid = oracle_sample["candidate_id"]
            lines.append(f"\n## Best-of-N sample compressed (C1) — {oracle_sample['c1_chars']} chars\n")
            lines.append("```\n" + oracle_sample["c1_text"][:2000] + ("\n... [truncated]" if len(oracle_sample["c1_text"]) > 2000 else "") + "\n```\n")
        ck_g = ck_text.get(gcid)
        if ck_g:
            lines.append(f"\n## Greedy CK text — {len(ck_g)} chars\n")
            lines.append("```\n" + ck_g[:2000] + ("\n... [truncated]" if len(ck_g) > 2000 else "") + "\n```\n")
        (out_root / f"{tid}__case_{idx}.md").write_text("\n".join(lines))
        n_written += 1
    print(f"[15] wrote {n_written} / 6 case studies -> {out_root}")
    if n_written < 6:
        missing = [k for k, v in pick.items() if v is None]
        print(f"     missing patterns: {missing}  "
              f"(some patterns may not occur in the data, especially if "
              f"baseline-pass / -fail counts are skewed)")


if __name__ == "__main__":
    main()
