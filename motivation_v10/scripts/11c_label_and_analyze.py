"""Stage 11c — label every chunk (v10 enriched schema) + compute
behavior advantage + pivot by chunk_type / functional_role / regression.

Reads:
  outputs/raw/v11_chunks.jsonl
  outputs/raw/v11_chunk_ablation_runs.jsonl
  data/v10_cases.jsonl

Writes:
  outputs/raw/v11_chunk_type_labels.jsonl  (one row per chunk)
  outputs/tables/v11_chunk_advantage.csv     (per-chunk LOO advantage)
  outputs/tables/v11_chunk_advantage_by_type.csv
  outputs/tables/v11_chunk_advantage_by_role.csv
  outputs/tables/v11_chunk_advantage_regression.csv
  outputs/reports/v11_top_behavior_chunks.md

Runs in EASMO/.venv.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd

from motivation_v10.data import (  # noqa
    ensure_outputs, read_jsonl, raw_path, table_path, REPORTS,
)
from motivation_v10.clients import make_client                       # noqa
from motivation_v10.chunk_label import label_chunk                   # noqa


def _read_jsonl(p):
    out = []
    if not Path(p).exists():
        return out
    for line in open(p):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# ----------------------------------------------------------------------
# Step 1: label every chunk via MiniMax (§17.5 enriched schema)
# ----------------------------------------------------------------------

def run_labels(args) -> None:
    chunks = read_jsonl(Path(args.chunks))
    cases = {c["case_id"]: c for c in read_jsonl(Path(args.cases))}
    out_path = Path(args.labels_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists():
        for r in _read_jsonl(out_path):
            done_ids.add(r["chunk_id"])
    pending = [c for c in chunks if c["chunk_id"] not in done_ids]
    print(f"[11c.1] labels: {len(done_ids)} done / {len(pending)} pending")
    if not pending:
        return

    client = make_client("minimax")

    def _do(ch):
        case = cases.get(ch["case_id"])
        if not case:
            return None
        lbl = label_chunk(
            chunk_id=ch["chunk_id"],
            chunk_text=ch["chunk_text"],
            user_instruction=case["user_instruction"],
            client=client,
            max_tokens=2048,
        )
        return {
            "chunk_id":                   lbl.chunk_id,
            "labeler_model":              lbl.labeler_model,
            "chunk_type":                 lbl.chunk_type,
            "contains_exact_literals":    lbl.contains_exact_literals,
            "contains_entity_list_form":  lbl.contains_entity_list_form,
            "contains_causal_relation":   lbl.contains_causal_relation,
            "contains_negative_evidence": lbl.contains_negative_evidence,
            "contains_action_outcome":    lbl.contains_action_outcome,
            "contains_runtime_binding":   lbl.contains_runtime_binding,
            "functional_role_guess":      lbl.functional_role_guess,
            "confidence":                 lbl.confidence,
            "one_sentence_rationale":     lbl.one_sentence_rationale,
            "error":                      lbl.error,
        }

    t0 = time.time(); n_done = 0; n_err = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_do, c): c for c in pending}
            for fut in as_completed(futs):
                rec = fut.result()
                if rec is None:
                    continue
                if rec.get("error"):
                    n_err += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_out.flush()
                n_done += 1
                if n_done % 30 == 0 or n_done <= 3:
                    eta = (time.time()-t0)/n_done * (len(pending)-n_done)
                    print(f"  [{n_done}/{len(pending)}] err={n_err} eta={eta:.0f}s",
                          flush=True)
    print(f"[11c.1] labels: {n_done} new ({n_err} errors); "
          f"elapsed {(time.time()-t0)/60:.1f} min")


# ----------------------------------------------------------------------
# Step 2: per-chunk leave-one-out behavioral advantage
# ----------------------------------------------------------------------

def compute_advantage(args) -> pd.DataFrame:
    """Return advantage dataframe: one row per (chunk_id), with score and pass advantage."""
    chunks = {c["chunk_id"]: c for c in read_jsonl(Path(args.chunks))}
    runs = _read_jsonl(args.runs)

    # Build full-context score per candidate_id
    full_score: Dict[str, float] = {}
    full_pass: Dict[str, bool] = {}
    minus_score: Dict[str, float] = {}
    minus_pass: Dict[str, bool] = {}
    for r in runs:
        if r.get("error"):
            continue
        if r["ablation_type"] == "full_context_control":
            cid = r["ablation_id"].rsplit("__full_control", 1)[0]
            full_score[cid] = r.get("score", 0.0)
            full_pass[cid] = bool(r.get("success"))
        elif r["ablation_type"] == "remove_chunk":
            # ablation_id == f"{candidate_id}__{chunk_id}__removed"
            ch_id = r.get("chunk_id")
            if ch_id is None:
                continue
            minus_score[ch_id] = r.get("score", 0.0)
            minus_pass[ch_id] = bool(r.get("success"))

    rows = []
    for ch_id, ch in chunks.items():
        cand_id = ch["candidate_id"]
        if cand_id not in full_score or ch_id not in minus_score:
            continue
        f_s = full_score[cand_id]; m_s = minus_score[ch_id]
        f_p = full_pass[cand_id];  m_p = minus_pass[ch_id]
        rows.append({
            "chunk_id":          ch_id,
            "candidate_id":      cand_id,
            "case_id":           ch["case_id"],
            "variant":           ch["variant"],
            "chunk_index":       ch["chunk_index"],
            "chunk_text":        ch["chunk_text"],
            "chunk_chars":       ch["chunk_chars"],
            "score_full":        f_s,
            "score_minus":       m_s,
            "success_full":      f_p,
            "success_minus":     m_p,
            "score_advantage":   f_s - m_s,
            "pass_advantage":    (1 if f_p else 0) - (1 if m_p else 0),
            "not_interpretable_due_to_full_fail": not f_p,
        })
    df = pd.DataFrame(rows)
    df.to_csv(table_path("v11_chunk_advantage.csv"), index=False)
    print(f"[11c.2] wrote per-chunk advantage table -> v11_chunk_advantage.csv "
          f"({len(df)} rows)")
    return df


# ----------------------------------------------------------------------
# Step 3: aggregate by chunk_type / functional_role / flags
# ----------------------------------------------------------------------

def aggregate(df: pd.DataFrame, labels: List[dict]) -> None:
    lbl_by_id = {r["chunk_id"]: r for r in labels}

    # Join
    rows = []
    for _, r in df.iterrows():
        lbl = lbl_by_id.get(r["chunk_id"], {})
        rows.append({**r.to_dict(), **lbl})
    df_full = pd.DataFrame(rows)

    # Aggregate by chunk_type
    agg_type = (
        df_full.groupby("chunk_type")
        .agg(
            n_chunks=("score_advantage", "size"),
            mean_score_advantage=("score_advantage", "mean"),
            median_score_advantage=("score_advantage", "median"),
            mean_pass_advantage=("pass_advantage", "mean"),
            frac_positive_advantage=("score_advantage", lambda s: (s > 0).mean()),
            contains_causal_relation_rate=("contains_causal_relation",
                                           lambda s: pd.Series(s).astype(bool).mean()),
            contains_runtime_binding_rate=("contains_runtime_binding",
                                            lambda s: pd.Series(s).astype(bool).mean()),
            contains_negative_evidence_rate=("contains_negative_evidence",
                                              lambda s: pd.Series(s).astype(bool).mean()),
        )
        .sort_values("mean_score_advantage", ascending=False)
        .reset_index()
    )
    agg_type.to_csv(table_path("v11_chunk_advantage_by_type.csv"), index=False)
    print(f"[11c.3] wrote chunk_advantage_by_type ({len(agg_type)} types)")

    # Aggregate by functional_role
    if "functional_role_guess" in df_full.columns:
        agg_role = (
            df_full.groupby("functional_role_guess")
            .agg(
                n_chunks=("score_advantage", "size"),
                mean_score_advantage=("score_advantage", "mean"),
                median_score_advantage=("score_advantage", "median"),
                mean_pass_advantage=("pass_advantage", "mean"),
                frac_positive_advantage=("score_advantage", lambda s: (s > 0).mean()),
            )
            .sort_values("mean_score_advantage", ascending=False)
            .reset_index()
        )
        agg_role.to_csv(table_path("v11_chunk_advantage_by_role.csv"), index=False)
        print(f"[11c.3] wrote chunk_advantage_by_role ({len(agg_role)} roles)")

    # Regression: linear OLS of score_advantage on categorical + numeric features
    # Lightweight version: just compute univariate correlation and a multivariate
    # R² via numpy (no statsmodels dependency).
    feature_cols = []
    rows_reg = []
    # Numeric features
    for col in ("chunk_chars", "chunk_index"):
        if col in df_full.columns:
            x = df_full[col].astype(float).values
            y = df_full["score_advantage"].astype(float).values
            if x.std() > 0:
                corr = float(np.corrcoef(x, y)[0, 1])
                rows_reg.append({"feature": col, "kind": "numeric",
                                  "univariate_pearson": corr})
    # Boolean flags
    for col in ("contains_exact_literals", "contains_entity_list_form",
                 "contains_causal_relation", "contains_negative_evidence",
                 "contains_action_outcome", "contains_runtime_binding"):
        if col in df_full.columns:
            x = df_full[col].astype(bool).astype(float).values
            y = df_full["score_advantage"].astype(float).values
            if x.std() > 0:
                corr = float(np.corrcoef(x, y)[0, 1])
                rows_reg.append({"feature": col, "kind": "bool",
                                  "univariate_pearson": corr})

    # Multivariate R² (combine all features as a numeric design matrix
    # with dummy-encoded chunk_type and functional_role).
    feat_cols_num = ["chunk_chars", "chunk_index",
                      "contains_exact_literals", "contains_entity_list_form",
                      "contains_causal_relation", "contains_negative_evidence",
                      "contains_action_outcome", "contains_runtime_binding"]
    feat_cols_num = [c for c in feat_cols_num if c in df_full.columns]
    X_num = df_full[feat_cols_num].fillna(0).astype(float).values
    # dummy
    if "chunk_type" in df_full.columns:
        X_type = pd.get_dummies(df_full["chunk_type"], prefix="type").astype(float).values
    else:
        X_type = np.zeros((len(df_full), 0))
    if "functional_role_guess" in df_full.columns:
        X_role = pd.get_dummies(df_full["functional_role_guess"], prefix="role").astype(float).values
    else:
        X_role = np.zeros((len(df_full), 0))
    X_full  = np.hstack([X_num, X_type, X_role])
    X_label = np.hstack([X_type, X_role])  # label-only features
    y = df_full["score_advantage"].astype(float).values

    def _r2(X, y):
        if X.shape[1] == 0:
            return 0.0
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        try:
            beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
            y_hat = X1 @ beta
            ss_res = ((y - y_hat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            if ss_tot == 0:
                return 0.0
            return 1 - ss_res / ss_tot
        except Exception:
            return 0.0

    r2_full   = _r2(X_full, y)
    r2_label  = _r2(X_label, y)
    r2_num    = _r2(X_num, y)

    rows_reg.append({"feature": "MULTIVARIATE_R2_FULL", "kind": "regression",
                      "univariate_pearson": r2_full})
    rows_reg.append({"feature": "MULTIVARIATE_R2_LABELS_ONLY", "kind": "regression",
                      "univariate_pearson": r2_label})
    rows_reg.append({"feature": "MULTIVARIATE_R2_NUMERIC_ONLY", "kind": "regression",
                      "univariate_pearson": r2_num})
    pd.DataFrame(rows_reg).to_csv(
        table_path("v11_chunk_advantage_regression.csv"), index=False
    )
    print(f"[11c.3] wrote regression table ({len(rows_reg)} feature rows)")

    # Top-20 chunks by behavior advantage
    REPORTS.mkdir(parents=True, exist_ok=True)
    top = df_full.sort_values("score_advantage", ascending=False).head(20)
    lines = ["# v11 — top 20 chunks by behavioral score advantage\n"]
    for _, r in top.iterrows():
        lines.append(f"## Δ = {r['score_advantage']:+.3f}  ({r['variant']}, case {r['case_id']}, type {r.get('chunk_type','?')}, role {r.get('functional_role_guess','?')})\n")
        lines.append(f"> {r.get('one_sentence_rationale','(no rationale)')}\n")
        lines.append("```\n" + r["chunk_text"][:600].rstrip() + ("..." if len(r["chunk_text"])>600 else "") + "\n```\n")
    (REPORTS / "v11_top_behavior_chunks.md").write_text("\n".join(lines))
    print(f"[11c.3] wrote v11_top_behavior_chunks.md")

    # Print Claim 4 verdict
    print(f"\n=== Spec §19.4 Claim 4 diagnostic ===")
    print(f"  Multivariate R² (labels only): {r2_label:.4f}")
    print(f"  Multivariate R² (numeric only): {r2_num:.4f}")
    print(f"  Multivariate R² (full):         {r2_full:.4f}")
    if r2_label < r2_full:
        print(f"  -> labels alone explain less than the full feature set.")
        print(f"     Behavior-based credit is the more reliable signal.")
    else:
        print(f"  -> labels alone match the full feature set; behavior advantage may not add information.")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("label", "advantage", "aggregate", "all"),
                    default="all")
    ap.add_argument("--cases", default=str(_REPO / "data" / "v10_cases.jsonl"))
    ap.add_argument("--chunks", default=str(raw_path("v11_chunks.jsonl")))
    ap.add_argument("--runs", default=str(raw_path("v11_chunk_ablation_runs.jsonl")))
    ap.add_argument("--labels_out",
                    default=str(raw_path("v11_chunk_type_labels.jsonl")))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ensure_outputs()

    if args.phase in ("label", "all"):
        run_labels(args)

    if args.phase in ("advantage", "all"):
        df = compute_advantage(args)
    else:
        df = pd.read_csv(table_path("v11_chunk_advantage.csv"))

    if args.phase in ("aggregate", "all"):
        labels = _read_jsonl(args.labels_out)
        aggregate(df, labels)


if __name__ == "__main__":
    main()
