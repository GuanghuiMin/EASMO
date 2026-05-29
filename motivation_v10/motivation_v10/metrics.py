"""Metrics for motivation_v9 (spec §5, §6, §10).

Reward definition (spec §5.1):
    R = score - lambda_len * normalized_length
    lambda_len = 0.02
    normalized_length = compressed_tokens_est / 1000
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LAMBDA_LEN = 0.02


def compute_reward(
    score: float, compressed_tokens_est: float,
    lambda_len: float = LAMBDA_LEN,
) -> float:
    if not np.isfinite(score) or score is None:
        return float("nan")
    n_tok = (compressed_tokens_est or 0) / 1000.0
    return float(score - lambda_len * n_tok)


def best_of_n_by_case(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("case_id", "compressor_model", "eval_context_round"),
) -> pd.DataFrame:
    """For each case × model × eval_round, compute greedy reward,
    best sample reward, best-of-N reward, gain, oracle_win."""
    rows = []
    for keys, grp in df.groupby(list(group_cols), dropna=False):
        if grp.empty:
            continue
        greedy = grp[grp["generation_type"] == "greedy"]
        samples = grp[grp["generation_type"] == "sample"]
        greedy_score = float(greedy["score"].iloc[0]) if len(greedy) else float("nan")
        greedy_success = bool(greedy["success"].iloc[0]) if len(greedy) else False
        greedy_length = float(greedy["compressed_tokens_est"].iloc[0]) if len(greedy) else float("nan")
        greedy_reward = (compute_reward(greedy_score, greedy_length)
                         if np.isfinite(greedy_score) else float("nan"))
        if len(samples):
            samples = samples.copy()
            samples["reward"] = samples.apply(
                lambda r: compute_reward(r["score"], r["compressed_tokens_est"]), axis=1)
            best_idx = samples["reward"].idxmax()
            best_sample_id = samples.loc[best_idx, "sample_id"]
            best_score = float(samples.loc[best_idx, "score"])
            best_success = bool(samples.loc[best_idx, "success"])
            best_length = float(samples.loc[best_idx, "compressed_tokens_est"])
            best_reward = float(samples.loc[best_idx, "reward"])
        else:
            best_sample_id = ""
            best_score = float("nan")
            best_success = False
            best_length = float("nan")
            best_reward = float("nan")
        best_of_n_reward = max(
            [r for r in (greedy_reward, best_reward) if np.isfinite(r)],
            default=float("nan"),
        )
        rows.append({
            "case_id": keys[0],
            "compressor_model": keys[1],
            "eval_context_round": keys[2],
            "greedy_success": greedy_success,
            "greedy_score": greedy_score,
            "greedy_length": greedy_length,
            "greedy_reward": greedy_reward,
            "best_sample_id": best_sample_id,
            "best_sample_success": best_success,
            "best_sample_score": best_score,
            "best_sample_length": best_length,
            "best_sample_reward": best_reward,
            "best_of_n_reward": best_of_n_reward,
            "best_of_n_gain_score": (best_score - greedy_score) if np.isfinite(best_score) else float("nan"),
            "best_of_n_gain_reward": (best_reward - greedy_reward) if np.isfinite(best_reward) else float("nan"),
            "oracle_win": bool(np.isfinite(best_reward) and best_reward > greedy_reward),
        })
    return pd.DataFrame(rows)


def best_of_n_summary(per_case: pd.DataFrame) -> pd.DataFrame:
    if per_case.empty:
        return pd.DataFrame()
    rows = []
    for keys, grp in per_case.groupby(["compressor_model", "eval_context_round"]):
        rows.append({
            "compressor_model": keys[0],
            "eval_context_round": keys[1],
            "n_cases": int(len(grp)),
            "greedy_pass_rate": float(grp["greedy_success"].mean()),
            "best_of_n_pass_rate": float(grp["best_sample_success"].mean()),
            "pass_gain_pp": float(grp["best_sample_success"].mean() - grp["greedy_success"].mean()) * 100,
            "greedy_mean_score": float(grp["greedy_score"].mean(skipna=True)),
            "best_of_n_mean_score": float(grp["best_sample_score"].mean(skipna=True)),
            "score_gain": float(grp["best_of_n_gain_score"].mean(skipna=True)),
            "greedy_mean_length": float(grp["greedy_length"].mean(skipna=True)),
            "best_of_n_mean_length": float(grp["best_sample_length"].mean(skipna=True)),
            "oracle_win_rate": float(grp["oracle_win"].mean()),
        })
    return pd.DataFrame(rows)


def reward_spread_by_case(df_behavior: pd.DataFrame) -> pd.DataFrame:
    """Spread of reward across the N samples per case × model × eval round."""
    df = df_behavior.copy()
    df["reward"] = df.apply(
        lambda r: compute_reward(r["score"], r["compressed_tokens_est"]), axis=1
    )
    samples = df[df["generation_type"] == "sample"]
    rows = []
    for keys, grp in samples.groupby(
        ["case_id", "compressor_model", "eval_context_round"], dropna=False,
    ):
        rows.append({
            "case_id": keys[0],
            "model": keys[1],
            "eval_context_round": keys[2],
            "num_candidates": int(len(grp)),
            "mean_score": float(grp["score"].mean()),
            "std_score": float(grp["score"].std(ddof=0)),
            "min_score": float(grp["score"].min()),
            "max_score": float(grp["score"].max()),
            "mean_reward": float(grp["reward"].mean()),
            "std_reward": float(grp["reward"].std(ddof=0)),
            "min_reward": float(grp["reward"].min()),
            "max_reward": float(grp["reward"].max()),
            "pass_rate_among_samples": float(grp["success"].mean()),
        })
    return pd.DataFrame(rows)


def c1_ck_transition(df_behavior: pd.DataFrame) -> pd.DataFrame:
    """For each candidate, classify (C1 pass, CK pass) into a 2×2 transition."""
    by_cand: Dict[Tuple, Dict[str, bool]] = {}
    for _, r in df_behavior.iterrows():
        key = (r["candidate_id"], r["compressor_model"], r["generation_type"])
        by_cand.setdefault(key, {})[r["eval_context_round"]] = bool(r["success"])
    rows = []
    for key, statuses in by_cand.items():
        if "C1" not in statuses or "CK" not in statuses:
            continue
        c1 = statuses["C1"]; ck = statuses["CK"]
        if c1 and ck:
            cls = "robust_pass"
        elif c1 and not ck:
            cls = "fragile_pass"
        elif (not c1) and ck:
            cls = "stress_improved"
        else:
            cls = "robust_fail"
        rows.append({
            "candidate_id": key[0], "compressor_model": key[1],
            "generation_type": key[2],
            "c1_success": c1, "ck_success": ck, "class": cls,
        })
    return pd.DataFrame(rows)


def c1_ck_fragility_by_model(transition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in transition.groupby(
        ["compressor_model", "generation_type"], dropna=False,
    ):
        n = len(grp)
        c_robust = int((grp["class"] == "robust_pass").sum())
        c_frag = int((grp["class"] == "fragile_pass").sum())
        c_impr = int((grp["class"] == "stress_improved").sum())
        c_rfail = int((grp["class"] == "robust_fail").sum())
        c1_pass = int(grp["c1_success"].sum())
        ck_pass = int(grp["ck_success"].sum())
        rows.append({
            "compressor_model": keys[0],
            "generation_type": keys[1],
            "n_candidates": n,
            "count_robust_pass": c_robust,
            "count_fragile_pass": c_frag,
            "count_stress_improved": c_impr,
            "count_robust_fail": c_rfail,
            "pass_rate_C1": c1_pass / n if n else 0.0,
            "pass_rate_CK": ck_pass / n if n else 0.0,
            "stress_drop_pp": (c1_pass - ck_pass) / n * 100 if n else 0.0,
            "fragility_rate": c_frag / max(c1_pass, 1),
        })
    return pd.DataFrame(rows)


def chunk_advantage(
    ablation_runs: pd.DataFrame,
    full_runs: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-chunk score / pass advantage.

    `ablation_runs` columns: candidate_id, case_id, chunk_id,
        ablation_type='remove_chunk', score, success, ...
    `full_runs` columns: candidate_id, case_id, ablation_type='full_context_control',
        score, success
    """
    full_idx = {}
    for _, r in full_runs.iterrows():
        full_idx[r["candidate_id"]] = (float(r["score"]), bool(r["success"]))
    rows = []
    for _, r in ablation_runs.iterrows():
        if r["ablation_type"] != "remove_chunk":
            continue
        score_full, succ_full = full_idx.get(r["candidate_id"], (float("nan"), False))
        score_minus = float(r["score"])
        succ_minus = bool(r["success"])
        chunk_score_adv = score_full - score_minus if np.isfinite(score_full) else float("nan")
        chunk_pass_adv = (1 if succ_full else 0) - (1 if succ_minus else 0)
        rows.append({
            "case_id": r["case_id"],
            "candidate_id": r["candidate_id"],
            "chunk_id": r["chunk_id"],
            "chunk_index": r.get("chunk_index"),
            "chunk_text": r.get("chunk_text", ""),
            "score_full": score_full, "score_minus_chunk": score_minus,
            "success_full": succ_full, "success_minus_chunk": succ_minus,
            "chunk_score_advantage": chunk_score_adv,
            "chunk_pass_advantage": chunk_pass_adv,
            "not_interpretable_due_to_full_fail": (not succ_full),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Normalize within candidate
    df["positive_adv"] = df["chunk_score_advantage"].clip(lower=0).fillna(0)
    df["chunk_adv_norm"] = 0.0
    for cid, sub in df.groupby("candidate_id"):
        s = sub["positive_adv"].sum()
        if s > 0:
            df.loc[df["candidate_id"] == cid, "chunk_adv_norm"] = sub["positive_adv"] / s
    return df


def chunk_advantage_by_type(
    chunk_adv: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Group chunk_advantage by labels.chunk_type and aggregate."""
    if chunk_adv.empty or labels.empty:
        return pd.DataFrame()
    joined = chunk_adv.merge(
        labels[["chunk_id", "chunk_type", "contains_causal_relation",
                "contains_exact_literals", "contains_negative_evidence"]],
        on="chunk_id", how="left",
    )
    # top-advantage chunks per candidate
    joined["is_top_adv"] = False
    for cid, sub in joined.groupby("candidate_id"):
        mask = sub["chunk_adv_norm"] >= 0.25
        if not mask.any():
            # top-1 fallback if any positive
            if sub["chunk_score_advantage"].max() > 0:
                top_idx = sub["chunk_score_advantage"].idxmax()
                joined.loc[top_idx, "is_top_adv"] = True
        else:
            joined.loc[sub.index[mask], "is_top_adv"] = True
    rows = []
    for ctype, grp in joined.groupby("chunk_type", dropna=False):
        rows.append({
            "chunk_type": ctype if pd.notna(ctype) else "UNLABELED",
            "n_chunks": int(len(grp)),
            "mean_score_advantage": float(grp["chunk_score_advantage"].mean(skipna=True)),
            "mean_pass_advantage": float(grp["chunk_pass_advantage"].mean(skipna=True)),
            "frac_positive_advantage": float((grp["chunk_score_advantage"] > 0).mean()),
            "frac_top_advantage": float(grp["is_top_adv"].mean()),
            "contains_causal_relation_rate": float(
                grp["contains_causal_relation"].astype(float).mean(skipna=True)
            ),
            "contains_exact_literals_rate": float(
                grp["contains_exact_literals"].astype(float).mean(skipna=True)
            ),
        })
    return pd.DataFrame(rows)


__all__ = [
    "LAMBDA_LEN",
    "compute_reward",
    "best_of_n_by_case",
    "best_of_n_summary",
    "reward_spread_by_case",
    "c1_ck_transition",
    "c1_ck_fragility_by_model",
    "chunk_advantage",
    "chunk_advantage_by_type",
]
