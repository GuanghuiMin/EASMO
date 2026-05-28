"""Metrics for motivation_v7 (spec §15 + §16).

Implements:

  * need-effect Δ_need per fact type, with bootstrap CIs;
  * surface-dominance logistic regression (multiple models);
  * Surface Dominance Index (SDI);
  * preference inversion rate (PIR);
  * Condition Responsiveness Score (CRS);
  * survival curves S_c(r), half-life, hazard, AUSC;
  * cross-model Kendall τ / Spearman ρ.

We use statsmodels for logistic regression to get McFadden R².
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm  # type: ignore
    from statsmodels.tools.sm_exceptions import PerfectSeparationError  # type: ignore
    _HAS_SM = True
except Exception:
    _HAS_SM = False


# ----------------------------------------------------------------------
# Bootstrap CI helper
# ----------------------------------------------------------------------


def _bootstrap_ci(
    values: Sequence[float],
    n_iter: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, v.size, size=(n_iter, v.size))
    samples = v[idx].mean(axis=1)
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


def _diff_bootstrap_ci(
    a: Sequence[float], b: Sequence[float],
    n_iter: int = 1000, alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return (float("nan"), float("nan"))
    samples = np.zeros(n_iter)
    for k in range(n_iter):
        ia = rng.integers(0, a.size, size=a.size)
        ib = rng.integers(0, b.size, size=b.size)
        samples[k] = a[ia].mean() - b[ib].mean()
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


# ----------------------------------------------------------------------
# Need-effect by fact type  (§15.1)
# ----------------------------------------------------------------------


def need_effect_by_type(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("compressor_model", "prompt_variant",
                                  "budget_chars", "fact_type"),
    n_boot: int = 500,
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for keys, grp in df.groupby(list(group_cols), dropna=False):
        n_v1 = grp[grp["need_label"] == 1]
        n_v0 = grp[grp["need_label"] == 0]
        if len(n_v1) == 0 and len(n_v0) == 0:
            continue
        retain_n = n_v1["retained_binary"].astype(float).mean() if len(n_v1) else float("nan")
        retain_u = n_v0["retained_binary"].astype(float).mean() if len(n_v0) else float("nan")
        score_n  = n_v1["retention_score"].astype(float).mean() if len(n_v1) else float("nan")
        score_u  = n_v0["retention_score"].astype(float).mean() if len(n_v0) else float("nan")
        if len(n_v1) and len(n_v0):
            lo, hi = _diff_bootstrap_ci(
                n_v1["retained_binary"].astype(float).values,
                n_v0["retained_binary"].astype(float).values,
                n_iter=n_boot, rng=rng,
            )
        else:
            lo, hi = (float("nan"), float("nan"))
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        coarse_group = grp["coarse_group"].iloc[0] if "coarse_group" in grp else ""
        row.update({
            "coarse_group": coarse_group,
            "n_needed": int(len(n_v1)),
            "n_unneeded": int(len(n_v0)),
            "retain_needed": float(retain_n),
            "retain_unneeded": float(retain_u),
            "delta_need": float(retain_n - retain_u) if (
                np.isfinite(retain_n) and np.isfinite(retain_u)) else float("nan"),
            "score_needed": float(score_n),
            "score_unneeded": float(score_u),
            "delta_need_score": float(score_n - score_u) if (
                np.isfinite(score_n) and np.isfinite(score_u)) else float("nan"),
            "ci_low": float(lo),
            "ci_high": float(hi),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Surface-dominance regression  (§15.2 + §15.3)
# ----------------------------------------------------------------------


def _fit_logit_with_r2(X: pd.DataFrame, y: np.ndarray) -> Tuple[float, dict]:
    """Fit a logistic regression and return (mcfadden_r2, coefs).
    Returns nan + empty dict if model fails."""
    if not _HAS_SM or len(X) == 0 or y.std() == 0:
        return float("nan"), {}
    Xc = sm.add_constant(X, has_constant="add")
    try:
        m = sm.Logit(y, Xc).fit(disp=False, method="bfgs", maxiter=100)
        coefs = {n: float(c) for n, c in zip(Xc.columns, m.params)}
        return float(m.prsquared), coefs
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError):
        return float("nan"), {}
    except Exception:
        return float("nan"), {}


def surface_dominance_regression(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("compressor_model", "prompt_variant", "budget_chars"),
) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(list(group_cols), dropna=False):
        y = grp["retained_binary"].astype(int).values
        need = grp[["need_label"]].astype(float)
        # One-hot encode fact_type (drop_first=True to avoid dummy trap)
        types = pd.get_dummies(grp["fact_type"], prefix="ft", drop_first=True)
        types = types.astype(float)
        # M_need
        r2_need, _ = _fit_logit_with_r2(need, y)
        # M_type
        r2_type, _ = _fit_logit_with_r2(types, y)
        # M_both
        Xb = pd.concat([need.reset_index(drop=True),
                        types.reset_index(drop=True)], axis=1)
        r2_both, coefs_both = _fit_logit_with_r2(Xb, y)
        sdi = float("nan")
        if np.isfinite(r2_type) and np.isfinite(r2_need):
            denom = r2_type + r2_need + 1e-8
            sdi = (r2_type - r2_need) / denom
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update({
            "n": int(len(grp)),
            "r2_need": float(r2_need),
            "r2_type": float(r2_type),
            "r2_both": float(r2_both),
            "sdi": float(sdi),
            "coef_need_in_both": coefs_both.get("need_label", float("nan")),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Preference Inversion Rate  (§15.4)
# ----------------------------------------------------------------------


def preference_inversion_rate(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("compressor_model", "prompt_variant",
                                  "budget_chars"),
    n_boot: int = 500,
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for keys, grp in df.groupby(list(group_cols), dropna=False):
        # within each case, pair needed-concrete vs unneeded-narrative
        needed_concrete = grp[
            (grp["need_label"] == 1) &
            (grp["coarse_group"].isin(["EXECUTABLE", "CONTROL"]))
        ]
        unneeded_narr = grp[
            (grp["need_label"] == 0) &
            (grp["coarse_group"] == "NARRATIVE")
        ]
        # Index by case for matching
        nc_by_case = needed_concrete.groupby("case_id")
        un_by_case = unneeded_narr.groupby("case_id")
        common = set(nc_by_case.groups) & set(un_by_case.groups)
        inversions: List[int] = []
        for case_id in common:
            for _, n_row in nc_by_case.get_group(case_id).iterrows():
                for _, u_row in un_by_case.get_group(case_id).iterrows():
                    inv = (
                        int(u_row["retained_binary"]) == 1 and
                        int(n_row["retained_binary"]) == 0
                    )
                    inversions.append(int(inv))
        if not inversions:
            continue
        pir = float(np.mean(inversions))
        lo, hi = _bootstrap_ci(inversions, n_iter=n_boot, rng=rng)
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update({
            "n_pairs": len(inversions),
            "preference_inversion_rate": pir,
            "ci_low": float(lo),
            "ci_high": float(hi),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Condition Responsiveness Score  (§15.5)
# ----------------------------------------------------------------------


def condition_responsiveness(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("compressor_model", "prompt_variant",
                                  "budget_chars", "fact_type"),
) -> pd.DataFrame:
    """For each fact with a needed/unneeded pair, CRS_f = score_needed - score_unneeded.
    Aggregated by fact_type."""
    rows = []
    df = df.copy()
    df_n = df[df["need_label"] == 1][["case_id", "fact_id", "fact_type",
                                       "coarse_group", "compressor_model",
                                       "prompt_variant", "budget_chars",
                                       "retention_score"]].rename(
        columns={"retention_score": "score_needed"})
    df_u = df[df["need_label"] == 0][["case_id", "fact_id",
                                       "compressor_model", "prompt_variant",
                                       "budget_chars", "retention_score"]].rename(
        columns={"retention_score": "score_unneeded"})
    merged = df_n.merge(df_u, on=["case_id", "fact_id", "compressor_model",
                                   "prompt_variant", "budget_chars"])
    if merged.empty:
        return pd.DataFrame()
    merged["crs"] = merged["score_needed"] - merged["score_unneeded"]
    for keys, grp in merged.groupby(list(group_cols), dropna=False):
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update({
            "coarse_group": grp["coarse_group"].iloc[0],
            "n_pairs": int(len(grp)),
            "mean_crs": float(grp["crs"].mean()),
            "median_crs": float(grp["crs"].median()),
            "frac_positive": float((grp["crs"] > 0).mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Iterative survival  (§16)
# ----------------------------------------------------------------------


def survival_by_round_type(
    df_iter: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("compressor_model", "prompt_variant",
                                  "budget_chars", "round", "fact_type"),
    n_boot: int = 500,
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for keys, grp in df_iter.groupby(list(group_cols), dropna=False):
        vals = grp["retained_binary"].astype(float).values
        score_vals = grp["retention_score"].astype(float).values
        if vals.size == 0:
            continue
        lo, hi = _bootstrap_ci(vals, n_iter=n_boot, rng=rng)
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update({
            "coarse_group": grp["coarse_group"].iloc[0] if "coarse_group" in grp else "",
            "n_facts": int(len(grp)),
            "survival_rate": float(vals.mean()),
            "survival_score_mean": float(score_vals.mean()),
            "ci_low": float(lo),
            "ci_high": float(hi),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def half_life_table(
    surv: pd.DataFrame,
    *,
    rounds_cap: int,
    threshold: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for keys, grp in surv.groupby(
        ["compressor_model", "prompt_variant", "budget_chars", "fact_type"],
        dropna=False,
    ):
        grp = grp.sort_values("round")
        below = grp[grp["survival_rate"] <= threshold]
        if len(below):
            half = int(below["round"].iloc[0])
            censored = False
        else:
            half = rounds_cap + 1
            censored = True
        rows.append({
            "compressor_model": keys[0],
            "prompt_variant":   keys[1],
            "budget_chars":     keys[2],
            "fact_type":        keys[3],
            "coarse_group":     grp["coarse_group"].iloc[0] if "coarse_group" in grp else "",
            "half_life":        half,
            "half_life_censored": censored,
            "n_rounds":         int(grp["round"].max()),
            "final_survival":   float(grp["survival_rate"].iloc[-1]),
        })
    return pd.DataFrame(rows)


def hazard_by_round_type(surv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in surv.groupby(
        ["compressor_model", "prompt_variant", "budget_chars", "fact_type"],
        dropna=False,
    ):
        grp = grp.sort_values("round")
        prev = None
        for _, r in grp.iterrows():
            cur = float(r["survival_rate"])
            if prev is None:
                hazard = 1.0 - cur
            else:
                hazard = 1.0 - cur / (prev + 1e-8) if prev > 0 else float("nan")
            rows.append({
                "compressor_model": keys[0],
                "prompt_variant":   keys[1],
                "budget_chars":     keys[2],
                "fact_type":        keys[3],
                "round": int(r["round"]),
                "hazard": float(hazard),
                "survival_rate": cur,
            })
            prev = cur
    return pd.DataFrame(rows)


def ausc_by_type(surv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in surv.groupby(
        ["compressor_model", "prompt_variant", "budget_chars", "fact_type"],
        dropna=False,
    ):
        ausc = float(grp["survival_rate"].sum())
        rows.append({
            "compressor_model": keys[0],
            "prompt_variant":   keys[1],
            "budget_chars":     keys[2],
            "fact_type":        keys[3],
            "ausc": ausc,
            "n_rounds": int(grp["round"].max()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Hierarchy rank stability  (§16.5)
# ----------------------------------------------------------------------


def hierarchy_rank_by_model(half_df: pd.DataFrame) -> pd.DataFrame:
    """Rank fact types within each (model, prompt, budget) by half_life desc.
    Ties broken by AUSC desc within the type set if available."""
    rows = []
    for keys, grp in half_df.groupby(
        ["compressor_model", "prompt_variant", "budget_chars"],
        dropna=False,
    ):
        ranked = grp.sort_values(
            ["half_life", "final_survival"], ascending=[False, False]
        ).reset_index(drop=True)
        for i, r in ranked.iterrows():
            rows.append({
                "compressor_model": keys[0],
                "prompt_variant":   keys[1],
                "budget_chars":     keys[2],
                "fact_type":        r["fact_type"],
                "coarse_group":     r["coarse_group"],
                "rank":             int(i + 1),
                "half_life":        int(r["half_life"]),
                "final_survival":   float(r["final_survival"]),
            })
    return pd.DataFrame(rows)


def cross_model_hierarchy_similarity(ranks: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Kendall τ and Spearman ρ between rank vectors across models."""
    from scipy.stats import kendalltau, spearmanr
    rows = []
    pivots = {}
    for (m, p, b), grp in ranks.groupby(
        ["compressor_model", "prompt_variant", "budget_chars"], dropna=False,
    ):
        pivots[(m, p, b)] = dict(zip(grp["fact_type"], grp["rank"]))
    keys = list(pivots.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            common = sorted(set(pivots[k1]) & set(pivots[k2]))
            if len(common) < 3:
                continue
            v1 = [pivots[k1][t] for t in common]
            v2 = [pivots[k2][t] for t in common]
            tau, tau_p = kendalltau(v1, v2)
            rho, rho_p = spearmanr(v1, v2)
            rows.append({
                "model_a": k1[0], "prompt_a": k1[1], "budget_a": k1[2],
                "model_b": k2[0], "prompt_b": k2[1], "budget_b": k2[2],
                "n_types": len(common),
                "kendall_tau": float(tau) if tau is not None else float("nan"),
                "kendall_p":  float(tau_p) if tau_p is not None else float("nan"),
                "spearman_rho": float(rho) if rho is not None else float("nan"),
                "spearman_p":   float(rho_p) if rho_p is not None else float("nan"),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Convergence / fixed-point (§16.6)
# ----------------------------------------------------------------------


def text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Cheap normalised levenshtein-substitute via SequenceMatcher
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def convergence_by_case(
    iter_compress: pd.DataFrame,
    iter_retention: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One row per (case, condition, model, prompt, budget). Detects
    when text + retained-fact set + length all stabilise."""
    rows = []
    key_cols = ["case_id", "condition_id", "compressor_model",
                "prompt_variant", "budget_chars"]
    for keys, grp in iter_compress.groupby(key_cols, dropna=False):
        grp = grp.sort_values("round")
        texts = list(grp["compressed_text"].fillna(""))
        rounds = list(grp["round"])

        # Retained fact sets per round if scores are available
        fact_sets: Dict[int, set] = {}
        if iter_retention is not None:
            sub = iter_retention[
                (iter_retention["case_id"] == keys[0]) &
                (iter_retention["condition_id"] == keys[1]) &
                (iter_retention["compressor_model"] == keys[2]) &
                (iter_retention["prompt_variant"] == keys[3]) &
                (iter_retention["budget_chars"] == keys[4])
            ]
            for r in rounds:
                f = sub[(sub["round"] == r) & (sub["retained_binary"])]
                fact_sets[r] = set(f["fact_id"].tolist())

        converged_at: Optional[int] = None
        for i in range(1, len(texts)):
            sim = text_similarity(texts[i], texts[i - 1])
            if fact_sets:
                fj = jaccard(fact_sets.get(rounds[i], set()),
                             fact_sets.get(rounds[i - 1], set()))
            else:
                fj = 1.0
            len_change = (
                abs(len(texts[i]) - len(texts[i - 1])) /
                max(len(texts[i - 1]), 1)
            )
            if sim >= 0.95 and fj >= 0.95 and len_change <= 0.02:
                converged_at = rounds[i]
                break

        row = {
            "case_id":         keys[0],
            "condition_id":    keys[1],
            "compressor_model": keys[2],
            "prompt_variant":  keys[3],
            "budget_chars":    keys[4],
            "converged":       converged_at is not None,
            "convergence_round": converged_at if converged_at else -1,
            "final_output_chars": int(grp["output_chars"].iloc[-1]),
            "final_round": int(grp["round"].iloc[-1]),
        }
        if fact_sets:
            final_round = rounds[-1]
            sub = iter_retention[
                (iter_retention["case_id"] == keys[0]) &
                (iter_retention["condition_id"] == keys[1]) &
                (iter_retention["compressor_model"] == keys[2]) &
                (iter_retention["prompt_variant"] == keys[3]) &
                (iter_retention["budget_chars"] == keys[4]) &
                (iter_retention["round"] == final_round)
            ]
            needed_subset    = sub[sub["need_label"] == 1]
            narrative_subset = sub[sub["coarse_group"] == "NARRATIVE"]
            exec_subset      = sub[sub["coarse_group"] == "EXECUTABLE"]
            row["final_fact_count"] = int(sub["retained_binary"].sum())
            row["needed_fact_recall_at_convergence"]      = float(
                needed_subset["retained_binary"].mean() if len(needed_subset) else float("nan"))
            row["narrative_fact_recall_at_convergence"]   = float(
                narrative_subset["retained_binary"].mean() if len(narrative_subset) else float("nan"))
            row["executable_fact_recall_at_convergence"]  = float(
                exec_subset["retained_binary"].mean() if len(exec_subset) else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "need_effect_by_type",
    "surface_dominance_regression",
    "preference_inversion_rate",
    "condition_responsiveness",
    "survival_by_round_type",
    "half_life_table",
    "hazard_by_round_type",
    "ausc_by_type",
    "hierarchy_rank_by_model",
    "cross_model_hierarchy_similarity",
    "convergence_by_case",
    "text_similarity",
    "jaccard",
]
