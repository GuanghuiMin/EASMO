"""Metrics for motivation_v8 (spec §13).

Reuses much of v7's metric machinery and adds:

  * §13.10 Fixed-point composition (per chain, at convergence round)
  * §13.11 Fixed-point need shift (Δ_need^∞ per fact type)
  * §13.13 Basin-of-attraction metrics (pairwise fact-jaccard,
    fact-type L1, contraction ratio)
  * §13.14 Budget compliance

We delegate single-round Δ_need / SDI / PIR / CRS / survival /
half-life / hazard / AUSC / hierarchy stability to local functions
that mirror v7 ``motivation_v7.metrics`` (no cross-package import to
keep v8 self-contained).
"""

from __future__ import annotations

from collections import defaultdict
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
# Bootstrap helpers
# ----------------------------------------------------------------------


def _bootstrap_ci(
    values: Sequence[float], n_iter: int = 1000, alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, v.size, size=(n_iter, v.size))
    samples = v[idx].mean(axis=1)
    return float(np.quantile(samples, alpha/2)), float(np.quantile(samples, 1-alpha/2))


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
        samples[k] = (a[rng.integers(0, a.size, size=a.size)].mean() -
                      b[rng.integers(0, b.size, size=b.size)].mean())
    return float(np.quantile(samples, alpha/2)), float(np.quantile(samples, 1-alpha/2))


# ----------------------------------------------------------------------
# Single-round metrics (mirror v7)
# ----------------------------------------------------------------------


def need_effect_by_type(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("model", "prompt_family",
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
        coarse = grp["coarse_group"].iloc[0] if "coarse_group" in grp else ""
        row.update({
            "coarse_group": coarse,
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
            "ci_low": float(lo), "ci_high": float(hi),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_logit_with_r2(X: pd.DataFrame, y: np.ndarray) -> Tuple[float, dict]:
    if not _HAS_SM or len(X) == 0 or y.std() == 0:
        return float("nan"), {}
    Xc = sm.add_constant(X, has_constant="add")
    try:
        m = sm.Logit(y, Xc).fit(disp=False, method="bfgs", maxiter=100)
        return float(m.prsquared), {n: float(c) for n, c in zip(Xc.columns, m.params)}
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError):
        return float("nan"), {}
    except Exception:
        return float("nan"), {}


def surface_dominance_regression(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("model", "prompt_family", "budget_chars"),
) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(list(group_cols), dropna=False):
        y = grp["retained_binary"].astype(int).values
        need = grp[["need_label"]].astype(float)
        types = pd.get_dummies(grp["fact_type"], prefix="ft", drop_first=True).astype(float)
        r2_need, _ = _fit_logit_with_r2(need, y)
        r2_type, _ = _fit_logit_with_r2(types, y)
        Xb = pd.concat([need.reset_index(drop=True), types.reset_index(drop=True)], axis=1)
        r2_both, coefs_both = _fit_logit_with_r2(Xb, y)
        sdi = float("nan")
        if np.isfinite(r2_type) and np.isfinite(r2_need):
            sdi = (r2_type - r2_need) / (r2_type + r2_need + 1e-8)
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update({
            "n": int(len(grp)),
            "r2_need": float(r2_need), "r2_type": float(r2_type),
            "r2_both": float(r2_both), "sdi": float(sdi),
            "coef_need_in_both": coefs_both.get("need_label", float("nan")),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def preference_inversion_rate(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("model", "prompt_family", "budget_chars"),
    n_boot: int = 500,
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for keys, grp in df.groupby(list(group_cols), dropna=False):
        nc = grp[(grp["need_label"] == 1) &
                  (grp["coarse_group"].isin(["EXECUTABLE", "CONTROL"]))]
        un = grp[(grp["need_label"] == 0) & (grp["coarse_group"] == "NARRATIVE")]
        nc_by = nc.groupby("case_id"); un_by = un.groupby("case_id")
        common = set(nc_by.groups) & set(un_by.groups)
        inv: List[int] = []
        for cid in common:
            for _, n_row in nc_by.get_group(cid).iterrows():
                for _, u_row in un_by.get_group(cid).iterrows():
                    inv.append(int(int(u_row["retained_binary"]) == 1 and
                                    int(n_row["retained_binary"]) == 0))
        if not inv:
            continue
        pir = float(np.mean(inv))
        lo, hi = _bootstrap_ci(inv, n_iter=n_boot, rng=rng)
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update({"n_pairs": len(inv), "preference_inversion_rate": pir,
                    "ci_low": float(lo), "ci_high": float(hi)})
        rows.append(row)
    return pd.DataFrame(rows)


def condition_responsiveness(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("model", "prompt_family",
                                  "budget_chars", "fact_type"),
) -> pd.DataFrame:
    df_n = df[df["need_label"] == 1][["case_id","fact_id","fact_type","coarse_group",
        "model","prompt_family","budget_chars","retention_score"]].rename(
        columns={"retention_score": "score_needed"})
    df_u = df[df["need_label"] == 0][["case_id","fact_id","model","prompt_family",
        "budget_chars","retention_score"]].rename(
        columns={"retention_score": "score_unneeded"})
    merged = df_n.merge(df_u, on=["case_id","fact_id","model","prompt_family","budget_chars"])
    if merged.empty:
        return pd.DataFrame()
    merged["crs"] = merged["score_needed"] - merged["score_unneeded"]
    rows = []
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
# Iterative metrics
# ----------------------------------------------------------------------


def survival_by_round_type(
    df_iter: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("model", "prompt_family", "init_type",
                                  "condition_type", "round", "fact_type"),
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
            "ci_low": float(lo), "ci_high": float(hi),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def half_life_table(
    surv: pd.DataFrame,
    *,
    rounds_cap: int,
    threshold: float = 0.5,
    group_cols: Sequence[str] = ("model", "prompt_family", "init_type",
                                  "condition_type", "fact_type"),
) -> pd.DataFrame:
    rows = []
    for keys, grp in surv.groupby(list(group_cols), dropna=False):
        grp = grp.sort_values("round")
        below = grp[grp["survival_rate"] <= threshold]
        if len(below):
            half = int(below["round"].iloc[0])
            cen = False
        else:
            half = rounds_cap + 1
            cen = True
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update({
            "coarse_group": grp["coarse_group"].iloc[0] if "coarse_group" in grp else "",
            "half_life": half,
            "half_life_censored": cen,
            "n_rounds": int(grp["round"].max()),
            "final_survival": float(grp["survival_rate"].iloc[-1]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def ausc_by_type(
    surv: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("model", "prompt_family", "init_type",
                                  "condition_type", "fact_type"),
) -> pd.DataFrame:
    rows = []
    for keys, grp in surv.groupby(list(group_cols), dropna=False):
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update({
            "ausc": float(grp["survival_rate"].sum()),
            "n_rounds": int(grp["round"].max()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def hazard_by_round_type(
    surv: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("model", "prompt_family", "init_type",
                                  "condition_type", "fact_type"),
) -> pd.DataFrame:
    rows = []
    for keys, grp in surv.groupby(list(group_cols), dropna=False):
        grp = grp.sort_values("round")
        prev = None
        for _, r in grp.iterrows():
            cur = float(r["survival_rate"])
            hazard = (1.0 - cur) if prev is None else (1.0 - cur / (prev + 1e-8)
                                                       if prev > 0 else float("nan"))
            row = {col: keys[i] for i, col in enumerate(group_cols)}
            row.update({
                "round": int(r["round"]),
                "hazard": float(hazard),
                "survival_rate": cur,
            })
            rows.append(row)
            prev = cur
    return pd.DataFrame(rows)


def hierarchy_rank_by_model_prompt(half_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in half_df.groupby(
        ["model", "prompt_family", "init_type", "condition_type"], dropna=False,
    ):
        ranked = grp.sort_values(
            ["half_life", "final_survival"], ascending=[False, False]
        ).reset_index(drop=True)
        for i, r in ranked.iterrows():
            row = {"model": keys[0], "prompt_family": keys[1],
                   "init_type": keys[2], "condition_type": keys[3]}
            row.update({
                "fact_type": r["fact_type"],
                "coarse_group": r["coarse_group"],
                "rank": int(i + 1),
                "half_life": int(r["half_life"]),
                "final_survival": float(r["final_survival"]),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def cross_model_prompt_hierarchy_similarity(ranks: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import kendalltau, spearmanr
    pivots = {}
    for keys, grp in ranks.groupby(
        ["model", "prompt_family", "init_type", "condition_type"], dropna=False,
    ):
        pivots[keys] = dict(zip(grp["fact_type"], grp["rank"]))
    rows = []
    keys_list = list(pivots.keys())
    for i, k1 in enumerate(keys_list):
        for k2 in keys_list[i + 1:]:
            common = sorted(set(pivots[k1]) & set(pivots[k2]))
            if len(common) < 3:
                continue
            v1 = [pivots[k1][t] for t in common]
            v2 = [pivots[k2][t] for t in common]
            tau, tau_p = kendalltau(v1, v2)
            rho, rho_p = spearmanr(v1, v2)
            rows.append({
                "model_a": k1[0], "prompt_a": k1[1],
                "init_a": k1[2],  "cond_a": k1[3],
                "model_b": k2[0], "prompt_b": k2[1],
                "init_b": k2[2],  "cond_b": k2[3],
                "n_types": len(common),
                "kendall_tau": float(tau) if tau is not None else float("nan"),
                "kendall_p": float(tau_p) if tau_p is not None else float("nan"),
                "spearman_rho": float(rho) if rho is not None else float("nan"),
                "spearman_p": float(rho_p) if rho_p is not None else float("nan"),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Fixed-point convergence + composition (spec §13.9–13.11)
# ----------------------------------------------------------------------


def _text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def convergence_by_chain(
    chains: pd.DataFrame,
    retention: pd.DataFrame,
) -> pd.DataFrame:
    """Detect when each chain reaches a fixed point.

    Both inputs must include ``chain_id`` and ``round``.
    """
    rows = []
    for chain_id, grp in chains.groupby("chain_id"):
        grp = grp.sort_values("round")
        # Build per-round retained-fact sets from retention table
        ret_chain = retention[retention["chain_id"] == chain_id]
        retained_by_round: Dict[int, set] = {}
        for r, sub in ret_chain.groupby("round"):
            retained_by_round[int(r)] = set(sub[sub["retained_binary"]]["fact_id"])

        contexts = list(grp["context_text"].fillna(""))
        rounds = list(grp["round"])

        converged_at: Optional[int] = None
        for i in range(1, len(contexts)):
            r_cur, r_prev = rounds[i], rounds[i - 1]
            sim = _text_similarity(contexts[i], contexts[i - 1])
            fj = _jaccard(retained_by_round.get(r_cur, set()),
                          retained_by_round.get(r_prev, set()))
            len_change = (
                abs(len(contexts[i]) - len(contexts[i - 1])) /
                max(len(contexts[i - 1]), 1)
            )
            if sim >= 0.95 and fj >= 0.95 and len_change <= 0.02:
                converged_at = r_cur
                break

        meta = grp.iloc[0]
        row = {
            "chain_id": chain_id,
            "case_id": meta.get("case_id"),
            "target_fact_id": meta.get("target_fact_id"),
            "target_fact_type": meta.get("target_fact_type"),
            "condition_type": meta.get("condition_type"),
            "prompt_family": meta.get("prompt_family"),
            "model": meta.get("model"),
            "init_type": meta.get("init_type"),
            "budget_chars": meta.get("budget_chars"),
            "converged": converged_at is not None,
            "convergence_round": converged_at if converged_at else -1,
            "final_round": int(grp["round"].iloc[-1]),
            "final_context_chars": int(grp["context_chars"].iloc[-1]),
        }
        # Per-group recall at fixed/final round
        fixed_round = converged_at if converged_at else int(grp["round"].iloc[-1])
        sub = ret_chain[ret_chain["round"] == fixed_round]
        row["final_fact_count"] = int(sub["retained_binary"].sum())
        for col, mask in [
            ("needed_fact_recall_fixed", sub["need_label"] == 1),
            ("narrative_fact_recall_fixed", sub["coarse_group"] == "NARRATIVE"),
            ("executable_fact_recall_fixed", sub["coarse_group"] == "EXECUTABLE"),
            ("control_fact_recall_fixed", sub["coarse_group"] == "CONTROL"),
        ]:
            seg = sub[mask]
            row[col] = float(seg["retained_binary"].mean()) if len(seg) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_point_composition_by_type(
    chains: pd.DataFrame,
    retention: pd.DataFrame,
    conv: pd.DataFrame,
) -> pd.DataFrame:
    """Per (model, prompt_family, init_type, condition_type, fact_type),
    retention at the *fixed* round of each chain."""
    # Mapping chain_id → fixed_round
    fixed_round_map = {}
    for _, r in conv.iterrows():
        cid = r["chain_id"]
        fr = int(r["convergence_round"]) if r["converged"] else int(r["final_round"])
        fixed_round_map[cid] = fr

    sub = retention[retention.apply(
        lambda r: r["round"] == fixed_round_map.get(r["chain_id"], -1), axis=1
    )]
    rows = []
    keys = ("model", "prompt_family", "init_type", "condition_type", "fact_type")
    for k, grp in sub.groupby(list(keys), dropna=False):
        row = {c: k[i] for i, c in enumerate(keys)}
        row.update({
            "coarse_group": grp["coarse_group"].iloc[0] if len(grp) else "",
            "n_facts": int(len(grp)),
            "survival_rate_fixed": float(grp["retained_binary"].mean()) if len(grp) else float("nan"),
            "survival_score_fixed": float(grp["retention_score"].mean()) if len(grp) else float("nan"),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_point_need_shift(
    chains: pd.DataFrame,
    retention: pd.DataFrame,
    conv: pd.DataFrame,
) -> pd.DataFrame:
    """Δ_need^∞ per fact type per (model, prompt_family)."""
    fixed_round_map = {}
    for _, r in conv.iterrows():
        cid = r["chain_id"]
        fr = int(r["convergence_round"]) if r["converged"] else int(r["final_round"])
        fixed_round_map[cid] = fr
    sub = retention[retention.apply(
        lambda r: r["round"] == fixed_round_map.get(r["chain_id"], -1), axis=1
    )]
    rows = []
    keys = ("model", "prompt_family", "fact_type")
    for k, grp in sub.groupby(list(keys), dropna=False):
        v1 = grp[grp["need_label"] == 1]
        v0 = grp[grp["need_label"] == 0]
        if v1.empty and v0.empty:
            continue
        m1 = float(v1["retained_binary"].mean()) if len(v1) else float("nan")
        m0 = float(v0["retained_binary"].mean()) if len(v0) else float("nan")
        row = {c: k[i] for i, c in enumerate(keys)}
        row.update({
            "coarse_group": grp["coarse_group"].iloc[0] if len(grp) else "",
            "n_needed": int(len(v1)), "n_unneeded": int(len(v0)),
            "recall_needed_fixed": m1,
            "recall_unneeded_fixed": m0,
            "delta_need_infty": (m1 - m0) if (np.isfinite(m1) and np.isfinite(m0))
                                else float("nan"),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Basin-of-attraction (§13.13)
# ----------------------------------------------------------------------


def basin_metrics(
    chains: pd.DataFrame,
    retention: pd.DataFrame,
    conv: pd.DataFrame,
) -> pd.DataFrame:
    """For each (case_id, model, prompt_family, condition_type) with
    multiple init_types, compute initial pairwise distances between the
    init contexts and final pairwise distances between fixed points.

    Distance metrics:
      - fact_jaccard_distance = 1 - Jaccard(retained_ids_a, retained_ids_b)
      - fact_type_l1_distance = L1 of normalised fact-type histograms
      - text_jaccard_distance = 1 - token-Jaccard (proxy for embedding sim)
    """
    fixed_round_map = {}
    for _, r in conv.iterrows():
        cid = r["chain_id"]
        fr = int(r["convergence_round"]) if r["converged"] else int(r["final_round"])
        fixed_round_map[cid] = fr

    # Per chain: initial context (round 0) + initial retained set, final context + retained set
    chain_state: Dict[str, dict] = {}
    for chain_id, grp in chains.groupby("chain_id"):
        grp = grp.sort_values("round")
        text_init = grp[grp["round"] == 0]["context_text"]
        text_init = text_init.iloc[0] if len(text_init) else ""
        fr = fixed_round_map.get(chain_id, int(grp["round"].iloc[-1]))
        text_fin_row = grp[grp["round"] == fr]
        text_fin = text_fin_row["context_text"].iloc[0] if len(text_fin_row) else ""
        meta = grp.iloc[0]
        chain_state[chain_id] = {
            "case_id": meta["case_id"],
            "model": meta["model"],
            "prompt_family": meta["prompt_family"],
            "condition_type": meta["condition_type"],
            "init_type": meta["init_type"],
            "text_init": text_init,
            "text_fin": text_fin,
            "fixed_round": fr,
        }

    # Per chain: retained_fact_id set at round 0 and at fixed_round
    ret_state: Dict[str, Dict[str, set]] = defaultdict(dict)
    type_hist: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for chain_id, grp in retention.groupby("chain_id"):
        for r in (0, chain_state.get(chain_id, {}).get("fixed_round", -1)):
            sub = grp[grp["round"] == r]
            ids = set(sub[sub["retained_binary"]]["fact_id"])
            ret_state[chain_id][f"round_{r}"] = ids
            hist = sub[sub["retained_binary"]]["fact_type"].value_counts().to_dict()
            type_hist[chain_id][f"round_{r}"] = hist

    def _tok_jaccard(a: str, b: str) -> float:
        toks_a = set(a.lower().split())
        toks_b = set(b.lower().split())
        return _jaccard(toks_a, toks_b)

    def _hist_l1(ha: Dict[str, int], hb: Dict[str, int]) -> float:
        keys = set(ha) | set(hb)
        if not keys:
            return 0.0
        sa = sum(ha.values()) or 1
        sb = sum(hb.values()) or 1
        return 0.5 * sum(abs(ha.get(k, 0) / sa - hb.get(k, 0) / sb) for k in keys)

    # Pair up chains by (case_id, model, prompt_family, condition_type)
    by_key: Dict[Tuple, List[str]] = defaultdict(list)
    for cid, st in chain_state.items():
        by_key[(st["case_id"], st["model"], st["prompt_family"], st["condition_type"])].append(cid)

    rows = []
    for key, chain_ids in by_key.items():
        if len(chain_ids) < 2:
            continue
        case_id, model, family, cond = key
        for i in range(len(chain_ids)):
            for j in range(i + 1, len(chain_ids)):
                a, b = chain_ids[i], chain_ids[j]
                init_a = chain_state[a]["init_type"]
                init_b = chain_state[b]["init_type"]
                # Initial pairwise distance (round 0)
                a0_ids = ret_state[a].get("round_0", set())
                b0_ids = ret_state[b].get("round_0", set())
                init_fj = 1.0 - _jaccard(a0_ids, b0_ids)
                init_l1 = _hist_l1(type_hist[a].get("round_0", {}),
                                    type_hist[b].get("round_0", {}))
                init_tj = 1.0 - _tok_jaccard(chain_state[a]["text_init"],
                                              chain_state[b]["text_init"])
                # Final pairwise distance
                a_fr = chain_state[a]["fixed_round"]
                b_fr = chain_state[b]["fixed_round"]
                af_ids = ret_state[a].get(f"round_{a_fr}", set())
                bf_ids = ret_state[b].get(f"round_{b_fr}", set())
                fin_fj = 1.0 - _jaccard(af_ids, bf_ids)
                fin_l1 = _hist_l1(type_hist[a].get(f"round_{a_fr}", {}),
                                   type_hist[b].get(f"round_{b_fr}", {}))
                fin_tj = 1.0 - _tok_jaccard(chain_state[a]["text_fin"],
                                              chain_state[b]["text_fin"])
                rows.append({
                    "case_id": case_id, "model": model,
                    "prompt_family": family, "condition_type": cond,
                    "init_a": init_a, "init_b": init_b,
                    "init_fact_jaccard_distance": init_fj,
                    "init_type_l1_distance": init_l1,
                    "init_text_distance": init_tj,
                    "fin_fact_jaccard_distance": fin_fj,
                    "fin_type_l1_distance": fin_l1,
                    "fin_text_distance": fin_tj,
                    "contraction_fact_jaccard": fin_fj / max(init_fj, 1e-8),
                    "contraction_type_l1": fin_l1 / max(init_l1, 1e-8),
                    "contraction_text": fin_tj / max(init_tj, 1e-8),
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Budget compliance (§13.14)
# ----------------------------------------------------------------------


def budget_compliance(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("model", "prompt_family", "budget_chars"),
    length_col: str = "compressed_chars",
    violation_col: str = "budget_violation",
) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(list(group_cols), dropna=False):
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update({
            "n": int(len(grp)),
            "violation_rate": float(grp[violation_col].astype(float).mean()) if violation_col in grp else float("nan"),
            "median_length": float(grp[length_col].median()) if length_col in grp else float("nan"),
            "p90_length": float(grp[length_col].quantile(0.9)) if length_col in grp else float("nan"),
            "p99_length": float(grp[length_col].quantile(0.99)) if length_col in grp else float("nan"),
        })
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "need_effect_by_type",
    "surface_dominance_regression",
    "preference_inversion_rate",
    "condition_responsiveness",
    "survival_by_round_type",
    "half_life_table",
    "ausc_by_type",
    "hazard_by_round_type",
    "hierarchy_rank_by_model_prompt",
    "cross_model_prompt_hierarchy_similarity",
    "convergence_by_chain",
    "fixed_point_composition_by_type",
    "fixed_point_need_shift",
    "basin_metrics",
    "budget_compliance",
]
