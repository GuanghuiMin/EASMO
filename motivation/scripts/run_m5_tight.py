"""Focused M5 ablation: is the selector itself *the* source of variance,
or can within-agent variance be driven to ~0 with stricter sampling?

Setting:
* T ∈ {0.0, 0.1}
* within_k = 10 (was 3)
* contexts = 10 (subsampled)
* budget = first config budget

If within-agent Jaccard climbs to ≥ 0.7, then the M2 cross-agent
overlap (~0.30) is comfortably below within, the gap exceeds 10pp,
and "policy-dependent memory" survives M5. If within stays at ~0.4,
the selector is intrinsically high-variance and M5 fails — we go to
Plan B / Plan C.

Usage:
    python -m scripts.run_m5_tight --config configs/default_locomo.yaml \\
        --temperatures 0.0 0.1 --within-k 10 --n-contexts 10
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from motivation.agents import get_agent
from motivation.data import load_contexts
from motivation.llm import MinimaxClient
from motivation.metrics import jaccard, tokenize_simple
from motivation.selector_ablation import _gen_candidates
from motivation.utils import load_config, seed_everything, setup_logging, write_json
from motivation.wandb_utils import WandBRun

_logger = setup_logging("scripts.run_m5_tight")


def _pairwise_jaccard(memories):
    if len(memories) < 2:
        return 0.0, 0
    jacs = []
    for a, b in combinations(memories, 2):
        jacs.append(jaccard(tokenize_simple(a), tokenize_simple(b)))
    return mean(jacs), len(jacs)


def _pairwise_sbert(memories):
    if len(memories) < 2:
        return 0.0, 0
    try:
        from motivation.semantic import precompute_embeddings
        import numpy as np
        emb = precompute_embeddings(memories)
    except Exception:
        return 0.0, 0
    sims = []
    for a, b in combinations(memories, 2):
        va, vb = emb.get(a), emb.get(b)
        if va is not None and vb is not None:
            sims.append(float(np.dot(va, vb)))
    return (mean(sims) if sims else 0.0), len(sims)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.1])
    parser.add_argument("--within-k", type=int, default=10)
    parser.add_argument("--n-contexts", type=int, default=10)
    parser.add_argument("--budget", type=int, default=0,
                        help="0 = first budget from config.budgets")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    seed = int(cfg["experiment"]["seed"])
    seed_everything(seed)
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "m5_tight.csv"

    llm_cfg = cfg["llm"]
    client = MinimaxClient(
        base_url=llm_cfg["base_url"], model_id=llm_cfg["model_id"],
        api_key=llm_cfg.get("api_key", "EMPTY"),
        max_concurrent=int(llm_cfg.get("max_concurrent", 8)),
    )
    if not client.ping():
        _logger.error("MiniMax endpoint not reachable.")
        sys.exit(2)

    contexts = load_contexts(cfg["data"])[: args.n_contexts]
    agents = [get_agent(a["scaffold"]) for a in cfg["agents"]]
    budget = args.budget or int(cfg["budgets"][0])

    _logger.info(
        "M5-tight: n_contexts=%d, budget=%d, within_k=%d, temperatures=%s",
        len(contexts), budget, args.within_k, args.temperatures,
    )

    with open(out_csv, "w", newline="", encoding="utf-8") as f, \
         WandBRun(cfg, name=f"{cfg['experiment']['name']}-m5tight",
                  job_type="m5tight") as run:
        w = csv.writer(f)
        w.writerow([
            "temperature", "context_id", "budget", "setting",
            "mean_jaccard", "mean_sbert", "n_pairs",
        ])
        all_rows = []

        for T in args.temperatures:
            _logger.info("=== Temperature %.2f ===", T)
            for ci, ctx in enumerate(contexts):
                # within-agent: K candidates per agent
                per_agent: dict[str, list[str]] = {}
                for ai, agent in enumerate(agents):
                    mems = _gen_candidates(
                        client, agent, ctx, budget, args.within_k,
                        seed=seed + 7919 * ci + 17 * ai,
                        temperature=T,
                    )
                    per_agent[agent.id] = mems
                    j, nj = _pairwise_jaccard(mems)
                    s, ns = _pairwise_sbert(mems)
                    all_rows.append({
                        "temperature": T, "context_id": ctx.context_id,
                        "budget": budget, "setting": f"within:{agent.id}",
                        "mean_jaccard": j, "mean_sbert": s, "n_pairs": nj,
                    })
                    w.writerow([T, ctx.context_id, budget, f"within:{agent.id}",
                                j, s, nj])
                    _logger.info(
                        "[T=%.2f | %s | %s] within  jacc=%.3f sbert=%.3f (n=%d)",
                        T, ctx.context_id, agent.id, j, s, nj,
                    )
                # cross-agent: pair every A.cand_i with every B.cand_j
                import numpy as np
                from motivation.semantic import precompute_embeddings
                all_memes = sum(per_agent.values(), [])
                emb = precompute_embeddings(all_memes) if all_memes else {}
                for a, b in combinations(agents, 2):
                    ma = per_agent.get(a.id, [])
                    mb = per_agent.get(b.id, [])
                    jacs, sims = [], []
                    for x in ma:
                        for y in mb:
                            jacs.append(jaccard(tokenize_simple(x), tokenize_simple(y)))
                            vx, vy = emb.get(x), emb.get(y)
                            if vx is not None and vy is not None:
                                sims.append(float(np.dot(vx, vy)))
                    mj = mean(jacs) if jacs else 0.0
                    ms = mean(sims) if sims else 0.0
                    all_rows.append({
                        "temperature": T, "context_id": ctx.context_id,
                        "budget": budget,
                        "setting": f"cross:{a.id}_vs_{b.id}",
                        "mean_jaccard": mj, "mean_sbert": ms, "n_pairs": len(jacs),
                    })
                    w.writerow([T, ctx.context_id, budget,
                                f"cross:{a.id}_vs_{b.id}",
                                mj, ms, len(jacs)])
                    _logger.info(
                        "[T=%.2f | %s | %s_vs_%s] cross  jacc=%.3f sbert=%.3f (n=%d)",
                        T, ctx.context_id, a.id, b.id, mj, ms, len(jacs),
                    )

        # Aggregate per-T
        summary = {}
        for T in args.temperatures:
            wj = [r["mean_jaccard"] for r in all_rows if r["temperature"] == T and r["setting"].startswith("within:")]
            cj = [r["mean_jaccard"] for r in all_rows if r["temperature"] == T and r["setting"].startswith("cross:")]
            ws = [r["mean_sbert"]   for r in all_rows if r["temperature"] == T and r["setting"].startswith("within:")]
            cs = [r["mean_sbert"]   for r in all_rows if r["temperature"] == T and r["setting"].startswith("cross:")]
            def _m(xs): return mean(xs) if xs else 0.0
            def _s(xs): return stdev(xs) if len(xs) > 1 else 0.0
            s_row = {
                "within_jaccard_mean":  _m(wj),
                "within_jaccard_std":   _s(wj),
                "cross_jaccard_mean":   _m(cj),
                "cross_jaccard_std":    _s(cj),
                "jaccard_gap":          _m(wj) - _m(cj),
                "within_sbert_mean":    _m(ws),
                "cross_sbert_mean":     _m(cs),
                "sbert_gap":            _m(ws) - _m(cs),
                "pass_lexical_10pp":    (_m(wj) - _m(cj)) > 0.10,
                "pass_semantic_10pp":   (_m(ws) - _m(cs)) > 0.10,
            }
            summary[f"T={T}"] = s_row
            run.summary(**{f"m5tight_T{T}_{k}": v for k, v in s_row.items()})
            print(f"\n=== T={T} ===")
            for k, v in s_row.items():
                if isinstance(v, float):
                    print(f"  {k:30s} {v:.3f}")
                else:
                    print(f"  {k:30s} {v}")

    write_json(out_dir / "m5_tight_summary.json", summary)
    client.close()


if __name__ == "__main__":
    main()
