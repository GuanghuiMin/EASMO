"""M3 driver — cross-agent transfer degradation.

Re-uses oracle memories from M1 (``oracle_memories.jsonl``), re-runs each
agent on each *other* agent's memory, measures task-drop, and correlates
that against a TV-divergence between the involved agents.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from motivation.agents import get_agent
from motivation.data import load_contexts
from motivation.llm import MinimaxClient
from motivation.metrics import linear_fit_r2, spearman
from motivation.transfer import (
    TransferRow, _policy_divergence, cross_agent_eval,
    mean_match_rate, mean_overlap_rate,
)
from motivation.utils import (
    load_config, read_jsonl, seed_everything, setup_logging, write_json,
)
from motivation.wandb_utils import WandBRun

_logger = setup_logging("scripts.run_m3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False
    out_dir = Path(cfg["experiment"]["output_dir"])
    oracle_path = out_dir / "oracle_memories.jsonl"
    if not oracle_path.exists():
        _logger.error("Missing %s — run M1 first.", oracle_path)
        sys.exit(2)

    seed = int(cfg["experiment"]["seed"])
    seed_everything(seed)

    llm_cfg = cfg["llm"]
    client = MinimaxClient(
        base_url=llm_cfg["base_url"], model_id=llm_cfg["model_id"],
        api_key=llm_cfg.get("api_key", "EMPTY"),
        request_timeout_s=float(llm_cfg.get("request_timeout_s", 180)),
        max_concurrent=int(llm_cfg.get("max_concurrent", 8)),
    )
    if not client.ping():
        _logger.error("MiniMax endpoint not reachable.")
        sys.exit(2)

    contexts = load_contexts(cfg["data"])
    ctx_by_id = {c.context_id: c for c in contexts}
    agents = {a["scaffold"]: get_agent(a["scaffold"]) for a in cfg["agents"]}
    agent_ids = sorted({a.id for a in agents.values()})

    # Group oracle records by (context_id, budget) → {agent_id: record}
    records = list(read_jsonl(oracle_path))
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in records:
        grouped[(r["context_id"], int(r["budget"]))][r["agent_id"]] = r

    samples_per_state = int(cfg["samples_per_state"])

    rows: list[TransferRow] = []
    csv_path = out_dir / "transfer_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "context_id", "budget", "source_agent", "target_agent",
            # binary (legacy)
            "action_match_self", "action_match_cross", "task_drop",
            # continuous (1 - TV) — added 2026-05-24 audit
            "overlap_self", "overlap_cross", "overlap_drop",
            "policy_divergence",
        ])

        with WandBRun(cfg, name=f"{cfg['experiment']['name']}-m3", job_type="m3") as run:
            t0 = time.time()
            done = 0
            total = sum(len(by_agent) * (len(by_agent) - 1) for by_agent in grouped.values())
            run.log({"setup/total_transfer_runs": total})

            # Cache full-context distributions so we don't recompute them.
            full_dists_cache: dict[tuple, dict] = {}
            div_cache: dict[tuple, float] = {}

            for (ctx_id, budget), by_agent in grouped.items():
                ctx = ctx_by_id.get(ctx_id)
                if ctx is None:
                    _logger.warning("Context %s missing from current pool; skipping.", ctx_id)
                    continue

                # Pre-compute π_target(·|s; full) for each target agent.
                target_full: dict[str, dict] = {}
                for agent_id, rec in by_agent.items():
                    spec = next(a for a in agents.values() if a.id == agent_id)
                    cache_key = (ctx_id, agent_id, "full")
                    if cache_key not in full_dists_cache:
                        full_dists_cache[cache_key] = cross_agent_eval(
                            client, spec, ctx.context_text, ctx,
                            samples_per_state=samples_per_state,
                            base_seed=seed + 99,
                        )
                    target_full[agent_id] = full_dists_cache[cache_key]

                for source_agent_id, src_rec in by_agent.items():
                    src_spec = next(a for a in agents.values() if a.id == source_agent_id)
                    for target_agent_id, tgt_rec in by_agent.items():
                        if target_agent_id == source_agent_id:
                            continue
                        tgt_spec = next(a for a in agents.values() if a.id == target_agent_id)

                        # δ(src, tgt) — policy divergence (cached per pair+context).
                        div_key = (ctx_id, source_agent_id, target_agent_id)
                        if div_key not in div_cache:
                            div_cache[div_key] = _policy_divergence(
                                client, src_spec, tgt_spec, ctx,
                                samples_per_state, base_seed=seed + 200,
                            )
                        div = div_cache[div_key]

                        # Self baseline: target_agent with its OWN oracle memory.
                        self_dists = cross_agent_eval(
                            client, tgt_spec, tgt_rec["memory_text"], ctx,
                            samples_per_state=samples_per_state,
                            base_seed=seed + 301,
                        )
                        self_match = mean_match_rate(target_full[target_agent_id], self_dists)
                        self_overlap = mean_overlap_rate(target_full[target_agent_id], self_dists)

                        # Cross: target_agent fed source_agent's memory.
                        cross_dists = cross_agent_eval(
                            client, tgt_spec, src_rec["memory_text"], ctx,
                            samples_per_state=samples_per_state,
                            base_seed=seed + 401,
                        )
                        cross_match = mean_match_rate(target_full[target_agent_id], cross_dists)
                        cross_overlap = mean_overlap_rate(target_full[target_agent_id], cross_dists)

                        row = TransferRow(
                            context_id=ctx_id, budget=budget,
                            source_agent=source_agent_id, target_agent=target_agent_id,
                            action_match_self=self_match,
                            action_match_cross=cross_match,
                            task_drop=self_match - cross_match,
                            overlap_self=self_overlap,
                            overlap_cross=cross_overlap,
                            overlap_drop=self_overlap - cross_overlap,
                            policy_divergence=div,
                        )
                        rows.append(row)
                        w.writerow([row.context_id, row.budget,
                                    row.source_agent, row.target_agent,
                                    row.action_match_self, row.action_match_cross,
                                    row.task_drop,
                                    row.overlap_self, row.overlap_cross,
                                    row.overlap_drop,
                                    row.policy_divergence])
                        f.flush()
                        done += 1
                        run.log({
                            "m3/progress/done": done,
                            "m3/progress/fraction": done / max(total, 1),
                            "m3/progress/elapsed_min": (time.time() - t0) / 60.0,
                        })

            # ------------------------------------------------------------------
            # Aggregate stats
            #
            # IMPORTANT — there are TWO ways to summarise the drop:
            #
            #   (a) Unconditional mean over ALL rows. Easy to compute, but
            #       it conflates two very different sources of zero-drop:
            #         * rows where the target's own oracle preserved
            #           behaviour and cross still preserved it
            #           (genuine "no drop"), AND
            #         * rows where the target's own oracle was ALSO 0
            #           (no signal to drop from — pure noise).
            #
            #   (b) Conditional mean over rows where ``self_match == 1.0``.
            #       This is the *publishable* number: "given that the
            #       agent's own oracle works on this triple, how often does
            #       swapping to another agent's oracle break it?".
            #
            # Both are reported so reviewers can see both.
            # ------------------------------------------------------------------

            # ---- Binary / legacy aggregate -----------------------------
            drops = [r.task_drop for r in rows]
            divs = [r.policy_divergence for r in rows]

            signal_rows = [r for r in rows if r.action_match_self >= 0.999]
            cond_drops = [1.0 - r.action_match_cross for r in signal_rows]
            cond_divs  = [r.policy_divergence       for r in signal_rows]
            mean_cond_drop = (sum(cond_drops) / len(cond_drops)) if cond_drops else 0.0

            # ---- Continuous / overlap aggregate (PRIMARY post-audit) ---
            OVERLAP_SIGNAL_THRESHOLD = 0.5  # mirror instance_noise_test
            ov_drops = [r.overlap_drop for r in rows]
            ov_signal_rows = [
                r for r in rows if r.overlap_self >= OVERLAP_SIGNAL_THRESHOLD
            ]
            ov_cond_drops = [
                1.0 - r.overlap_cross for r in ov_signal_rows
            ]
            ov_cond_divs = [r.policy_divergence for r in ov_signal_rows]
            mean_ov_cond_drop = (
                sum(ov_cond_drops) / len(ov_cond_drops)
            ) if ov_cond_drops else 0.0

            summary = {
                "m3_total_runs": len(rows),

                # ---- Binary (legacy) ----
                "m3_mean_task_drop": (sum(drops) / len(drops)) if drops else 0.0,
                "m3_signal_rows": len(signal_rows),
                "m3_signal_fraction": len(signal_rows) / max(len(rows), 1),
                "m3_mean_conditional_drop": mean_cond_drop,
                "m3_conditional_spearman_drop_vs_div": spearman(cond_divs, cond_drops),
                "m3_conditional_linear_fit_r2": linear_fit_r2(cond_divs, cond_drops),
                "m3_spearman_drop_vs_div": spearman(divs, drops),
                "m3_linear_fit_r2": linear_fit_r2(divs, drops),

                # ---- Continuous overlap (PRIMARY) ----
                "m3_overlap_signal_threshold": OVERLAP_SIGNAL_THRESHOLD,
                "m3_overlap_signal_rows": len(ov_signal_rows),
                "m3_overlap_signal_fraction": (
                    len(ov_signal_rows) / max(len(rows), 1)
                ),
                "m3_overlap_mean_drop_unconditional": (
                    (sum(ov_drops) / len(ov_drops)) if ov_drops else 0.0
                ),
                "m3_overlap_mean_conditional_drop": mean_ov_cond_drop,
                "m3_overlap_conditional_spearman_drop_vs_div": spearman(
                    ov_cond_divs, ov_cond_drops
                ),
                "m3_overlap_conditional_linear_fit_r2": linear_fit_r2(
                    ov_cond_divs, ov_cond_drops
                ),
            }
            summary["m3_pass_unconditional_drop_15"] = (
                summary["m3_mean_task_drop"] > 0.15
            )
            summary["m3_pass_conditional_drop_15"] = (mean_cond_drop > 0.15)
            summary["m3_pass_signal_fraction_10"] = (
                summary["m3_signal_fraction"] >= 0.10
            )
            summary["m3_pass_overlap_conditional_drop_15"] = (
                mean_ov_cond_drop > 0.15
            )
            summary["m3_pass_overlap_signal_fraction_10"] = (
                summary["m3_overlap_signal_fraction"] >= 0.10
            )

            # Per-(source -> target) breakdown (conditional + unconditional).
            by_pair: dict[tuple, list] = defaultdict(list)
            for r in rows:
                by_pair[(r.source_agent, r.target_agent)].append(r)
            pair_breakdown = []
            for (src, tgt), rs in sorted(by_pair.items()):
                d = [r.task_drop for r in rs]
                v = [r.policy_divergence for r in rs]
                signal = [r for r in rs if r.action_match_self >= 0.999]
                cond_d = [1.0 - r.action_match_cross for r in signal]
                mean_drop = sum(d) / len(d) if d else 0.0
                mean_cd = sum(cond_d) / len(cond_d) if cond_d else 0.0

                ov_signal = [
                    r for r in rs
                    if r.overlap_self >= OVERLAP_SIGNAL_THRESHOLD
                ]
                ov_cond_d = [1.0 - r.overlap_cross for r in ov_signal]
                ov_d = [r.overlap_drop for r in rs]
                mean_ov_drop = sum(ov_d) / len(ov_d) if ov_d else 0.0
                mean_ov_cd = sum(ov_cond_d) / len(ov_cond_d) if ov_cond_d else 0.0

                pair_breakdown.append({
                    "source_agent": src, "target_agent": tgt,
                    "n": len(rs),
                    "n_signal": len(signal),
                    "n_overlap_signal": len(ov_signal),
                    "mean_task_drop_unconditional": mean_drop,
                    "mean_conditional_drop": mean_cd,
                    "mean_overlap_drop_unconditional": mean_ov_drop,
                    "mean_overlap_conditional_drop": mean_ov_cd,
                    "mean_policy_div": sum(v) / len(v) if v else 0.0,
                })
                run.log({
                    f"m3/task_drop/{src}_to_{tgt}": mean_drop,
                    f"m3/conditional_drop/{src}_to_{tgt}": mean_cd,
                    f"m3/overlap_conditional_drop/{src}_to_{tgt}": mean_ov_cd,
                })
            summary["m3_pair_breakdown"] = pair_breakdown
            run.summary(**summary)
            run.log_table(
                "m3/transfer_rows",
                columns=["context_id", "budget", "source_agent", "target_agent",
                         "action_match_self", "action_match_cross", "task_drop",
                         "overlap_self", "overlap_cross", "overlap_drop",
                         "policy_divergence"],
                rows=[[r.context_id, r.budget, r.source_agent, r.target_agent,
                       r.action_match_self, r.action_match_cross, r.task_drop,
                       r.overlap_self, r.overlap_cross, r.overlap_drop,
                       r.policy_divergence] for r in rows],
            )
            run.log_artifact(str(csv_path), artifact_name="transfer_results", artifact_type="dataset")

    write_json(out_dir / "m3_summary.json", summary)
    client.close()
    _logger.info("M3 summary: %s", summary)


if __name__ == "__main__":
    main()
