"""M3 augmented with LLM-as-judge task-success scoring.

Reuses ``outputs/<exp>/oracle_memories.jsonl`` (no oracle re-search). For
each (context, budget) group and each (source_agent, target_agent) pair,
samples N answers from the target agent given the source's compressed
memory, judges each against the gold answer, and reports a
task-success-rate-based transfer drop.

This gives a more direct "does compression preserve task correctness"
metric than the action-distribution-match in M3.

Outputs:
    outputs/<exp>/transfer_judge.csv
    outputs/<exp>/m3_judge_summary.json
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

from motivation.agents import get_agent, sample_action_distribution
from motivation.data import load_contexts
from motivation.judge import judge_batch
from motivation.llm import MinimaxClient
from motivation.utils import (
    load_config, read_jsonl, seed_everything, setup_logging, write_json,
)
from motivation.wandb_utils import WandBRun

_logger = setup_logging("scripts.run_m3_judge")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--max-budgets", type=int, default=0,
                        help="if >0, restrict to first N budgets (saves time)")
    parser.add_argument("--samples", type=int, default=0,
                        help="override samples_per_state (default: from config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    seed = int(cfg["experiment"]["seed"])
    seed_everything(seed)

    out_dir = Path(cfg["experiment"]["output_dir"])
    oracle_path = out_dir / "oracle_memories.jsonl"
    if not oracle_path.exists():
        _logger.error("Missing %s — run M1 first.", oracle_path)
        sys.exit(2)

    n_samples = args.samples or int(cfg["samples_per_state"])

    llm_cfg = cfg["llm"]
    client = MinimaxClient(
        base_url=llm_cfg["base_url"], model_id=llm_cfg["model_id"],
        api_key=llm_cfg.get("api_key", "EMPTY"),
        max_concurrent=int(llm_cfg.get("max_concurrent", 8)),
    )
    if not client.ping():
        _logger.error("MiniMax endpoint not reachable.")
        sys.exit(2)

    contexts = load_contexts(cfg["data"])
    ctx_by_id = {c.context_id: c for c in contexts}

    # Sub-set budgets if requested.
    budgets = list(cfg["budgets"])
    if args.max_budgets > 0:
        budgets = budgets[: args.max_budgets]
    budgets_set = {int(b) for b in budgets}

    records = list(read_jsonl(oracle_path))
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in records:
        b = int(r["budget"])
        if b not in budgets_set:
            continue
        grouped[(r["context_id"], b)][r["agent_id"]] = r
    _logger.info(
        "Loaded %d oracle records, grouped into %d (context, budget) cells.",
        len(records), len(grouped),
    )

    agents = {a["scaffold"]: get_agent(a["scaffold"]) for a in cfg["agents"]}

    csv_path = out_dir / "transfer_judge.csv"
    f_out = open(csv_path, "w", newline="", encoding="utf-8")
    w = csv.writer(f_out)
    w.writerow([
        "context_id", "budget", "source_agent", "target_agent",
        "success_self", "success_cross", "judge_drop",
        "n_samples",
    ])

    rows: list[dict] = []
    success_cache: dict[tuple, float] = {}      # (ctx_id, target_id, memory_text) -> success rate

    def _eval_target_on_memory(ctx, target_spec, memory_text, gold_answer, base_seed):
        key = (ctx.context_id, target_spec.id, memory_text)
        if key in success_cache:
            return success_cache[key]
        dist, results = sample_action_distribution(
            client, target_spec, memory_text,
            ctx.probe_states[0].state_text,
            n=n_samples, question=ctx.question,
            base_seed=base_seed, task_type=ctx.task_type,
        )
        candidates = [r.text for r in results if r.error is None and r.text]
        if not candidates:
            success_cache[key] = 0.0
            return 0.0
        score, _ = judge_batch(client, ctx.question, gold_answer, candidates)
        success_cache[key] = score
        return score

    with WandBRun(cfg, name=f"{cfg['experiment']['name']}-m3judge", job_type="m3judge") as run:
        t0 = time.time()
        done = 0
        total = sum(len(by_agent) * (len(by_agent) - 1) for by_agent in grouped.values())
        run.log({"setup/total_transfer_runs": total, "setup/n_samples": n_samples})

        for (ctx_id, budget), by_agent in grouped.items():
            ctx = ctx_by_id.get(ctx_id)
            if ctx is None:
                _logger.warning("Context %s missing from current pool; skipping.", ctx_id)
                continue
            if not ctx.probe_states:
                continue
            gold = ctx.probe_states[0].gold_action

            # Pre-compute self-success per target agent.
            self_success: dict[str, float] = {}
            for agent_id, rec in by_agent.items():
                spec = next(a for a in agents.values() if a.id == agent_id)
                self_success[agent_id] = _eval_target_on_memory(
                    ctx, spec, rec["memory_text"], gold,
                    base_seed=seed + 101,
                )

            for src_id, src_rec in by_agent.items():
                src_mem = src_rec["memory_text"]
                for tgt_id in by_agent:
                    if tgt_id == src_id:
                        continue
                    tgt_spec = next(a for a in agents.values() if a.id == tgt_id)
                    cross = _eval_target_on_memory(
                        ctx, tgt_spec, src_mem, gold,
                        base_seed=seed + 202,
                    )
                    s_self = self_success[tgt_id]
                    drop = s_self - cross
                    row = {
                        "context_id": ctx_id,
                        "budget": budget,
                        "source_agent": src_id,
                        "target_agent": tgt_id,
                        "success_self": s_self,
                        "success_cross": cross,
                        "judge_drop": drop,
                        "n_samples": n_samples,
                    }
                    rows.append(row)
                    w.writerow([row["context_id"], row["budget"],
                                row["source_agent"], row["target_agent"],
                                row["success_self"], row["success_cross"],
                                row["judge_drop"], row["n_samples"]])
                    f_out.flush()
                    done += 1
                    run.log({
                        "m3judge/progress/done": done,
                        "m3judge/progress/fraction": done / max(total, 1),
                        "m3judge/progress/elapsed_min": (time.time() - t0) / 60.0,
                    })

        # ---- Aggregate stats ----
        all_drops = [r["judge_drop"] for r in rows]
        all_self  = [r["success_self"] for r in rows]
        all_cross = [r["success_cross"] for r in rows]

        # Conditional on self_success > 0 (target's own memory works to some degree)
        signal = [r for r in rows if r["success_self"] >= 0.5]
        cond_drops = [r["judge_drop"] for r in signal]

        summary = {
            "m3judge_total_runs": len(rows),
            "m3judge_mean_self_success":  sum(all_self) / len(all_self) if all_self else 0.0,
            "m3judge_mean_cross_success": sum(all_cross) / len(all_cross) if all_cross else 0.0,
            "m3judge_mean_drop":          sum(all_drops) / len(all_drops) if all_drops else 0.0,
            "m3judge_signal_rows":        len(signal),
            "m3judge_signal_fraction":    len(signal) / max(len(rows), 1),
            "m3judge_mean_conditional_drop": sum(cond_drops) / len(cond_drops) if cond_drops else 0.0,
            "m3judge_n_samples":          n_samples,
        }
        summary["m3judge_pass_unconditional_drop_15"] = summary["m3judge_mean_drop"] > 0.15
        summary["m3judge_pass_conditional_drop_15"]   = summary["m3judge_mean_conditional_drop"] > 0.15

        # Per-pair
        by_pair = defaultdict(list)
        for r in rows:
            by_pair[(r["source_agent"], r["target_agent"])].append(r)
        pair_breakdown = []
        for (src, tgt), rs in sorted(by_pair.items()):
            d = [r["judge_drop"] for r in rs]
            sig = [r for r in rs if r["success_self"] >= 0.5]
            cd = [r["judge_drop"] for r in sig]
            pair_breakdown.append({
                "source_agent": src, "target_agent": tgt,
                "n": len(rs),
                "n_signal": len(sig),
                "mean_drop_unconditional": sum(d) / len(d) if d else 0.0,
                "mean_conditional_drop":   sum(cd) / len(cd) if cd else 0.0,
            })
            run.log({
                f"m3judge/drop_unconditional/{src}_to_{tgt}": pair_breakdown[-1]["mean_drop_unconditional"],
                f"m3judge/drop_conditional/{src}_to_{tgt}":   pair_breakdown[-1]["mean_conditional_drop"],
            })
        summary["m3judge_pair_breakdown"] = pair_breakdown
        run.summary(**summary)

    f_out.close()
    write_json(out_dir / "m3_judge_summary.json", summary)
    client.close()
    _logger.info("M3 judge summary: %s", summary)


if __name__ == "__main__":
    main()
