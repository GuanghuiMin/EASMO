"""M1 driver — oracle memory discovery for every (context, agent, budget).

Usage:
    python -m scripts.run_m1 --config configs/smoke.yaml
    python -m scripts.run_m1 --config configs/default.yaml --no-wandb
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make the package importable when running as a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from motivation.agents import get_agent
from motivation.data import load_contexts
from motivation.llm import MinimaxClient
from motivation.oracle import find_oracle_memory
from motivation.utils import (
    append_jsonl, load_config, seed_everything, setup_logging, write_json,
)
from motivation.wandb_utils import WandBRun

_logger = setup_logging("scripts.run_m1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--limit-contexts", type=int, default=0,
                        help="hard cap on # contexts (0 = use config value)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    seed = int(cfg["experiment"]["seed"])
    seed_everything(seed)
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_path = out_dir / "oracle_memories.jsonl"
    # Truncate any prior file so reruns are reproducible.
    if oracle_path.exists():
        oracle_path.unlink()

    _logger.info("Output dir: %s", out_dir.resolve())

    # 1. Bring up MiniMax client + smoke-check it.
    llm_cfg = cfg["llm"]
    client = MinimaxClient(
        base_url=llm_cfg["base_url"],
        model_id=llm_cfg["model_id"],
        api_key=llm_cfg.get("api_key", "EMPTY"),
        request_timeout_s=float(llm_cfg.get("request_timeout_s", 180)),
        max_concurrent=int(llm_cfg.get("max_concurrent", 8)),
        retry_attempts=int(llm_cfg.get("retry_attempts", 3)),
        retry_backoff_s=float(llm_cfg.get("retry_backoff_s", 5)),
    )
    if not client.ping():
        _logger.error("MiniMax endpoint %s is not reachable. Abort.", llm_cfg["base_url"])
        sys.exit(2)
    _logger.info("MiniMax reachable at %s (model=%s)", llm_cfg["base_url"], llm_cfg["model_id"])

    # 2. Load contexts.
    contexts = load_contexts(cfg["data"])
    if args.limit_contexts > 0:
        contexts = contexts[: args.limit_contexts]
    _logger.info("Loaded %d contexts.", len(contexts))

    # 3. Load agent specs.
    agents = [get_agent(a["scaffold"]) for a in cfg["agents"]]
    budgets = list(cfg["budgets"])
    samples_per_state = int(cfg["samples_per_state"])
    m1_cfg = cfg["m1"]
    total_triples = len(contexts) * len(agents) * len(budgets)

    # 4. W&B init.
    with WandBRun(cfg, name=f"{cfg['experiment']['name']}-m1", job_type="m1") as run:
        run.log({"setup/total_contexts": len(contexts),
                 "setup/total_agents": len(agents),
                 "setup/total_budgets": len(budgets),
                 "setup/total_triples": total_triples})

        rows_for_table: list[list] = []
        t0 = time.time()
        triples_done = 0
        pass_count = 0
        per_agent_rates: dict[str, list[float]] = {a.id: [] for a in agents}

        for ctx_i, ctx in enumerate(contexts):
            for agent in agents:
                baseline_dists = None  # reuse across budgets to save calls
                for budget in budgets:
                    triples_done += 1
                    _logger.info(
                        "[%d/%d] ctx=%s agent=%s budget=%d",
                        triples_done, total_triples, ctx.context_id, agent.id, budget,
                    )
                    t_triple = time.time()
                    result = find_oracle_memory(
                        client, agent, ctx, budget,
                        samples_per_state=samples_per_state,
                        candidate_top_k=int(m1_cfg.get("candidate_top_k", 5)),
                        seed=seed + ctx_i,
                        action_match_pass=float(m1_cfg.get("action_match_pass", 0.85)),
                        baseline_dists=baseline_dists,
                    )
                    elapsed = time.time() - t_triple

                    append_jsonl(oracle_path, result.to_dict())
                    rows_for_table.append([
                        ctx.context_id, agent.id, budget,
                        round(result.action_match_rate, 3),
                        round(result.mean_kl_to_full, 3),
                        result.memory_tokens, round(elapsed, 1),
                    ])
                    per_agent_rates[agent.id].append(result.action_match_rate)
                    if result.pass_action_match_85:
                        pass_count += 1
                    run.log({
                        f"m1/action_match/{agent.id}_B{budget}": result.action_match_rate,
                        f"m1/mean_kl/{agent.id}_B{budget}": result.mean_kl_to_full,
                        f"m1/memory_tokens/{agent.id}_B{budget}": result.memory_tokens,
                        "m1/progress/triples_done": triples_done,
                        "m1/progress/fraction_done": triples_done / max(total_triples, 1),
                        "m1/progress/elapsed_min": (time.time() - t0) / 60.0,
                        "m1/progress/pass_85_count": pass_count,
                    })

        # 5. Final logging.
        run.log_table(
            "m1/per_triple",
            columns=["context_id", "agent_id", "budget",
                     "action_match", "mean_kl", "memory_tokens", "elapsed_s"],
            rows=rows_for_table,
        )
        summary = {
            "m1_total_triples": total_triples,
            "m1_pass_count": pass_count,
            "m1_pass_fraction": pass_count / max(total_triples, 1),
            "m1_total_minutes": (time.time() - t0) / 60.0,
        }
        for agent_id, rates in per_agent_rates.items():
            if rates:
                summary[f"m1_mean_action_match/{agent_id}"] = sum(rates) / len(rates)
        run.summary(**summary)
        write_json(out_dir / "m1_summary.json", summary)
        if run.enabled:
            run.log_artifact(str(oracle_path), artifact_name="oracle_memories", artifact_type="dataset")

    client.close()
    _logger.info("Done. Oracle memories at %s", oracle_path)


if __name__ == "__main__":
    main()
