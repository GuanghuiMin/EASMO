"""M5 driver — selector-consistency ablation.

This is the small but critical experiment that rules out the rebuttal
"the M2 differences are just selector sampling noise".
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from motivation.data import load_contexts
from motivation.llm import MinimaxClient
from motivation.selector_ablation import run_selector_ablation
from motivation.utils import load_config, seed_everything, setup_logging, write_json
from motivation.wandb_utils import WandBRun

_logger = setup_logging("scripts.run_m5")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--budget", type=int, default=0,
                        help="single budget to ablate over; 0 = take first from config.budgets")
    parser.add_argument("--limit-contexts", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    seed = int(cfg["experiment"]["seed"])
    seed_everything(seed)

    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

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
    if args.limit_contexts > 0:
        contexts = contexts[: args.limit_contexts]

    scaffolds = [a["scaffold"] for a in cfg["agents"]]
    budget = args.budget or int(cfg["budgets"][0])
    within_k = int(cfg.get("m5", {}).get("within_k", 3))

    with WandBRun(cfg, name=f"{cfg['experiment']['name']}-m5", job_type="m5") as run:
        rows = run_selector_ablation(
            client, contexts, scaffolds, budget,
            within_k=within_k, seed=seed,
        )

        csv_path = out_dir / "selector_consistency.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["context_id", "budget", "setting",
                        "mean_jaccard", "mean_sbert", "n_pairs"])
            for r in rows:
                w.writerow([r.context_id, r.budget, r.setting,
                            r.mean_jaccard, r.mean_sbert, r.n_pairs])

        # Aggregate within vs cross
        within_jacs, cross_jacs = [], []
        within_sbert, cross_sbert = [], []
        for r in rows:
            if r.setting.startswith("within:"):
                within_jacs.append(r.mean_jaccard)
                within_sbert.append(r.mean_sbert)
            elif r.setting.startswith("cross:"):
                cross_jacs.append(r.mean_jaccard)
                cross_sbert.append(r.mean_sbert)

        def _mean(xs): return sum(xs) / len(xs) if xs else 0.0
        within_j = _mean(within_jacs); cross_j = _mean(cross_jacs)
        within_s = _mean(within_sbert); cross_s = _mean(cross_sbert)
        gap_j = within_j - cross_j
        gap_s = within_s - cross_s

        summary = {
            "m5_within_agent_jaccard_mean": within_j,
            "m5_cross_agent_jaccard_mean":  cross_j,
            "m5_jaccard_gap":               gap_j,
            "m5_within_agent_sbert_mean":   within_s,
            "m5_cross_agent_sbert_mean":    cross_s,
            "m5_sbert_gap":                 gap_s,
            "m5_pass_lexical_10pp_gap":     gap_j > 0.10,
            "m5_pass_semantic_10pp_gap":    gap_s > 0.10,
        }
        run.summary(**summary)
        run.log_table(
            "m5/per_setting",
            columns=["context_id", "budget", "setting",
                     "mean_jaccard", "mean_sbert", "n_pairs"],
            rows=[[r.context_id, r.budget, r.setting,
                   r.mean_jaccard, r.mean_sbert, r.n_pairs] for r in rows],
        )

    write_json(out_dir / "m5_summary.json", summary)
    client.close()
    _logger.info("M5 summary: %s", summary)


if __name__ == "__main__":
    main()
