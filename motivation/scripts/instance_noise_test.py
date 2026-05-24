"""Instance-noise ablation driver — the hinge test for Path D.

For each context at a chosen budget, generate K=3 oracle-memory candidates
*per agent* using M1's exact selector regime (T=0.6, variation hint),
then test the same target agent against its OWN seed-variant memories
("within-agent drop") vs the SAME target against other agents'
candidate-0 memories ("cross-agent drop"). The ratio cross/within tells
us whether M3's transfer drop is driven by policy or by selector seed
noise.

Outputs:
    outputs/<exp>/instance_noise.csv
    outputs/<exp>/instance_noise_summary.json
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from motivation.data import load_contexts
from motivation.instance_noise import run_instance_noise_test
from motivation.llm import MinimaxClient
from motivation.utils import (
    load_config, seed_everything, setup_logging, write_json,
)
from motivation.wandb_utils import WandBRun

_logger = setup_logging("scripts.instance_noise_test")


def bootstrap_ci(values, n_iter=5000, seed=42):
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_iter)
    )
    return statistics.mean(values), means[int(0.025 * n_iter)], means[int(0.975 * n_iter)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--budget", type=int, default=512,
                        help="single budget to ablate (M3 strongest signal: 128/512)")
    parser.add_argument("--n-contexts", type=int, default=15)
    parser.add_argument("--candidates-per-agent", type=int, default=3,
                        help="K candidates per agent; ≥2 needed for within-agent variants")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    seed = int(cfg["experiment"]["seed"])
    seed_everything(seed)

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
    scaffolds = [a["scaffold"] for a in cfg["agents"]]
    samples_per_state = int(cfg["samples_per_state"])

    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "instance_noise.csv"

    _logger.info(
        "Instance-noise test: n=%d contexts, B=%d, K=%d candidates/agent, N=%d samples",
        len(contexts), args.budget, args.candidates_per_agent, samples_per_state,
    )

    t0 = time.time()
    with WandBRun(cfg, name=f"{cfg['experiment']['name']}-instance-noise",
                  job_type="instance-noise") as run:
        run.log({
            "setup/n_contexts": len(contexts),
            "setup/budget": args.budget,
            "setup/candidates_per_agent": args.candidates_per_agent,
            "setup/samples_per_state": samples_per_state,
        })
        rows = run_instance_noise_test(
            client, contexts, scaffolds, args.budget,
            candidates_per_agent=args.candidates_per_agent,
            samples_per_state=samples_per_state,
            seed=seed,
        )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "context_id", "budget", "target_agent",
                "self_match", "within_match_mean", "cross_match_mean",
                "within_drop", "cross_drop",
                "n_within", "n_cross", "n_target_candidates",
            ])
            for r in rows:
                w.writerow([r.context_id, r.budget, r.target_agent,
                            r.self_match, r.within_match_mean, r.cross_match_mean,
                            r.within_drop, r.cross_drop,
                            r.n_within, r.n_cross, r.n_target_candidates])
        _logger.info("Wrote %s (%d rows)", csv_path, len(rows))

        # Aggregate stats — both unconditional and conditional (on signal rows)
        all_within = [r.within_drop for r in rows]
        all_cross  = [r.cross_drop  for r in rows]
        signal     = [r for r in rows if r.self_match >= 0.999]
        sig_within = [r.within_drop for r in signal]
        sig_cross  = [r.cross_drop  for r in signal]

        wm, wlo, whi = bootstrap_ci(all_within)
        cm, clo, chi = bootstrap_ci(all_cross)
        swm, swlo, swhi = bootstrap_ci(sig_within)
        scm, sclo, schi = bootstrap_ci(sig_cross)
        ratio_all   = (cm / wm)  if wm > 1e-6 else float("inf")
        ratio_signal = (scm / swm) if swm > 1e-6 else float("inf")

        summary = {
            "n_rows":          len(rows),
            "n_signal_rows":   len(signal),

            "unconditional_within_drop_mean": wm,  "unconditional_within_ci": [wlo, whi],
            "unconditional_cross_drop_mean":  cm,  "unconditional_cross_ci":  [clo, chi],
            "unconditional_ratio_cross_over_within": ratio_all,

            "conditional_within_drop_mean":  swm, "conditional_within_ci": [swlo, swhi],
            "conditional_cross_drop_mean":   scm, "conditional_cross_ci":  [sclo, schi],
            "conditional_ratio_cross_over_within": ratio_signal,
        }

        # Path-D decision
        if ratio_signal >= 3.0:
            summary["path_d_verdict"] = "STRONG — Spotlight-defensible"
        elif ratio_signal >= 1.5:
            summary["path_d_verdict"] = "WEAK — Path D borderline"
        else:
            summary["path_d_verdict"] = "M3 likely instance-noise — Path D dies, fall back to Path B"

        write_json(out_dir / "instance_noise_summary.json", summary)
        run.summary(**{k: v for k, v in summary.items()
                       if not isinstance(v, list)})

        # Pretty print
        print(f"\n===== Instance-noise ablation summary (budget={args.budget}) =====")
        print(f"  n_rows={len(rows)}  n_signal={len(signal)}  "
              f"({100*len(signal)/max(len(rows),1):.0f}% signal)")
        print(f"\n  ALL rows:")
        print(f"    within drop = {wm:.3f}  CI [{wlo:.3f}, {whi:.3f}]")
        print(f"    cross  drop = {cm:.3f}  CI [{clo:.3f}, {chi:.3f}]")
        print(f"    ratio cross / within = {ratio_all:.2f}x")
        print(f"\n  CONDITIONAL (self_match==1.0):")
        print(f"    within drop = {swm:.3f}  CI [{swlo:.3f}, {swhi:.3f}]")
        print(f"    cross  drop = {scm:.3f}  CI [{sclo:.3f}, {schi:.3f}]")
        print(f"    ratio cross / within = {ratio_signal:.2f}x")
        print(f"\n  Path-D verdict: {summary['path_d_verdict']}")
        print(f"\n  Elapsed: {(time.time() - t0)/60:.1f} min")

    client.close()


if __name__ == "__main__":
    main()
