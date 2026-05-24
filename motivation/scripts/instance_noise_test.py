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
                # binary
                "self_match", "within_match_mean", "cross_match_mean",
                "within_drop", "cross_drop",
                # continuous (1 - TV)
                "self_overlap", "within_overlap_mean", "cross_overlap_mean",
                "within_overlap_drop", "cross_overlap_drop",
                "n_within", "n_cross", "n_target_candidates",
            ])
            for r in rows:
                w.writerow([
                    r.context_id, r.budget, r.target_agent,
                    r.self_match, r.within_match_mean, r.cross_match_mean,
                    r.within_drop, r.cross_drop,
                    r.self_overlap, r.within_overlap_mean, r.cross_overlap_mean,
                    r.within_overlap_drop, r.cross_overlap_drop,
                    r.n_within, r.n_cross, r.n_target_candidates,
                ])
        _logger.info("Wrote %s (%d rows)", csv_path, len(rows))

        # ---- Binary (legacy) aggregate ----------------------------------
        all_within = [r.within_drop for r in rows]
        all_cross  = [r.cross_drop  for r in rows]
        bin_signal = [r for r in rows if r.self_match >= 0.999]
        bsm_within = [r.within_drop for r in bin_signal]
        bsm_cross  = [r.cross_drop  for r in bin_signal]

        wm, wlo, whi = bootstrap_ci(all_within)
        cm, clo, chi = bootstrap_ci(all_cross)
        bswm, bswlo, bswhi = bootstrap_ci(bsm_within)
        bscm, bsclo, bschi = bootstrap_ci(bsm_cross)
        ratio_all_bin   = (cm / wm)  if wm > 1e-6 else None
        ratio_signal_bin = (bscm / bswm) if bswm > 1e-6 else None

        # ---- Continuous overlap aggregate (PRIMARY) ---------------------
        # signal threshold: target's own candidate-0 produces an action
        # distribution that's at least 50% overlapping (TV ≤ 0.5) with the
        # target's full-context distribution. Loose enough to admit rows
        # where M1 binary match would have been 0 due to one flipped
        # argmax sample.
        OVERLAP_SIGNAL_THRESHOLD = 0.5

        all_within_o = [r.within_overlap_drop for r in rows]
        all_cross_o  = [r.cross_overlap_drop  for r in rows]
        ov_signal = [r for r in rows if r.self_overlap >= OVERLAP_SIGNAL_THRESHOLD]
        osm_within = [r.within_overlap_drop for r in ov_signal]
        osm_cross  = [r.cross_overlap_drop  for r in ov_signal]

        wmo, wloo, whio = bootstrap_ci(all_within_o)
        cmo, cloo, chio = bootstrap_ci(all_cross_o)
        oswm, oswlo, oswhi = bootstrap_ci(osm_within)
        oscm, osclo, oschi = bootstrap_ci(osm_cross)
        ratio_all_ov   = (cmo / wmo)  if wmo > 1e-6 else None
        ratio_signal_ov = (oscm / oswm) if oswm > 1e-6 else None

        summary = {
            "n_rows":               len(rows),
            "n_signal_rows_binary": len(bin_signal),
            "n_signal_rows_overlap": len(ov_signal),
            "overlap_signal_threshold": OVERLAP_SIGNAL_THRESHOLD,

            # Binary (legacy)
            "binary_unconditional_within_drop_mean": wm,
            "binary_unconditional_within_ci": [wlo, whi],
            "binary_unconditional_cross_drop_mean":  cm,
            "binary_unconditional_cross_ci":  [clo, chi],
            "binary_unconditional_ratio_cross_over_within": ratio_all_bin,
            "binary_conditional_within_drop_mean":  bswm,
            "binary_conditional_within_ci": [bswlo, bswhi],
            "binary_conditional_cross_drop_mean":   bscm,
            "binary_conditional_cross_ci":  [bsclo, bschi],
            "binary_conditional_ratio_cross_over_within": ratio_signal_bin,

            # Continuous overlap (primary)
            "overlap_unconditional_within_drop_mean": wmo,
            "overlap_unconditional_within_ci": [wloo, whio],
            "overlap_unconditional_cross_drop_mean":  cmo,
            "overlap_unconditional_cross_ci":  [cloo, chio],
            "overlap_unconditional_ratio_cross_over_within": ratio_all_ov,
            "overlap_conditional_within_drop_mean":  oswm,
            "overlap_conditional_within_ci": [oswlo, oswhi],
            "overlap_conditional_cross_drop_mean":   oscm,
            "overlap_conditional_cross_ci":  [osclo, oschi],
            "overlap_conditional_ratio_cross_over_within": ratio_signal_ov,
        }

        # Path-D decision — driven by *continuous overlap* signal.
        MIN_SIGNAL_ROWS = 5
        primary_signal_n = len(ov_signal)
        primary_ratio = ratio_signal_ov
        primary_within = oswm
        primary_cross  = oscm

        if primary_signal_n < MIN_SIGNAL_ROWS:
            summary["path_d_verdict"] = (
                f"INSUFFICIENT SIGNAL — only {primary_signal_n} overlap "
                f"signal rows (need ≥ {MIN_SIGNAL_ROWS}); re-run with "
                "more contexts or a budget where M1 self-overlap is "
                "non-trivial"
            )
        elif primary_within is None or abs(primary_within) <= 1e-6:
            summary["path_d_verdict"] = (
                "DEGENERATE — within-agent overlap drop is ~0; "
                "denominator unstable, cannot judge ratio"
            )
        elif primary_ratio is None:
            summary["path_d_verdict"] = "DEGENERATE — ratio undefined"
        elif primary_ratio >= 3.0:
            summary["path_d_verdict"] = (
                "STRONG — overlap ratio ≥ 3× ⇒ M3 cross-agent drop is "
                "policy-driven, not seed-noise"
            )
        elif primary_ratio >= 1.5:
            summary["path_d_verdict"] = (
                "WEAK — overlap ratio in [1.5, 3); Path D borderline"
            )
        else:
            summary["path_d_verdict"] = (
                "DEAD — overlap ratio < 1.5; M3 cross-agent drop is "
                "indistinguishable from selector seed noise. Retreat "
                "to Path B."
            )

        write_json(out_dir / "instance_noise_summary.json", summary)
        run.summary(**{k: v for k, v in summary.items()
                       if not isinstance(v, list)})

        # Pretty print — show both metrics side by side so it's obvious
        # whether the binary metric was the bottleneck.
        print(f"\n===== Instance-noise ablation summary (budget={args.budget}) =====")
        print(f"  n_rows={len(rows)}  "
              f"binary signal={len(bin_signal)}  "
              f"overlap signal={len(ov_signal)} "
              f"(@self_overlap≥{OVERLAP_SIGNAL_THRESHOLD})")
        print("\n  --- BINARY top-1 match (legacy / sanity) ---")
        print(f"  ALL rows:    within={wm:.3f}  cross={cm:.3f}  "
              f"ratio={ratio_all_bin}")
        print(f"  CONDITIONAL: within={bswm:.3f}  cross={bscm:.3f}  "
              f"ratio={ratio_signal_bin}")
        print("\n  --- CONTINUOUS overlap = 1-TV (PRIMARY) ---")
        print(f"  ALL rows:    within drop={wmo:.3f}  cross drop={cmo:.3f}  "
              f"ratio={ratio_all_ov}")
        print(f"  CONDITIONAL: within drop={oswm:.3f}  cross drop={oscm:.3f}  "
              f"ratio={ratio_signal_ov}")
        print(f"\n  Path-D verdict: {summary['path_d_verdict']}")
        print(f"\n  Elapsed: {(time.time() - t0)/60:.1f} min")

    client.close()


if __name__ == "__main__":
    main()
