"""M2 driver — cross-agent overlap (Jaccard / sentence / TF-IDF) and a few
illustrative saliency heatmaps. Consumes ``outputs/<exp>/oracle_memories.jsonl``
produced by M1.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from motivation.overlap import pairwise_overlap, OverlapRow
from motivation.utils import load_config, read_jsonl, setup_logging, write_json
from motivation.wandb_utils import WandBRun

_logger = setup_logging("scripts.run_m2")


def _stats(vals):
    if not vals:
        return (float("nan"), float("nan"))
    if len(vals) < 2:
        return (mean(vals), 0.0)
    return (mean(vals), stdev(vals))


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

    records = list(read_jsonl(oracle_path))
    _logger.info("Loaded %d oracle records", len(records))

    rows: list[OverlapRow] = pairwise_overlap(records)
    csv_path = out_dir / "overlap_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["context_id", "budget", "agent_a", "agent_b",
                    "token_jaccard", "sentence_jaccard", "tfidf_cosine",
                    "sbert_cosine",
                    "memory_a_tokens", "memory_b_tokens"])
        for r in rows:
            w.writerow([r.context_id, r.budget, r.agent_a, r.agent_b,
                        r.token_jaccard, r.sentence_jaccard, r.tfidf_cosine,
                        r.sbert_cosine,
                        r.memory_a_tokens, r.memory_b_tokens])
    _logger.info("Wrote %s (%d rows)", csv_path, len(rows))

    # Aggregate by (agent pair) and by (agent pair, budget).
    by_pair: dict[tuple, list[OverlapRow]] = defaultdict(list)
    for r in rows:
        by_pair[(r.agent_a, r.agent_b)].append(r)
    by_pair_budget: dict[tuple, list[OverlapRow]] = defaultdict(list)
    for r in rows:
        by_pair_budget[(r.agent_a, r.agent_b, r.budget)].append(r)

    pair_summary = []
    for (a, b), rs in sorted(by_pair.items()):
        mt, st = _stats([r.token_jaccard for r in rs])
        ms, ss = _stats([r.sentence_jaccard for r in rs])
        mc, sc = _stats([r.tfidf_cosine for r in rs])
        msb, ssb = _stats([r.sbert_cosine for r in rs])
        pair_summary.append({
            "agent_a": a, "agent_b": b,
            "n": len(rs),
            "token_jaccard_mean": mt, "token_jaccard_std": st,
            "sentence_jaccard_mean": ms, "sentence_jaccard_std": ss,
            "tfidf_cosine_mean": mc, "tfidf_cosine_std": sc,
            "sbert_cosine_mean": msb, "sbert_cosine_std": ssb,
        })
    all_token_means = [p["token_jaccard_mean"] for p in pair_summary]
    all_sbert_means = [p["sbert_cosine_mean"] for p in pair_summary]
    overall = {
        "mean_pairwise_token_jaccard": (sum(all_token_means) / len(all_token_means)) if all_token_means else float("nan"),
        "mean_pairwise_sbert_cosine":  (sum(all_sbert_means) / len(all_sbert_means)) if all_sbert_means else float("nan"),
        "pass_jaccard_under_0.4": (
            all(p["token_jaccard_mean"] < 0.4 for p in pair_summary) if pair_summary else False
        ),
        # New: "lexical low AND semantic low" — defuses 'just paraphrasing' rebuttal
        "pass_sbert_under_0.75": (
            all(p["sbert_cosine_mean"] < 0.75 for p in pair_summary) if pair_summary else False
        ),
    }
    write_json(out_dir / "m2_pair_summary.json", {"pairs": pair_summary, "overall": overall})

    with WandBRun(cfg, name=f"{cfg['experiment']['name']}-m2", job_type="m2") as run:
        run.log_table(
            "m2/overlap_rows",
            columns=["context_id", "budget", "agent_a", "agent_b",
                     "token_jaccard", "sentence_jaccard", "tfidf_cosine",
                     "sbert_cosine",
                     "memory_a_tokens", "memory_b_tokens"],
            rows=[[r.context_id, r.budget, r.agent_a, r.agent_b,
                   r.token_jaccard, r.sentence_jaccard, r.tfidf_cosine,
                   r.sbert_cosine,
                   r.memory_a_tokens, r.memory_b_tokens] for r in rows],
        )
        for p in pair_summary:
            label = f"{p['agent_a']}_vs_{p['agent_b']}"
            run.log({
                f"m2/token_jaccard_mean/{label}": p["token_jaccard_mean"],
                f"m2/sbert_cosine_mean/{label}": p["sbert_cosine_mean"],
                f"m2/tfidf_cosine_mean/{label}": p["tfidf_cosine_mean"],
            })
        run.summary(**overall)
        run.log_artifact(str(csv_path), artifact_name="overlap_matrix", artifact_type="dataset")

    _logger.info("M2 overall: %s", overall)


if __name__ == "__main__":
    main()
