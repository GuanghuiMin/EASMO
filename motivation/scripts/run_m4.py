"""M4 driver — train a small classifier to predict source-agent from memory text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from motivation.classifier import train_eval_agent_classifier
from motivation.utils import load_config, read_jsonl, setup_logging, write_json
from motivation.wandb_utils import WandBRun

_logger = setup_logging("scripts.run_m4")


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
        _logger.error("Missing %s - run M1 first.", oracle_path)
        sys.exit(2)

    records = list(read_jsonl(oracle_path))
    mems = [r["memory_text"] for r in records if r.get("memory_text")]
    labels = [r["agent_id"] for r in records if r.get("memory_text")]
    if len(mems) < 6:
        _logger.error("Need >= 6 oracle memories for a meaningful split (have %d).", len(mems))
        sys.exit(2)

    result = train_eval_agent_classifier(mems, labels, seed=int(cfg["experiment"]["seed"]))
    write_json(out_dir / "m4_classifier.json", result.to_dict())

    with WandBRun(cfg, name=f"{cfg['experiment']['name']}-m4", job_type="m4") as run:
        run.summary(**result.to_dict())
        run.log_table(
            "m4/confusion_matrix",
            columns=["true_vs_pred"] + result.label_names,
            rows=[[lbl] + row for lbl, row in zip(result.label_names, result.confusion_matrix)],
        )

    _logger.info("M4 accuracy = %.3f (pass>0.70 = %s)", result.test_accuracy, result.pass_70)


if __name__ == "__main__":
    main()
