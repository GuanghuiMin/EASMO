# EASMO

Companion code and experimental artifacts for the EASMO project on
**policy-dependent prompt compression for agentic systems**.

## Layout

```
EASMO/
├── motivation/         # motivation experiments (M1–M5 + ablations)
│   ├── docs/           # design spec + live results / interpretation
│   ├── motivation/     # importable package
│   ├── scripts/        # CLI drivers
│   ├── configs/        # YAML experiment configs
│   └── outputs/        # experiment artifacts (CSV, JSON, JSONL, figs)
└── README.md           # this file
```

## Motivation experiments

* **Design spec**: [`motivation/docs/01_experiments_spec.md`](motivation/docs/01_experiments_spec.md)
* **Live results + interpretation**: [`motivation/docs/02_results_and_interpretation.md`](motivation/docs/02_results_and_interpretation.md)
* **Code README**: [`motivation/README.md`](motivation/README.md)

## Live results dashboard

Weights & Biases project:
[easmo-motivation](https://wandb.ai/guanghui_min-university-of-virginia/easmo-motivation)

## Reproducing the motivation pipeline

```bash
source .venv/bin/activate  # or your own env
cd motivation

# Smoke check (~10 min)
python -m scripts.smoke_test --config configs/smoke_locomo.yaml --no-wandb

# Full clean re-run (6 budgets, ~10-12 h per dataset)
python -m scripts.run_all --config configs/wide_locomo.yaml
python -m scripts.run_all --config configs/wide_longmemeval.yaml

# Path-D / T2 hinge (instance-noise ablation, ~1 h)
python -m scripts.instance_noise_test --config configs/default_longmemeval.yaml \
    --budget 512 --n-contexts 10 --candidates-per-agent 3

# Path-C / T1 monotonicity verdict, after wide_* finishes
python -m scripts.budget_regime_test --runs outputs/wide_longmemeval
python -m scripts.budget_regime_test --runs outputs/wide_locomo
```

See the experiments spec for the underlying hypotheses (T1, T2) and the
results doc for the current numbers + verdict.
