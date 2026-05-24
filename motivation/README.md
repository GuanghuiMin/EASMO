# Motivation Experiments — Policy-Dependent Memory

Implementation of the four motivation experiments (M1–M4) described in
[`/workspace/guidances/ICLR27_Memory_motivation_experiments.md`](../../guidances/ICLR27_Memory_motivation_experiments.md).

Compared to the original spec, **everything (oracle selector + the three
agent scaffolds) runs through the in-house MiniMax-M2.5 vLLM endpoint** at
`http://10.183.22.68:8005/v1`. This means no local GPU is needed; the
agents are *policy-distinct because of their scaffolds (ReAct / Plan /
Reflexion)*, not because of different base weights.

## Layout

```
motivation/
├── README.md                  # ← you are here
├── configs/
│   ├── default.yaml           # full experiment (M1: 500 contexts × 3 agents × 4 budgets)
│   └── smoke.yaml             # tiny smoke test (2 contexts × 3 agents × 1 budget)
├── motivation/                # importable package
│   ├── __init__.py
│   ├── llm.py                 # MiniMax client + retry / streaming
│   ├── agents.py              # A_react / A_plan / A_cot scaffolds
│   ├── data.py                # load contexts + probe states (AppWorld first; LongMemEval/LoCoMo plug-in points)
│   ├── oracle.py              # M1: behavior-aware selector + scoring
│   ├── metrics.py             # token / sentence Jaccard, TF-IDF cosine, action-match, KL, Spearman
│   ├── overlap.py             # M2: cross-agent overlap + saliency
│   ├── transfer.py            # M3: cross-agent transfer eval
│   ├── classifier.py          # M4: TinyBERT-style probe
│   ├── wandb_utils.py         # W&B run / table / progress helpers
│   └── utils.py               # tokens, logging, file I/O
├── scripts/
│   ├── run_m1.py              # M1 driver (CLI)
│   ├── run_m2.py              # M2 driver
│   ├── run_m3.py              # M3 driver
│   ├── run_m4.py              # M4 driver
│   ├── run_all.py             # M1 → M2 → M3 → M4 in sequence
│   └── smoke_test.py          # 2 ctx × 3 agents × 1 budget end-to-end smoke
└── outputs/                   # populated at runtime: oracle_memories.jsonl, overlap_matrix.csv, …
```

## Quick start

```bash
source /workspace/EASMO/.venv/bin/activate

# 1. Smoke test (~5–10 min on MiniMax, sanity-checks the whole pipeline)
python -m scripts.smoke_test --config configs/smoke.yaml

# 2. Full M1 oracle discovery
python -m scripts.run_m1 --config configs/default.yaml

# 3. M2 overlap + saliency (reuses M1 outputs)
python -m scripts.run_m2 --config configs/default.yaml

# 4. M3 cross-agent transfer eval (reuses M1)
python -m scripts.run_m3 --config configs/default.yaml

# 5. M4 (stretch) classifier
python -m scripts.run_m4 --config configs/default.yaml

# Or all at once:
python -m scripts.run_all --config configs/default.yaml
```

## W&B tracking

Every experiment opens a W&B run under project `easmo-motivation`. To
disable W&B (e.g. for a quick local sanity check), set
`WANDB_MODE=disabled` or `--no-wandb` on the CLI.

Login once before first use:

```bash
wandb login
```

What gets logged per experiment:

* **M1**: per-context oracle-search progress bar, action-match rate per
  (agent, budget), pass-criteria gate (≥85%), final
  `oracle_memories.jsonl` artifact.
* **M2**: 3×3 Jaccard / TF-IDF cosine heatmaps, per-budget saliency
  Spearman ρ, illustrative saliency heatmaps.
* **M3**: task-drop-vs-divergence scatter, linear-fit R², full
  `transfer_results.csv`.
* **M4**: confusion matrix, test accuracy.

## Mapping to the spec

| Spec item             | Replacement / note                              |
|-----------------------|-------------------------------------------------|
| GPT-4o oracle selector| MiniMax-M2.5 with the same selector prompt      |
| Qwen2.5-7B agents     | MiniMax-M2.5 with 3 distinct system prompts     |
| 3 weeks / $500-1000   | A few hours / $0 (MiniMax is on internal vLLM)  |
| AppWorld 100 ctx      | bootstrapped from `acon/.../tasks/`             |
| LongMemEval, LoCoMo   | plug-in points in `motivation/data.py` (TODO)   |

## Pass-criteria gates (re-stated)

| Exp | Pass | Fallback (Plan B) |
|-----|------|--------------------|
| M1  | oracle action-match ≥ 85%, task ≥ 90% of full-context | budget too small → up B |
| M2  | mean Jaccard < 0.4, ≥2/3 ρ < 0.5 | Jaccard > 0.6 OR all ρ > 0.7 → drop Spotlight pitch |
| M3  | mean cross-agent drop > 15%, ρ > 0.5, R² > 0.5 | drop < 5% → drop Spotlight pitch |
| M4  | test acc > 70% (random=33%) | — bonus only |

If M2 *or* M3 fails the pass criterion we should escalate before going further.
