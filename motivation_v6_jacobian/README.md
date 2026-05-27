# motivation_v6_jacobian — Jacobian / active-subspace diagnostics

This track tests whether agent context compression can be cast as
**preserving the downstream-policy Jacobian active subspace**.

Three experiments answer three orthogonal questions:

| Experiment | Question | Method |
|---|---|---|
| **A** | Does a single white-box backward pass recover the same signal as v4's expensive leave-one-span-out probe? | Embedding-gradient saliency on Qwen3-4B-Instruct-2507 |
| **B** | Does the context-to-decision Jacobian have a low-rank active subspace? | SVD over Jacobian-weighted middle-layer activations |
| **C** | Is there a low-dimensional compression upper bound for this target? | Optimise k ∈ {4,8,16,32,64} soft tokens to match full-context NLL |
| **D** *(optional)* | Are gradient-ranked spans behaviourally useful? | MiniMax-M2.5 downstream agent on gradient-ranked text contexts |

## Run

```bash
# Stop the vLLM Qwen server first (it occupies ~70GB of the H100).
kill "$(cat /workspace/qwen3-vllm/server.pid)"

# Run full pipeline (A + B + C + D, all 30 v4 tasks, 12K ctx, soft k∈{4,8,16,32,64})
nohup bash scripts/run_all.sh \
    > outputs/sprint_logs/runall_full.log 2>&1 &
disown
```

Knobs in `scripts/run_all.sh` (env vars):

| Var | Default |
|---|---|
| `PYBIN` | `/workspace/EASMO/.venv/bin/python` |
| `ACONPY` | `/workspace/acon/.venv/bin/python` |
| `QWEN_MODEL_PATH` | local snapshot of `Qwen/Qwen3-4B-Instruct-2507` |
| `MAX_CONTEXT_TOKENS` | `12000` |
| `N_CASES` | unset (all 30 v4 tasks) |
| `LAYER_INDEX` | unset (uses `N/2 = 18`) |
| `SOFT_KS` | `4,8,16,32,64` |
| `SOFT_STEPS` | `200` |
| `RUN_EXP_D` | `0` (set to `1` for downstream sanity) |

## Layout

```text
motivation_v6_jacobian/
├── README.md
├── docs/
│   ├── 01_experimental_design.md
│   ├── 02_gradient_definitions.md
│   ├── 03_soft_token_oracle.md
│   └── 04_results_summary.md          # written after run by stage 08
├── motivation_v6_jacobian/            # python package
│   ├── data.py
│   ├── prompts.py
│   ├── gradients.py
│   ├── hooks.py
│   ├── active_subspace.py
│   ├── soft_tokens.py
│   ├── metrics.py
│   └── plotting.py
├── scripts/
│   ├── 01_build_cases.py
│   ├── 02_compute_jacobian_saliency.py
│   ├── 03_compare_to_v4_sensitivity.py
│   ├── 04_active_subspace_spectrum.py
│   ├── 05_soft_token_oracle.py
│   ├── 06_aggregate.py
│   ├── 07_plot.py
│   ├── 08_write_report.py
│   ├── 09_compose_jacobian_contexts.py     # Exp D
│   ├── 10_run_jacobian_downstream.py       # Exp D
│   ├── 11_summarise_jacobian_downstream.py # Exp D
│   ├── _model_loader.py
│   └── run_all.sh
└── outputs/
    ├── raw/        JSONL + active-vector npz
    ├── tables/     CSV
    ├── figures/    PNG + PDF
    ├── reports/    results_summary.md
    └── sprint_logs/
```

## Inputs reused from earlier rounds

* `motivation_v4/outputs/raw/history_spans.jsonl` — chronological span text per task.
* `motivation_v4/outputs/raw/reference_decision_states.jsonl` — teacher-forcing target.
* `motivation_v4/outputs/raw/span_sensitivity_scores.jsonl` — black-box v4 comparator.
* `motivation_v4/outputs/raw/compressed_contexts.jsonl` — ACON-style baseline text for Experiment C.
* `motivation_v3/outputs/motivation_full_trajectories.jsonl` — task instructions.

## Status

See `docs/04_results_summary.md` after the pipeline completes.
