# motivation_v3 — Compression-method comparison (LLM compressors → behavioral utility)

This track tests three claims via behavior-level evaluation on AppWorld:

1. **Natural-language summaries are an inefficient interface for tool-use agents.** Agents often need exact symbolic evidence: IDs, variable values, entity bindings, constraints, and action outcomes.
2. **Prompted compression can miss task-useful evidence.** A summary may look reasonable yet drop executable details the downstream agent needs.
3. **Compression utility should be measured behaviorally.** A compressed context is useful iff it helps the same fixed agent complete the task with fewer steps, fewer tokens, and fewer API calls.

## Spec

[`docs/01_compression_experiments_spec.md`](docs/01_compression_experiments_spec.md) (symlinked to `motivation_v2/user_feedback/compression_experiments.md`).

## Layout

```
motivation_v3/
├── motivation_v3/                 # python package
│   ├── data.py                    # load successful AppWorld trajectories
│   ├── prompts.py                 # the 5 spec prompts (compression × 3, evidence label, audit, recovery)
│   ├── compressors.py             # build NL summary / ACON-style / symbolic evidence
│   ├── evidence.py                # behavioral evidence labelling + audit
│   ├── runner.py                  # run downstream agent with a compressed context
│   └── metrics.py                 # token count, ID/binding/constraint/action_outcome counters
├── scripts/
│   ├── 01_select_consumers.py     # pick 30 successful dev trajectories
│   ├── 02_build_compressions.py   # Exp 1: 3 compression methods per task
│   ├── 03_label_evidence.py       # Exp 2 step 1: behavioral usefulness labels
│   ├── 04_audit_compressions.py   # Exp 2 step 2: audit
│   ├── 05_run_downstream.py       # Exp 3: agent runs (7 conditions × 2 budgets)
│   ├── 06_label_recovery.py       # Exp 3: recovery API call labels
│   ├── 07_aggregate_tables.py     # Tables 1, 2, 3
│   ├── 08_plot_figures.py         # 3 PDFs
│   └── 09_write_report.py         # motivation_results.md
└── outputs/
    ├── motivation_full_trajectories.jsonl
    ├── motivation_symbolic_units.jsonl
    ├── motivation_behavioral_evidence.jsonl
    ├── motivation_compressed_contexts.jsonl
    ├── motivation_behavior_runs.jsonl
    ├── motivation_audits.jsonl
    ├── tables/{table1,table2,table3}.csv
    ├── figures/fig_*.pdf
    └── motivation_results.md
```

## Conditions

* AppWorld split: **dev** (56 tasks).
* Target n: **30 successful full-context trajectories**.
* Compressor model and downstream agent model: **MiniMaxAI/MiniMax-M2.5** (single executor; spec compliance the same as motivation_v2).
* Compressed methods (Exp 1 + Exp 3):
  * `task_aware_summary` — NL summary, task-aware (Exp 1.A).
  * `acon_style_summary` — structured sections (Exp 1.B).
  * `symbolic_evidence` — JSON unit list rendered as bracketed block (Exp 1.C).
* Behavior conditions (Exp 3, 7 conditions × 2 budgets):
  * `full_context`, `task_aware_summary`, `acon_style_summary`, `symbolic_evidence`, `wrong_task_symbolic_same_app`, `wrong_task_symbolic_cross_app`, `no_context`.
* Budgets: `max_steps ∈ {15 (loose), 8 (strict)}`.

## Reproduction

```bash
# Top-level pipeline (sequential within stages)
PYBIN=/workspace/EASMO/.venv/bin/python
ACONPY=/workspace/acon/.venv/bin/python

cd /workspace/EASMO/motivation_v3
$ACONPY scripts/01_select_consumers.py        # generate dev full-context trajectories
$PYBIN  scripts/02_build_compressions.py      # 3 compressions per task (Exp 1)
$PYBIN  scripts/03_label_evidence.py          # behavioral usefulness labels (Exp 2.1)
$PYBIN  scripts/04_audit_compressions.py      # compression-vs-evidence audit (Exp 2.2)
$ACONPY scripts/05_run_downstream.py          # 7-condition × 2-budget agent runs (Exp 3)
$PYBIN  scripts/06_label_recovery.py          # recovery API call labels (Exp 3 post-hoc)
$PYBIN  scripts/07_aggregate_tables.py        # Tables 1-3
$PYBIN  scripts/08_plot_figures.py            # 3 PDFs
$PYBIN  scripts/09_write_report.py            # motivation_results.md
```
