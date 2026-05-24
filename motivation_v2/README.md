# motivation_v2 — AppWorld-based motivation experiments

This is the active motivation experiment track for EASMO as of
**2026-05-24**. The previous track lives in `../motivation/` and is
deprecated as primary motivation evidence (see
`../motivation/docs/02_results_and_interpretation.md` deprecation
banner). The wide_*/instance_noise runs there will produce data
that is repurposed as the "long-memory QA generalisation" appendix
(§8 of `docs/new_motivation.md`).

## Why a new track

The previous design built T1+T2 on top of LLM-as-selector compressing
LongMemEval/LoCoMo for three same-LLM-different-prompt "agents". That
setup is internally inconsistent: T2 (LLM selectors don't policy-
condition) being true makes the T1 hinge test (cross-policy memory
behaviourally distinct) impossible to satisfy with LLM-generated
memories — there is no ground-truth oracle to anchor T1 independently.

`docs/new_motivation.md` (in this repo at the parent level —
`../motivation/docs/new_motivation.md` — see the audit-fix in-line
notes) replaces this with:

* **Real agentic benchmark** — AppWorld (multi-step tool-use, app-
  state structured, executable success criterion).
* **Execution-derived ground-truth memory `m*_exec`** — derived
  deterministically from successful trajectories (touched APIs / DB
  rows / final-state-evaluator references), not LLM-generated. T1
  is now testable independently of T2.
* **Policy = task family** (calendar / email / shopping / …),
  defined by which app the final-state evaluator interrogates, not
  by ReAct/Plan/CoT scaffold variants.

## Infrastructure dependency

The AppWorld + agent-runner pipeline is **not** re-implemented here.
We consume outputs from:

```
/workspace/acon/                  # Microsoft ACON (arXiv 2510.00615)
├── .venv/                        # working pydantic v1 / sqlmodel 0.0.10 venv
├── experiments/appworld/
│   ├── run.py / run_all.py       # agent runner end-to-end
│   ├── data/{tasks,api_docs,base_dbs,datasets}/  # AppWorld data
│   └── outputs/<exp>/<split>/task_<id>/
│       ├── appworld_trajectory.json
│       ├── env_history.json
│       ├── results.json
│       └── ...
└── src/productive_agents/        # AppWorldEnv, AppWorldAgent
```

`motivation_v2` lives in the EASMO repo and **reads** acon's
trajectory outputs to build memory units / `m*_exec` / policy
families, then re-runs acon's `AppWorldAgent` with a compressed-
memory variant to measure downstream task success.

## Layout

```
motivation_v2/
├── README.md                  # this file
├── docs/                      # design docs specific to v2 (numbers, decisions)
├── configs/                   # YAML configs per experiment
├── motivation_v2/             # python package
│   ├── data.py                # acon trajectory → motivation_v2 data structures
│   ├── units.py               # raw env state → discrete memory units
│   ├── exec_memory.py         # m*_exec_minimal / m*_exec_trajectory builders
│   ├── policy_family.py       # task → policy family classifier (final-state app set)
│   ├── compressors.py         # m_recent / m_freq / BM25 / embedding / prompted variants
│   ├── runner.py              # wraps acon's AppWorldAgent with a compressed-memory mode
│   └── metrics.py             # task success, first-action match, API/arg F1, edit distance
├── scripts/
│   ├── generate_full_context_trajectories.py  # wraps acon run_all.py for AppWorld train
│   ├── run_m1.py              # compression-pressure sweep
│   ├── run_m2.py              # prompted selector gap
│   ├── run_m3.py              # cross-policy heatmap (single-app subset)
│   └── run_m4.py              # memory-unit ablation (swap with filler)
└── outputs/<config>/<split>/  # results parallel to acon's layout
```

## Roadmap

1. Smoke-test acon end-to-end on 1 task in this environment (T3).
2. Build `data.py` + `units.py` + `policy_family.py` against the
   3 existing successful trajectories (`82e2fac_1`, `82e2fac_2`,
   `82e2fac_3` from `acon/.../stage1_v0/train_tiny/`).
3. Build `exec_memory.py` (both variants).
4. Build `compressors.py` (m_recent + m_freq).
5. Build `runner.py` (compressed-memory executor wrapper).
6. End-to-end smoke on 1 task with `m*_exec_trajectory(B=512)`.
7. Generate full-context trajectories on AppWorld train (89 tasks)
   — long-pole, ~22 h sequential.
8. Pilot M1 + M2 (per `docs/new_motivation.md` §10.2).
