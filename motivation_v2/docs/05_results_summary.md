# EASMO motivation results — current snapshot

> Snapshot: 2026-05-26 9:20 AM PT.
> Latest data point: 2026-05-24 9:43 PM PT (commit `a1b2464`); no new
> runs since.
>
> **Purpose**: a self-contained results-focused doc you can read in
> 10 minutes to decide next steps. For full design rationale see
> `01_experimental_design.md`. For role-projection definitions see
> `03_role_memory_extractors.md`. For prompts see
> `02_strategy_prompts.md` and `04_multi_stage_role_setup.md`.

## 1. What we set out to test

Two empirical theses about compact memory in multi-agent systems:

* **T1 — role-conditional memory matters**:
  * **T1a (structural)**: different agent roles (planner / tool-user
    / coder / verifier) want orthogonal compressed memories of the
    same upstream context.
  * **T1b (efficiency / capability)**: in deployment-realistic
    bounded-budget setups, wrong-role memory imposes a measurable
    cost (efficiency tax under loose budget; capability drop under
    tight budget).
* **T2 — prompted LLM selectors fail**: even with explicit role +
  task descriptions, off-the-shelf LLMs produce surface-uniform
  compressions that don't reproduce the role-projected oracle's
  selectivity.

## 2. Setup at a glance

| Component | Choice | Note |
|---|---|---|
| Benchmark | AppWorld (ACON repo) | 90 train tasks; 60 single-app; 5 task families (spotify/file_system/phone/simple_note/+venmo from dev) |
| Executor model | MiniMax-M2.5 (in-house vLLM) | Single executor — cross-executor robustness is the only open gap |
| Policy axis 1 (control) | Strategy variant: direct / verify / explore | Same task, same model, different system prompts |
| Policy axis 2 (data) | Task variant: spotify task A vs B | Same domain, different sub-task |
| Policy axis 3 (headline) | Role projection: tool / code / plan / verify | 4 deterministic slicing rules per trajectory |
| Policy axis 3 (upgrade) | Real role agents: planner → executor → verifier | 3-agent pipeline on the same task, independent LLM outputs |
| Prompt baseline (T2) | Same model, role-aware compression prompt | 4 role prompts × 4 budgets × 83 tasks |
| Memory budgets | B ∈ {128, 256, 512, 1024, 2048} | "Plumbing floor" emerges below B≈256 |
| Jaccard metric | Entity-token (lowercased alphanumeric ≥ 4 chars, stopword-filtered) | Robust to LLM paraphrase vs structured memory |

Total trajectories on disk:
* Pilot: 90 × 3 strategies = 208 successful runs (direct 92%, verify 67%, explore 72%)
* Cross-task transfer: 72 cells (cap=50) + 72 (cap=15) + 72 (cap=8) = 216 cells
* T2 prompted: 1,328 prompted-memory cells
* Multi-stage: 18 task × 3-agent pipeline = 54 LLM calls + 18 trajectories

## 3. Results

### 3.1 Three-tier Jaccard hierarchy (the central structural finding)

Cross-X Jaccard at B=512 (smaller = more divergent):

| Axis varied | What's held constant | Mean Jaccard | Interpretation |
|---|---|---|---|
| Strategy (direct/verify/explore) | task, role, model | **0.91** | agent style ≠ memory driver (control) |
| Task (spotify-A vs spotify-B) | role, model | **0.17** | tasks demand different facts |
| Role (tool/code/plan/verify) | task, model | **0.04** | roles want orthogonal memory ★ |

Per-pair detail at the role level (B=512, projection-based):

| pair | Jaccard |
|---|---|
| tool – code | 0.000 |
| tool – plan | 0.060 |
| tool – verify | 0.054 |
| code – plan | 0.000 |
| code – verify | 0.000 |
| plan – verify | 0.099 |

Within-role cross-task Jaccard (per-role transferability):

| Role | B=128 | B=256 | B=512 | B=1024 |
|---|---|---|---|---|
| `code` | 0.426 | 0.409 | **0.409** | 0.409 |
| `verify` | 0.110 | 0.086 | 0.105 | 0.105 |
| `tool` | 0.121 | 0.099 | 0.089 | 0.093 |
| `plan` | 0.016 | 0.075 | 0.072 | 0.072 |

**Implication**: code-pattern memory transfers freely across tasks
(0.41); fact-level memory (tool / verify / plan) is task-specific
(0.07–0.11). Within a role, you can train a compressor on a small
diverse task set and have it generalise.

Reproduce: `analyze_role_overlap.py` and `analyze_task_overlap.py`.

### 3.2 Multi-stage real-agent role orthogonality (closes "projection" critique)

Critique: the 0.04 role-projection result might be an artefact of
slicing rules. We re-tested with three actually-running LLM agents
(planner → executor → verifier) on n=18 spotify tasks. Two of the
four role memories are now pure agent outputs, not projections.

Cross-role Jaccard (n=18 tasks, real agent outputs):

| pair | mean | median | min | max |
|---|---|---|---|---|
| plan – tool | 0.048 | 0.049 | 0.022 | 0.110 |
| plan – code | 0.031 | 0.000 | 0.000 | 0.141 |
| **plan – verify** ★ | **0.123** | 0.102 | 0.000 | 0.250 |
| tool – code | 0.032 | 0.000 | 0.000 | 0.282 |
| tool – verify | 0.065 | 0.061 | 0.024 | 0.103 |
| code – verify | 0.022 | 0.000 | 0.000 | 0.119 |
| **mean** | **0.054** | — | — | — |
| Reference: projection baseline | 0.036 | — | — | — |

**Headline**: 0.054 mean (1.5× the projection baseline). plan↔verify
specifically — both pure independent LLM agent outputs — gives 0.123.
Even when both agents reference the same task and same final answer,
they share only ~12% of significant entity tokens. Reviewer's
"projections forced the orthogonality" critique closed.

Reproduce: `analyze_multi_stage_overlap.py`.

### 3.3 T2 prompted compressor failure (n=1328 cells)

Asked MiniMax to compress trajectories with role-conditioned prompts
(canonical templates in `prompted_memory.py::_PROMPT_TEMPLATES`).
Then computed cross-role Jaccard on the LLM's output vs the
projected oracle.

Cross-role Jaccard for prompted memory (B=512):

| pair | prompted | oracle (`m_role`) | ratio |
|---|---|---|---|
| tool – code | 0.190 | 0.000 | ∞ |
| tool – plan | 0.301 | 0.060 | 5.0× |
| tool – verify | 0.317 | 0.054 | 5.9× |
| code – plan | 0.147 | 0.000 | ∞ |
| code – verify | 0.089 | 0.000 | ∞ |
| plan – verify | 0.250 | 0.099 | 2.5× |
| **mean** | **0.216** | **0.036** | **6.0×** |

**Verdict**: STRONG T2 — prompted memory is **6.0× more uniform**
across roles than the projection oracle.

Per-role recall (Jaccard between prompted memory and projected oracle
of the same role, mean across budgets):

| role | recall |
|---|---|
| `tool` | 0.253 |
| `plan` | 0.213 |
| `verify` | 0.138 |
| **`code`** ★ | **0.049** |

**Headline finding for §T2 of the paper**: prompted compressor
captures only **5%** of code-pattern oracle when explicitly told
"compress for a coding agent". The LLM responds with API call lists,
not control-flow patterns — it doesn't understand the abstraction
level needed for transferability across tasks. This is the cleanest
piece of T2 evidence and arguably the paper's most quotable single
finding.

Reproduce: `analyze_prompted_overlap.py`.

### 3.4 Capped-budget capability cost (T1b strong)

Cross-task transfer with `max_iter` artificially capped to simulate
deployment-realistic bounded inference budget.

| condition (B=512) | cap=50 (default) | cap=15 (deployment-realistic) | cap=8 (stress) |
|---|---|---|---|
| self memory | 100% | **100%** | 17% |
| within-generator | 100% | 83% | 0% |
| within-app cross-gen | 100% | 83% | 33% |
| cross-app | 100% | **67%** | 17% |
| **drop self − cross_app** | +0pp | **+33pp** | +0pp |

Same comparison at B=128 (plumbing floor):

| | cap=50 | cap=15 |
|---|---|---|
| self | 100% | 100% |
| cross-app | 100% | **50%** |
| drop | +0pp | **+50pp** |

**Headline**: at deployment-realistic max_iter=15:
* B=512 cross-app memory: 100% → 67% (-33pp success drop)
* B=128 cross-app memory: 100% → 50% (-50pp success drop)
* Wrong memory converted from "+40% efficiency tax" (under cap=50)
  to "**measurable capability loss**" under bounded budget.
* cap=8 too tight (self memory only 17%) — collapse, no useful signal.

Reproduce: `analyze_capped_xtask.py`.

### 3.5 Strategy success-rate side finding

| Strategy | Success | Median iters |
|---|---|---|
| `direct` | 83/90 (92%) | 19 |
| `verify` | 60/90 (67%) | ~45 (≈ ceiling) |
| `explore` | 65/90 (72%) | 38 |

Forced cross-validation (`verify` strategy) drops task completion by
**25 percentage points** vs `direct`. Side panel for the paper:
*strategy specification interacts with task success, not just trajectory length*.

## 4. Scorecard

| # | Criterion | Status | Number |
|---|---|---|---|
| 1 | Cross-role Jaccard ≤ 0.10 (projection) | ✅ | 0.036 |
| 2 | Cross-task within-role pattern (code high, others low) | ✅ | 0.41 vs 0.07–0.11 |
| 3 | Cross-task transfer plumbing-floor pattern | ✅ | B=128 flat / B=512 +40% iter |
| 4 | T2 closure ratio ≥ 5× | ✅ | 6.0× |
| 5 | Cross-executor robustness | ⏳ | pending Qwen endpoint |
| 6 | Multi-stage real-agent role orthogonality | ✅ | 0.054 mean / plan-verify 0.123 |
| 7 | Capped-budget capability drop | ✅ | +33pp at B=512 (cap=15) |

**6 of 7 fully achieved** — only #5 (cross-executor) is external-dependency-blocked.

## 5. Honest assessment of what we have

**What's strong**:
* Three-tier Jaccard hierarchy is deterministic, reproducible, large-N.
* T2's 5% code recall is sharp, surprising, paper-quotable.
* Cap=15 capability drop is real-deployment-relevant.
* Multi-stage validation closes the projection-vs-agent critique cleanly.

**What's still soft**:
* **Single executor (MiniMax)**. The 5% code-recall finding might be MiniMax-specific. Need at least one other model (Qwen / GPT-4o-mini) for cross-executor robustness.
* **AppWorld is single-agent benchmark**. Multi-stage pipeline addresses this somewhat, but a true multi-agent benchmark (ChatDev, MetaGPT, AutoGen) would be stronger.
* **Code role on AppWorld is sparse**. 8/18 tasks have median 0 code-pattern tokens — the code-role finding holds on tasks WITH patterns but is fragile sample-size-wise.
* **Small N on cross-task transfer**. 6 consumers per condition → CIs are wide. Worth scaling to 18-24 consumers for paper-quality stats.
* **No EASMO method yet**. Motivation is strong; the actual learned compressor that addresses T1+T2 hasn't been built or evaluated.

**Tier estimate** based on current data alone:
* Findings/short paper: ✓ already publishable
* ICLR/NeurIPS main conference: ~70% with cross-executor added
* Spotlight: ~40%; needs cross-executor AND a working EASMO method (or a second benchmark)

## 6. Possible next steps (decision matrix)

### A. Keep the motivation paper as-is and add only cross-executor

* Run all role / prompted-memory analyses on Qwen2.5-7B trajectories.
* Compute time: ~6 h to regenerate trajectories + ~30 min for analyses.
* Implementation cost: low (re-use all existing scripts; just point at new endpoint).
* Outcome: 7/7 scorecard, but still motivation-only paper. Findings/Workshop tier.

### B. Add a second benchmark (multi-agent)

* Candidates:
  * **ChatDev**: software development with PM/architect/programmer/reviewer roles
  * **MetaGPT**: software engineering team with explicit role specialisation
  * **AutoGen multi-agent demos**: planner+coder+executor+critic
* Replicates three-tier Jaccard hierarchy on a real multi-agent benchmark (not synthetic 3-stage AppWorld).
* Compute time: depends on benchmark; rough estimate 3–7 days infrastructure + 1–2 days runs.
* Implementation cost: medium-high (each benchmark has its own infra).
* Outcome: kills "AppWorld is single-agent" critique; pushes towards spotlight.

### C. Build the EASMO method (and evaluate it)

* Train a role-conditional compressor (LoRA on MiniMax-M2.5 or smaller backbone).
* Training signal options:
  1. Distillation from `m_role*` projections (supervised, easiest).
  2. RLVR on capped-budget task success (harder, more impactful).
  3. Hybrid: warm-start from distillation, fine-tune with RLVR.
* Evaluation: compare EASMO vs prompted compressor (T2 baseline) vs `m_role*` (oracle) on:
  * Cross-role Jaccard (does EASMO produce role-orthogonal memory?)
  * Capability cost (does EASMO close the +33pp gap?)
  * Generalisation (does EASMO trained on spotify transfer to phone/file_system?)
* Compute time: training 1–2 days; eval 1 day.
* Implementation cost: medium-high (need training loop, infra for fine-tuning).
* Outcome: turns paper from "we found a problem" into "we proposed and evaluated a solution". Spotlight-tier potential.

### D. Scale up data on existing experiments

* Run cross-task transfer on 24 consumers instead of 6 (4× sample, ~6 h compute).
* Run multi-stage pipeline on 60 tasks instead of 18 (3× sample, ~3 h compute).
* Tightens CIs on existing findings; doesn't add new findings.
* Useful supplementary work, low-leverage on its own.

## 7. My read on the priority order

If I had to rank the moves by expected paper-tier upgrade per unit effort:

1. **Cross-executor (A)** — cheap, removes a clear reviewer attack, unlocks 7/7. Should do this regardless of paper direction.
2. **EASMO method (C)** — biggest tier upgrade if you want spotlight; ties motivation to a working method.
3. **Second benchmark (B)** — most rigorous; highest implementation cost.
4. **Scale-up (D)** — only if reviewers ask in revision.

If you want spotlight: A + C (and optionally B for robustness).
If you want main-conference quick: A + light C (a minimal EASMO baseline trained via distillation).
If you want findings paper: A only.

## 8. What would change my view

* If you can demonstrate Qwen reproduces three-tier hierarchy: bumps confidence main-conference → 80%.
* If you can demonstrate EASMO method beats prompted by ≥ 20pp on capability test: bumps spotlight → 60%.
* If multi-agent benchmark (ChatDev) reproduces cross-role 0.05 Jaccard: bumps spotlight → 70%.
* If 5% code recall reverses on a different model (e.g., Qwen2.5-72B does it well): we lose the cleanest T2 quote, paper has to lean on the 6× ratio instead. Manageable, not fatal.

## 9. Files to point at

| Doc | Purpose |
|---|---|
| `01_experimental_design.md` (888 lines) | Full design rationale, all sections |
| `02_strategy_prompts.md` (153 lines) | Strategy prompts (paper appendix) |
| `03_role_memory_extractors.md` (201 lines) | Role projection definitions (paper §3) |
| `04_multi_stage_role_setup.md` (176 lines) | Multi-stage pipeline spec |
| `05_results_summary.md` (this file) | Decision-ready snapshot |

## 10. Key data files

| Path | Content |
|---|---|
| `outputs/mv2_pilot/` | 3-strategy pilot, 270 trajectories |
| `outputs/mv2_pilot/role_overlap.json` | Cross-role + cross-task Jaccards |
| `outputs/mv2_pilot/prompted_memories.jsonl` | T2 1,328 cells |
| `outputs/mv2_pilot/prompted_overlap_final.json` | T2 verdict numbers |
| `outputs/mv2_xtask/transfer_results.jsonl` | Cap=50 baseline, 72 cells |
| `outputs/mv2_xtask_cap15/transfer_results.jsonl` | Cap=15 capability drop, 72 cells |
| `outputs/mv2_xtask_cap8/transfer_results.jsonl` | Cap=8 stress, 72 cells |
| `outputs/mv2_multi_stage_pilot/pipeline_summary.jsonl` | Multi-stage 18 tasks |
| `acon/.../outputs/MiniMaxAI_MiniMax-M2.5_mv2_*` | Raw AppWorld trajectories |

All analysers are named `analyze_*.py` under `motivation_v2/scripts/`.
Each is a standalone Python script you can re-run anytime — none
require LLM calls (deterministic post-processing only).
