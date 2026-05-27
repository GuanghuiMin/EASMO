# motivation_v4 — Experimental Design

> Track: `EASMO/motivation_v4/`
> Spec: [`Motivation_Experiment_v4.md`](../Motivation_Experiment_v4.md)
> Snapshot: 2026-05-27 11:10 AM PT. All stages 01–10 complete.

## 0. One-paragraph framing

motivation_v3 showed that *structural compression metrics do not predict
behavioral utility* — symbolic_evidence had the highest evidence
coverage (99.5%) but the lowest downstream success (57%/30% at
cap=15/cap=8). motivation_v4 proposes and tests a different
selection signal: **decision-state sensitivity** of individual
history spans, computed by leave-one-span-out probing of the
downstream agent itself. The hypothesis is that a span matters iff
removing it changes the agent's inferred decision state. v4 builds
the signal end-to-end on the same 30 dev trajectories as v3 and
evaluates extractive span-selection methods against summary, recency,
random, and no-context baselines.

## 1. Claim under test

> **For LLM agents, context importance should be measured by whether a
> history span changes the downstream agent's decision state, not by
> textual salience, evidence coverage, or summary completeness.**

Formally, for full history *h*, candidate span *u<sub>i</sub>*, and fixed
downstream agent π:

> *I*(*u<sub>i</sub>* ; *h*, π) = *d* ( *q*<sub>π</sub>(*h*) , *q*<sub>π</sub>(*h* \\ *u<sub>i</sub>*) )

where *q*<sub>π</sub>(·) is a structured **decision-state probe** and *d*
measures disagreement. The experiment is a *diagnostic*, not a method
training run. No RL, no selector training, no prompt-format hand-tuning.

## 2. Scope

| | |
|---|---|
| Benchmark | AppWorld |
| Split | **dev** (56 tasks → 30 successful full-context trajectories, ranked by trajectory length) |
| Downstream / probe / judge model | **MiniMaxAI/MiniMax-M2.5** (single executor; temperature 0 for probe + judge, 0.2 for compressor where relevant) |
| Number of spans | **620** (mean 20.7 spans/task; range 16–29) |
| Per-span tokens | mean 274, range 51–1,535 |
| LLM cost | 30 reference probes + 620 ablation probes + 618 LLM-judge calls + 360 agent runs = **1,268 LLM-only calls + 360 agent cells** |
| Wall-clock | ~1h 30m on shared vLLM endpoint (8 workers LLM, 6 workers downstream) |

The downstream agent prompt is the **same** USER turn as v3, so the four
reused conditions (task_aware_summary, acon_style_summary,
truncated_full_context, no_context) are apples-to-apples comparable
across v3 and v4 tables.

## 3. Methods (compression conditions)

Six **NEW** span-based conditions implemented in this round, plus four
conditions reused from v3 for the same 30 tasks:

| Condition | Source | Description |
|---|---|---|
| `high_sensitivity_spans` | NEW (v4) | Greedy fill by sensitivity-per-token, descending |
| `low_sensitivity_spans` | NEW (v4) | Greedy fill by sensitivity-per-token, ascending |
| `recent_spans` | NEW (v4) | Most recent spans first, greedy fill |
| `random_spans_seed{1,2,3}` | NEW (v4) | Uniform random selection with three different seeds |
| `task_aware_summary` | reused v3 | NL summary with published-convention preserve-IDs prompt |
| `acon_style_summary` | reused v3 | Structured sections (TASK STATE / IDENTIFIERS / BINDINGS / etc.) |
| `truncated_full_context` | reused v3 | First 12K chars of rendered trajectory |
| `no_context` | reused v3 | Empty memory_text |

All span-based contexts are rendered in original chronological order
wrapped in `[SELECTED_HISTORY_SPANS] ... [/SELECTED_HISTORY_SPANS]`.

For full details on per-method composer logic and budget computation
see [`04_conditions_setup.md`](04_conditions_setup.md).

## 4. Pipeline (10 stages)

| Stage | Script | Output | Wall-clock |
|---|---|---|---|
| 01 | `01_make_spans.py` | `raw/history_spans.jsonl` (620 rows) | <1 min |
| 02 | `02_reference_probes.py` | `raw/reference_decision_states.jsonl` (30 rows) | 1.5 min |
| 03 | `03_ablation_probes.py` | `raw/span_ablation_probes.jsonl` (620 rows) | 27.5 min |
| 04 | `04_judge_distance.py` | `raw/span_judge_distances.jsonl` (618 rows) | 20.8 min |
| 05 | `05_compute_sensitivity.py` | `raw/span_sensitivity_scores.jsonl` + `tables/span_sensitivity_scores.csv` | <1 min |
| 06 | `06_compose_contexts.py` | `raw/compressed_contexts.jsonl` (180 rows: 30 tasks × 6 NEW methods) | <1 min |
| 07 | `07_run_downstream.py` | `raw/behavior_runs.jsonl` (360 cells: 30 × 6 × 2) | ~40 min |
| 08 | `08_aggregate_tables.py` | 4 spec tables in `tables/` | <1 min |
| 09 | `09_plot_figures.py` | 4 spec figures (PDF + PNG) in `figures/` | <1 min |
| 10 | `10_write_report.py` | `reports/decision_probe_results.md` | <1 min |

Top-level orchestrator: `scripts/run_all.sh`.

## 5. Decision-state probe (§4 of spec)

The probe asks the same MiniMax-M2.5 model to *infer the next-action
decision state* from a given context — not to solve the task. The
output is a structured JSON document with 9 fields (`active_subgoal`,
`completed_actions`, `active_constraints`, `candidate_objects`,
`avoid_objects`, `missing_information`, `next_action_type`,
`next_action_arguments`, `confidence`).

The probe is run twice per (task, span):

* **Reference** (once per task, Stage 02): probe sees the rendered
  full trajectory (capped at 18K chars). Produces *q*<sub>π</sub>(*h*).
* **Ablation** (once per span, Stage 03): probe sees the full
  trajectory minus the span being scored. Produces *q*<sub>π</sub>(*h* \\ *u<sub>i</sub>*).

For the exact prompt text see
[`02_probe_prompts.md`](02_probe_prompts.md). For schema and parser
behaviour see [`03_sensitivity_definitions.md`](03_sensitivity_definitions.md).

## 6. Distance and sensitivity score (§6 of spec)

A span's sensitivity combines two independent measurements:

* **Rule-based distance** (8 component features over the JSON state):
  weighted aggregate per spec §6.1.
* **LLM-judge distance**: a separate MiniMax-M2.5 call (different
  prompt) compares reference vs ablated states and returns
  `meaningful_change` + `severity ∈ {none, low, medium, high}`. The
  severity maps to a 0–1 score per spec §6.2.

Final score per spec:

> `final_sensitivity = 0.5 × rule_norm + 0.5 × judge_score`

All components and the rule-based feature definitions are in
[`03_sensitivity_definitions.md`](03_sensitivity_definitions.md).

## 7. Construction of span-based contexts (§7 of spec)

For each task we set the per-task token budget to the average
`task_aware_summary` token count from v3 (defaults to 400 when
unavailable; in practice ~315 tokens). Each span-based method then
greedy-fills under that budget by a different ranking signal:

* `high_sensitivity_spans`: argmax sensitivity-per-token
* `low_sensitivity_spans`: argmin sensitivity-per-token
* `recent_spans`: argmax step_id
* `random_spans_seed{1,2,3}`: shuffled with the given seed

All four methods then **emit the selected spans in original
chronological order** so the downstream agent sees a coherent
sub-trajectory, not a jumbled list.

## 8. Downstream evaluation (§8 of spec)

For each (task, method, budget) cell we run the same fixed MiniMax-M2.5
downstream agent (v3's prompt) with `max_steps ∈ {15 (loose), 8
(strict)}` and the spec's downstream-agent USER turn that explicitly
tells the agent "Use exact IDs from the compressed context when
reliable; call tools to verify if missing." The 4 reused v3 conditions
contribute 240 cells; the 6 NEW v4 conditions contribute 360 cells.
Total **600 agent runs** in the merged Table 2.

## 9. Main questions and headline answers (preview; full §10 in 05)

| # | Question | Verdict |
|---|---|---|
| Q1 | Does decision-state sensitivity predict behavior better than structural coverage? | **Partially.** High-sens beats low-sens (+23pp loose, +10pp strict) and random (+7pp / +4pp), so the signal is real. But it does NOT beat recent_spans. |
| Q2 | Are high-sensitivity spans more useful than recent spans? | **No** for AppWorld task continuation. Recent_spans tops the span-based methods at both budgets (47% vs 40% loose; 30% vs 20% strict). |
| Q3 | Can extractive span selection compete with summaries? | **No, not in this round.** Summaries (NL 70%, ACON 67%, truncated_full 87% at cap=15) outperform every span-based method by a wide margin. |
| Q4 | Does low-sensitivity context behave like no_context? | **Yes.** low_sensitivity_spans = 17% / 10% ≈ no_context = 17% / 0%. Confirms the signal is doing something — removing high-impact spans collapses behavior. |

## 10. Success criteria (spec §13) — 4 of 6 met

The experiment is declared useful if at least 2 of 6 criteria hold:

| # | Criterion | Met? | Number |
|---|---|---|---|
| 1 | High-sens > low-sens | ✅ | +23pp / +10pp (loose / strict) |
| 2 | High-sens > random | ✅ | +7pp / +4pp |
| 3 | High-sens > recent | ❌ | −7pp / −10pp |
| 4 | Sensitivity correlates with success better than recency/coverage | ❌ (weak) | best r = 0.20 (max-sens); recency r ≈ 0; all weak |
| 5 | Removing high-sens changes meaningful fields | ✅ | 39% of spans LLM-judge-rated severity=high |
| 6 | Top sensitivity spans not the longest / most recent | ✅ | mean recency rank 0.474 (≈ trajectory middle) |

> Spec passes at 4/6. The headline framing for the paper is "decision-state probing identifies a meaningful signal, but recency is a surprisingly strong baseline in this benchmark." See §6 of [`05_results_summary.md`](05_results_summary.md) for interpretation.

## 11. Caveats / methodological warnings

* **Self-judgement bias.** The probe, the LLM-judge, and the downstream
  agent are all the same MiniMax-M2.5 model. Spec acknowledges this is
  unavoidable for the decision-state framing. Cross-model probes (e.g.
  GPT-4o-mini probe + MiniMax downstream) deferred per separate user
  direction.
* **Probe is a proxy, not ground truth.** The reference decision state
  is the same model's interpretation under full context; it can be
  systematically wrong.
* **Span granularity = 1 step.** Multi-step dependencies (e.g. "step 4
  defines a variable, step 9 reads it") are partially captured: removing
  step 4 will change the ablated probe at step 4, but the f1-style
  comparison may not penalise it heavily.
* **`truncated_full_context` is itself a compression**. We render at most
  12K chars of the trajectory because the agent context window can't
  fit the literal full trace. Paper should label this honestly.
* **Span-level ablation costs O(n_spans) probes per task** — 620 probes
  for 30 dev tasks. This is feasible at dev scale but does not scale to
  millions of trajectories without batching or amortisation.
* **`high > recent` failure is benchmark-specific.** In AppWorld
  trajectories the most recent spans naturally cluster around
  state-changing API calls near the completion step, so recency is a
  particularly strong baseline here. The same probe approach on dialogue
  / long-horizon recall benchmarks (where recency doesn't trivially win)
  would be a fairer test.

## 12. File layout

```
motivation_v4/
├── Motivation_Experiment_v4.md       (spec, user-authored)
├── README.md                         (top-level)
├── docs/
│   ├── 01_experimental_design.md     (this file)
│   ├── 02_probe_prompts.md           (paper-appendix-ready prompt text)
│   ├── 03_sensitivity_definitions.md (rule-based components + final-score formula)
│   ├── 04_conditions_setup.md        (the 10 compression conditions, budgets, composer logic)
│   └── 05_results_summary.md         (decision-ready snapshot with final numbers)
├── motivation_v4/                    (python package)
│   ├── __init__.py
│   ├── data.py                       (v3-reuse helpers + paths)
│   ├── prompts.py                    (probe + judge + downstream)
│   ├── spans.py                      (trajectory → spans)
│   ├── probe.py                      (decision-state probe + LLM-judge)
│   ├── distance.py                   (rule-based distance + aggregation)
│   ├── compose.py                    (high/low/recent/random span pickers)
│   └── runner.py                     (downstream agent wrapper)
├── scripts/
│   ├── 01_make_spans.py
│   ├── 02_reference_probes.py
│   ├── 03_ablation_probes.py
│   ├── 04_judge_distance.py
│   ├── 05_compute_sensitivity.py
│   ├── 06_compose_contexts.py
│   ├── 07_run_downstream.py
│   ├── 08_aggregate_tables.py
│   ├── 09_plot_figures.py
│   ├── 10_write_report.py
│   └── run_all.sh
└── outputs/
    ├── raw/      (7 JSONL: spans / references / ablations / judges / sensitivity / contexts / behavior)
    ├── tables/   (5 CSV: 4 spec tables + span_sensitivity_scores)
    ├── figures/  (4 figures × PDF + PNG)
    ├── reports/  (decision_probe_results.md)
    └── sprint_logs/
```

## 13. Reproduction

```bash
ACONPY=/workspace/acon/.venv/bin/python
PYBIN=/workspace/EASMO/.venv/bin/python

cd /workspace/EASMO/motivation_v4
bash scripts/run_all.sh     # ~1h 30m wall-clock end-to-end
```

Individual stages can be re-run independently; each script reads its
inputs from `outputs/raw/` and writes to a single canonical file
(idempotent).
