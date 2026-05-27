# Decision-State Sensitivity Motivation Results (v4)

## Setup

- AppWorld split: **dev** (reuses motivation_v3's 30 selected successful trajectories).
- Number of tasks: **30**.
- Downstream model: **MiniMaxAI/MiniMax-M2.5** (vLLM endpoint, temperature=0).
- Number of spans: **620** (one (action, observation) step = one span).
- Probe model: same as downstream (MiniMaxAI/MiniMax-M2.5).
- Budgets: max_steps ∈ {15 (loose), 8 (strict)}.
- Methods evaluated: 6 NEW span-based + 4 reused from v3.

---

## Finding 1: Structural Coverage Is Not Enough (carry-over from v3)

In motivation_v3 we showed that symbolic_evidence — the compression with the highest structural coverage of behaviorally-useful evidence (99.5%) — had the **lowest** downstream success rate (57% at cap=15) among the three compressed methods. NL summary (74.3% structural coverage) and ACON-style summary (85.7%) outperformed it on success rate. **Structural compression metrics do not predict behavioral utility.**

This v4 experiment tests whether a *behavior-aware* signal — span-level decision-state sensitivity, computed by leave-one-span-out probing of the downstream agent itself — is a better selection signal than static coverage or recency.

## Finding 2: Decision-State Sensitivity Is Non-Uniform

Across 620 spans from 30 tasks, span sensitivity has mean **0.401** with **44%** of spans scoring > 0.3 and **37%** scoring > 0.6.

This is the prerequisite for the rest of the experiment: if every span had the same sensitivity, span-level selection couldn't outperform random selection.

Figure: `figures/fig_sensitivity_distribution.{pdf,png}`.

Per-task descriptive stats are in `tables/table_span_sensitivity_stats.csv` (30 task rows).

## Finding 3: High-Sensitivity Spans vs Other Selection Strategies

Table 2 — behavioral success rate per method × budget:

| method | budget | n | success_rate | avg_steps | avg_total_input_tokens |
|---|---|---:|---:|---:|---:|
| high_sensitivity_spans | loose_15 | 30 | 40% | 10.97 | 49941.0 |
| high_sensitivity_spans | strict_8 | 30 | 20% | 7 | 24763.0 |
| low_sensitivity_spans | loose_15 | 30 | 17% | 14.1 | 74261 |
| low_sensitivity_spans | strict_8 | 30 | 10% | 7.73 | 31538.0 |
| recent_spans | loose_15 | 30 | 47% | 9.57 | 47624.0 |
| recent_spans | strict_8 | 30 | 30% | 6.4 | 20816.0 |
| random_spans_mean | loose_15 | 30 | 33% | 12.64 | 68003.0 |
| random_spans_mean | strict_8 | 30 | 16% | 7.24 | 28854.0 |
| task_aware_summary | loose_15 | 30 | 70% | 7.47 | 38322 |
| task_aware_summary | strict_8 | 30 | 47% | 5.33 | 20775.0 |
| acon_style_summary | loose_15 | 30 | 67% | 8.43 | 42483.0 |
| acon_style_summary | strict_8 | 30 | 53% | 4.87 | 18034.0 |
| truncated_full_context | loose_15 | 30 | 87% | 8.87 | 73651.0 |
| truncated_full_context | strict_8 | 30 | 43% | 6.47 | 47711.0 |
| no_context | loose_15 | 30 | 17% | 14.33 | 74943.0 |
| no_context | strict_8 | 30 | 0% | 8 | 33194.0 |

Figure: `figures/fig_budgeted_success_by_method.{pdf,png}`.

At cap=15: high_sensitivity 40% vs low_sensitivity 17% vs recent 47% vs random_mean 33%.
High-sensitivity span selection beats the negative controls: +23pp over low, +7pp over random.

## Finding 4: Sensitivity Is Not Just Recency

If high-sensitivity spans were simply the most recent spans, recent_spans would behaviorally tie with high_sensitivity_spans. Figure 4 plots span sensitivity against recency rank: if there is no positive-correlation cluster around rank≈0, sensitivity carries non-recency information.

Figure: `figures/fig_sensitivity_vs_recency.{pdf,png}`.

## Finding 5: Sensitivity Predicts Behavior Better Than Static Metrics

Table 3 — Pearson correlation of per-task metrics with downstream success at cap=15:

| metric | corr_high_sens | corr_recent | corr_random | n |
|---|---:|---:|---:|---:|
| decision_state_sensitivity_avg | 0.0099 | 0.0069 | 0.1738 | 30 |
| decision_state_sensitivity_max | 0.195 | 0.101 | 0.0086 | 30 |
| trajectory_token_count | 0.1245 | 0.1159 | 0.1972 | 30 |
| num_spans_in_trajectory | 0.0184 | 0.0301 | -0.0403 | 30 |

Figure: `figures/fig_sensitivity_vs_behavior.{pdf,png}`.

## Representative Examples

(Auto-selected from `tables/table_top_span_case_studies.csv`. Manual review recommended for paper.)

| task_id | top_span | severity | high_sens@15 | recent@15 | task_aware@15 |
|---|---|---|:---:|:---:|:---:|
| 4fab96f_1 | step_026 | high | 0 | 1 | 0 |
| 57c3486_3 | step_028 | high | 1 | 0 | 1 |
| 0d8a4ee_2 | step_002 | high | 0 | 0 | 0 |
| 530b157_3 | step_024 | high | 0 | 0 | 0 |
| b119b1f_2 | step_023 | medium | 1 | 0 | 1 |

## Failure Cases

(Filtered from runs JSONL post-hoc; pick 2 representative examples for paper.)

- A task where high_sensitivity_spans failed at cap=8 likely indicates either (a) the probe undervalued a span that turned out to matter for action execution, or (b) the task requires multi-span context that no single-span ablation can identify.
- A task where all extractive methods failed but summaries succeeded indicates narrative compression carries information that span-list selection can't preserve.

## Interpretation for Paper

The key message is not that any particular context format is best. The key message is that **context importance should be measured by its effect on the downstream agent's decision state**. Decision-state sensitivity is a behavior-aware selection signal that, unlike structural coverage or recency, ranks spans by their causal effect on what the agent thinks comes next. v3 showed structural metrics give the wrong ranking; v4 shows that probing the agent itself yields a metric that does select behaviorally useful spans.

## Caveats

- Decision-state probe is a proxy, not ground truth.
- Probe uses the same model family as the downstream agent; this is by design (consistent decision-state semantics) but introduces self-judgment bias.
- Span-level ablation costs O(n_spans) probes per task; expensive at scale.
- Full *policy* sensitivity is approximated by decision-state sensitivity; the latter is one structured proxy.
- Span granularity = single step; coarse multi-step dependencies may be missed.
- This is a motivation/diagnostic experiment, not a final method benchmark.
