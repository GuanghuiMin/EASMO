# motivation_v4 — Sensitivity definitions

> Spec reference: §3 (spans), §4 (probe), §6 (distance + aggregation).
> Implementation: `motivation_v4/spans.py`, `motivation_v4/probe.py`,
> `motivation_v4/distance.py`.

## 1. Span definition (spec §3)

**A span is one trajectory step** (one (thought, action, observation)
unit produced by the AppWorld agent). We pick this granularity because:

* It matches the natural unit of agent reasoning — each step is a
  single LLM call.
* Leave-one-out at step granularity is interpretable ("remove the step
  that fetched playlist 42") and produces O(n_steps) probes per task,
  which is feasible at dev scale (620 probes for 30 tasks).
* Finer granularity (per-action or per-observation) would multiply the
  probe cost without obvious behavioural payoff.

### Rendered span format

```
[STEP <n>]
Thought:
<leading # ... # comment lines from the action, if any>
Action:
<rest of the Python action code, capped at 600 chars>
API calls:
  - <up to 8 apis.<app>.<fn>(...) extracted from action, each capped at 240 chars>
Observation:
<step.output, capped at 600 chars>
[/STEP <n>]
```

Each field is capped to keep span text size bounded; the ablated probe
context needs to be comparable across with-span and without-span runs.
Token counts use `tiktoken cl100k_base` when available, else
`max(1, len(text)//4)` as a fallback.

### Observed span statistics

| Metric | Value (n=620 spans / 30 tasks) |
|---|---|
| Spans per task (mean / min / max) | 20.7 / 16 / 29 |
| Tokens per span (mean / median / max) | 274 / 245 / 1,535 |
| Total per-task trajectory tokens | ~5,690 average (range ~3,500–14,500) |

The 30 dev tasks selected by v3 are deliberately length-biased toward
medium-to-long trajectories (ranked by success-at-direct-strategy
iteration count), so all 30 have at least 16 spans — enough leverage
for leave-one-out to produce meaningful sensitivity variation.

## 2. Decision-state schema (spec §4)

The probe normalises every LLM response into a fixed dict with 9
top-level keys (see `motivation_v4/probe.py::_normalise_decision_state`):

| Field | Type | Distance semantic |
|---|---|---|
| `active_subgoal` | string | (not in rule-based score; appears in judge's free-text reasoning) |
| `completed_actions` | list of {action, object, evidence} | F1 over (action⊕object) entity keys |
| `active_constraints` | list of {constraint, evidence} | F1 over constraint entity keys |
| `candidate_objects` | list of {object_id, object_type, reason, required_action} | F1 over (object_id⊕object_type) keys |
| `avoid_objects` | list of {object_id, object_type, reason} | F1 over (object_id⊕object_type) keys |
| `missing_information` | list of strings | increase indicator (ablated has > reference items) |
| `next_action_type` | string | exact-match changed indicator (entity-token equality) |
| `next_action_arguments` | dict (or list) | F1 over (key=value) entity keys |
| `confidence` | enum {high, medium, low} | drop indicator (high→medium→low) |

Entity keys are normalised via lowercased alphanumeric tokens, length
≥ 2, joined with spaces, truncated to 160 chars. This canonical form
matches across paraphrase ("the playlist with ID 42" → "playlist 42")
and is what the F1 loss is computed on.

## 3. Rule-based distance (spec §6.1)

For each (task, span) we compute 8 component features comparing the
reference decision state vs the ablated decision state:

| Feature | Type | Range | Spec weight |
|---|---|---|---|
| `next_action_type_changed` | indicator | 0 / 1 | 2.0 |
| `next_action_arguments_f1_loss` | continuous | 0 — 1 | 2.0 |
| `candidate_objects_f1_loss` | continuous | 0 — 1 | 1.5 |
| `active_constraints_f1_loss` | continuous | 0 — 1 | 1.5 |
| `avoid_objects_f1_loss` | continuous | 0 — 1 | 1.0 |
| `completed_actions_f1_loss` | continuous | 0 — 1 | 1.0 |
| `missing_information_increase` | indicator | 0 / 1 | 1.0 |
| `confidence_drop` | indicator | 0 / 1 | 0.5 |

F1 loss is `1 − F1(reference_keys, ablated_keys)`. When both sets are
empty F1 loss is 0 (no change). When one is empty but not the other,
F1 loss is 1.

### Weighted aggregate

```
weighted_sensitivity =
    2.0  · next_action_type_changed
  + 2.0  · next_action_arguments_f1_loss
  + 1.5  · candidate_objects_f1_loss
  + 1.5  · active_constraints_f1_loss
  + 1.0  · avoid_objects_f1_loss
  + 1.0  · completed_actions_f1_loss
  + 1.0  · missing_information_increase
  + 0.5  · confidence_drop
```

The total weight is 10.5. We normalise to 0–1 by dividing:

```
rule_norm = min(weighted_sensitivity / 10.5, 1.0)
```

## 4. LLM-judge distance (spec §6.2)

Stage 04 runs a separate MiniMax-M2.5 call with the LLM-judge prompt
([`02_probe_prompts.md`](02_probe_prompts.md) §2). The judge emits:

| Field | Used as |
|---|---|
| `meaningful_change` (bool) | recorded but not in final score |
| `severity` (none/low/medium/high) | mapped to `judge_score` 0.0 / 0.25 / 0.6 / 1.0 (spec table) |
| `changed_fields` (list) | recorded for Table 4 case studies |
| `reason` (string) | recorded for qualitative examples |

## 5. Final span sensitivity

Per spec §6.2:

```
final_sensitivity = 0.5 · rule_norm + 0.5 · judge_score
```

This is the score used by all downstream stages (composing `high_sens` /
`low_sens` contexts, correlation tables, sensitivity-vs-recency figures).

### Observed distribution (n=618 spans with parseable ablation probes)

| Quantile | final_sensitivity |
|---|---|
| min | 0.000 |
| median | 0.231 |
| mean | 0.401 |
| max | 0.948 |
| std | 0.367 |

Threshold counts:

| Threshold | % of spans |
|---|---|
| > 0.0 | 54% |
| > 0.3 | 44% |
| > 0.6 | 37% |

LLM-judge severity breakdown:

| severity | count | % |
|---|---|---|
| high | 240 | 39% |
| medium | 28 | 5% |
| low | 63 | 10% |
| none | 287 | 46% |

The bimodal-ish shape (large `none` cluster + large `high` cluster)
is exactly what the experiment needs: it means the probe really is
differentiating spans, not assigning a uniform "everything matters
equally" score. If sensitivity were ~constant across spans, the
`high_sens` vs `low_sens` contexts would be identical and the
comparison would be uninformative.

## 6. Sensitivity-recency relationship (Q2 / Figure 4)

The point of `high_sens` competing against `recent_spans` is to
demonstrate that sensitivity captures *non-recency* information. If
the most sensitive spans were always the most recent, the probe
adds no value over a recency heuristic.

| Metric | Value |
|---|---|
| Pearson correlation between sensitivity and recency rank (0=most recent, 1=oldest) | **−0.085** |
| Mean recency rank of top-3 sensitivity spans per task (0=most recent) | **0.474** (≈ trajectory middle) |

The near-zero correlation and middle-of-trajectory location of the top
sensitivity spans confirm that decision-state sensitivity is a
distinct signal from recency. (Whether it is a *better behavioural*
signal is a separate question — see [`05_results_summary.md`](05_results_summary.md)
§Q2.)
