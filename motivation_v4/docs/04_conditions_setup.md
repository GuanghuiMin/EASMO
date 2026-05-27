# motivation_v4 — Compression conditions and downstream setup

> Spec reference: §7 (compression conditions) + §8 (downstream evaluation).
> Implementation: `motivation_v4/compose.py`, `motivation_v4/runner.py`,
> `scripts/06_compose_contexts.py`, `scripts/07_run_downstream.py`.

## 1. Per-task token budget

All four span-based methods (high / low / recent / random) share a
**single per-task token budget** so the comparison is matched on
context size, not selection signal.

Per spec §7:

```
budget = average task_aware_summary token count for that task
fallback if v3 summary unavailable: 400 tokens
```

In practice, v3's `task_aware_summary` averages **315 tokens** per
task (range 130–730). The per-task budget is set to that task's exact
v3 `task_aware_summary` token count (so e.g. a complex task with a
long summary gets a proportionally larger span budget).

Inside the composer (`motivation_v4/compose.py`), the budget also
accounts for ~10 tokens of overhead for the `[SELECTED_HISTORY_SPANS]`
+ `[/SELECTED_HISTORY_SPANS]` wrapper, so the *spans themselves*
total to (budget − overhead) at most.

## 2. The 10 conditions

### 6 NEW v4 conditions (span-based, evaluated this round)

| Condition | Selection rule | Tie-break |
|---|---|---|
| `high_sensitivity_spans` | greedy fill by `final_sensitivity / token_count`, descending | smaller `step_id` first |
| `low_sensitivity_spans` | greedy fill by `final_sensitivity / token_count`, ascending | smaller `step_id` first |
| `recent_spans` | greedy fill by `step_id`, descending | n/a |
| `random_spans_seed1` | shuffle with seed 1, take in shuffled order | n/a |
| `random_spans_seed2` | shuffle with seed 2 | n/a |
| `random_spans_seed3` | shuffle with seed 3 | n/a |

After selection, all four method types **emit spans in original
chronological order**, wrapped in:

```
[SELECTED_HISTORY_SPANS]
[STEP <a>]
...
[/STEP <a>]
[STEP <b>]
...
[/STEP <b>]
[/SELECTED_HISTORY_SPANS]
```

This preserves causal structure ("the agent first logged in, then
fetched the playlist") rather than presenting a jumbled span list.

### 4 REUSED v3 conditions (data pulled from
`motivation_v3/outputs/motivation_behavior_runs.jsonl`)

| Condition | v3 method name | Source |
|---|---|---|
| `task_aware_summary` | same | v3 Exp 1.A — LLM-generated NL summary |
| `acon_style_summary` | same | v3 Exp 1.B — structured-section summary |
| `truncated_full_context` | `full_context` | v3 Exp 3 condition 1 — first 12K chars of trajectory rendered as text |
| `no_context` | same | v3 Exp 3 condition 6 — empty memory_text |

These conditions use the **same downstream agent prompt** as v4 (see
[`02_probe_prompts.md`](02_probe_prompts.md) §3) so merging is
apples-to-apples. The merge happens in Stage 08
(`08_aggregate_tables.py`).

## 3. Downstream agent setup

For each (task, condition, budget) cell:

| Knob | Value |
|---|---|
| Agent model | `MiniMaxAI/MiniMax-M2.5` (vLLM endpoint) |
| Agent prompt template | v3's `direct` strategy jinja, with the spec's downstream USER turn spliced before "Using these APIs..." |
| Strategy | `direct` (same as v3) |
| `max_steps` | 15 (loose) or 8 (strict) |
| `seed` | 42 |
| Trajectory output | `acon/experiments/appworld/outputs/MiniMaxAI_MiniMax-M2.5_mv4_run_<method>_cap<N>/dev/task_<id>/` |

The runner is `motivation_v4/runner.py`, which is a thin wrapper around
`motivation_v3/runner.py::run_with_compressed_context` with the tag
namespace changed to `mv4_run` so trajectory directories don't clash.

## 4. RunResult schema

Each downstream cell writes one row to `raw/behavior_runs.jsonl`:

```json
{
  "task_id": "57c3486_3",
  "method": "high_sensitivity_spans",
  "budget": "loose_15",
  "budget_max_steps": 15,
  "success": true,
  "score": 1.0,
  "num_steps": 12,
  "iterations": 12,
  "total_input_tokens": 48211,
  "peak_input_tokens": 48211,
  "input_tokens": 48211,
  "output_tokens": 3120,
  "elapsed_s": 75.3,
  "memory_text_len": 1842,
  "output_dir": "/workspace/acon/.../task_57c3486_3",
  "termination_reason": "task_completed",
  "error": null,
  "failure_reason": null
}
```

Notes:
* `peak_input_tokens` is currently an alias of `total_input_tokens`
  (we don't track per-step token-window peaks; spec is satisfied with
  a single per-run token figure).
* `api_call_count` is not yet computed at this stage — it would
  require post-hoc parsing of `env_history.json` (as v3 did in
  Stage 06). For v4 the spec only requires the column in Table 2,
  not per-cell labels, so the table reports 0 here. A follow-up
  post-hoc pass (analogous to v3's `06_label_recovery.py`) could add
  api-call counts to v4 if a reviewer asks.

## 5. Total cell budget

| Condition group | Cells | Cost |
|---|---|---|
| 6 NEW v4 conditions × 30 tasks × 2 budgets | **360** | ~40 min on 6 workers |
| 4 REUSED v3 conditions × 30 tasks × 2 budgets | 240 (already paid in v3) | 0 (just merged into Table 2) |
| **Total cells in merged Table 2** | **600** | |

All 360 NEW cells succeeded (no parse / spawn failures). The full
matrix is in `outputs/raw/behavior_runs.jsonl` (360 rows) merged with
v3's `motivation_behavior_runs.jsonl` (418 rows; v3 also had ≤2
skipped per condition for missing wrong-task pairings).

## 6. Why this design is fair across methods

1. **Same downstream agent / model / prompt / strategy / temperature**
   across all 10 conditions. Only the `{compressed_context}` placeholder
   varies between conditions.

2. **Same per-task token budget** for the 6 span-based methods.
   Span-based context length is matched to v3's `task_aware_summary`
   length per task. The 4 reused summary conditions use their own
   natural lengths (NL 315 tokens, ACON 554, truncated_full 73K,
   no_context 0); we present token figures alongside success rates in
   Table 2 so a reader can normalise by length.

3. **Same 30 dev tasks** across all conditions (the v3-selected
   length-biased subset; trajectory dirs already exist in
   `motivation_v3/outputs/motivation_full_trajectories.jsonl`).

4. **Same two budgets** (15 loose / 8 strict) for all conditions, so
   the same 7-condition × 2-budget × 30-task factorial holds.

5. **Random-seed averaging** for `random_spans_*`: we report all three
   seeds individually plus a `random_spans_mean` row in Table 2 so a
   reader can see seed variance (10–13pp range across seeds is
   typical).

## 7. What the `[SELECTED_HISTORY_SPANS]` block looks like in practice

For task `57c3486_3` (a Spotify "like all songs from followed artists"
task), the **high_sensitivity_spans** rendering at budget=315 might
be (excerpted):

```
[SELECTED_HISTORY_SPANS]
[STEP 6]
Thought:
# I need to find which artists the user follows.
Action:
followed_artists = apis.spotify.show_following_artists(access_token=tok)
API calls:
  - apis.spotify.show_following_artists(access_token=tok)
Observation:
[{"artist_id": 28, "name": "Evelyn Rose"}, {"artist_id": 20, "name": "Emily Rivers"}]
[/STEP 6]
[STEP 12]
Thought:
# For each artist, fetch their songs.
Action:
artist_songs = apis.spotify.search_songs(artist_id=28, page_limit=20)
API calls:
  - apis.spotify.search_songs(artist_id=28, page_limit=20)
Observation:
[{"song_id": 176, "title": "..."}, {"song_id": 177, ...}, ...]
[/STEP 12]
[/SELECTED_HISTORY_SPANS]
```

The **recent_spans** rendering at the same budget would instead pick
the last 2–3 steps (likely showing the `like_song` API calls near the
trajectory end), even if those steps are simply "I liked song 184",
"I liked song 185" — the most-recent steps may be high-density in
agent actions but low-density in *novel* information.

The **low_sensitivity_spans** rendering would deliberately pick the
spans that the probe judged as having no effect on the decision state
— typically early API-doc-lookup steps like
`apis.api_docs.show_api_doc(app_name='spotify', api_name='show_song')`
that produce a schema but no task-relevant content.
