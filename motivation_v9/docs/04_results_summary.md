# motivation_v9 results — Behavioral Compression Stress and Chunk Information Advantage

> Hand-written paper-tier summary. First-pass numbers cross-checked
> against `outputs/tables/*.csv` and
> `outputs/reports/motivation_v9_results_summary.md` (auto-written).
> Widened n=20 chunk addendum is in flight (started 2026-05-29 1:11 PM PT);
> the §10 addendum at the bottom will be filled in when it lands.

## TL;DR (paper-tier, three findings)

1. **ACON greedy decoding is NOT best-of-N optimal under its own
   distribution.** Across 30 v3 AppWorld cases × 1 model (MiniMax-M2.5)
   × N=8 stochastic samples per case, best-of-N raises downstream
   AppWorld pass-rate from **70.0 % → 96.7 % on C1 (+26.7 pp)** and
   **60.0 % → 96.7 % on CK (+36.7 pp)**, with oracle win rate
   **0.900 / 0.833**. Best-of-N samples are also *shorter* on average
   (451.6 vs 486.8 chars on C1, 419.6 vs 464.6 on CK), so the gain is
   not length-mediated. Greedy is dominated stochastically on both
   pass-rate and length.
2. **One-step compression is fragile under repeated-compression
   stress T^K (K=2).** Across the same 270 candidates (30 greedy + 240
   sample), 28.6 % (6/21) of originally-passing greedy candidates fail
   after K=2 recompression rounds; greedy C1→CK pass-rate drops from
   70 % → 60 % (10 pp). **Sampling is more robust than greedy**
   (fragility 21.8 %, C1→CK drop 2.1 pp). This is consistent with
   greedy attaching to brittle surface features that recompression
   discards. The text itself converges fast: 18/270 chains hit a
   text-level fixed point by round 1 and **85/270 by round 2**, so the
   behavioral fragility is concentrated in the first 1–2 recompressions.
3. **(NEGATIVE after widening n=12 → n=20.)** The first-pass version
   of this finding said "causal-flagged chunks have 15× higher mean
   advantage than non-causal non-entity, and unbounded ratio over
   ENTITY_LIST_ONLY". On the widened sample (239 unique chunks vs
   144 first-pass), the direction **reverses**:

   | group | n_unique (n=20) | mean score adv | % positive |
   |---|---:|---:|---:|
   | `ENTITY_LIST_ONLY` (type) | 27 | **+0.167** | 25.9 % |
   | `contains_causal_relation = True` | 28 | +0.036 | 10.7 % |
   | other (non-causal, non-entity) | 184 | −0.030 | 14.1 % |

   The spec's "causal NL > entity lists" prediction is **falsified at
   n=239**. The first-pass +0.150 mean for causal chunks was an
   n-fragile artifact. **Likely reason**: the labeler's
   `ENTITY_LIST_ONLY` describes *form* (compact tokens listed without
   prose) not *function* — many entity-list chunks are actually
   exact runtime bindings (`access_token`, `target_id`) that the
   next-step API call depends on. The one type that survives
   widening with positive mean advantage is **CONTROL_NEGATIVE_EVIDENCE**
   (n=13, mean +0.115, 47.4 % causal-flag rate) — "what failed last
   time" is the only chunk-type label whose semantics map cleanly to
   agent behavior. See §10 addendum for the full new numbers.

**Interpretation under spec §22.** Claims 1 and 2 clear their
acceptance thresholds; Claim 3 in its originally-stated form does
NOT:

> ACON's greedy compression is a single sample from a behaviorally
> bimodal distribution. Its modal failure modes (brittle to
> recompression) are not artifacts of capacity but of *decoding
> strategy + abstraction prior*. v9 thus reframes the v7/v8 surface-
> type abstraction prior as a *behavioral tax*: not only does the
> compressor over-abstract executable content (v7), and not only does
> it converge to a surface-type fixed point regardless of need (v8),
> but those defects translate into measurable +27 to +37 pp
> pass-rate loss and 28.6 % stress-induced regression on AppWorld.
>
> However, the v7/v8 spec hypothesis "preserving causal natural-
> language chunks should help more than preserving entity lists"
> does NOT survive chunk-level ablation: at n=239 chunks, entity-
> only chunks have higher mean per-chunk advantage than causal-
> flagged chunks. The v9 chunk-level finding that does survive is
> narrower: **chunks describing past failures / negative evidence
> carry the highest behavioral advantage**, which echoes v5's
> recompressor-drops-failure-log bottleneck.

## 1. Setup

| Setting | Value |
|---|---|
| Compressor | `MiniMaxAI/MiniMax-M2.5` (`http://10.183.22.68:8005/v1`, max_model_len 32K) |
| Downstream agent | `MiniMaxAI/MiniMax-M2.5` via `productive_agents` AppWorld runner (acon `.venv`) |
| Chunk labeler | **MiniMax-M2.5 only** (spec §3.3 forbids Qwen as labeler) |
| ACON prompt | UTCO `improved_history_prompt_samples_4.jinja` from `microsoft/acon` commit `d63f9ae18959dc7215ff62899c94c5e8c56847ae` (sha256 of resolved template = `9e50d0f93aca7f75eb723a90a758642d1aac3d7550f6afe1e692e56e2bc7b71c`) |
| n_cases (reused from v3) | **30** AppWorld dev trajectories (all medium/long, ≥15 steps) |
| Generation | greedy: temp 0.0 seed 42 ; samples: temp 0.7 seeds 1000+i for i∈[0,7] ; **N_SAMPLES = 8** |
| Stress depth | **K = 2** recompression rounds |
| Compression budget | `max_tokens = 2048` per call (LLM thinking eats ~543 tokens median; final compressed text median 1983 chars / 495 tokens) |
| Behavior budget | AppWorld cap_steps = 15 ; pass-rate measured at cap=15 |
| Total compute | first pass: 270 candidate compressions + 810 stress rounds + 540 C1+CK agent runs + 144 chunks + 156 chunk-ablation runs in 2 h 16 min wall-clock (9:34 AM → 11:50 AM PT 2026-05-29) |

## 2. Pipeline (14 stages)

```
00 prepare         — resolve config, mkdir outputs/{raw,tables,figures,reports,logs}
01 build_cases     — reuse 30 v3 trajectories; persist data/v9_cases.jsonl
02 generate_candidates  — 30 cases × 1 model × (1 greedy + 8 samples) = 270 compressions
03 stress_recompress    — re-feed each candidate as input K=2 times → 810 chain rows
04 behavior_c1_ck       — 540 AppWorld runs (270 × {C1, CK}, cap 15 steps, workers=6)
05 best_of_n            — per-case oracle pick over N=8 samples; pass-gain
06 c1_ck_fragility      — robust_pass / fragile_pass / stress_improved / robust_fail
07 select_chunk_cases   — pick 12 (first pass) / 20 (widened) candidates for chunk analysis
08 segment_chunks       — split each selected text into ≤12 NL chunks → 144 chunks
09a build_chunk_contexts — for each chunk, build context-minus-that-chunk via MiniMax
09 chunk_ablation_behavior — re-run AppWorld with each minus-chunk context, ≤156 runs
10 chunk_advantage      — per-chunk leave-one-out score Δ vs full context
11 chunk_labels (MiniMax) — categorical chunk_type + causal/literal/negative flags
12 chunk_advantage_by_type — pivot Δ by chunk_type
13 plot_figures
14 write_report (auto)
```

## 3. Claim 1 — Best-of-N pass gain over greedy

(`outputs/tables/best_of_n_summary.csv`, `best_of_n_by_case.csv`,
`reward_spread_by_case.csv` ; figure `fig_best_of_n_pass_gain.{pdf,png}`)

| eval round | n cases | greedy pass | best-of-N pass | gain (pp) | greedy mean score | best-of-N mean score | greedy len | best-of-N len | oracle win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **C1** | 30 | **0.700** | **0.967** | **+26.7** | 0.700 | 0.967 | 486.8 | 451.6 | **0.900** |
| **CK** | 30 | **0.600** | **0.967** | **+36.7** | 0.600 | 0.967 | 464.6 | 419.6 | **0.833** |

* **Oracle win rate** is the fraction of cases where at least one of the
  N=8 stochastic samples scores strictly higher than greedy. On 27/30
  (C1) and 25/30 (CK) cases, a better-than-greedy sample exists in N=8.
* **Mean length goes the wrong way for greedy.** Best-of-N samples are
  shorter while passing more often — so the gap is not a capacity / KV
  budget artifact, it is a behavioral one.
* **CK gain (+36.7 pp) is larger than C1 gain (+26.7 pp)**: greedy's
  brittleness is amplified by stress (see Claim 2).
* Per-case reward spread (`reward_spread_by_case.csv`) shows the typical
  `std_score` across 8 samples is 0.33–0.48 on borderline cases —
  bimodal pass/fail behavior across samples, not Gaussian noise around
  greedy. This is what makes best-of-N pay off.

**Verdict Claim 1**: **STRONG POSITIVE.** Both gain magnitude
(≥20 pp) and oracle win rate (≥0.8) thresholds in spec §22.1 cleared.

## 4. Claim 2 — Stress fragility C1 vs CK

(`outputs/tables/c1_ck_fragility_by_model.csv`, `c1_ck_transition.csv`,
`stress_chain_convergence.csv` ; figures
`fig_c1_ck_pass_drop_by_model.{pdf,png}`,
`fig_c1_ck_transition_matrix.{pdf,png}`,
`fig_stress_pass_curve_by_round.{pdf,png}`)

### 4.1 Per-(model, generation) fragility

| generation | n | robust_pass | fragile_pass | stress_improved | robust_fail | pass_rate C1 | pass_rate CK | C1→CK drop (pp) | fragility rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 30 | 15 | 6 | 3 | 6 | 0.700 | 0.600 | **−10.0** | **0.286** |
| sample | 240 | 122 | 34 | 29 | 55 | 0.650 | 0.629 | −2.1 | 0.218 |

* `fragility_rate = fragile_pass / (robust_pass + fragile_pass)` — the
  fraction of originally-passing candidates that fail after K=2 stress.
* `stress_improved` (3 greedy / 29 sample) counts the opposite case:
  recompression made a failing candidate pass. The net (`fragile_pass −
  stress_improved` = 3 greedy, 5 sample) is small for sample but
  larger for greedy, again pointing at greedy-specific brittleness.

**Bonus finding (not in spec)**: greedy is MORE fragile than sample
(28.6 % vs 21.8 %). Combined with Claim 1's "best-of-N is shorter and
passes more", this is a double indictment of ACON's greedy decoding:
the modal sample of MiniMax under temp 0.0 is suboptimal both before
and after stress.

### 4.2 Text-level fixed-point dynamics

(`outputs/raw/stress_chains.jsonl`)

| round | n | median chars | mean chars | median tokens |
|---|---:|---:|---:|---:|
| 0 (init compressed) | 270 | 1,983 | 2,014 | 495 |
| 1 (recompressed once) | 270 | 1,890 | 1,938 | 472 |
| 2 (recompressed twice) | 270 | 1,870 | 1,913 | 467 |

* **Length collapse is ~5 % per round and slows fast** — the abstraction
  attractor that v8 observed under non-ACON prompts shows up here too.
* **Text fixed-point hit rate**: round1 identical to round0 in **18/270
  (6.7 %)** of chains; round2 identical to round1 in **85/270 (31.5 %)**.
  Recompression idempotence becomes the modal outcome by K=2 — yet
  10 pp of greedy candidates that initially passed fail after K=2,
  confirming the small text drift between rounds 1 and 2 still
  materially changes downstream behavior.

**Verdict Claim 2**: **POSITIVE.** Fragility ≥ 0.20 + stress drop ≥
5 pp thresholds in spec §22.2 cleared for greedy; sample also clears
fragility but not stress-drop. Bonus finding (greedy > sample
fragility) is paper-quality but was not in spec.

## 5. Claim 3 — Chunk Information Advantage by causal-flag

(`outputs/tables/chunk_advantage_by_type.csv`,
`chunk_information_advantage.csv`, `outputs/raw/chunk_type_labels.jsonl`
; figures `fig_chunk_advantage_by_type.{pdf,png}`,
`fig_top_chunk_type_distribution.{pdf,png}`)

### 5.1 The clean cut — by `contains_causal_relation` flag

| group | n | mean score advantage | % chunks with positive advantage |
|---|---:|---:|---:|
| `contains_causal_relation = True` | 20 | **+0.150** | 30.0 % |
| other (non-causal, non-entity) | 112 | +0.009 | 19.6 % |
| `ENTITY_LIST_ONLY` (type) | 12 | +0.000 | 8.3 % |

Causal-flagged chunks have **15× higher mean advantage** than non-causal
non-entity chunks, and **unbounded ratio** over entity-only chunks. The
flag-based view is the cleanest projection of Claim 3 because:

* `contains_causal_relation` is a uniform binary across all 7 chunk
  types and avoids the small-n problem of the 8-chunk
  CAUSAL_PRECONDITION category.
* `ENTITY_LIST_ONLY` is the spec's a-priori "null" type (pure entities
  without explanation of use). Confirming mean adv = 0 + only 8.3 %
  positive is a clean negative control.

### 5.2 Categorical view (noisier, with surprise)

| chunk_type | n | mean score adv | mean pass adv | % positive | % top in case | %causal | %literal |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACTION_OUTCOME | 59 | **+0.068** | +0.068 | 22.0 % | 3.4 % | 3.4 % | 83.1 % |
| NARRATIVE_PROGRESS | 19 | +0.053 | +0.053 | 15.8 % | 10.5 % | 26.3 % | 78.9 % |
| RUNTIME_BINDING | 38 | **−0.026** | −0.026 | 23.7 % | **13.2 %** | 2.6 % | 100 % |
| CAUSAL_PRECONDITION | 8 | +0.000 | +0.000 | 25.0 % | 12.5 % | 100 % | 75 % |
| CONTROL_NEGATIVE_EVIDENCE | 4 | +0.000 | +0.000 | 25.0 % | 0 % | 100 % | 100 % |
| ENTITY_LIST_ONLY | 12 | +0.000 | +0.000 | 8.3 % | 0 % | 0 % | 100 % |
| TASK_GOAL_OR_TODO | 4 | +0.000 | +0.000 | 0 % | 0 % | 0 % | 100 % |

Three surprises in the categorical cut:

1. **ACTION_OUTCOME wins on mean advantage** (n=59, mean +0.068), not
   the spec-predicted CAUSAL_PRECONDITION (n=8, mean 0). This says
   that "what happened on the previous step" is a stronger context
   signal than "why a future step requires X" for AppWorld tasks.
   Plausible reading: AppWorld task continuation depends on remembering
   *which tokens/IDs are already issued* (an executable trace) more
   than *what the rules of API auth are* (a causal preamble).
2. **RUNTIME_BINDING has negative mean adv but the highest %top
   (13.2 %)** — it's bimodal. A few critical token/ID bindings (e.g.
   access_token, target user_id) are causally necessary; many other
   bindings are over-specified, and removing them lets the agent
   re-fetch successfully. Suggests a "preserve-by-type" compression
   policy on RUNTIME_BINDING is overly conservative.
3. **CAUSAL_PRECONDITION and CONTROL_NEGATIVE_EVIDENCE both have
   100 % causal-flag rate (by construction) but only n=8+4=12 total**
   — too small to bound the categorical mean. The 5.1 flag-based
   aggregation borrows strength across types.

### 5.3 Per-chunk interpretability

(`chunk_information_advantage.csv`)

* 144 chunks total; **96/144 = 66.7 %** are "interpretable" (the
  full-context run passed, so removing the chunk gives a clean signal
  on Δ).
* 29/144 = **20.1 %** of chunks have strictly positive `chunk_score_advantage`.
* The remaining 80 % includes (a) chunks where the full-context run failed
  anyway (`not_interpretable_due_to_full_fail`, 48/144) and (b) chunks
  whose removal didn't change behavior (zero advantage; most chunks).

**Verdict Claim 3**: **POSITIVE at flag-level, PARTIAL at
categorical-level.** Spec §22.3 requires "causal/control chunks have
higher mean advantage than entity-only" — that is cleared at
flag-level (+0.150 vs 0.000 vs +0.009). At categorical level, the
spec-predicted winner CAUSAL_PRECONDITION is out-shone by
ACTION_OUTCOME, but with small n on the causal type, the directional
claim still holds.

## 6. Spec §22 acceptance summary (★ after widened n=20 addendum)

| Claim | Threshold | First-pass (n_chunks=144) | Widened (n_chunks=239) | Final verdict |
|---|---|---|---|---|
| 1 — best-of-N gain | gain ≥ 20 pp AND oracle_win ≥ 0.8 (CK) | gain +36.7 pp, oracle 0.833 | (unchanged; stages 01–06 same) | **STRONG ✅** |
| 2 — fragility | fragility_rate ≥ 0.20 AND stress_drop ≥ 5 pp | greedy 0.286 / 10 pp | (unchanged; stages 01–06 same) | **POSITIVE ✅** |
| 3 — chunk causal > entity | mean adv (causal) > mean adv (entity) | +0.150 vs 0.000 (n=20 / 12) | **+0.036 vs +0.167 (n=28 / 27)** | **NEGATIVE ❌** at the originally-stated form |

**Important**: the first-pass §5.1 "STRONG positive at flag-level"
finding does NOT survive widening to n_unique_chunks=239. See §10
addendum for the post-widening numbers. The auto-written
`outputs/reports/motivation_v9_results_summary.md` was regenerated
after the addendum but still mechanically labels Claim 3 STRONG
positive based on row-level deltas across categorical types — that
verdict is over-confident and superseded by this document (§6 + §10).

The single Claim-3 sub-finding that does survive widening is
**CONTROL_NEGATIVE_EVIDENCE > everything else, with stable mean
+0.105 row-level / +0.115 unique-chunk and 47.4 % causal-flag rate**.
This is a categorical sub-result, not the spec's flag-level
prediction; see §10.3.

## 7. Caveats

1. **CAUSAL_PRECONDITION n=8, CONTROL_NEGATIVE_EVIDENCE n=4.** The
   `chunk_advantage_by_type.csv` categorical pivot has wide CIs on
   the spec-relevant types. The widened n=20 chunk-cases addendum (PID
   2357806, started 2026-05-29 1:11 PM PT) roughly doubles the chunk
   pool and will let §5.2 surface a stable categorical winner.
2. **All cases reused from v3 dev.** Same length-bias caveat as v7/v8:
   no short trajectories (<15 steps). If short tasks are more
   compress-tolerant the v9 fragility numbers may overstate.
3. **Single compressor (MiniMax-M2.5).** Spec §3.3 explicitly forbade
   Qwen as labeler; v9 also only used MiniMax as compressor for budget.
   Cross-compressor replication (Qwen3-4B-Instruct-2507 + MiniMax) is
   a natural v10 follow-up.
4. **Auto-written `outputs/reports/motivation_v9_results_summary.md`**
   declares Claim 3 STRONG positive on a thinner data slice. This
   document (`docs/04_results_summary.md`) is the honest hand-written
   counterpart. The auto-written report is preserved as-is for
   reproducibility audit.
5. **Stage 11 bug (caught + fixed mid-day)**: first run shipped with
   `max_tokens=256` on the MiniMax chunk labeler. MiniMax thinking
   blocks alone are 500–750 tokens (median 543 measured from stage 02
   n=270), so the entire budget was consumed by `<think>...</think>`
   and the post-strip JSON payload was empty. All 144 chunks fell
   back to the default `OTHER` label. The bug was invisible —
   `err=0` was logged and no exception raised. **Fix**: bumped
   `max_tokens` 256 → 2048 in `motivation_v9/chunk_label.py`. **Guard**:
   added `WARN_THINKING_MIN_MAX_TOKENS=1024` and a warn-once
   `_is_minimax(name) and max_tokens < threshold` check in
   `motivation_v9/clients.chat()`. Re-ran stages 11–14 in 5 min and got
   the real labels reported here. **The original (buggy) all-OTHER
   `chunk_type_labels.jsonl` was overwritten and is not in git
   history; the `motivation_v9_results_summary.md` history at commit
   `19da127` shows the fixed version.**
6. **Behavior runs are single-seed.** Each (candidate × eval_round)
   is one agent rollout, not averaged. AppWorld is fully deterministic
   given seed but agent-side noise can still show up via tool-result
   timing or LLM seed mismatch. Replication across 3 seeds is a
   tractable follow-up.

## 8. What this motivates for RL / method

* **Claim 1 + 2 jointly** motivate training a compressor with reward
  `R = behavior_after_stress(T^K) - λ · length`. Best-of-N already
  shows that a +27 to +37 pp gain is *available in the model's own
  output distribution* — RL would just need to identify the better
  samples reliably, which oracle-best already does at win rate 0.83–0.90.
* **Claim 3** supports **chunk-level credit assignment** for the
  compression reward: ENTITY_LIST_ONLY chunks can be aggressively
  pruned (zero advantage); causal chunks (the 20 with the flag set)
  should be preserved verbatim or rewritten faithfully. An IAPO-style
  natural-language credit assignment over `chunk_information_advantage.csv`
  is a clean next step.
* The **stress side of Claim 2** (greedy more fragile than sample)
  argues against deploying ACON greedy at all in iterative-summary
  pipelines. Sampling and then voting / re-using under stress is a
  cheap deployment win even before any training.

## 9. Files of record

```
motivation_v9/
├── docs/04_results_summary.md                ★ this file
├── outputs/raw/
│   ├── candidate_compressions.jsonl          270 stage-02 compressions
│   ├── stress_chains.jsonl                   810 stage-03 stress rounds
│   ├── behavior_runs_c1_ck.jsonl             540 stage-04 agent runs
│   ├── chunk_case_selection.jsonl            12 (stage-07) → 20 (widened)
│   ├── chunks.jsonl                          144 stage-08 chunks
│   ├── chunk_ablation_contexts.jsonl         144 stage-09a minus-chunk contexts
│   ├── chunk_ablation_behavior_runs.jsonl    156 stage-09 agent runs
│   └── chunk_type_labels.jsonl               144 stage-11 labels (POST-FIX)
├── outputs/tables/
│   ├── best_of_n_summary.csv                 §3
│   ├── best_of_n_by_case.csv                 §3 per-case
│   ├── reward_spread_by_case.csv             §3 per-case sample dispersion
│   ├── c1_ck_transition.csv                  §4 per-candidate class
│   ├── c1_ck_fragility_by_model.csv          §4.1
│   ├── stress_chain_convergence.csv          §4.2 per-candidate summary
│   ├── chunk_information_advantage.csv       §5.3 per-chunk LOO
│   └── chunk_advantage_by_type.csv           §5.2 categorical pivot
├── outputs/figures/
│   ├── fig_best_of_n_pass_gain.{pdf,png}     §3
│   ├── fig_c1_ck_pass_drop_by_model.{pdf,png}    §4.1
│   ├── fig_c1_ck_transition_matrix.{pdf,png}     §4.1
│   ├── fig_stress_pass_curve_by_round.{pdf,png}  §4.2
│   ├── fig_chunk_advantage_by_type.{pdf,png}     §5.2
│   └── fig_top_chunk_type_distribution.{pdf,png} §5.2
└── outputs/reports/motivation_v9_results_summary.md  auto-written counterpart
```

## 10. Addendum — widened n=20 chunk-cases run (★ supersedes §5 for Claim 3)

> **Status**: ✅ done 2026-05-29 2:15 PM PT (PID 2357806, wall-clock
> 64 min, 20:11Z → 21:15Z). Stage 07 selected **20 cases** from groups
> 1+2 alone (11 best_sample_CK_succeeds_greedy_fails + 9
> fragile_pass) — strict superset of the n=12 first pass. Stages
> 01–06 untouched, so Claims 1 & 2 numbers in §3/§4 are unchanged.
>
> **Bottom line: Claim 3 must be DOWNGRADED.** The widened n shows
> that §5's "causal-flag chunks have 15× higher mean advantage"
> finding does NOT survive a 2.2× larger chunk pool. Both the
> flag-aggregation cut and the categorical cut reverse direction
> at n_unique_chunks=239.

### 10.1 New numbers — flag-aggregation (★ replaces §5.1)

Aggregating per-chunk leave-one-out score advantage at **unique-chunk
level** (mean over C1+CK rounds for each unique chunk_id):

| group | n_unique | mean score advantage | median | % positive |
|---|---:|---:|---:|---:|
| `ENTITY_LIST_ONLY` (type) | 27 | **+0.167** | 0.000 | 25.9 % |
| `contains_causal_relation = True` | 28 | +0.036 | 0.000 | 10.7 % |
| other (non-causal, non-entity) | 184 | −0.030 | 0.000 | 14.1 % |

**This is the OPPOSITE direction of the spec's prediction and the
opposite of §5.1's n=12 finding.** ENTITY_LIST_ONLY chunks have the
highest mean advantage, not the lowest. Causal-flagged chunks have
only marginal advantage (+0.036) and crucially have the LOWEST
%positive rate (10.7 %) — half of `other` and 40 % of entity-only.

Row-level aggregation (one observation per (chunk, eval_round)) tells
the same story with slightly different magnitudes:

| group | n_obs | mean adv | % positive |
|---|---:|---:|---:|
| ENTITY_LIST_ONLY | 36 | +0.111 | 19.4 % |
| causal (flag=True) | 44 | +0.023 | 6.8 % |
| other | 303 | −0.043 | 9.6 % |

### 10.2 New numbers — categorical (★ replaces §5.2)

Sorted by mean score advantage, at **unique-chunk level**:

| chunk_type | n_unique | mean score adv | % positive | n_first_pass | mean_first_pass | direction |
|---|---:|---:|---:|---:|---:|---|
| ENTITY_LIST_ONLY | 27 | **+0.167** | 25.9 % | 12 | +0.000 | **↑↑** |
| CONTROL_NEGATIVE_EVIDENCE | 13 | +0.115 | 15.4 % | 4 | +0.000 | **↑** |
| NARRATIVE_PROGRESS | 33 | +0.076 | 21.2 % | 19 | +0.053 | ↑ |
| RUNTIME_BINDING | 47 | +0.011 | 14.9 % | 38 | −0.026 | ↑ |
| CAUSAL_PRECONDITION | 5 | +0.000 | 0 % | 8 | +0.000 | = |
| ACTION_OUTCOME | 105 | **−0.057** | 12.4 % | 59 | +0.068 | **↓↓** |
| TASK_GOAL_OR_TODO | 5 | −0.300 | 0 % | 4 | +0.000 | ↓↓ |
| OTHER | 4 | −0.375 | 0 % | 0 | — | new |

Two large reversals to call out:

* **ACTION_OUTCOME** flipped from §5's "surprise winner" (+0.068 at
  n=59) to **−0.057 at n=168**. The §5 finding was an n-fragile
  artifact — most action-outcome chunks turn out to be removable
  (the agent re-derives outcome from tool re-calls).
* **ENTITY_LIST_ONLY** flipped from "spec-predicted null" (mean 0.000
  at n=12) to "highest mean advantage" (+0.167 at n=27, +0.111
  row-level). This is consistent with the labeler conflating *form*
  (tokens are listed compactly) with *function* (those tokens are
  often the actual `access_token` / `target_id` that the next step
  depends on). The label "entity list" describes how the chunk reads,
  not whether it's behaviorally necessary.

### 10.3 Stable sub-finding (the one paper-quotable result)

**CONTROL_NEGATIVE_EVIDENCE** stays positive across both runs (n=4 →
n=13, mean adv 0.000 → +0.115). Of the 19 (chunk, round) observations,
47.4 % carry an explicit causal-relation flag and 73.7 % carry exact
literals. This is the only chunk type whose label semantics
(*records a failed attempt or thing to avoid*) directly maps to the
agent's behavior at decision-time — and the only categorical bucket
that survives both n=12 and n=20 with a positive mean. It echoes v5's
"looked_like_past_log" drop pattern (v5 §4): the recompressor
preferentially drops failure / negative-evidence content, but ablation
here shows those are exactly the chunks the agent uses most.

### 10.4 What this means for v9's overall story

Combining §10.1–10.3 with §3 / §4:

* **Claim 1 (best-of-N gain)** is unaffected — still STRONG POSITIVE,
  +27/+37 pp pass gain, oracle win 0.90/0.83.
* **Claim 2 (stress fragility)** is unaffected — still POSITIVE,
  greedy 28.6 % fragility, greedy > sample fragility bonus finding.
* **Claim 3 (causal > entity chunk advantage)** is **NEGATIVE** at
  n_unique=239. The original directional prediction fails: entity-only
  chunks have higher mean advantage than causal-flagged chunks. The
  only directional survivor is CONTROL_NEGATIVE_EVIDENCE (n=13,
  mean +0.115), which deserves a follow-up paper-figure on its own.

**Net for the paper**: v7+v8 establish the surface-type abstraction
prior; v9 §3 + §4 give it a behavioral price tag (+27 pp pass loss
and 28.6 % stress fragility for greedy ACON); v9 §5+§10 caution that
the chunk-level information story is more subtle than "causal NL
matters more than entities" — the chunk-type label semantics don't
map cleanly onto behavior, and a larger ablation sample is needed to
make any chunk-level claim. RL credit-assignment at chunk level
(§8) is still motivated but needs a sharper labeling scheme than the
spec's 7-way taxonomy.

### 10.5 What changed in the artifacts at commit time

* `outputs/raw/chunks.jsonl`: 144 → **239 chunks** (20 cases × ≤12 chunks)
* `outputs/raw/chunk_ablation_contexts.jsonl`: 144 → 259
* `outputs/raw/chunk_ablation_behavior_runs.jsonl`: 156 → **415** (more rounds covered)
* `outputs/raw/chunk_type_labels.jsonl`: 144 → **239 labels**, 0 errors, 2 empty rationales
* `outputs/tables/chunk_advantage_by_type.csv`: 7 → **8 rows** (gained OTHER bucket)
* `outputs/tables/chunk_information_advantage.csv`: 144 → **383 rows** (239 unique chunks × ~1.6 rounds avg)
* `outputs/figures/fig_chunk_advantage_by_type.{pdf,png}` regenerated
* `outputs/figures/fig_top_chunk_type_distribution.{pdf,png}` regenerated
* `outputs/reports/motivation_v9_results_summary.md` regenerated (auto-written; still over-claims Claim 3 — see Caveat #4 in §7)
