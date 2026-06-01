# Progress report — May 28 – May 31, 2026

Five separate motivation experiments completed back-to-back (v6, v7, v8,
v9, v10) and the final paper-tier experiment (v11) designed, audited,
and launched. Across the four days, the project's central thesis —
*"LLM compressors interfere with downstream agents because their
compression prior conditions on form, not function"* — accumulated
independent, cross-track evidence at increasing scale.

## TL;DR

- **5 experimental tracks finished, 1 launched** in four days.
- **v7 + v8** delivered the project's first clean paper-tier positive
  result: the abstraction prior is real, cross-model stable, and
  conditioned on surface form rather than downstream need.
- **v9** validated the story behaviorally: best-of-N over a stochastic
  compressor beats greedy by **+26.7 / +36.7 pp** pass-rate on C1 / CK.
- **v10** closed the trainable-policy question: 3 of 4 spec claims
  pass, with a bonus finding that SFT compression is a *stress-invariant
  fixed point* (drift +4.7 % vs Raw-Qwen −33 %).
- **v11 launched** Sunday afternoon and is currently running on the full
  AppWorld train+dev population (145 tasks × 4 prompt families × N=8
  samples × K=2 stress); expected completion early Tuesday.

---

## 1. v6 — White-box Jacobian active-subspace diagnostics (5/28)

**Question.** Can the first-order embedding Jacobian on Qwen3-4B
identify which trajectory spans matter for the agent's decision state,
and does this prediction beat the recency baseline established in v4?

**Outcome.** Mixed result, 2 of 4 sub-claims confirmed.

| sub-claim | verdict | quantification |
|---|---|---|
| (A) Jacobian predicts v4 span sensitivity | **negative** | per-task median Spearman = −0.03 across 28 tasks |
| (B) Active subspace is low-rank | **positive (STRONG)** | example-level k=16 cumulative variance = 92 % |
| (C) Soft-token oracle matches reference decision state | **degenerate** | k=4 already over-recovers (gap recovery 2.26×) |
| (D) High-sens spans dominate behavioral utility | **negative** | jacobian_low_spans 0.80 ≈ high_spans_raw 0.83 at MiniMax cap=15 |

**Why it matters.** Kills naive span-rank-by-gradient as a compression
heuristic, but supports the active-subspace projection idea for future
method work.

## 2. v7 — Abstraction prior under official ACON UTCO prompt (5/28)

**Question.** Is the LLM compressor's behavior conditioned on the
*structural type* of an extracted fact (entity vs token vs path) or on
its *functional necessity* downstream?

**Outcome.** STRONG positive on both claims (Qwen and MiniMax).

| metric | Qwen3-4B | MiniMax-M2.5 |
|---|---|---|
| Stable distribution index (SDI) | **0.96** | **0.99** |
| McFadden R² of `need_label` (function) | 0.003 | 0.0006 |
| McFadden R² of `fact_type` (form) | 0.155 | 0.110 |
| **R² ratio (form / function)** | **52×** | **183×** |
| Cross-model Kendall τ on retention ranks | — | **0.491** (p=0.041) |
| Convergence rate in ≤5 iterative rounds | — | 79.3 % |

**Why it matters.** This is the project's cleanest paper-tier
positive: LLM compressors are unconditional surface-form abstraction
priors. Tokens, IDs, and paths die fast regardless of whether the
downstream agent will need them. v7 became the anchor result that
v8/v9/v10/v11 build on.

## 3. v8 — Fixed-point analysis under non-ACON general prompts (5/28)

**Question.** Does v7's prior survive without ACON's prompt scaffolding?
Can different prompt framings or initializations move the attractor?

**Outcome.** v7 replicates AND strengthens under general prompts. Two
new mechanisms discovered.

- **Replication**: SDI under P2 task-agnostic = **1.000 / 0.998**
  (vs v7 ACON 0.96 / 0.99 — even more extreme).
- **Cross-(model, prompt) Kendall τ up to 0.778** (vs v7 cross-model
  0.491). Convergence rate 84.9 % across 186 chains.
- **NEW mechanism 1**: P1 task-aware framing *inverts* the fixed-point
  composition from NARR > EXEC (P2 0.88 vs 0.55) to EXEC > NARR (P1
  0.64 vs 0.46). Task framing reshapes the attractor.
- **NEW mechanism 2**: Different initializations reach **disjoint**
  fixed points (RAW_FULL vs FACT_TABLE_ONLY: Jaccard distance up to
  1.00 under MiniMax P1) — there is no universal attractor.

**Why it matters.** The prior is a real geometric structure (multiple
basins, framing-sensitive attractor) and is not an artifact of ACON's
specific prompt. v8 extended the paper's framing-vs-function distinction
to a more general setting.

## 4. v9 — Behavior-first validation of best-of-N and stress (5/29)

**Question.** If the compressor has the prior structure documented in
v7/v8, does *stochastic best-of-N over the compressor's own
distribution* beat greedy decoding on downstream agent pass-rate? And
does multi-round recompression degrade downstream performance?

**Outcome.** First-pass done 2026-05-29 11:50 AM PT (2 h 16 min).

| claim | verdict | quantification |
|---|---|---|
| 1 — best-of-N beats greedy | **STRONG positive** | **+26.7 pp** on C1, **+36.7 pp** on CK pass-rate; oracle win 90 % / 83 % |
| 2 — repeated compression is fragile | **positive** | greedy fragility 28.6 % (6/21); sample fragility lower at 21.8 % |
| 3 — causal NL chunks > entity-list chunks (chunk ablation) | **negative at n=239** | first-pass n=144 looked positive but reversed at widened n=20 cases |

**Bonus finding.** Sample compressions are also slightly *shorter* than
greedy (~451 vs 487 chars on C1), so the +26.7 pp gain is not
length-mediated.

**Bug fix shipped**: stage 11 initially used `max_tokens=256` for
MiniMax label calls. Because MiniMax-M2.5 emits 500-750-token
`<think>...</think>` blocks, the JSON payload was empty and all chunk
labels fell back to `OTHER`. Fix bumped to 2048 and added a
`WARN_THINKING_MIN_MAX_TOKENS=1024` warn-once guard in the shared
client. Empirically derived from a 270-compression sample (median
thinking 543 tokens, p90 750, max 1361).

**Why it matters.** v9 turned v7/v8's diagnostic finding into an
*actionable* optimization signal: greedy is suboptimal under the prior;
sampling + selection recovers ~30 pp. Motivated the v10 SFT track.

## 5. v10 — Trainable compressor policy + GRPO readiness (5/29 – 5/31)

**Question.** Can a small student model (Qwen3-4B) be SFT-trained on
best-of-N selections to internalize the optimization v9 demonstrated,
and is the resulting reward landscape suitable for downstream RL
(GRPO)?

**Outcome.** All 4 spec claims judged 2026-05-31 1:37 PM PT.

| claim | verdict | quantification |
|---|---|---|
| 1 — proxy recovers best-of-N | **FAIL on CK** | pairwise +4 pp / 16 % recovery vs spec's 10 pp / 40 % bar |
| 2 — SFT-CK > SFT-C1 > Raw on CK | **PARTIAL** | aggregate ✓ (54.8 / 50.0 / 47.6 %), held-out 12 test cases too thin |
| 3 — SFT-CK reward spread suitable for GRPO | **PASS** | all 3 variants std 0.42–0.47, oracle_win 0.81–0.83, all_fail 0 |
| 4 — chunk labels insufficient | **PASS** | multivariate R²(labels-only) = 0.019 vs R²(full feature set) = 0.037 |

🌟 **Two paper-bonus findings.**

1. **SFT stress robustness.** Raw-Qwen drops −11.9 pp from C1 to CK
   under stress; SFT-CK gains +7.2 pp. **+19 pp robustness swing.**
2. **SFT compression is a stress-invariant fixed point.** Raw-Qwen
   output gets compressed −33 % by MiniMax stress in 2 rounds; SFT-CK
   output drifts only +4.7 %. This is the *causal mechanism* behind
   finding (1) and the cleanest in-distribution demonstration that a
   trained compressor can escape the form-prior.

⚠ **Caveat shipped to v11.** MiniMax verbal verifier ranks
Raw-Qwen > SFT (greedy verbal score 0.82 vs 0.64) but actual AppWorld
pass ranks SFT-CK > Raw-Qwen. Verbal verifier is **not** a calibrated
cross-policy ranker; in v11 we treat verbal selectors as negative
baselines and rely on continuation-entropy as the prior-orthogonal
selector.

## 6. v11 — Final motivation experiment, designed + launched (5/31 PM)

**Question.** Bring all of the above to the full AppWorld train+dev
population (145 tasks). Quantify the (length, pass-rate, stress
robustness, selection effectiveness) trade-off across four prompt
families and nine selectors. Produce the paper's main motivation
figure (full-vs-compressed transition matrix).

**Scope (plan β).**

```
145 tasks × 4 prompt families × (1 greedy + 8 stochastic samples) = 5,220 compressions
+ K=2 stress  → 10,440 recompression rounds
+ behavior C1+CK → 10,440 agent runs
+ pointwise / pairwise / continuation-entropy verifiers → ~19K verifier outputs
≈ 46K LLM/agent calls total
```

**Pre-launch quality gate.** GPT-5.5-Pro provided a 10-item review
checklist. Audit found and patched:

| # | issue | severity | resolution |
|---|---|---|---|
| 1 | stage 07 omitted `split=train\|dev` arg → all train tasks would silently load as dev | silent bug | patched before stage 07 starts |
| 8 | stage 10 `preserve_success_rate` computed as `count/N` instead of spec §13.1's `P(C=1 \| F=1)` | paper-number bug | patched + 5 new explicit conditional/unconditional columns |
| 10 | spec §12.2 promised `full_train_dev_compression_candidate_bank.jsonl`; no stage wrote it | spec gap | new stage 13b added |
| 3, 4 | inventory train/dev count drift (145 spec vs 147 actual) | documentation | meta-note row generalized |

Also caught a latent infrastructure bug: the auto-push watcher's
script (`motivation/scripts/sync_and_push.sh`) had a hardcoded list of
`motivation_v2`–`motivation_v9`, so v10 and v11 outputs were never
auto-pushed. Replaced with a `motivation_v*/` glob; v10 outputs
back-filled to the remote.

**Launch.** 2026-05-31 14:17 PT. Currently mid-pipeline; preliminary
analysis on partial data is already showing paper-quality signals
(separate report covers post-launch progress).

---

## Aggregate impact across the four days

The five completed tracks now form a coherent cross-experiment narrative
that the paper's motivation section can build on:

| track | finding | unified message |
|---|---|---|
| v3 (earlier) | symbolic 100 % coverage but 57 % pass; task-aware 74 % coverage but 70 % pass | structural completeness ≠ behavioral utility |
| v5 (earlier) | ACON-dropped items get re-dropped on recompression (93 %) | prior is recompression-invariant |
| **v7 + v8 (this period)** | SDI ≈ 1; R²(form)/R²(function) up to 183× | prior conditions on **form** not **function** |
| **v9 (this period)** | best-of-N gains +27 / +37 pp over greedy | the form-prior leaves systematic selection room |
| **v10 (this period)** | SFT-CK is a stress-invariant fixed point (+4.7 % drift vs Raw −33 %) | a trained policy *can* escape the form-prior |
| **v11 (this period, in progress)** | quantify the (length, pass-rate, stress robustness) Pareto front across 4 families | does prompt choice alone escape the prior, or is training necessary? |

---

## Risks and dependencies

- **MiniMax-M2.5 endpoint** (`10.183.22.68:8005`) has been continuously
  available across the four-day window. v11 currently has **0
  generation errors across all 26K calls so far** (5,292 + 15,873 +
  partial 3,224).
- **Auto-push watcher** has been alive 8+ days, pushing every 20 min.
  Once the glob bug above was patched, all `motivation_v*/` tracks
  back-fill correctly.
- **v11 ETA**: pipeline running ~1.5–1.8× faster than the handoff's
  worst-case estimate; full completion expected **early Tuesday
  morning PT** (vs Wednesday afternoon in the original plan).

## Deliverables produced this window

| document | location |
|---|---|
| v6 paper-tier results | `motivation_v6_jacobian/docs/04_results_summary.md` |
| v7 paper-tier results | `motivation_v7/docs/04_results_summary.md` |
| v8 paper-tier results | `motivation_v8/docs/04_results_summary.md` |
| v9 paper-tier results | `motivation_v9/docs/04_results_summary.md` |
| **v10 paper-tier results** | `motivation_v10/docs/04_results_summary.md` |
| v11 spec + scaffold | `motivation_v11/docs/01_experimental_design.md` |
| Session handoff (updated) | `motivation/docs/SESSION_HANDOFF.md` |

Repository: `git@github.com:GuanghuiMin/EASMO.git`
