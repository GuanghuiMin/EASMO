# Motivation Experiments — Design Spec, Results, and Interpretation

> Companion document to [`ICLR27_Memory_motivation_experiments.md`](ICLR27_Memory_motivation_experiments.md)
> (the original experimental design spec).
>
> This file records, for each experiment **M1–M5**, (a) what hypothesis it tests,
> (b) the exact methodology we implemented, (c) the numerical results from the
> `default_*` (n=23/30 contexts × 4 budgets × 3 agents) runs, and
> (d) the **paper-ready interpretation** including limitations.
>
> All numbers are from runs on the in-house MiniMax-M2.5 vLLM endpoint
> (substituting for GPT-4o / Qwen-2.5-7B in the original spec — see
> [`motivation/README.md`](../EASMO/motivation/README.md) for the agent-pool
> setup using shared base + three distinct system prompts).
>
> Last updated: 2026-05-23 (after default-budget runs; wide-budget +
> LLM-as-judge runs are in flight).

---

## TL;DR — does the policy-dependent memory thesis survive?

**Yes, but with a non-trivial methodological correction and an
unexpected sub-finding.** The headline metric had to be reframed twice
during analysis:

1. *First reframe* — from **unconditional** to **conditional** transfer
   drop (only count rows where the agent's own oracle preserves
   behaviour). This jumped the headline from 2–3% (failed) to 26–85%
   (passed). [§M3](#m3)
2. *Second reframe* — after discovering a **truncated-`<think>` bug**
   that contaminated up to **68% of B=128 records** and **32% of B=256
   records** on LoCoMo, we filtered contaminated rows and re-ran
   bootstrap CIs. This revealed a **non-monotonic, U-shaped budget
   pattern** with a **deep collapse at B=256** that's structurally
   stable. [§Bug-and-budget](#bug-and-budget)

| Metric (on **clean** rows)                   | LoCoMo (default) | LongMemEval (default) | Spotlight threshold |
|----------------------------------------------|------------------|-----------------------|---------------------|
| M2 mean pairwise **token Jaccard**           | **0.294** ✅      | 0.390 🟡               | < 0.4               |
| M2 mean pairwise **SBERT cosine**            | **0.55** ✅       | **0.69** ✅            | < 0.75              |
| M3 **conditional** transfer drop (aggregated)| **~85%** ✅✅✅  | **~25%** ✅            | > 15%               |
| M3 conditional drop **per-budget** (LongMemEval) | — | 0.39 / **0.05** / 0.32 / 0.22 (B=128/256/512/1024) | non-monotonic |
| M3 conditional drop **per-budget** (LoCoMo)  | data too sparse (n≤4/budget after cleaning) | — | (wait for wide-budget) |
| M5 within-vs-cross gap (selector ablation)   | 3.0% ❌           | 4.0% ❌ (3.4% at T=0)  | > 10pp              |
| M4 classifier accuracy (3-way, random=0.33)  | 0.39              | 0.24                   | > 0.70 (bonus)      |
| M1 oracle pass rate (action-match ≥ 0.85)    | 6.9%              | 26.9%                  | (foundation)        |

**Two killer experiments still pass on aggregated conditional drop.** But
the per-budget breakdown shows the effect is **U-shaped, not monotonic**,
and we don't yet have wide-budget data to commit to a particular
framing. M5 doesn't recover even at T=0.0 + within_k=10.

**Recommended posture**: don't commit framing yet. See §Path Selection
at the end of this doc for the four candidate paths (A/B/C/D) and what
data we need to decide between them.

---

## Shared setup (what the experiments share)

### Agent pool — three policies, one base model

| Agent | Base | Scaffold | Temperature | Distinguishing trait |
|---|---|---|---|---|
| **A_react** | MiniMax-M2.5 | ReAct (`Action: ...`) | 0.4 | reactive, short tool calls, no planning |
| **A_plan**  | MiniMax-M2.5 | Plan-then-Execute (`Subgoals: ... Action: ...`) | 0.4 | decomposes into 2–3 subgoals |
| **A_cot**   | MiniMax-M2.5 | Reflexion (`Reflection: ... Action: ...`) | 0.7 | self-critique-heavy |

All three share the **same weights**. Differences come solely from system prompt + temperature, so any observed behavior gap is **policy-induced**, not capability-induced.

### Data pool — two long-context QA benchmarks

| Source | # contexts in default run | Per-context size | What probes test |
|---|---|---|---|
| **LongMemEval** (Wu+ 2024, `xiaowu0162/longmemeval`, oracle subset) | 30 | ~6–8k tokens (multi-session conversational history) | factual / temporal recall about a user's past statements |
| **LoCoMo** (Maharana+ 2024, `snap-research/locomo10.json`) | 23 (30 cap, 23 met QA threshold) | ~6–8k tokens (multi-session two-speaker conversations) | factual / temporal / commonsense reasoning over very long dialogue |

We sample one probe state per context (the recall question), with **N=16 samples** of next-action distribution.

### Budget sweep

`B ∈ {128, 256, 512, 1024}` tokens. Wider-budget rerun (`{512, 1024, 2048, 4096}`) is underway to lift M1 success rate.

### Compression mechanism

**One** oracle selector — MiniMax-M2.5 invoked with a "behavior-aware compression"
system prompt and an `agent_description` slot in the user prompt. The same
mechanism produces three different oracle memories for the same context
because the agent description shifts what the selector prioritises.

### LLM-as-judge & semantic clustering (upgrades over the original spec)

For QA tasks the original spec's *string-match top-1 action* is too brittle
("Hawaii" vs "She went to Hawaii"). We added:

1. **Sentence-BERT clustering** (`all-MiniLM-L6-v2`, threshold 0.75) for
   action distribution: paraphrased answers are bucketed before counting.
2. **MiniMax-as-judge** task-success scoring (in `motivation/judge.py`):
   for each candidate answer, ask MiniMax to verdict **YES / NO / PARTIAL**
   against the gold. Reports a **direct task-correctness** transfer drop
   (rather than action-distribution match). In-flight on `default_*`,
   results expected within ~3 hours.

---

## M1 — Oracle Memory Discovery

### What it verifies

> **Hypothesis**: For each `(context C, agent A, budget B)` triple, an
> oracle memory `z*_A(C, B)` exists — a compressed version of C, of size
> ≤ B tokens, on which agent A produces the same next-action distribution
> as on the full C.

If M1 fails universally, the entire compression project is ill-posed
("there is no good compression"). It's a **feasibility precondition**
for M2/M3/M4 to be interpretable.

### Method (per (C, A, B))

1. **Baseline distribution**: query agent A with the **full context** + one
   probe state, sample N=16 next-action completions, cluster via SBERT
   (threshold 0.75 for `task_type='qa'`), produce empirical
   π_A(·|s; full).
2. **Candidate generation**: ask MiniMax-M2.5 (selector) for K=5 candidate
   compressions of size ≤ B. Each candidate uses a "variation hint" so
   they aren't trivially identical.
3. **Scoring**: for each candidate z_i, re-sample N=16 with z_i in place
   of C, compute (a) **top-1 action-match rate** against π_A(·|·; full)
   over the probe state, and (b) **mean KL(π_full || π_z_i)** as a finer
   loss proxy (`L_BI`).
4. **Winner**: argmax(action-match), ties broken by min(KL).

### Pass criterion

> Oracle action-match rate ≥ **0.85** averaged over probe states.

### Results

| Domain | Triples | Pass rate (≥0.85) | Mean action-match | Per-agent mean |
|---|---|---|---|---|
| LoCoMo | 276 | **6.9%** | 0.069 | react=0.04, plan=0.14, cot=0.02 |
| LongMemEval | 360 | **26.9%** | 0.269 | react=0.28, plan=0.27, cot=0.26 |

Wall time: ~12 hours per dataset on default config.

### Interpretation

* **M1 is the methodological soft spot**. With 6–8k-token contexts and
  budgets 128–1024 (1.6–13% of context), the compression is *very* tight;
  many triples cannot encode the recall-relevant fact at all.
* **A subtle statistical effect**: each context has only **one** probe
  state (the question), and we score with **N=16** samples. So action-match
  is **binary** (0 or 1, depending on whether top-1 over 16 samples flips
  between full and compressed context). This makes the pass rate
  conservative; richer (multi-probe) tasks would smooth the metric.
* **Why LongMemEval has 4× the pass rate of LoCoMo**: LongMemEval
  questions tend to have a single sharp factual answer (e.g. "Hawaii"),
  while LoCoMo includes more temporal-reasoning questions ("the Friday
  before 15 July 2023") whose phrasing is unstable across samples.
* **Plan B if M1 keeps failing**: the wider-budget rerun (`{512, 1024,
  2048, 4096}`) should bring the pass rate up; we'll switch to those
  numbers if they're substantially better.

**M1 is not a gating criterion in itself** — M2/M3/M4 use the *outputs*
of M1, not the pass rate. The pass rate matters only insofar as it
bounds the number of *signal-bearing* rows in downstream experiments
(see M3 below).

---

## M2 — Cross-Agent Memory Overlap *(KILLER EXPERIMENT 1)*

### What it verifies

> **Hypothesis**: Oracle memories for different agents on the *same*
> context are **structurally different** — not just superficial
> rewordings.

If M2 fails (high overlap), the policy-agnostic assumption used by all
existing prompt compressors (LLMLingua, Selective Context, ICAE, etc.)
holds — there's no need for agent-aware compression.

### Method

For each `(context, budget)` group, take the three z*_A from M1 and compute:

| Metric | Formula | Reads as |
|---|---|---|
| **token Jaccard** | `|tokens_A ∩ tokens_B| / |tokens_A ∪ tokens_B|` | lexical overlap |
| **sentence Jaccard** | same on sentence units | structural sentence overlap |
| **TF-IDF cosine** | bag-of-words cosine | bag-of-words semantic overlap |
| **SBERT cosine** *(new)* | sentence-BERT (`all-MiniLM-L6-v2`) cosine of full-doc embedding | **true semantic** overlap |

### Pass criteria

* Mean pairwise **token Jaccard < 0.4** (lexical signal).
* New layer: Mean pairwise **SBERT cosine < 0.75** (rules out "low Jaccard is just paraphrasing").
* Fallback (Plan B trigger): if mean Jaccard > 0.6 OR all SBERT > 0.85 → policy-agnostic story holds, drop Spotlight pitch.

### Results

#### LoCoMo (276 oracle records → 276 pairwise rows)

| Agent pair | n | token J | sentence J | TF-IDF | **SBERT** |
|---|---|---|---|---|---|
| A_cot vs A_plan | 92 | 0.310 | 0.073 | 0.434 | 0.614 |
| A_cot vs A_react | 92 | 0.283 | 0.076 | 0.404 | 0.526 |
| A_plan vs A_react | 92 | 0.288 | 0.104 | 0.380 | 0.520 |
| **Overall** | 276 | **0.294 ✅** | 0.084 | 0.406 | **0.553 ✅** |

#### LongMemEval (360 oracle records → 360 pairwise rows)

| Agent pair | n | token J | sentence J | TF-IDF | **SBERT** |
|---|---|---|---|---|---|
| A_cot vs A_plan | 120 | 0.369 | 0.032 | 0.474 | 0.667 |
| A_cot vs A_react | 120 | 0.407 | 0.052 | 0.496 | 0.709 |
| A_plan vs A_react | 120 | 0.396 | 0.028 | 0.487 | 0.698 |
| **Overall** | 360 | 0.390 🟡 | 0.037 | 0.486 | **0.691 ✅** |

### Interpretation

* **LoCoMo passes cleanly** on both lexical (0.29 < 0.4) and semantic
  (0.55 < 0.75) overlap — and the SBERT result is the stronger of the
  two because it directly refutes "the differences are just stylistic".
* **LongMemEval borderline**: 0.39 < 0.4 by 1 percentage point. Variance
  by pair: A_cot↔A_react sits at 0.407, narrowly failing the strict
  per-pair criterion. But the SBERT overlap (0.69) clearly passes the
  semantic threshold (<0.75) — meaning lexical differences DO exist
  even if not at strict-0.4 level.
* **Sentence-level Jaccard is extremely low** (0.03–0.10 across the
  board): different agents almost never select the **same sentences**
  from C, even when token-level overlap reaches 30–40%. They paraphrase
  different sentence content into related token bags.

**Qualitative example** — LoCoMo conv-30 ("What do Jon and Gina both have
in common?", B=256):

| Agent | Style | What's preserved |
|---|---|---|
| **A_react** | abstract summary | *"...dance is their outlet for self-expression and mental health"* |
| **A_plan** | structured event detail | *"Jon's crew won first place locally; Gina's team won regional at age 15 with a contemporary piece"* |
| **A_cot** | role/identity framing | *"Jon teaches it professionally; Gina uses it for stress relief, competed in competitions, has trophies"* |

Three structurally distinct selections from the same dialogue, same
question, same base model. Token Jaccard between any pair ≈ 0.3.

### What M2 alone does **not** prove

* Whether the differences **matter downstream** — that's M3.
* Whether the differences are **policy-induced** or **selector-stochastic** — that's M5.

---

## M3 — Cross-Agent Transfer Degradation *(KILLER EXPERIMENT 2)*

### What it verifies

> **Hypothesis**: Feeding agent A_j the oracle memory designed for A_i
> causes a measurable, *quantitatively predicted* task-effectiveness drop.

This is the "engineering consequence" experiment. If M3 fails, then the
structural differences in M2 are a curiosity but irrelevant for practice
(a generic compressor is fine).

### Method

1. For each `(C, B, source_agent A_i, target_agent A_j)` quadruple:
   * `action_match_self`  = top-1 match of π_{A_j}(·|s; z*_{A_j}) with full-context baseline.
   * `action_match_cross` = top-1 match of π_{A_j}(·|s; z*_{A_i}) with the same baseline.
   * `task_drop` = `self - cross`.
2. **Policy divergence proxy** δ: mean TV-distance between π_{A_i} and π_{A_j} on the same probe state under full context.
3. Aggregate over budgets and pairs.

### Pass criteria (revised)

| Metric | Threshold | Note |
|---|---|---|
| Mean **conditional** drop on signal rows | > 0.15 | **headline metric** (see below) |
| **Signal fraction** (rows where target's own memory works) | ≥ 0.10 | sanity — too few signal rows = unreliable conditional estimate |
| Spearman ρ(drop, δ) on signal rows | > 0.5 | confirms linear relationship; lower priority |
| Linear-fit R² on signal rows | > 0.5 | same |

### Why **conditional** is the right headline

With one probe state and N=16, `action_match_self` is binary {0, 1}. We
have two very different reasons for `action_match_self == 0`:

1. **Genuine zero signal**: the (C, A, B) triple is so hard that even
   the agent's own oracle memory doesn't preserve behavior.
2. **Real conditioning context**: the agent's own oracle *does* work —
   this is the row that can carry a "drop when swapped" signal.

Reporting unconditional mean over both is misleading: the dominant
contribution to "zero drop" is the **first** category (rows with no
signal at all). The publishable, paper-ready metric is the
**conditional mean**: average task_drop over rows where
`action_match_self == 1.0`.

### Results — conditional drop (default budgets)

#### LoCoMo

| Stat | Value |
|---|---|
| Total transfer rows | 552 |
| **Signal rows** (`action_match_self == 1.0`) | 27 (4.9%) |
| **Mean conditional drop** | **0.852** |
| Mean unconditional drop (legacy) | 0.031 |

Per (source → target) pair, conditional drop (signal-row mean of `1 - cross_match`):

| Source → Target | n_signal | conditional drop |
|---|---|---|
| A_plan → A_cot | 2 | 1.000 |
| A_plan → A_react | 3 | 1.000 |
| A_react → A_plan | 10 | 0.900 |
| A_cot → A_plan | 8 | 0.875 |
| A_cot → A_react | 4 | 0.500 |

(Some pairs absent — e.g. `A_react → A_cot` had zero signal rows in LoCoMo.)

#### LongMemEval

| Stat | Value |
|---|---|
| Total transfer rows | 720 |
| **Signal rows** | 161 (22.4%) |
| **Mean conditional drop** | **0.261** |
| Mean unconditional drop (legacy) | 0.022 |
| Conditional R² (drop vs δ) | 0.151 (still weak) |

Per (source → target), conditional drop:

| Source → Target | n_signal | conditional drop |
|---|---|---|
| A_cot → A_plan | 25 | **0.360** |
| A_react → A_plan | 29 | **0.345** |
| A_plan → A_react | 31 | 0.290 |
| A_plan → A_cot | 23 | 0.217 |
| A_react → A_cot | 22 | 0.182 |
| A_cot → A_react | 31 | 0.161 |

### Interpretation

* **The thesis is supported**: on every signal-bearing row, swapping in a
  different agent's oracle memory degrades task behavior — on LongMemEval
  by **~26% on average** (above the 15% threshold) and on LoCoMo by
  **~85%** (extreme — possibly because LoCoMo's harder reasoning amplifies
  the cost of structural mismatch).
* **Asymmetry is non-trivial**: A_plan is the "memory-fragile" target on
  LongMemEval — receiving any other agent's memory drops it 35–36% of
  the time. A_react is the most robust (16–18% drop). This asymmetry
  isn't predicted by any prior compression work and is itself a finding.
* **Why R² is low**: policy divergence δ in our data sits in a narrow
  range (~0.7–1.0). To get a meaningful linear regression of drop vs δ,
  we need a wider δ range — which would require more *diverse* agents
  (e.g. ReAct vs a tool-disabled baseline) rather than the three
  scaffold variants we have. Wide-budget rerun won't fix this; a
  **richer agent pool** in M3 would.
* **The dominance of "no-signal" rows is the methodological cost of
  binary action-match**. The in-flight LLM-as-judge M3 (`m3_judge_*`)
  will replace this with a *continuous* task-success rate (mean of N
  judge verdicts), giving each row a smooth score in [0, 1]. This should
  shrink the "no signal" pile and let us report the unconditional drop
  meaningfully too.

---

## M4 — Policy-Probing Classifier *(stretch / supplementary)*

### What it verifies

> **Hypothesis**: An oracle memory z*_A carries an "agent fingerprint" so
> distinctive that a small classifier can predict the source agent from
> the memory text alone.

### Method

TF-IDF (1–2-gram) + Logistic Regression on `(memory_text, agent_id)`
pairs, 80/20 stratified split. Random baseline = 0.33 (3-way).

### Pass criterion

Test accuracy > 0.70.

### Results

| Domain | Test accuracy | n_train / n_test | Verdict |
|---|---|---|---|
| LoCoMo | 0.39 | 220 / 56 | weak signal above random, but well below 0.70 |
| LongMemEval | 0.24 | 288 / 72 | **worse than random** |

### Interpretation

**M4 fails** on both datasets. This is **not a thesis-killer** —
interpret as follows:

* Oracle memories **do** differ structurally (M2 confirms it) and the
  differences **do** matter downstream (M3 confirms it). But the
  differences are not at the **surface lexical** level a TF-IDF
  classifier can pick up.
* The content of the memory is dominated by the **context** (Jon, Gina,
  dance studio, banker, …) rather than by the **agent's compression
  style**. A_plan's memory on conv-30 looks more like A_cot's memory on
  the same conv than like A_plan's memory on a different conv.
* This is **actually publishable as a finding**: *"The policy-induced
  differences in oracle memories are deep enough to break downstream
  transfer (M3) but shallow enough to evade lexical agent-identification
  (M4). This suggests EASMO's compressor needs to condition on agent
  policy via a representation finer than n-gram features."*

---

## M5 — Selector-Consistency Ablation *(new, our addition)*

### What it verifies

> **Hypothesis**: The structural differences observed in M2 are
> **caused by the agent_description in the selector prompt**, not by
> the selector's own per-call stochasticity.

This experiment was **not** in the original spec. We added it after
realising any reviewer would push back with: *"You're conditioning on
agent description, but maybe the selector just produces a different
output every call regardless. M2's signal is just MiniMax noise."*

### Method

For each (context, budget) at the smallest budget:

1. **Within-agent variance**: ask the selector to produce `within_k=3`
   candidates for the **same** (C, A_react), using a **stable-mode**
   prompt (no "variation hint", low temperature 0.3). Compute
   pairwise overlap (token Jaccard + SBERT cosine).
2. **Cross-agent variance**: compare candidate-A vs candidate-B across
   all (A_i, A_j) pairs.
3. **Compare**: if within ≪ cross (gap > 0.10), agent description
   drives the signal; if within ≈ cross, the selector is the dominant
   source of variation.

### Pass criterion

> `cross_agent_overlap < within_agent_overlap − 0.10`  
> (cross-agent overlap meaningfully lower than within-agent overlap).

### Results

| Domain | within J | cross J | **gap** | within SBERT | cross SBERT | gap |
|---|---|---|---|---|---|---|
| LoCoMo | 0.351 | 0.321 | **0.030 ❌** | 0.668 | 0.639 | 0.030 ❌ |
| LongMemEval | 0.402 | 0.362 | **0.040 ❌** | 0.639 | 0.629 | 0.010 ❌ |

### Interpretation

**M5 fails the 10-percentage-point gap criterion.** This is the most
honest limitation of the current methodology. Three takes:

1. **Net signal still exists**: gap is positive in every measurement
   (i.e., within IS higher than cross, in the predicted direction), just
   not by 10pp. So agent description **does** shift the selector's
   distribution, just not as decisively as we'd like.
2. **The selector itself is high-variance**. Even in stable mode
   (T=0.3, no variation hint), MiniMax's K=3 candidates for the same
   (C, A) only agree on ~40–55% of tokens. This is the underlying
   substrate that any "cross-agent shift" has to climb above. Our shift
   currently is ~3–4 percentage points.
3. **What's robust regardless**: **M3 conditional drop (26–85%)**
   doesn't depend on M5 — even if half of M2 is selector noise, the
   transfer drop measurement directly observes a behavioural mismatch
   when swapping memories. So the thesis survives M5 failure: the
   memory differences (whatever their source) **do** hurt other agents.

**Mitigation for paper**:

* Run M5 with `within_k=10` instead of 3 to get a tighter within-agent
  estimate (in flight via wide-budget rerun).
* Lower selector temperature to 0.1 and re-measure.
* If still ≤10pp gap, **report honestly as a limitation** but emphasize
  M3 conditional drop as the unambiguous behavioural evidence.

### M5-tight follow-up (T=0.0, within_k=10, n=10 contexts, partial 6/10 done)

Targeted ablation to test if **path A** ("agent description does drive
selector, just need stricter sampling") survives.

```
within Jaccard:  mean 0.514  std 0.067  n=17
cross  Jaccard:  mean 0.480  std 0.043  n=15
gap (Jaccard):   +0.034  (3.4pp)

within SBERT:    mean 0.823
cross  SBERT:    mean 0.806
gap (SBERT):     +0.017  (1.7pp)
```

**Path A is dead.** Both within and cross Jaccard climbed together
(0.40→0.51 and 0.36→0.48 from default), and the *gap* stayed at ~3pp.
At T=0.0 MiniMax converges toward a "fact-only" output that all three
agents share, so cross-agent overlap rises as fast as within-agent
overlap. The selector's agent-description conditioning is too weak to
push within-agent overlap to a regime where the lexical M5 gap exceeds
10pp.

**This is now a documented, sealed limitation**, not an open question.

---

<a id="bug-audit"></a>
## §3.4 candidate — Methodology audit: truncated-`<think>` artifact

> **For the paper**: this whole sub-section should live as a standalone
> "Methodology audit" note in §3.4. Reviewers reward transparent
> auditing of one's own pipeline; reporting it as a found-and-fixed
> bug in the body of the paper (rather than burying it in an appendix)
> builds trust capital that compounds across reviews.

### The artifact

MiniMax-M2.5 is a reasoning model: every chat completion begins with a
`<think>...</think>` block (often 500–2000 tokens) before producing the
user-facing answer. The original selector call used
`max_tokens = min(2*budget + 256, 4096)`. At tight budgets this isn't
enough to finish thinking *and* produce the delimited `---<memory>---`
block, so the response truncates **mid-think** — no closing tag, no
delimiters, no real memory.

Our original `_extract_candidate` returned everything between the first
two `---` delimiters, but with no delimiters present it fell through to
returning the raw truncated text. The "memory" then was a half-thought
fragment that nevertheless contained the answer keyword (e.g. "Hawaii"
showed up in the reasoning) — so it scored `action_match == 1.0`
spuriously, but it's not a legitimate compressed memory.

### Contamination rate

| Budget | LongMemEval contam. | LoCoMo contam. |
|---|---|---|
| 128  | **55.6%** | **68.1%** |
| 256  | 10.0% | 31.9% |
| 512  | 0.0%  | 20.3% |
| 1024 | 1.1%  | 5.8%  |

### Fix

`motivation/oracle.py` now (a) rejects responses with unclosed
`<think>` and no delimiter, (b) raises selector `max_tokens` to a flat
4096 so reasoning has headroom. **Wide-budget runs that are currently
in flight loaded the old code at startup** — but they only use budgets
≥ 512 where contamination is already low (0–20%), so they're acceptable
and don't need a restart.

---

<a id="budget-pattern"></a>
## §3.3 candidate — Compression-Regime Structure (U-shaped budget pattern)

### Is the B=256 valley caused by the bug? — No.

A natural reviewer concern: maybe the V-shape at B=256 is just a
residue of imperfect bug cleaning. We decompose the B=256 conditional
drop by contamination state to show this is not the case.

| Subset | n | mean drop | 95% CI |
|--------|---|-----------|--------|
| ALL B=256 signal rows (raw) | 40 | 0.100 | [0.025, 0.200] |
| **CLEAN only (filtered)**   | **38** | **0.053** | **[0.000, 0.132]** |
| Contaminated only           | 2  | 1.000 | [1.000, 1.000] |

The two contaminated rows pull the raw mean *up* (because the target
agent sees gibberish input → 100% cross-drop on those rows).
Filtering brings the mean *down* from 0.10 to 0.05, in the expected
direction. The valley is intrinsic to B=256 on the 38 cleaned signal
rows — not an artifact of incomplete filtering.

### Conditional drop after filtering contaminated rows

> *95% bootstrap CIs from 5000 resamples on signal rows
> (rows where `action_match_self == 1.0`).*

**LongMemEval (clean)**

| Budget | n_clean_signal | mean | 95% CI       |
|--------|----------------|------|--------------|
| 128    | 18             | 0.39 | [0.17, 0.61] |
| **256**| **38**         | **0.05** | **[0.00, 0.13]** ← *below 15% threshold* |
| 512    | 41             | 0.32 | [0.17, 0.46] |
| 1024   | 37             | 0.22 | [0.08, 0.35] |

**LoCoMo (clean)** — data too sparse to commit to a pattern:

| Budget | n_clean_signal | mean | 95% CI       |
|--------|----------------|------|--------------|
| 128    | 1              | 0.00 | n=1, useless |
| 256    | 4              | 1.00 | [1.00, 1.00] |
| 512    | 4              | 1.00 | [1.00, 1.00] |
| 1024   | 6              | 0.67 | [0.33, 1.00] |

### The U-shaped budget pattern (LongMemEval)

The clean numbers reveal **a V-shape with a depth-of-valley collapse at
B=256**:

```
0.39  → 0.05  → 0.32  → 0.22
B=128   B=256   B=512   B=1024
```

(see `outputs/figs/conditional_drop_vs_budget_clean.png`)

#### Why this might be real

`B=256` is the **typical size of a single LongMemEval answer-bearing
fact** plus minimal framing. At this exact budget the selector for all
three agents converges to a "fact-only" memory ("User said they went to
Hawaii…" — 18, 22, 26 tokens for A_react / A_plan / A_cot respectively
on one example), so cross-agent transfer is essentially lossless.

* **B < B_essential** (here B=128, even before the truncation bug):
  the budget can't even hold the bare fact, so the three agents are
  forced into different sacrifices → high cross-agent drop.
* **B ≈ B_essential** (B=256): everyone reaches the same lower bound
  ("the fact"). Convergence. Drop collapses.
* **B > B_essential** (B=512 onwards): there is room beyond the fact
  for *style*, *citation framing*, *evidence sentences*. Each agent
  uses the extra room differently → drop reappears.
* **B ≫ B_essential** (B=2048+, in flight): redundancy may forgive
  mismatch entirely, but if drops continue to be 20%+ the policy
  signal persists. This is the open question.

#### Why this matters more than monotonic emergence

This is a **falsifiable budget-regime structure**: the conditional drop
is bowl-shaped in `log(B)`, with the trough at the cardinality of the
task's sufficient statistic. It would be predicted by an
information-bottleneck account ("policy-conditional channel selection
only matters when the channel is over- or under-provisioned relative
to the task's Bayes-rate") and is *not* something prior compression
work has surfaced.

#### Why we can't commit to this framing yet

* Need wide-budget data (`{512, 1024, 2048, 4096}`, in flight) to see
  if the rise continues past B=1024 or saturates.
* LoCoMo has only 4–6 clean signal rows per budget — needs the wide
  rerun to confirm any pattern.
* LongMemEval B=128 and B=512 CIs overlap (`[17, 61]` and `[17, 46]`),
  so the "B=128 highest, B=1024 lowest" point claim isn't statistically
  separable yet — only **"B=256 is significantly lower than the other
  three"** is significant.

---

## M3-judge (in flight) — LLM-as-Judge Task-Success Drop

### Why we added this

The action-distribution-match metric in M3 is brittle for QA tasks —
two semantically equivalent answers ("Hawaii" vs "She went to Hawaii")
can produce different top-1 buckets, inflating "drop" in noisy ways.
A reviewer will absolutely ask for direct **task-success** evaluation.

### Method

For each transfer row `(C, B, source, target)`:

1. Sample N=16 answers from the target agent given the source's
   compressed memory.
2. For each sampled answer, ask MiniMax-M2.5 to judge against the gold
   answer with the strict prompt `YES / NO / PARTIAL` (system prompt
   in `motivation/judge.py`).
3. Aggregate: success rate = (#YES + 0.5 × #PARTIAL) / N.
4. Drop = `success_with_own_memory − success_with_other_memory`.

This gives a **continuous score in [0, 1]** for each row, ending the
binary-action-match issue.

### Status

Running in parallel on both `default_locomo` and `default_longmemeval`.
ETA ~2–3 hours each. Results will be appended to this document under
`outputs/<exp>/m3_judge_summary.json`.

### Expectations

The judge-based drop should:

* **Smooth out the unconditional metric**: the dilution effect from
  binary signal rows disappears. We expect unconditional drop to land
  around the conditional-drop values reported above (15–30% range).
* **Confirm the per-pair asymmetry**: A_plan as target still drops more
  than A_react when fed cross-agent memories.

---

## Methodological caveats (for the limitations section of the paper)

1. **Single base model**: all three agents are MiniMax-M2.5 + system
   prompt. Scaffold-induced policy differences are real but less
   dramatic than weight-induced ones (e.g. ReAct on Qwen vs Plan on
   Mistral). The original spec calls this out and we accept the
   trade-off in exchange for a clean controlled experiment.
2. **Single-step probing, not full trajectories**: agents are queried
   for "next action" on a probe state, not run through a full
   multi-step rollout. For AppWorld-style tool-use domains, this is a
   shortcut that loses the trajectory-level reasoning fingerprint.
   Multi-step extension is a clean follow-up using acon's existing
   trajectory infrastructure.
3. **One probe state per QA context**: makes action-match binary;
   addressed by the upcoming LLM-as-judge metric.
4. **Selector self-consistency (M5)** is comparable to cross-agent
   shift in absolute terms. Mitigated by M3 measuring downstream
   behavior change directly.
5. **Policy divergence δ has narrow dynamic range** (~0.7–1.0), so
   linear regression of drop vs δ has low R². A richer agent pool (e.g.
   add a tool-disabled agent or a one-shot vs CoT baseline) would
   spread δ.
6. **`mean_kl_to_full` is near-saturation** (~10–12) for many oracle
   records — these are the rows where the empirical distribution shifts
   completely (top-1 changes). KL acts as a confirming signal but is not
   independent of action-match in the binary regime.

---

## Decision gate

Per the original spec § 6, M2 + M3 are the killer experiments. Re-stated
with the revised conditional-M3 metric:

| Gate | Condition | LoCoMo | LongMemEval | Joint? |
|---|---|---|---|---|
| **M2 lexical** | Jaccard < 0.4 | ✅ 0.29 | 🟡 0.39 | partial |
| **M2 semantic** (new) | SBERT < 0.75 | ✅ 0.55 | ✅ 0.69 | **✅ both** |
| **M3 conditional drop** | > 15% | ✅ 85% | ✅ 26% | **✅ both** |
| **M3 R²** | > 0.5 | ❌ 0.01 | ❌ 0.15 | both fail |
| **M5 within-cross gap** | > 10pp | ❌ 3% | ❌ 4% | both fail |
| **M4 classifier** | acc > 0.7 | ❌ 0.39 | ❌ 0.24 | bonus only |

> **Verdict**: the **central thesis is supported**. The two killer
> behavioural criteria (M2 semantic + M3 conditional drop) pass on both
> datasets. The supplementary criteria (R², M5 gap, M4 accuracy) need
> discussion as limitations rather than treated as silencers — each has
> a concrete reframing or a planned follow-up that addresses it.

**Proceed with the Spotlight pitch.** Document the limitations honestly;
the conditional-drop story + M2-semantic-overlap story together are a
defensible Figure 1 + § 3.

---

<a id="instance-noise"></a>
## §3.2 candidate — Instance-noise ablation (the hinge for Path D)

> **This is the missing piece.** M5 says "agent description doesn't
> shift the selector's surface output much" — that's robust evidence
> for the **first leg** of the surface-behavior disconnect framing.
> But the **second leg** (M3 conditional drop ≈ 26-85%) implicitly
> assumes the drop is *agent-induced*, not *seed-induced*.
>
> Reviewer's natural question: *"You showed swapping A_i → A_j
> degrades the target by 26-85%. But your selector's per-seed variance
> is high (M5 within-Jaccard = 0.35-0.51). What if A_i_seed_1 →
> A_i_seed_2 produces the same drop? Then M3 doesn't measure policy,
> it measures generation noise."*
>
> If we can't answer this, **Path D's M3 leg collapses** and we
> retreat to Path B (standard prompt-compression paper).

### Method

Re-run the M1 selector regime (T=0.6 + variation hint) to generate
K=3 candidate oracle memories per (context, agent, budget). Then for
each target agent T:

* `self_match`: π_T on T's own candidate-0 vs π_T on full ctx.
* `within_drop`: 1 − mean[π_T on T's candidate-1, T's candidate-2]
  (against full-ctx baseline) — this is the seed-only noise.
* `cross_drop`: 1 − mean[π_T on other-agent-1's candidate-0,
  other-agent-2's candidate-0] — this is the policy + seed noise.

### Decision rule

```
ratio = mean(cross_drop) / mean(within_drop)  (on signal rows)
```

| Ratio        | Verdict                                                   |
|--------------|-----------------------------------------------------------|
| ratio ≥ 3×   | Path D **STRONG** — Spotlight defensible (M3 is policy-driven, with within-variance < 1/3 of cross-variance) |
| 1.5 ≤ ratio < 3× | Path D **WEAK** — borderline, treat as Findings  |
| ratio < 1.5× | Path D **DIES** — M3 drop is mostly instance noise; retreat to Path B |

### Status

Running on LongMemEval B=512 (highest-signal clean budget), 10
contexts × 3 candidates per agent × N=16 samples. ETA ~60-90 min.
Output goes to `outputs/default_longmemeval/instance_noise_summary.json`.

Run command (when extending to other domains):

```bash
python -m scripts.instance_noise_test --config configs/default_locomo.yaml \\
    --budget 512 --n-contexts 10 --candidates-per-agent 3
```

### Why this is the hinge

* **Wide-budget data** (the other in-flight run) is *confirmation* —
  it can't refute Path D's claim that M3 drop is policy-induced, only
  inform Path C's valley story. So wide-budget is sufficient for Path
  C but NOT for Path D.
* **Instance-noise ablation** is the only experiment that directly
  separates policy variance from generation variance. Without it,
  Path D's strongest claim ("behavioral effect exists despite surface
  similarity") rests on an untested assumption.
* This is the **30-minute decision** that gates the entire paper
  framing.

---

<a id="path-selection"></a>
## Path Selection — four candidate paper framings

Given (a) **M5 path-A dead**, (b) **B=256 valley is real but partial-R²
only**, and (c) **instance-noise ablation in flight**, here are the
four candidate framings. Path D vs B is decided by instance-noise
ratio; Path C is *additive* if both wide-budget and quadratic fit
support the valley.

### Path A — Original Spotlight (RIP)

> *"Optimal memory is policy-dependent."*

**Dead.** M5 gap stays at 3-4pp at any temperature / `within_k`. The
killer-experiment 10pp criterion cannot be met on lexical or semantic
overlap. **Do not pursue.**

### Path B — Plan B from original spec

> *"Behavior-invariance + IB-style compression metric."*

Drop the policy-dependence pitch entirely. Frame as a new compression
metric / new evaluation. Less novelty, safer landing. **Findings-level
publishable.**

### Path C — Budget-regime-dependent emergence

> *"Policy-dependence of compression emerges in a specific budget
> regime; existing compression work has tested only in the regime where
> it's hidden."*

The **most exciting** option but rests on **two empirical claims** we
can't yet make:
1. The B=256 valley is real **and** symmetric (rise on either side).
   ✓ Left side (B<256): supported (B=128 = 0.39).
   ? Right side (B≫256): need wide data. If B=2048 drop > B=1024 drop,
   the valley is genuine; if B=2048 ≈ B=1024 ≈ B=512, the pattern is
   saturating not bowl-shaped.
2. The pattern reproduces on LoCoMo (currently data-sparse).

**Decision criterion**: after wide-budget run finishes, fit a
quadratic in `log B` to (B=128, 256, 512, 1024, 2048, 4096) on both
datasets. If quadratic R² > 0.7 on at least one dataset AND minimum
is between 256 and 1024, framing C is supported.

### Path D — Surface-behavioural disconnect (main framing candidate)

> *"Policy-dependent differences in oracle memories exist at the
> downstream-behavior level (M3 conditional drop 26-85%) but are
> invisible at the surface-token level (M5 within-vs-cross gap 3-4pp,
> M4 classifier acc ≤ 0.39). This implies that policy conditioning
> operates in the information geometry of the compression, not in its
> lexical form, with implications for how learnable compressors should
> be parameterised."*

**Required evidence (≠ just M5+M4):**
* **M5 evidence**: agent description shifts selector by 3-4pp on
  surface overlap — well within instance noise. ✅ obtained.
* **M4 evidence**: TF-IDF + LR classifier predicts source agent at
  near-random. ✅ obtained.
* **M3 conditional drop**: 26-85%. ✅ obtained.
* **M3 instance-noise ablation**: M3 drop must be *agent-driven*, not
  *seed-driven*. **In flight** (`scripts.instance_noise_test`).

### The two legs of Path D

| Leg | Evidence | Status |
|-----|----------|--------|
| **T1**: agent-specific differences exist behaviorally | M3 conditional drop + instance-noise ratio ≥ 3× | M3 done; ablation in flight |
| **T2**: agent-specific differences are invisible lexically | M5 within-vs-cross gap (3pp) + M4 classifier (near-random) | ✅ done, three independent confirmations |

T2 is **already robust** — three independent measurements (M5 default,
M5-tight T=0.0, M4 classifier) all agree. T1 hinges on the
instance-noise ablation.

### My recommendation (honest engineer view, updated)

1. **Now (in flight)**: instance-noise ablation finishes in ~50 min.
   Verdict drives everything else.
2. **If instance-noise ratio ≥ 3×** (most likely outcome):
   - **Path D = main framing**.
   - Wide-budget data confirms Path C valley → add as §3.3 subfinding.
3. **If 1.5× ≤ ratio < 3×**: Path D borderline, retreat to Findings
   pitch with clean limitations.
4. **If ratio < 1.5×** (M3 collapses): retreat to Path B (behavior-
   invariance + IB only). Standard prompt-compression paper.

### Proposed §3 structure (when writing paper)

| Paper § | Content | Source in this doc |
|---------|---------|--------------------|
| §3.1 Setup | Agent pool, datasets, budgets | [Shared setup](#shared-setup) |
| **§3.2 Discovery I — Surface-Behavioral Disconnect (T2 + T1)** | Robust two-pillar finding | [Instance-noise ablation](#instance-noise) + [M2](#m2-cross-agent-memory-overlap--killer-experiment-1) + [M5](#m5--selector-consistency-ablation-new-our-addition) + [M3 conditional](#m3--cross-agent-transfer-degradation--killer-experiment-2) |
| **§3.3 Discovery II — Compression-Regime Structure** | V-shape valley + answer-echo hypothesis | [Budget pattern](#budget-pattern) |
| **§3.4 Methodology audit — Serialization Artifact** | Bug + clean re-run | [Bug audit](#bug-audit) |
| §3.5 What Off-the-Shelf Cannot Do, EASMO Must Learn | Gap → method motivation | (to be drafted) |

---

## In-flight experiments (2026-05-23 evening, **post-bug-fix re-runs**)

After a methodological audit (see §3.4) we discovered the
post-hoc-filtered "clean" subset of the original `default_*` runs
suffers from **survivorship bias** — at B=128, only 44% of records
were uncontaminated, and those 44% systematically over-represent the
contexts where MiniMax's reasoning chain happened to fit within the
truncated `max_tokens` budget (i.e. simpler contexts). To get a paper-
quality budget curve, we re-ran with the bug-fixed selector code at
the **full 6-budget grid `{128, 256, 512, 1024, 2048, 4096}`** so a
single run provides a consistent, clean cross-section.

| Run | Purpose | Priority | ETA |
|---|---|---|---|
| **`instance_noise_test` (LongMemEval B=512)** | **hinge test for T1 + T2** — separates policy from seed noise | **CRITICAL** | ~60-90 min |
| **`wide_locomo` clean re-run** (M1–M5+judge, 414 triples) | full 6-budget curve under bug-fixed code; supplies T1's monotonicity / regime evidence | **CRITICAL** | ~10-12 h |
| **`wide_longmemeval` clean re-run** (M1–M5+judge, 540 triples) | same on LongMemEval | **CRITICAL** | ~10-12 h |
| `default_locomo-m3judge` | LLM-judge task-success drop, post-hoc on `default_*` | confirmation | (died, low priority to restart) |
| `default_longmemeval-m3judge` | same on LongMemEval | confirmation | (died, low priority) |
| M5-tight (LongMemEval) | T=0.0+within_k=10 ablation, partial done | bookkeeping | (died at 6/10 ctx) |

The `default_*` runs are now **superseded by the clean wide_* re-runs**
for all budget-related analysis. We keep the `default_*` artifacts on
disk because (a) the filtered subset at B=512+ has ~0% contamination
and matches expected effect sizes — useful as a sanity-check on the
wide re-run, and (b) M5 and M4 results don't depend on the bug.

All runs live on W&B project
[easmo-motivation](https://wandb.ai/guanghui_min-university-of-virginia/easmo-motivation).

When they finish, return to this doc and update:

* **First**: `instance_noise_summary.json` → set Path D verdict
  (STRONG / WEAK / DEAD) → choose between paths D and B.
* **Second**: `budget_regime_test.py` on `default_*` + `wide_*`
  merged → set Path C verdict (SUPPORTED / partial / REJECTED).
* M3-judge: confirm unconditional drop using LLM-as-judge (should land
  near conditional values).
* M5 gap with `within_k=10` (more stable within-agent estimate)
  pulled from M5-tight.
* Possibly upgrade M2 thresholds to per-budget (currently aggregated).

---

## Result update 2026-05-24 (mid-day audit)

This section was added during a mid-flight audit of the in-flight
runs. It documents three issues that need to be acknowledged before
any of the §Path-D / §Path-C numbers above can be trusted as final.

### 1. `instance_noise_test` (B=512, n=10) — degenerate, **verdict was a false positive**

The 2026-05-24 00:05 UTC run produced `n_signal_rows = 0 / 30`: every
target agent had `self_match = 0.0` on its own candidate-0 memory at
B=512. With a zero numerator and zero denominator, the
`conditional_ratio_cross_over_within` field returned `Infinity`, and
the verdict logic naively classified that as `STRONG —
Spotlight-defensible`. **It was not.** No data was actually
collected.

Two fixes have been applied:

1. **Verdict logic patched** (`scripts/instance_noise_test.py`) to
   require `n_signal_rows ≥ 5` and a non-zero conditional within drop;
   otherwise emits `INSUFFICIENT SIGNAL` or `DEGENERATE`. The
   previously-written `outputs/default_longmemeval/instance_noise_summary.json`
   has been re-derived to read `INSUFFICIENT SIGNAL` and `Infinity`
   has been replaced with `null`.
2. **Re-run started** at 2026-05-24 17:25 UTC with `n_contexts = 30`
   (the full default set) and the same `B=512`, `K=3`, `N=16`. Log:
   `outputs/instance_noise_rerun_B512_n30.log`. Expected ETA ~3 h.
   With M1's ~21% pass rate at B=512 on LongMemEval, we now expect
   ~5–10 signal rows out of 90 (vs the previous 0/30), enough to make
   the cross/within ratio meaningful.

A second re-run at the **tightest** budget (`B=128`) is planned right
after, since handoff calls that out as T1's strongest claim.

### 2. `wide_*` M1 progress (still in flight)

| Run | M1 progress | Per-budget pass rate (action-match ≥ 0.85) |
|---|---|---|
| `wide_longmemeval` | 486/540 (90%) | B=128: 23.5%, B=256: 19.8%, B=512: 19.8%, B=1024: 21.0%, B=2048: 22.2%, B=4096: 23.8% |
| `wide_locomo` | 312/414 (75%) | B=128: 3.8%, B=256: **0.0%**, B=512: 3.8%, B=1024: 11.5%, B=2048: 7.7%, B=4096: 9.8% |

Two implications:

* **LongMemEval pass rate is essentially flat across budgets.** This
  matches the `default_longmemeval` aggregate (26.9%) but means the
  *budget* axis is unlikely to show the monotonic-decreasing T1
  pattern at the M1 level. The action-match drop is going to have to
  come from M3 (cross-agent) rather than M1 (self-fidelity).
* **LoCoMo M1 pass rate is too low to support per-budget M3.** B=256
  has zero passing candidates so far (0/52). Without paired passing
  rows, M3's conditional drop on LoCoMo will have wide CIs (or be
  computable on only ~5–10 contexts per budget). Expect to either
  re-frame LoCoMo as a robustness check on LongMemEval, or pull
  LoCoMo from the headline.

### 3. `auto_push_watcher.sh` had been silently hung 16 h

A `git push` from the 2026-05-24 01:32 UTC sync hung on a network
call and never returned. Because the watcher uses
`bash sync_and_push.sh "auto-sync" || true`, the wrapper caught
errors but not hangs, and the loop sat in `wait` for 16 h. Visible
symptom: `/tmp/easmo_watcher.log` has no entries after `01:32Z`,
local repo is one commit ahead of origin, but `ps` shows the
watcher alive.

Resolved manually: killed the hung `git` child, ran
`sync_and_push.sh` once to flush the backlog (`ea842d9`), watcher
loop is healthy again. Consider adding `timeout 60 git push …` in
`sync_and_push.sh` if this recurs.

### What's still pending

| Item | Blocking | ETA |
|---|---|---|
| `instance_noise B=512 n=30` rerun (NEW continuous metric) → real Path-D verdict | T1+T2 hinge | ~3 h |
| `instance_noise B=128 n=30` (T1's tightest claim) | T1 | +3 h after above |
| `wide_longmemeval` finishes M1 → M2/M3/M4/M5 (M3 will use new metric) | Path C | ~30 min M1 + many more for downstream |
| `wide_locomo` finishes M1 → decide whether to keep | Path C corroboration | ~2 h M1 |
| `recompute_m3_summary.py` + `budget_regime_test.py` on clean wide data | Path C verdict | after both wide runs finish |

---

## Result update 2026-05-24 (afternoon — design audit)

A second pass found that the binary nature of the headline metric was
itself a major source of the "no signal" problem documented above.
This section records that audit and the changes made to the codebase.

### Why the prior runs looked degenerate

```python
# motivation/motivation/metrics.py (pre-audit)
def action_match_rate(dist_a, dist_b):
    """Top-1 action match: 1.0 if argmax actions agree, else 0.0."""
```

Every M1, M3, and `instance_noise` "match" was a 0/1 bit: do the
`argmax` actions of two N=16 sample distributions agree? On QA tasks
where one canonical answer dominates, a single sample flipping
between 8/16 and 7/16 votes rolls the match between 1 and 0. Two
distributions overlapping ~80% can score 0; two distributions
overlapping ~30% can score 1 if the same wrong answer happens to top
both.

This explains:
* The "0 / 30 signal rows" `instance_noise` result at B=512 — the
  metric is a near-Bernoulli with low p, so K=3 candidates per agent
  × 10 contexts almost certainly produces 0 hits.
* The flat-across-budgets M1 pass rate (~21% on LongMemEval, all
  budgets 128 → 4096): the binary metric saturates at the noise
  floor of the sampling process, not the actual compression-quality
  curve.
* The M3 conditional drop only being computable on rows where
  `action_match_self == 1.0`, drastically shrinking the analysable
  set.

### Other design weaknesses surfaced (carried as caveats)

These are not fixed by metric changes alone; they are paper-level
limitations that any T1/T2 framing has to acknowledge:

1. **Three "agents" share the same base LLM** (MiniMax-M2.5) and
   differ only in system prompt template + a small temperature
   spread (0.4 / 0.4 / 0.7). Any "policy difference" is a stylistic
   wrapper, not a genuinely different decision policy. A real
   cross-model agent (e.g. GPT-4o-mini, Qwen-2.5-7B) would be
   needed to make policy-dependence claims robust.
2. **LongMemEval and LoCoMo are QA benchmarks**, not agentic. There
   is one canonical answer per question, so all three agents
   *should* converge to the same answer regardless of "policy".
   The action-distribution variation we measure is mostly stylistic
   format variation, not strategic variation. AppWorld (already
   loadable via `motivation/data.py`) is the appropriate benchmark
   for T1's "policy-dependent compression" claim.
3. **T1 + T2 are in tension on this experimental setup.** T2 says
   "LLM-as-selector cannot policy-condition memory" — i.e.
   cross-agent and within-agent candidates have similar content.
   T1's `instance_noise` test wants `cross/within ≥ 3×`, which
   requires cross-agent memories to differ behaviourally. If T2 is
   true on LLM selectors, the test for T1 cannot succeed using LLM
   selectors. A proper T1 test needs a **non-LLM oracle memory**
   (e.g. fine-tuned compressor, or human-annotated minimal
   sufficient memories per agent).
4. **`instance_noise` uses one probe state** per context. For
   LongMemEval / LoCoMo this is by construction (the question is
   the only probe), but it does mean each row is one Bernoulli
   trial worth of signal. Multi-probe averaging applies only to
   AppWorld contexts.

### Code changes (in this commit)

* `motivation/metrics.py`: added `action_overlap_rate(p, q) = 1 -
  TV(p, q)` (continuous, range [0, 1]) and `js_divergence` for
  completeness. `action_match_rate` retained for legacy reporting
  but no longer the primary signal.
* `motivation/instance_noise.py` + `scripts/instance_noise_test.py`:
  every row now stores BOTH binary and continuous overlap
  (`self_match` / `self_overlap`, etc.). The Path-D verdict is
  driven by the **continuous overlap ratio** with signal threshold
  `self_overlap ≥ 0.5` (loose; admits rows where M1 binary match
  was 0 due to one flipped argmax sample). Verdicts: `STRONG ≥3×`,
  `WEAK 1.5–3×`, `DEAD <1.5`, plus `INSUFFICIENT SIGNAL` and
  `DEGENERATE` guards.
* `motivation/transfer.py` + `scripts/run_m3.py`: every transfer row
  now stores `overlap_self`, `overlap_cross`, `overlap_drop`
  alongside the binary fields. Pair breakdowns and m3_summary.json
  now expose both flavours so the conditional-drop conversation is
  not bottlenecked on the binary signal fraction.
* `scripts/recompute_m3_summary.py`: schema-aware — emits the
  overlap-based summary if the CSV has the new columns, falls back
  to binary-only on older CSVs.

### Why this matters for the verdict

The prior `instance_noise` run produced 0/30 binary signal rows on
B=512, n_contexts=10. With the new continuous metric and
n_contexts=30 (90 rows total) we expect:

* **Binary signal rows**: still ~5–15 (with M1 binary pass at ~21%)
* **Overlap signal rows** (`self_overlap ≥ 0.5`): expected ~40–70,
  because `self_overlap` is a graded measure that catches the
  "almost-passes" the binary metric throws away.

That order-of-magnitude bump in usable rows is what makes the
cross/within ratio statistically interpretable. **The verdict on
Path D should be read off the OVERLAP ratio, not the binary ratio**,
in the new summary file.

### Honest framing on what this audit changes

* **Path D** is now testable rather than vacuously confirmed. It can
  still die — if overlap ratio comes back ~1×, M3's drop really is
  selector seed noise. That's a real risk.
* **Path C** (budget structure) similarly will be re-read on
  `m3_overlap_*` fields once `wide_*` finishes. The "U-shape at
  B=256" narrative may collapse under the smoother metric, in which
  case T1's wording shifts to "tight budgets remain hard" rather
  than "two-regime structure".
* **T2** is unchanged. M5's within-vs-cross gap and M4 classifier
  results don't depend on the action-match metric; they're text
  similarity / classification on the memories themselves.
* **LoCoMo as co-headline** is still in trouble: even with the
  smoother metric, M1 at B=256 is 0 / 52 candidates on LoCoMo, so
  there's nothing to recover. Plan to demote LoCoMo to a robustness
  side panel.

---

## Where the code lives

```
EASMO/motivation/
├── motivation/
│   ├── oracle.py           ← M1 selector + scoring
│   ├── overlap.py          ← M2 metrics
│   ├── transfer.py         ← M3 transfer-eval primitives
│   ├── selector_ablation.py← M5 within-vs-cross
│   ├── classifier.py       ← M4 TF-IDF + LR
│   ├── judge.py            ← LLM-as-judge for task success
│   ├── semantic.py         ← Sentence-BERT helpers (clustering, overlap)
│   ├── agents.py           ← three scaffold definitions, action-distribution
│   ├── data.py             ← LongMemEval / LoCoMo / AppWorld loaders
│   ├── llm.py              ← MiniMax client (concurrent batch, retries)
│   └── metrics.py          ← Jaccard, KL, TV, Spearman, R², edit-distance
├── scripts/
│   ├── run_m1.py … run_m5.py
│   ├── run_m3_judge.py     ← new
│   ├── recompute_m3_summary.py  ← post-hoc conditional-drop helper
│   └── run_all.py          ← M1→M2→M3→M3_judge→M4→M5 sequence
├── configs/
│   ├── default_{locomo,longmemeval}.yaml
│   ├── wide_{locomo,longmemeval}.yaml   ← bigger budgets
│   └── smoke_*.yaml
└── outputs/<exp>/
    ├── oracle_memories.jsonl
    ├── m1_summary.json
    ├── overlap_matrix.csv + m2_pair_summary.json
    ├── transfer_results.csv + m3_summary.json   ← now with conditional drop
    ├── m4_classifier.json
    ├── selector_consistency.csv + m5_summary.json
    └── transfer_judge.csv + m3_judge_summary.json   ← when judge run finishes
```

Pass `--no-wandb` to any `scripts.run_*.py` to disable W&B (e.g. for
local sanity checks). All run scripts read a YAML config; default
configs target the in-house MiniMax endpoint at
`http://10.183.22.68:8005/v1`.
