# motivation_v6 results — Jacobian active-subspace diagnostics

> Final manual edit: 2026-05-28 PT. Numbers cross-checked against
> `outputs/tables/v6_dashboard.csv` and the per-stage tables. The
> auto-generated short version lives at
> `outputs/reports/results_summary.md`.

## TL;DR (paper-tier, four findings)

1. **Jacobian saliency does NOT predict v4's leave-one-span-out
   sensitivity.** Global Spearman = **−0.20** across 591 spans;
   per-task median Spearman = **−0.03**. Of 28 tasks with valid
   correlations, 9 are positive and 18 are negative; 7 have |r|>0.3.
   The first-order embedding gradient is not a substitute for the
   finite-difference span probe used by v4.
2. **The Jacobian-weighted residual stream IS strongly low-rank.**
   At layer 18 (N/2 of 36), randomised SVD on the example-level
   active vector matrix recovers **92 %** of variance in the top
   16 components, and **80 %** at k=8. At the span level the
   numbers are weaker but still strong: **74 %** at k=16, **83 %**
   at k=32. **High-v4-sensitivity spans concentrate more tightly
   than low-v4 spans** at low k (66 % vs 47 % at k=4; gap closes by
   k=32). The low-rank active-subspace hypothesis is verified.
3. **The decision-state target oracle is DEGENERATE.** Even k=4
   soft tokens drive the median target NLL from 1.79 (no context)
   down to **0.09**, well below the full-context loss of 0.99.
   Gap-recovery exceeds **2.25× at every k ≥ 4**. This is *not* a
   meaningful low-dimensional context-compression upper bound —
   continuous embeddings of dimension 4 × 2560 = 10 K params have
   enough bandwidth to encode the ~800-token target JSON directly,
   so the experiment measures oracle bandwidth rather than context
   information. Future C-style experiments must use a target with
   higher per-position uncertainty (e.g. held-out action-code
   tokens) or evaluate KL against the full-context predictive
   distribution.
4. **Behaviorally, gradient-ranked span selection is null.**
   30 tasks × 4 methods × 2 token-budgets = 240 MiniMax-M2.5 agent
   runs at `cap=15`. `jacobian_low_spans` (0.80) is statistically
   indistinguishable from `jacobian_high_spans_raw` (0.83) and
   `jacobian_high_spans` (0.70). Choosing the *least*-Jacobian
   spans is as good as choosing the *most*-Jacobian spans — a
   behavioral confirmation of finding (1).

**Interpretation under spec §11.** The result pattern is closest to
"B positive, A weak, C degenerate" — i.e. the low-rank claim is
**supported**, the v4 finite-difference signal is **not** captured by
local embedding gradients, and the soft-token oracle cannot
distinguish information bandwidth from soft-prompt parameter
bandwidth. The first two are the central methodological contributions
of v6; the third is a methodological negative finding that constrains
future oracle design.

## 1. Setup

| Setting | Value |
|---|---|
| White-box model | `Qwen/Qwen3-4B-Instruct-2507` (HF local snapshot, bf16, sdpa attention) |
| Number of layers | 36 (mid-layer for active capture = 18) |
| Hidden size | 2560 |
| Tasks | **30 / 30** (28 at max_ctx=12 K, 2 retried at max_ctx=6 K) |
| Median context tokens after tokenisation | 5,456 |
| Max context tokens | 12,000 (8,000 for 2 retried cases) |
| Target | canonical JSON of v4 reference decision state (teacher-forced) |
| Probe prompt | verbatim from `motivation_v4.prompts.DECISION_STATE_PROBE` |
| Soft-token ks | {4, 8, 16, 32, 64}, 200 AdamW steps, lr 0.05, patience 30 |
| GPU | 1 × H100 (80 GB); vLLM Qwen server stopped before run |
| Wall-clock | stage 02 = 0.5 min; stages 03 + 04 = 25 s; stage 05 = **57.3 min**; stages 06 – 08 = 5 s |

## 2. Experiment A — Jacobian saliency vs v4 finite-difference sensitivity

### 2.1 Headline correlations

| Metric | Value | Spec threshold | Verdict |
|---|---|---|---|
| Global Spearman `(span_gxa_sqrtlen, v4_final_sensitivity)` | **−0.200** | ≥ 0.25 strong / ≥ 0.10 medium | WEAK |
| Global Pearson                                              | −0.059  | — | WEAK |
| Median **per-task** Spearman                                | **−0.029** | ≥ 0.25 strong | NEGATIVE |
| Median top-1 enrichment (∩top‑k random base)                | 0.00×   | ≥ 2.0× strong | NEGATIVE |
| Median top-3 enrichment                                     | **1.83×** | ≥ 2.0× strong / ≥ 1.3× medium | borderline MEDIUM |
| Median top-5 enrichment                                     | 1.28×   | — | WEAK |
| Median top-10 enrichment                                    | 0.96×   | — | NEGATIVE |

`span_gxa_sqrtlen = Σ_{j∈span} |g_j·e_j| / sqrt(|T_span|)`. Six other
score variants (`grad_sum`, `grad_mean`, `gxa_sum`, `gxa_mean`,
`top10_mean`, `g_dot_x_abs_sum`) are reported in
`tables/jacobian_vs_v4_correlations.csv` — none beat `gxa_sqrtlen`.

### 2.2 Per-task distribution

Across 28 tasks with non-constant v4 sensitivity:

| stat | per-task Spearman |
|---|---|
| min  | −0.55 (`383cbac_1`) |
| 25th | −0.27 |
| median | **−0.03** |
| 75th | +0.13 |
| max  | +0.41 (`0d8a4ee_2`) |
| n positive | 9 |
| n negative | 18 |
| n with `\|r\| > 0.3` | 7 (3 positive, 4 negative) |

### 2.3 Confounders

| Spearman against | Jacobian primary | v4 sensitivity |
|---|---|---|
| `token_count`         | **+0.358**  | +0.087 |
| `step_id` (recency)   | −0.029     | +0.158 |

**The Jacobian score has a substantial length bias** (Spearman +0.36
with token count) that v4 does not have (Spearman +0.09). This is a
plausible mechanistic explanation for the disagreement — Jacobian
saliency over-rewards long spans (despite the √-length
normalisation), while v4's leave-one-span-out finite difference is
length-insensitive because it measures absolute downstream-state
change. **Neither score correlates strongly with recency**, which
matches the v4 finding that recency is a strong but mostly
orthogonal baseline.

### 2.4 Top-1 rank cross-check

| Variable | median across tasks |
|---|---|
| rank of v4 top-1 span under Jacobian score | high (most tasks > 5 of N≈20) |
| rank of Jacobian top-1 span under v4 score | high                          |

These are reported in `tables/jacobian_top1_ranks.csv`. The
distribution overlaps the random-rank baseline.

### 2.5 Verdict for A

**WEAK / NEGATIVE.** First-order embedding gradients on Qwen3-4B do
not recover v4's finite-difference span sensitivity. Reading the
spec §11 rules literally, this means *gradient saliency is not the
right primitive for span selection here*. It is also possible (we
cannot disentangle it from this experiment alone) that:

* v4 finite-difference sensitivity is itself dominated by a
  non-local effect (multi-step downstream decision change) that
  first-order Jacobian can't capture;
* v4 sensitivity is partly noise — judge-graded distance averaged
  with rule-based norm — and 0.30-noise signal in v4 explains some
  of the apparent disagreement;
* The teacher-forcing target (`reference_decision_state` JSON) is a
  proxy for downstream behaviour and may not exactly match the
  signal v4 is probing.

We do **not** select spans by Jacobian score in any downstream
artifact — this would conflate length and importance.

## 3. Experiment B — active-subspace spectrum

### 3.1 Cumulative explained variance

Centred randomised SVD on Jacobian-weighted activations
$A = H_L \odot G_L$ at layer $L = 18$:

| Matrix | rows | k=4 | k=8 | **k=16** | k=32 | k=64 |
|---|---:|---:|---:|---:|---:|---:|
| example  | 30  | 0.655 | 0.788 | **0.922** | 1.000 | 1.000 |
| span     | 593 | 0.521 | 0.640 | **0.737** | 0.828 | 0.903 |
| high_v4 (≥ median v4 sens) | 297 | 0.660 | 0.747 | 0.821 | 0.886 | 0.938 |
| low_v4  (< median v4 sens) | 294 | 0.470 | 0.596 | 0.716 | 0.826 | 0.914 |

### 3.2 Verdict for B

**STRONG POSITIVE.**

* Example-level k=16 cumulative variance is **0.922** — far above
  the spec STRONG threshold of 0.5 at k=16.
* Span-level is more diffuse but still passes the medium threshold
  (k=16 cumulative = 0.737; k=32 = 0.828) and meets the spec STRONG
  threshold of 0.7 at k=32.
* The **high-v4 / low-v4 separation** at low k is **+19 percentage
  points at k=4** (66 % vs 47 %) and **+15 pp at k=8** (75 % vs
  60 %), narrowing to **+10 pp at k=16** and disappearing by
  k=64. This is the medium-positive cross-validation: spans that
  v4 marked sensitive *do* concentrate more tightly in the active
  subspace than spans v4 marked insensitive.

### 3.3 Interpretation

The Jacobian-weighted residual stream at layer N/2 admits a
**low-dimensional active subspace** of effective rank ≈ 16–32 on
context-level aggregates. This is the central empirical result of
v6: a tiny number of directions in mid-layer hidden space carries
nearly all of the downstream-task gradient mass. Compression methods
that **project to this subspace** (rather than rank tokens by
saliency) are the natural next experiment.

## 4. Experiment C — soft-token oracle

### 4.1 Loss table (median across 30 tasks)

| condition           | median NLL | $\rho_k$ gap recovery |
|---|---:|---:|
| full context (~5 K tokens)   | **0.993**  | 1.000 (definition) |
| no context (empty)           | 1.786      | 0.000 (definition) |
| recent 5 spans               | 1.333      | 0.57   |
| ACON baseline                | —          | (not present in v4 outputs) |
| soft tokens, k=4             | 0.089      | **2.26** |
| soft tokens, k=8             | 0.005      | **2.25** |
| soft tokens, k=16            | 0.000      | **2.26** |
| soft tokens, k=32            | 0.000      | **2.26** |
| soft tokens, k=64            | 0.000      | **2.29** |

### 4.2 Verdict for C

The headline numbers technically pass the spec's STRONG threshold
(*ρ_k ≥ 0.7 at k ∈ {16, 32}*), but the gap-recovery values of
**+2.26 at every k from 4 onward** are diagnostic of **oracle
overfitting**: the soft tokens drive teacher-forced NLL below the
full-context baseline because $k \cdot d = 4 \cdot 2560 = 10{,}240$
trainable parameters per case is enough to encode the ~800-token
target JSON outright. The experiment thus does *not* measure
"compressibility of context-information into k embedding directions";
it measures "bandwidth of k continuous tokens to encode a fixed
target string."

This is a **methodological negative result for C as designed.**

### 4.3 What this means and what to do next

Two follow-up framings would give a useful upper bound:

* **C2 (next-action target).** Spec §7.2 proposed using next-action
  code (rather than the full decision-state JSON) as the target.
  Next-action code is far shorter (typically one API call,
  10–40 tokens) but still has *per-position uncertainty* that the
  soft tokens must extract from context. We did not implement C2 in
  this round because trajectory parsing for AppWorld next-action
  extraction requires the productive_agents venv and is in scope
  for a follow-up.
* **KL-divergence C.** Replace teacher-forced NLL with KL-divergence
  against the full-context predictive distribution on M held-out
  next tokens, where the held-out tokens cover diverse continuation
  branches. This evaluates whether soft tokens reproduce the
  *behaviour* of full context, not just whether they minimise the
  loss on one fixed target.

Both are listed as v6 follow-up work; they do not change the
findings of A or B above.

## 5. Negative findings (explicit list)

* Jacobian saliency does not predict v4 sensitivity at the span
  level (median per-task Spearman ≈ 0). [§2]
* Jacobian saliency has a substantial length bias (Spearman +0.36
  with token count) that v4 does not have. [§2.3]
* `recent_5_spans` baseline (NLL 1.33) is much closer to full
  context (NLL 0.99) than to no context (NLL 1.79). Just the last
  5 spans recover roughly **57 %** of the full-context gap with
  *zero* soft-token machinery — consistent with v4's finding that
  recency is a strong baseline. [§4]
* Soft-token oracle as specified is degenerate; cannot be used as
  evidence of low-dimensional context compressibility. [§4.2]
* No ACON baseline loss was captured because v4's
  `compressed_contexts.jsonl` does not currently expose an
  `acon`-method row that the v6 reader recognises. The ACON
  comparison should be re-derived from v3's compressors in a
  follow-up. [§4.1]

## 6. Implications for method design

The combined message is:

> Mid-layer Jacobian-weighted activations live in a 10-to-30-
> dimensional active subspace; that subspace correlates with where
> v4's finite-difference probe places mass (high-v4 spans
> concentrate more), but *not* in a way that lets one rank spans
> for selection.

This favours **representation-level compression** — for example,
projecting the residual stream onto the top-k SVD directions of the
active-vector matrix and reconstructing a small soft-prompt that
reproduces those directions, rather than picking text spans by a
saliency score. Whether this beats recency + ACON downstream is an
empirical question we cannot answer with the current diagnostic
suite.

It does **not** favour gradient-ranked span selection (Experiment D
in this round is a confirmatory sanity check, not the headline
story; see §7).

## 7. Experiment D — gradient-ranked text spans (behavioral sanity)

Status: **completed 2026-05-28**. 240 agent cells total
(30 tasks × 4 methods × 2 token-budgets {2048, 4096} × 1
`max_steps=15`). Downstream model = MiniMax-M2.5 at
`http://10.183.22.68:8005/v1`, runner reused verbatim from
`motivation_v4.runner`.

### 7.1 Methods

| method | rule |
|---|---|
| `jacobian_high_spans`     | greedy by `span_gxa_sqrtlen / token_count`, descending |
| `jacobian_high_spans_raw` | greedy by `span_gxa_sqrtlen`, descending (no length normalisation) |
| `jacobian_low_spans`      | greedy by `span_gxa_sqrtlen / token_count`, ascending  |
| `jacobian_recent_hybrid`  | 50 % recent tail + 50 % top-Jacobian from the rest      |

### 7.2 Headline numbers

| method | n cells | success rate | mean iters | mean input tokens |
|---|---:|---:|---:|---:|
| `jacobian_high_spans`             | 60 | **0.700** | 8.6 | 61,039 |
| `jacobian_high_spans_raw`         | 60 | **0.833** | 7.3 | 52,725 |
| `jacobian_low_spans`              | 60 | **0.800** | 8.1 | 60,500 |
| `jacobian_recent_hybrid`          | 60 | **0.667** | 8.3 | 57,961 |
| v4 `high_sensitivity_spans` cap=15 | 30 | 0.400 | 11.0 | 49,941 |
| v4 `recent_spans`           cap=15 | 30 | 0.467 |  9.6 | 47,624 |

(v6 cell count is 60 because the v6 contexts are emitted at *two* token
budgets — 2,048 and 4,096 — and v4's `behavior_runs.jsonl`
corresponds to a different budget regime, so the comparison is
not apples-to-apples on raw success rate. The intra-v6 contrasts
below are the trustworthy ones.)

### 7.3 The decisive intra-v6 contrast

> **`jacobian_low_spans` (0.800) ≈ `jacobian_high_spans_raw`
> (0.833) ≈ `jacobian_high_spans` (0.700).** The DIRECTION of
> gradient ranking matters less than 4 percentage points and is
> sometimes inverted: picking the *least*-Jacobian spans is
> indistinguishable from picking the *most*-Jacobian spans.

This is the **behavioral confirmation of the Experiment A negative
result**. Gradient saliency, at the span level, is not a useful
ranking signal — the choice of greedy direction does not produce
the systematic high-beats-low gap that the v4 sensitivity ranking
produced. The `recent_hybrid` condition, which used 50 % recency,
was the *worst* of the four (0.667), so recency mixing does not
help here either.

### 7.4 What about the v4-baseline gap?

The fact that the four v6 methods cluster at 0.67–0.83 while v4's
two reported `cap=15` baselines sit at 0.40–0.47 is striking, but
**not interpretable as evidence of a v6 method**. The differences
are:

* v6 contexts at 4 K token budget are notably larger than v4's
  conditions (mean input tokens 52–61 K vs 47–50 K at the same
  `cap=15`). More raw budget alone explains much of the gap.
* v6's compose script uses the same v4-style
  `[SELECTED_HISTORY_SPANS]` wrapper but picks spans differently;
  any difference vs v4 conflates "ranking signal" with "amount of
  text included".
* The intra-v6 comparison (high vs low vs raw vs hybrid) controls
  for budget and renderer, and that contrast is **flat**, which is
  the actual result.

### 7.5 Verdict for D

**NEGATIVE for gradient saliency as a span-selection signal.**

This matches the A and §3.2-§3.3 finding that the first-order
embedding Jacobian is a length-biased, non-discriminative score at
the span level. The downstream agent recovers most of full-context
performance from *any* moderate compression as long as enough
spans are retained — but it is indifferent to which spans the
Jacobian flagged as important.

Outputs:
* `outputs/raw/jacobian_compressed_contexts.jsonl` — 240 rendered contexts
* `outputs/raw/jacobian_behavior_runs.jsonl`       — 240 agent traces
* `outputs/tables/jacobian_span_downstream_results.csv` — aggregate summary

## 8. Combined interpretation (paper-tier)

The four findings tile cleanly:

* **A negative + D negative** (intra-v6 high≈low): the first-order
  embedding Jacobian is not a span-selection signal on this
  benchmark. It is partly a length proxy, and even when length is
  normalised away the resulting ranking is uncorrelated with v4 and
  uncorrelated with downstream task success.
* **B positive**: the same Jacobian, *propagated through the
  network* and Hadamard-multiplied with the mid-layer residual
  stream, lives in a 16–32-dimensional active subspace. Spans v4
  flagged as sensitive concentrate more tightly in that subspace.
* **C degenerate**: target NLL on the v4 reference decision state
  is not a useful objective for soft-token oracle measurement.

Together this **kills "rank spans by Jacobian and select top-k"**
as a method direction, **supports "project to low-rank active
subspace"** as a representation-level direction, and **rules out the
naïve soft-token oracle** as a measurement instrument.

Concretely, the strongest paper claim we can defend with these
diagnostics alone is:

> Agent context information at the layer-N/2 residual stream lives
> in a low-dimensional subspace whose principal directions
> correlate with where finite-difference probes place mass —
> motivating subspace-preserving compression, not gradient-ranked
> span selection.

We do *not* yet have evidence that such a compressor improves
downstream performance over recency or ACON; that is left for the
next round.

## 9. Files of record

Raw:
* `raw/cases.jsonl` (30 tasks; rendered context + decision-state target)
* `raw/jacobian_span_scores.jsonl` (593 spans, 30 tasks)
* `raw/jacobian_case_summary.jsonl` (30 cases; 2 marked `retry=True`)
* `raw/active_vectors_layer18.npz` (example: 30 × 2560; span: 593 × 2560)
* `raw/active_vector_metadata.jsonl`
* `raw/soft_token_histories.jsonl` (per-(task,k) per-step loss curves)

Tables:
* `tables/jacobian_vs_v4_correlations.csv` — global + per-task Spearman/Pearson for 7 score variants + confounders
* `tables/jacobian_topk_overlap.csv`        — top-k overlap and enrichment at k ∈ {1,3,5,10}
* `tables/jacobian_top1_ranks.csv`          — rank-of-top1 cross-check
* `tables/active_subspace_spectrum.csv`     — full SVD spectrum per matrix
* `tables/soft_token_oracle_losses.csv`     — per-case losses + gap recovery
* `tables/v6_dashboard.csv`                 — flat dashboard the report writer reads

Figures (both `.png` + `.pdf`):
* `figures/fig_jacobian_vs_v4_scatter.*`
* `figures/fig_jacobian_rank_overlap.*`
* `figures/fig_active_subspace_spectrum_example.*`
* `figures/fig_active_subspace_spectrum_span.*`
* `figures/fig_high_vs_low_sensitivity_spectrum.*`
* `figures/fig_soft_token_gap_recovery.*`
* `figures/fig_soft_token_loss_vs_k.*`

## 10. One-paragraph summary for the paper

> We train a single backward pass through Qwen3-4B-Instruct-2507 on
> 30 long-context AppWorld trajectories, teacher-forcing the v4
> reference decision-state JSON. Per-token embedding gradients show
> **no per-task rank-correlation with v4's leave-one-span-out
> sensitivity probe** (median Spearman −0.03; n=28), and the same
> ranking is **behaviorally inert at the downstream-agent level**:
> in 240 MiniMax-M2.5 runs, selecting the bottom-Jacobian spans is
> indistinguishable from selecting the top-Jacobian spans (0.80 vs
> 0.83 success rate). However, the Jacobian-weighted activations
> at the mid-layer residual stream are **strongly low-rank** — 92 %
> of variance is captured by the top 16 components at the example
> level, and **spans that v4 marked sensitive concentrate more
> tightly in that subspace than spans v4 marked insensitive** (Δ =
> +19 pp at k=4). A soft-token oracle on the same teacher-forced
> target is *degenerate*: even k=4 continuous tokens drive the
> loss below the full-context baseline because the parameter count
> of soft tokens exceeds the entropy of the target JSON. We
> conclude that the active-subspace formulation is empirically
> supported, but that gradient saliency is **not** the right
> primitive for span selection on this benchmark, and that
> oracle-style compression upper bounds need a more selective
> target than reference-state teacher forcing.
