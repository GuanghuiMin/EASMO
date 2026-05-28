# Motivation v6 Results: Jacobian Active-Subspace Diagnostics

> Auto-written by `scripts/08_write_report.py` at 2026-05-28 00:23Z.

## 1. Setup

* model path: `/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
* number of tasks (v4 reuse): **28**
* max context tokens (white-box): **12000**
* median ctx tokens after tokenization: **5456**
* target type: canonical v4 reference decision-state JSON (teacher-forced)
* mid-layer for active capture: **layer N/2 (default)**

## 2. Experiment A — Jacobian saliency vs v4 finite-difference sensitivity

| Metric | Value |
|---|---|
| Global Spearman (primary score = `span_gxa_sqrtlen`) | **-0.206** |
| Global Pearson | -0.058 |
| Median per-task Spearman | **-0.029** |
| Median top-1 enrichment over random | 0.000× |
| Median top-3 enrichment over random | **1.833×** |
| Median top-5 enrichment over random | 1.280× |
| Confounder: Jacobian ↔ token_count Spearman | 0.358 |
| Confounder: Jacobian ↔ step_id (recency) Spearman | -0.029 |
| Confounder: v4 ↔ token_count Spearman | 0.087 |

**Verdict A:** MEDIUM positive

Figures: `figures/fig_jacobian_vs_v4_scatter.{png,pdf}`,
`figures/fig_jacobian_rank_overlap.{png,pdf}`.

## 3. Experiment B — Active subspace spectrum (layer N/2 (default))

Cumulative explained variance of randomised SVD on
Jacobian-weighted activations:

| Matrix | k=4 | k=8 | k=16 | k=32 | k=64 |
|---|---|---|---|---|---|
| example  | 0.662 | 0.796 | **0.931** | — | — |
| span     | 0.525 | 0.644 | **0.741** | 0.831 | 0.904 |
| high_v4  | 0.652 | 0.739 | 0.815 | 0.881 | 0.935 |
| low_v4   | 0.482 | 0.608 | 0.729 | 0.838 | 0.924 |

High-vs-low cumulative variance gap at k=16: **0.086**.

**Verdict B:** **STRONG POSITIVE**

Figures: `figures/fig_active_subspace_spectrum_example.{png,pdf}`,
`figures/fig_active_subspace_spectrum_span.{png,pdf}`,
`figures/fig_high_vs_low_sensitivity_spectrum.{png,pdf}`.

## 4. Experiment C — Soft-token oracle

Median target NLL across cases:

| Method | median NLL | gap recovery |
|---|---|---|
| full context | **0.993** | 1.000 |
| no context | 1.786 | 0.000 |
| recent (last 5 spans) | 1.333 | 1.333 |
| ACON baseline | — | — |
| soft k=4 | 0.089 | 2.256 |
| soft k=8 | 0.005 | 2.252 |
| soft k=16 | **0.000** | **2.258** |
| soft k=32 | **0.000** | **2.258** |
| soft k=64 | 0.000 | 2.286 |

**Verdict C:** **STRONG POSITIVE**

Figures: `figures/fig_soft_token_gap_recovery.{png,pdf}`,
`figures/fig_soft_token_loss_vs_k.{png,pdf}`.

## 5. Negative findings

- Jacobian-vs-v4 correlation against token-length: 0.358; if
  this is large, Jacobian scores are partially a length proxy.
- Jacobian-vs-step_id (recency) Spearman: -0.029; large
  positive means scores covary with recency.
- low_v4 spans cumulative variance at k=16 ≈ 0.729 — if not
  much smaller than high_v4 (0.815), then the active subspace
  isn't really concentrating on the spans v4 flagged.

## 6. Implications for method design

(See interpretation rules §11 of the spec.)

- If A & B positive: motivates an active-subspace-preserving compressor.
- If B & C positive but A weak: motivates soft-memory / representation-
  level compression, NOT span selection.
- If A positive but B/C weak: gradients are useful as a saliency signal
  but the low-rank claim is unsupported.

## 7. Files of record

Tables:
- `tables/jacobian_vs_v4_correlations.csv`
- `tables/jacobian_topk_overlap.csv`
- `tables/jacobian_top1_ranks.csv`
- `tables/active_subspace_spectrum.csv`
- `tables/soft_token_oracle_losses.csv`
- `tables/v6_dashboard.csv`

Figures:
- `figures/fig_jacobian_vs_v4_scatter.{png,pdf}`
- `figures/fig_jacobian_rank_overlap.{png,pdf}`
- `figures/fig_active_subspace_spectrum_example.{png,pdf}`
- `figures/fig_active_subspace_spectrum_span.{png,pdf}`
- `figures/fig_high_vs_low_sensitivity_spectrum.{png,pdf}`
- `figures/fig_soft_token_gap_recovery.{png,pdf}`
- `figures/fig_soft_token_loss_vs_k.{png,pdf}`

Raw artifacts:
- `raw/cases.jsonl`
- `raw/jacobian_span_scores.jsonl`
- `raw/jacobian_case_summary.jsonl`
- `raw/active_vectors_layer*.npz`
- `raw/active_vector_metadata.jsonl`
- `raw/soft_token_histories.jsonl`
