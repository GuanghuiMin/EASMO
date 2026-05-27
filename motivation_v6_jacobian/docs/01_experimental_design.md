# motivation_v6 — Experimental design

## 0. Framing

`motivation_v4` showed a real but unsatisfactory signal:

> Leave-one-span-out sensitivity beats random and low-sensitivity baselines,
> but **recency** (just keep the last few spans) beats it.

`motivation_v5` showed that ACON-style compressors drop concrete state that an
audit model can re-surface — but the story still depended on prompt
engineering on top of MiniMax.

v6 stops asking LLMs *which spans matter* and asks the fixed white-box
model *which context directions the downstream objective locally
depends on*. Formally, for context representation $r(h)$ and downstream
target $\mathcal{L}(h, x)$,
\[
\Delta z \approx J_r\,\Delta r,\quad J_r = \partial z_\theta(h, x)/\partial r,
\]
and the hypothesis is

> Long agent histories affect future behaviour through a low-dimensional
> active subspace of the downstream policy Jacobian. Effective
> compression should preserve this active subspace and discard
> directions in the Jacobian null space.

## 1. Questions

* **Q1 / Exp A.** Can a single white-box backward pass recover the same
  signal as v4 leave-one-span-out sensitivity?
* **Q2 / Exp B.** Is the downstream Jacobian low-rank?
* **Q3 / Exp C.** Is there a soft-token compression upper bound?
* **Q4 / Exp D (optional).** Are gradient-ranked text spans
  behaviourally useful when fed to a downstream agent?

## 2. White-box model and target

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` (HF, local snapshot) |
| Layers | 36 (active-subspace capture defaults to layer 18 = `N // 2`) |
| Hidden size | 2560 |
| Tokenizer | `Qwen2Tokenizer` fast, with offset mapping |
| Attention | `attn_implementation="sdpa"` (no flash-attn dependency) |
| Dtype | bf16 weights; fp32 soft-token master copy |

We do **not** quantise weights, because gradients through quantised
linear layers may complicate the interpretation of the Jacobian.

Target text is canonical JSON of v4's reference decision state:

```python
target_text = json.dumps(reference_decision_state, ensure_ascii=False, sort_keys=True)
```

Cross-entropy is computed only over target tokens (prompt labels set
to `-100`).

## 3. Data plan

| | Stage | # cases | Wall-clock (est) |
|---|---|---:|---:|
| A | Jacobian saliency over 30 v4 tasks                                                                 | 30 | 15–30 min |
| B | SVD on Jacobian-weighted activations (free; captured during A)                                    | 30 | 1–5 min  |
| C | Soft-token oracle, k ∈ {4,8,16,32,64}, 200 steps                                                   | 30 | 3–5 h    |
| D | Compose 4 conditions × 30 tasks, MiniMax downstream agent run                                     | 120 runs | 3–4 h |

All experiments reuse `motivation_v4/outputs/raw/*.jsonl` so we do not
re-render trajectories or re-probe MiniMax.

## 4. Tokenisation and span alignment (Exp A)

Spans already carry `[STEP N]\n...` sentinels in their text (rendered
by v4 stage 01). To map per-token gradients back to spans without
relying on substring search:

1. Build the rendered context by concatenating span texts with `"\n\n"`.
2. Use the fast tokenizer with `return_offsets_mapping=True` to obtain
   `(char_start, char_end)` for every token.
3. For each span, collect tokens whose char range overlaps the span's
   char range (computed from the rendered context).
4. Build the full input by chat-templating
   `build_probe_prompt(instruction, "<<<CTX_REPLACE_ME>>>")` and
   swapping the sentinel-token range for the real context-token range.

This guarantees the per-span token-index ranges are exact.

## 5. Truncation policy

If the rendered context exceeds `--max_context_tokens` (default
**12,000**) under Qwen's tokenizer, we drop spans from the **front**
(oldest first) until it fits. We then realign char→token offsets on
the truncated context. Truncation is logged per-task in
`outputs/raw/jacobian_case_summary.jsonl` so any downstream
correlation analysis can audit which tasks were affected.

## 6. Saliency formulae (Exp A)

For each context token $j$ we compute

```text
grad_norm[j]   = || ∂L/∂e_j ||₂
gxa_abs[j]     = Σ |(∂L/∂e_j) ⊙ e_j|         (Hadamard, then sum-abs)
g_dot_x_abs[j] = |Σ (∂L/∂e_j) ⊙ e_j|
```

For each span $i$ over its token index set $T_i$:

```text
span_grad_sum    = Σ_{j∈T_i} grad_norm[j]
span_grad_mean   = mean
span_gxa_sum     = Σ_{j∈T_i} gxa_abs[j]
span_gxa_mean    = mean
span_gxa_sqrtlen = span_gxa_sum / sqrt(|T_i|)      <-- primary
span_top10_mean  = mean of top-10 gxa_abs in span
```

Reason for `sqrtlen` as the primary score: raw sum over-rewards long
spans; mean under-weights diffuse-but-important spans. The
square-root normalisation matches the variance scaling of an
independent-tokens-sum-of-magnitudes baseline.

## 7. Active-subspace capture (Exp B)

In the same backward pass as A we register a forward hook on the
target transformer block (`layer_index = N/2 = 18`), retain the
intermediate gradient, and after `loss.backward()` materialise

```text
A[j,:] = H_{L,j} ⊙ G_{L,j}        (per-token active vector)
A_span = Σ_{j∈span} A[j,:]         (sum across span tokens)
A_example = Σ_{j∈context} A[j,:]
```

We save two stacked matrices, `example` (n_tasks × hidden_dim) and
`span` (n_spans_total × hidden_dim), centre them, and run randomised
SVD with `n_components = 256`. Cumulative explained variance is the
headline.

We additionally split the span matrix by the median v4
`final_sensitivity` and compute the spectrum of the **high**-v4-mass
and **low**-v4-mass halves. If the high group concentrates more
tightly, that is the medium-positive result for B.

## 8. Soft-token oracle (Exp C)

Per spec §7. Input arrangement is

```text
<chat-templated probe with EMPTY context>
<k soft embedding vectors>
<target JSON tokens>
```

Soft tokens are initialised at `mean(input_embed_weights) + 0.02·N(0,I)`
in fp32; on every step we cast to bf16, splice into `inputs_embeds`,
forward + backward, and AdamW-step the fp32 master copy. Loss is
teacher-forced CE over target tokens only.

Default `num_steps = 200`, `lr = 0.05`, early stop with patience 30.

Baselines (no soft-token training; pure forward):

| baseline | context |
|---|---|
| `full` | entire v4 history (truncated to 12K) |
| `no` | empty |
| `recent` | last 5 spans |
| `acon` | v4's `acon_baseline` compressed_text if present, else NaN |

Headline metric is gap recovery
\[
\rho_k = \frac{L_\text{no} - L_\text{soft,k}}{L_\text{no} - L_\text{full}}.
\]

`ρ_k ≥ 0.7` at `k ≤ 32` ⇒ STRONG positive for C.

## 9. Optional Experiment D

For each task, build four conditions under fixed token budgets
{2048, 4096}:

| method | rule |
|---|---|
| `jacobian_high_spans`     | greedy by `span_gxa_sqrtlen / token_count`, descending |
| `jacobian_high_spans_raw` | greedy by `span_gxa_sqrtlen`,           descending |
| `jacobian_low_spans`      | greedy by `span_gxa_sqrtlen / token_count`, ascending  |
| `jacobian_recent_hybrid`  | 50 % recent tail + 50 % top-Jacobian among the rest    |

Picked spans are emitted in chronological order, wrapped in
`[SELECTED_HISTORY_SPANS] … [/SELECTED_HISTORY_SPANS]` to match v4's
runner input format. We then call v4's existing MiniMax-M2.5 runner
(reused via `motivation_v4.runner.run_with_compressed_context`) under
`acon/.venv` and aggregate success rates.

Comparison is against v4's `recent_spans` and `high_sensitivity_spans`
runs (same budget) in `motivation_v4/outputs/raw/behavior_runs.jsonl`.

## 10. Success criteria

* **A** STRONG: median per-task Spearman ≥ 0.25 OR median top-3 enrichment ≥ 2.0×.
* **A** MEDIUM: top spans cleanly separated from low-sensitivity even if correlation is weak.
* **B** STRONG: example or span matrix cumulative-explained-variance at k=16 ≥ 0.5 (or ≥ 0.7 at k=32).
* **B** MEDIUM: high-v4 spectrum is visibly steeper than low-v4.
* **C** STRONG: median `ρ_k` at k ∈ {16, 32} ≥ 0.7.
* **C** MEDIUM: soft tokens beat textual ACON/recent baselines on target NLL.

`docs/04_results_summary.md` is written automatically by
`scripts/08_write_report.py` and tags each verdict.

## 11. Non-goals (verbatim from spec §2)

- No prompt optimisation, no YAML schema design.
- No new ACON-style compressor.
- No LLM audit taxonomy.
- No "ask Qwen whether this span is important."
- No span selection method as the **main** result.
- No MiniMax gradients (it is API-only).
