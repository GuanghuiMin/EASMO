# Gradient definitions for motivation_v6

## 1. Notation

Let $E \in \mathbb{R}^{T \times d}$ be the input embedding tensor
($T$ = sequence length, $d$ = hidden dim). Write $E_j$ for row $j$
(the embedding of the $j$-th token).

Let $\mathcal{L}$ be the teacher-forced cross-entropy on the target
JSON tokens only — prompt and context labels are set to `-100`. For
target token positions $t \in \mathcal{T}_\text{tgt}$,
\[
\mathcal{L} = \frac{1}{|\mathcal{T}_\text{tgt}|}
              \sum_{t \in \mathcal{T}_\text{tgt}}
              -\log p_\theta\!\left(y_t \mid y_{<t}, E\right).
\]

We always do **one** forward + **one** backward per case. Weights are
frozen; only `E.requires_grad_(True)` is set.

## 2. Per-token saliency

Three scalars per context token $j$:

| name | formula |
|---|---|
| `grad_norm[j]`   | $\big\|\partial \mathcal{L}/\partial E_j\big\|_2$ |
| `gxa_abs[j]`     | $\sum_d |(\partial \mathcal{L}/\partial E_j)_d \cdot E_{j,d}|$ |
| `g_dot_x_abs[j]` | $\big|\langle \partial \mathcal{L}/\partial E_j, E_j \rangle\big|$ |

`gxa_abs` is the Hadamard-then-sum-abs variant; it is invariant to the
sign of the contribution and therefore robust to the "negative tokens
cancel positive tokens" pathology that pure dot-products suffer. It
is the gradient-input analogue of an integrated-gradients term but
with a fixed reference at zero.

## 3. Per-span aggregation

For span $i$ with token index set $T_i$:

```text
span_grad_sum    = Σ_{j∈T_i} grad_norm[j]
span_grad_mean   = (1/|T_i|) Σ_{j∈T_i} grad_norm[j]
span_gxa_sum     = Σ_{j∈T_i} gxa_abs[j]
span_gxa_mean    = (1/|T_i|) Σ_{j∈T_i} gxa_abs[j]
span_gxa_sqrtlen = Σ_{j∈T_i} gxa_abs[j] / sqrt(|T_i|)         <-- primary
span_top10_mean  = mean of top-min(10, |T_i|) entries of gxa_abs[j]
span_g_dot_x_abs_sum = Σ_{j∈T_i} g_dot_x_abs[j]
```

Why `sqrtlen` is primary:

* Raw sum scales linearly with span length and is dominated by long
  spans regardless of per-token importance.
* Mean down-weights important spans that contain a few decisive
  tokens diluted by boilerplate.
* `Σ / sqrt(|T_i|)` is the right scaling under the null
  $g_j$ ∼ iid with finite variance — sum of squared values of a
  $|T_i|$-token span grows linearly, but its standard deviation grows
  as $\sqrt{|T_i|}$. So `Σ / sqrt(|T_i|)` is centred on zero under
  the null and grows with the per-token excess.

All variants are reported in `outputs/tables/jacobian_vs_v4_correlations.csv`
for ablation.

## 4. Active-subspace vectors

We register a forward hook on transformer block $L$ (default
$L = N/2$). The hook calls `output[0].retain_grad()`, so after
`loss.backward()` the Python tensor object inside the cache has
`.grad` populated. Let $H \in \mathbb{R}^{T \times d}$ be the
residual stream at layer $L$ and $G \in \mathbb{R}^{T \times d}$ its
gradient.

Per token, span, and example:

```text
A[j,:]    = H[j,:] ⊙ G[j,:]            (Hadamard product)
A_span    = Σ_{j∈span}     A[j,:]
A_example = Σ_{j∈context}  A[j,:]
```

We stack these into matrices and run randomised SVD on the
centred matrices:

```python
X = A - A.mean(axis=0, keepdims=True)
_, S, _ = randomized_svd(X, n_components=256, random_state=0)
explained = S**2 / (S**2).sum()
cumulative = np.cumsum(explained)
```

The headline metric is `cumulative[k-1]` at $k \in \{4, 8, 16, 32, 64\}$.

## 5. Memory budget

For $T = 12{,}000$, $d = 2560$, bf16, Qwen3-4B (36 layers):

* Weights: 4.0 B × 2 = 8 GB.
* Activation tensors stored for backward: with gradient checkpointing
  and `attn_implementation="sdpa"`, ~10–15 GB at $T = 12$K.
* `inputs_embeds.grad`: $T \cdot d \cdot 4$ B (fp32 grad on bf16 tensor
  ⇒ 120 MB; transformers casts grads up).
* Captured layer hidden + grad: $T \cdot d \cdot 4$ B each ⇒ 480 MB.

Total peak ≈ 25–30 GB; well within an 80 GB H100 if we stop the vLLM
server first.

## 6. Numerical caveats

1. `bf16` gradients are noisy. We cast to fp32 immediately after
   reading them off the tensor (`g = inputs_embeds.grad[…].float()`)
   to avoid noise propagating into the per-span aggregates.
2. `randomized_svd` is non-deterministic without a seed; we pin
   `random_state=0`.
3. The Hadamard product $H \odot G$ is not invariant to per-feature
   rescaling (e.g. RMSNorm scaling) — but Qwen's RMSNorm has been
   absorbed into the residual stream by the time the hook fires, so
   the captured vectors are in the "natural" residual basis.
