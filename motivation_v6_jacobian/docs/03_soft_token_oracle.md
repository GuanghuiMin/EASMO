# Soft-token oracle (Experiment C)

## 1. Purpose

This experiment is an *upper bound*, not a deployable method. It asks
the following question:

> Given a fixed model, a fixed downstream target (the v4 reference
> decision state), and a long history, can we find $k$ continuous
> vectors in the input-embedding space whose effect on the target NLL
> is close to that of the full history?

If yes, then natural-language compressors are bottlenecked by *the
text parameterisation*, not by the bandwidth of $k$ vectors. If no,
then the information is genuinely high-dimensional and no
compression at $k$ tokens can close the gap.

## 2. Input arrangement

The full input fed to Qwen3-4B-Instruct-2507 is

```text
[chat-template prompt with EMPTY context]   ← n_prefix tokens, frozen
[k soft tokens]                              ← inputs_embeds, trainable
[target JSON tokens + EOS]                   ← n_target tokens, frozen
```

The CE label is set to `-100` on the first `n_prefix + k` positions
and to the target tokens for the remaining `n_target` positions.
Attention mask covers all positions.

Compared to the spec, the only deviation is that we always append a
trailing EOS to the target so the model sees a natural stop signal.

## 3. Initialisation

```python
soft_master = mean_input_embedding + 0.02 * torch.randn_like
soft_master = soft_master.float()                 # fp32 master copy
soft_master.requires_grad_(True)
```

* `mean_input_embedding` is the per-feature mean of the embedding
  matrix. Sampling from this prior puts soft tokens on the natural
  manifold of word embeddings instead of leaving them at zero.
* We always keep an fp32 master copy; on every step we cast to bf16
  for the forward pass. This avoids the bf16 + AdamW
  `epsilon`-rounding pathology where the update vanishes once
  $\|m_t\| < \epsilon$.

## 4. Optimiser

```python
optimizer = AdamW([soft_master], lr=0.05, weight_decay=0.0)
```

Step:

```python
soft = soft_master.to(torch.bfloat16)
inputs_embeds = torch.cat([prefix_embeds, soft.unsqueeze(0), target_embeds], dim=1)
out = model(inputs_embeds=inputs_embeds, attention_mask=mask, labels=labels, use_cache=False)
loss = out.loss
loss.backward()
optimizer.step(); optimizer.zero_grad()
```

Early-stop when no improvement of more than `min_delta = 1e-4` for
`patience = 30` steps. Hard cap `num_steps = 200`.

## 5. Per-k cost

Per case the forward pass is short — `n_prefix + k + n_target` ≈
700 + 64 + 800 ≈ 1500 tokens for $k = 64$. On an H100 a single
forward+backward over 1500 tokens of Qwen3-4B in bf16 is roughly
0.2 s. So 200 steps per $k$ × 5 $k$ values × 30 tasks ≈ 30000 step
iterations × 0.2 s ≈ 6000 s ≈ 100 min for the optimisation. Adding
the four baseline forwards (no backward) per case adds ≈ 30 cases ×
4 × 0.3 s = 36 s.

Realistic wall-clock estimate is 2–3 h; the user-facing run budget
should reserve ~5 h to allow for slow tokens and OOM retries.

## 6. Metrics

Per case we save the final loss for each $k$ and the per-step loss
history (`outputs/raw/soft_token_histories.jsonl`). The aggregation
script reports

* per-$k$ median final loss across cases;
* per-$k$ median gap recovery
  $\rho_k = (L_\text{no} - L_\text{soft,k}) / (L_\text{no} - L_\text{full})$;
* per-case sign of the soft-vs-recent comparison and soft-vs-ACON
  comparison.

`ρ_k` is undefined (NaN) for cases where $L_\text{no} < L_\text{full}$.
This shouldn't happen on average but does occur on some short or
already-self-evident tasks. We treat those as missing rather than
clamping to 0 or 1, so the median is computed over the defined
subset.

## 7. Failure modes

* **Soft tokens diverge.** If the loss explodes the fp32 master copy
  protects us from NaN-poisoning, but the optimiser may still get
  stuck. Spec-suggested defaults are conservative; if needed, drop
  `lr` to 0.02 or 0.01.
* **Soft tokens degenerate to a single direction.** Visible in the
  active-subspace SVD of the soft tokens themselves (out of scope for
  this round). If observed in follow-up work, add a small spectral
  regulariser.
* **Memory peak exceeds GPU.** Lower `max_context_tokens` for the
  baseline forward pass (the soft-token training itself uses very
  little memory).
