# Motivation v6: Jacobian / Active-Subspace Diagnostics for Agent Context Compression

**Track:** `EASMO/motivation_v6_jacobian/`  
**Goal:** test whether agent context compression can be studied as *preserving the downstream policy Jacobian active subspace*, rather than prompt optimization, YAML summaries, or span-selection heuristics.

This is a diagnostic experiment, not a new compressor yet.

---

## 0. One-paragraph framing

Previous rounds showed two important but unsatisfactory facts:

1. `motivation_v4` decision-state probing found a real span-level signal: high-sensitivity spans beat random and low-sensitivity spans, but they do **not** beat recent spans or summary methods. This suggests that scalar span importance is real but not the right compression primitive.
2. `motivation_v5` failure auditing found many ACON-style failures where concrete state is recovered by an audit model and then dropped again by recompression. This is useful phenomenology, but the audit/taxonomy story remains too prompt-shaped and too heuristic.

For v6, we stop optimizing prompts and stop asking LLM auditors what is important. Instead, we ask the fixed white-box model directly:

> Which context directions does the downstream objective locally depend on?

Formally, for a fixed model \(\pi_\theta\), context representation \(r(h)\), and downstream target loss \(\mathcal{L}(h, x)\), the first-order effect of context perturbation is

\[
\Delta z \approx J_r \Delta r, \quad J_r = \frac{\partial z_\theta(h,x)}{\partial r}.
\]

The core hypothesis is:

> Long agent histories affect future behavior through a low-dimensional active subspace of the downstream policy Jacobian. Effective compression should preserve this active subspace and discard directions in the Jacobian null space.

This experiment validates or falsifies that hypothesis using Qwen3-4B as a white-box model.

---

## 1. What this experiment must answer

### Q1. Can white-box Jacobian saliency recover the same signal as v4 leave-one-span-out sensitivity?

Use Qwen3-4B gradients to score context tokens/spans in a **single backward pass**. Compare these scores against the existing v4 finite-difference / LLM-judge span sensitivity.

This checks whether the expensive black-box probe in v4 can be replaced by a local Jacobian surrogate.

### Q2. Is the downstream context Jacobian low-rank?

Collect Jacobian-weighted activation vectors over many AppWorld contexts and compute the singular-value spectrum.

This checks whether the “low-dimensional active subspace” story is empirically true.

### Q3. Is there a soft-token compression upper bound?

Optimize a small number of continuous soft tokens to match full-context behavior. This asks whether the effective decision information in a long context is compressible at all, independent of natural-language rendering.

If 16/32 soft tokens recover most of the full-context next-action loss, then the bottleneck is not information-theoretic; it is the parameterization of compression as natural-language summaries.

---

## 2. Non-goals

Do **not** do any of the following in v6:

- No prompt optimization.
- No YAML/checkpoint schema design.
- No new ACON-style compressor prompt.
- No LLM audit taxonomy.
- No “ask Qwen whether this span is important.”
- No span selection method as the main result.
- No MiniMax gradients; MiniMax is black-box and can only be used for optional downstream validation.

The primary objects are gradients, Jacobians, activation vectors, spectra, and soft-token optimization.

---

## 3. Inputs and existing artifacts

Use existing EASMO outputs as much as possible.

### Required inputs

```text
/workspace/EASMO/motivation_v4/outputs/raw/history_spans.jsonl
/workspace/EASMO/motivation_v4/outputs/raw/reference_decision_states.jsonl
/workspace/EASMO/motivation_v4/outputs/raw/span_sensitivity_scores.jsonl
/workspace/EASMO/motivation_v4/outputs/raw/compressed_contexts.jsonl
/workspace/EASMO/motivation_v4/outputs/raw/behavior_runs.jsonl
```

### Optional inputs

```text
/workspace/EASMO/motivation_v5/outputs/raw/merged_case_audits.jsonl
/workspace/EASMO/motivation_v5/data/sampled_cases.jsonl
```

Use v5 only for optional analysis on ACON failure cases. The core v6 experiment should be runnable on v4 data alone.

### Model

Use a **white-box local Hugging Face model**, not the vLLM OpenAI API endpoint, because gradients are required.

Default:

```text
QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-Qwen/Qwen3-4B-Instruct}
```

If the repo has a local checkpoint path, prefer that. The script must accept `--model_path`.

Recommended loading:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
```

If memory is tight, use a smaller sample first. Do **not** switch to quantized 4-bit for the primary gradient experiment unless necessary; gradients through quantized weights may complicate interpretation.

---

## 4. Directory layout to create

```text
motivation_v6_jacobian/
├── README.md
├── docs/
│   ├── 01_experimental_design.md
│   ├── 02_gradient_definitions.md
│   ├── 03_soft_token_oracle.md
│   └── 04_results_summary.md
├── motivation_v6_jacobian/
│   ├── __init__.py
│   ├── data.py
│   ├── prompts.py
│   ├── gradients.py
│   ├── hooks.py
│   ├── active_subspace.py
│   ├── soft_tokens.py
│   ├── metrics.py
│   └── plotting.py
├── scripts/
│   ├── 01_build_cases.py
│   ├── 02_compute_jacobian_saliency.py
│   ├── 03_compare_to_v4_sensitivity.py
│   ├── 04_active_subspace_spectrum.py
│   ├── 05_soft_token_oracle.py
│   ├── 06_aggregate.py
│   ├── 07_plot.py
│   ├── 08_write_report.py
│   └── run_all.sh
└── outputs/
    ├── raw/
    ├── tables/
    ├── figures/
    └── reports/
```

---

## 5. Experiment A — Jacobian saliency vs v4 finite-difference sensitivity

### 5.1 Purpose

Test whether a single white-box backward pass gives a span-importance signal comparable to v4’s expensive leave-one-span-out sensitivity.

v4’s finite-difference signal is imperfect but meaningful. The question here is not whether it is the final compression method. The question is whether the Jacobian contains the same behavioral signal.

### 5.2 Case construction

Script: `scripts/01_build_cases.py`

For each task in v4:

1. Load all spans from `history_spans.jsonl`.
2. Reconstruct the full context as chronological rendered spans.
3. Load the v4 reference decision state for that task.
4. Canonicalize the target JSON as compact text:

```python
target_text = json.dumps(reference_decision_state, ensure_ascii=False, sort_keys=True)
```

5. Build a Qwen-compatible probe prompt using the v4 decision-state probe content, but do not generate. We only teacher-force the target JSON.

The full input sequence should be:

```text
<probe prompt with full context>
<target JSON>
```

Labels should ignore all prompt/context tokens and compute cross-entropy only on the target JSON tokens.

Output:

```text
outputs/raw/cases.jsonl
```

Schema:

```json
{
  "task_id": "...",
  "task_instruction": "...",
  "context_text": "...",
  "target_text": "...",
  "spans": [
    {
      "span_id": 0,
      "step_id": 0,
      "span_text": "...",
      "char_start": 0,
      "char_end": 1234,
      "token_count": 250,
      "v4_final_sensitivity": 0.72
    }
  ]
}
```

### 5.3 Token-to-span alignment

Because the gradient is computed over tokens, we need to aggregate token saliency back to spans.

Do this by constructing the context text with explicit sentinel markers:

```text
[STEP 0]
...
[/STEP 0]
[STEP 1]
...
[/STEP 1]
```

After tokenization, map each token to a span by decoding offsets if fast tokenizer offsets are available. If offsets are unavailable, use robust approximate alignment:

1. Tokenize each span separately.
2. Tokenize the full context.
3. Use marker token positions to locate each span region.
4. Store token index ranges per span.

The output `cases.jsonl` must include `span_token_start` and `span_token_end` after alignment if possible.

### 5.4 Gradient computation

Script: `scripts/02_compute_jacobian_saliency.py`

For each case:

1. Build `input_ids` for prompt + target.
2. Get input embeddings:

```python
embed_layer = model.get_input_embeddings()
inputs_embeds = embed_layer(input_ids)
inputs_embeds.requires_grad_(True)
```

3. Forward with `inputs_embeds` and labels that ignore prompt tokens:

```python
labels = input_ids.clone()
labels[:, :target_start_idx] = -100
out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
loss = out.loss
loss.backward()
```

4. Extract gradients for context-token positions only:

```python
g = inputs_embeds.grad[0, context_token_indices, :]          # dL / d embedding
e = inputs_embeds.detach()[0, context_token_indices, :]
```

5. Compute token scores:

```python
grad_norm = torch.linalg.norm(g, dim=-1)
gxa_abs = torch.sum(torch.abs(g * e), dim=-1)
g_dot_x_abs = torch.abs(torch.sum(g * e, dim=-1))
```

6. Aggregate to span scores:

For each span token set \(T_i\):

```python
span_grad_sum = grad_norm[T_i].sum()
span_grad_mean = grad_norm[T_i].mean()
span_gxa_sum = gxa_abs[T_i].sum()
span_gxa_mean = gxa_abs[T_i].mean()
span_gxa_sqrtlen = gxa_abs[T_i].sum() / sqrt(len(T_i))
span_top10_mean = topk(gxa_abs[T_i], k=min(10, len(T_i))).mean()
```

Primary score for ranking:

```text
jacobian_score = span_gxa_sqrtlen
```

Reason: raw sum over-rewards long spans; mean over-penalizes diffuse but important spans. The `sqrtlen` normalization is a compromise. Keep all variants for ablation.

Output:

```text
outputs/raw/jacobian_span_scores.jsonl
```

Schema:

```json
{
  "task_id": "...",
  "span_id": 7,
  "step_id": 7,
  "token_count": 245,
  "loss": 1.234,
  "v4_final_sensitivity": 0.82,
  "span_grad_sum": 0.0,
  "span_grad_mean": 0.0,
  "span_gxa_sum": 0.0,
  "span_gxa_mean": 0.0,
  "span_gxa_sqrtlen": 0.0,
  "span_top10_mean": 0.0,
  "jacobian_rank": 1,
  "v4_sensitivity_rank": 3
}
```

### 5.5 Metrics

Script: `scripts/03_compare_to_v4_sensitivity.py`

Compute:

1. Spearman correlation between each gradient score and v4 final sensitivity, per task and globally.
2. Pearson correlation, per task and globally.
3. Top-k overlap:

```text
k ∈ {1, 3, 5, budget-matched}
```

4. Enrichment over random:

```text
expected_random_topk_overlap = k^2 / n_spans
observed / expected
```

5. Rank of v4 top-1 under Jacobian score.
6. Rank of Jacobian top-1 under v4 score.
7. Correlation with recency and token length, to ensure Jacobian is not only a length/recency proxy.

Outputs:

```text
outputs/tables/jacobian_vs_v4_correlations.csv
outputs/tables/jacobian_topk_overlap.csv
outputs/figures/fig_jacobian_vs_v4_scatter.png
outputs/figures/fig_jacobian_rank_overlap.png
```

### 5.6 Success criteria for Experiment A

Strong positive:

```text
median per-task Spearman(jacobian_score, v4_sensitivity) >= 0.25
OR
Top-3 overlap enrichment >= 2.0x random
```

Medium positive:

```text
Jacobian top spans are clearly separated from low-sensitivity spans,
even if correlation is weak.
```

Negative:

```text
Jacobian scores mostly correlate with token length or recency,
or show no enrichment over random.
```

Do not hide a negative result. If the local Jacobian does not predict v4 sensitivity, that is scientifically useful: it says finite-difference context effects are highly nonlinear or prompt-dependent.

---

## 6. Experiment B — Active subspace spectrum of the downstream Jacobian

### 6.1 Purpose

Test whether the context-to-decision Jacobian has a low-dimensional active subspace.

This is the most important “low-rank” validation. We are not claiming facts or spans are low-rank. We are testing whether **Jacobian-weighted context representations** concentrate in a low-dimensional subspace.

### 6.2 Representation choice

Use a middle transformer layer residual stream, default:

```text
layer_index = floor(num_layers / 2)
```

Also allow `--layer_index` override.

For each case, capture hidden states \(H_L\) and gradients \(G_L = \partial \mathcal{L}/\partial H_L\) at that layer for context tokens.

Compute Jacobian-weighted activation vectors:

Per token:

\[
a_j = H_{L,j} \odot G_{L,j}
\]

Per span:

\[
a_{span} = \sum_{j \in span} a_j
\]

Per example:

\[
a_{example} = \sum_{j \in context} a_j
\]

### 6.3 Hook implementation

Script: `scripts/04_active_subspace_spectrum.py`

Implementation guidance:

```python
cache = {}

def forward_hook(module, inp, out):
    # out may be tuple depending on model
    hidden = out[0] if isinstance(out, tuple) else out
    hidden.retain_grad()
    cache["hidden"] = hidden

handle = model.model.layers[layer_index].register_forward_hook(forward_hook)

out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
loss = out.loss
loss.backward()

H = cache["hidden"].detach()[0, context_token_indices, :].float()
G = cache["hidden"].grad.detach()[0, context_token_indices, :].float()
A = H * G
```

Make this robust to Qwen model module naming. If `model.model.layers` fails, inspect `model.named_modules()` and implement a helper to find the transformer blocks.

### 6.4 Matrices to save

Save two matrices:

1. Example-level active vectors:

```text
A_example: shape [n_cases, hidden_dim]
```

2. Span-level active vectors:

```text
A_span: shape [n_spans_total, hidden_dim]
```

Also save metadata mapping rows to task/span.

Output:

```text
outputs/raw/active_vectors_layer{L}.npz
outputs/raw/active_vector_metadata.jsonl
```

### 6.5 SVD / PCA

Compute SVD on centered matrices:

```python
X = A - A.mean(axis=0, keepdims=True)
U, S, Vt = randomized_svd(X, n_components=min(256, min(X.shape)-1))
explained = S**2 / np.sum(S**2)
cumulative = np.cumsum(explained)
```

Use randomized SVD for speed.

Run separately for:

- example-level active vectors;
- span-level active vectors;
- high-v4-sensitivity spans only;
- low-v4-sensitivity spans only.

### 6.6 Outputs

```text
outputs/tables/active_subspace_spectrum.csv
outputs/figures/fig_active_subspace_spectrum_example.png
outputs/figures/fig_active_subspace_spectrum_span.png
outputs/figures/fig_high_vs_low_sensitivity_spectrum.png
```

CSV schema:

```json
{
  "matrix": "example|span|high_v4|low_v4",
  "layer_index": 18,
  "component": 1,
  "singular_value": 123.4,
  "explained_variance": 0.12,
  "cumulative_explained_variance": 0.12
}
```

### 6.7 Success criteria for Experiment B

Strong positive:

```text
top 16 components explain >= 50% of active-vector variance
OR
top 32 components explain >= 70%
```

Medium positive:

```text
high-sensitivity spans have a more concentrated spectrum than low-sensitivity spans.
```

Negative:

```text
spectrum decays slowly; no meaningful low-rank structure.
```

This experiment is the gate for the low-rank story. If there is no spectral decay, do not claim low-rank active subspace.

---

## 7. Experiment C — Soft-token oracle compression upper bound

### 7.1 Purpose

Test whether a small number of continuous tokens can replace long context for a fixed downstream objective.

This is not a deployable method. It is an upper bound that answers:

> Is the information in long context compressible into a low-dimensional continuous state at all?

### 7.2 Targets

Use two target options. Implement both if possible; prioritize C1.

#### C1. Decision-state target

Same as Experiment A:

```text
target_text = canonical v4 reference decision-state JSON
```

This is easiest and aligns with v4.

#### C2. Next-action target

If you can reliably extract next actions from trajectories, use next action code as target.

For a boundary \(t\):

```text
context = prefix up to step t
target = action code at step t+1
```

This is more policy-like, but requires robust trajectory parsing.

### 7.3 Baselines to score

For every case, compute target NLL under:

1. Full context.
2. No context.
3. Recent spans context.
4. ACON-style summary context if available from v4.
5. Optimized soft tokens with k in `{4, 8, 16, 32, 64}`.

### 7.4 Soft-token optimization

Script: `scripts/05_soft_token_oracle.py`

For each case and k:

1. Initialize k trainable embeddings:

```python
soft = torch.nn.Parameter(torch.randn(k, hidden_dim, device=device, dtype=torch.bfloat16) * 0.02)
```

Better initialization option:

```python
soft = mean_input_embedding + 0.02 * noise
```

2. Build input embeddings:

```text
[prompt without context prefix tokens]
[soft tokens]
[target JSON tokens]
```

Use `inputs_embeds` by concatenating:

```python
prefix_embeds = embed(input_ids_prefix)
target_embeds = embed(input_ids_target)
inputs_embeds = torch.cat([prefix_embeds, soft[None, :, :], target_embeds], dim=1)
```

3. Labels ignore prefix and soft tokens; compute CE only on target tokens.

4. Optimize only `soft`, freeze model.

```python
optimizer = torch.optim.AdamW([soft], lr=0.05, weight_decay=0.0)
for step in range(num_steps):
    loss = model(inputs_embeds=inputs_embeds, attention_mask=mask, labels=labels).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Suggested defaults:

```text
num_steps = 200
lr = 0.05
patience = 30
min_delta = 1e-4
```

Use fp32 master copy of soft embeddings if bf16 optimization is unstable.

### 7.5 Metrics

For each case:

```text
full_loss
no_context_loss
recent_loss
acon_loss
soft_loss_k4
soft_loss_k8
soft_loss_k16
soft_loss_k32
soft_loss_k64
```

Compute recovery of the full-vs-no-context gap:

\[
\mathrm{gap\_recovery}_k =
\frac{L_{no} - L_{soft,k}}{L_{no} - L_{full}}
\]

Clamp for reporting only to avoid weird negative values, but keep raw values too.

Also compute token-normalized loss and exact next-API-name accuracy if using next-action targets.

### 7.6 Outputs

```text
outputs/tables/soft_token_oracle_losses.csv
outputs/figures/fig_soft_token_gap_recovery.png
outputs/figures/fig_soft_token_loss_vs_k.png
```

### 7.7 Success criteria for Experiment C

Strong positive:

```text
k=16 or k=32 soft tokens recover >= 70% of the full-vs-no-context loss gap.
```

Medium positive:

```text
soft tokens with k<=32 beat ACON/recent text on target NLL.
```

Negative:

```text
even k=64 soft tokens do not recover the full-context loss gap.
```

If the soft oracle fails, the low-dimensional compression story is likely weak for this target/model setup.

---

## 8. Optional Experiment D — Gradient-ranked text spans as a sanity check

This is optional and should not be the main story.

Construct contexts under the same per-task budget as v4:

- `jacobian_high_spans`: greedy fill by `span_gxa_sqrtlen / token_count` or `span_gxa_sqrtlen`, test both.
- `jacobian_low_spans`: bottom-ranked spans.
- `jacobian_recent_hybrid`: 50% recent tail + 50% top Jacobian spans.

Then run the same MiniMax downstream agent as v4.

This checks whether gradient saliency is behaviorally useful. But even if it fails, Experiments B/C may still support the active-subspace framing.

Output:

```text
outputs/raw/jacobian_compressed_contexts.jsonl
outputs/raw/jacobian_behavior_runs.jsonl
outputs/tables/jacobian_span_downstream_results.csv
```

---

## 9. `run_all.sh`

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

PYBIN=${PYBIN:-/workspace/EASMO/.venv/bin/python}
MODEL_PATH=${QWEN_MODEL_PATH:-Qwen/Qwen3-4B-Instruct}
ROOT=${ROOT:-/workspace/EASMO/motivation_v6_jacobian}

cd "$ROOT"

$PYBIN scripts/01_build_cases.py \
  --v4_dir /workspace/EASMO/motivation_v4 \
  --out outputs/raw/cases.jsonl \
  --max_cases 30

$PYBIN scripts/02_compute_jacobian_saliency.py \
  --model_path "$MODEL_PATH" \
  --cases outputs/raw/cases.jsonl \
  --out outputs/raw/jacobian_span_scores.jsonl \
  --max_context_tokens 12000

$PYBIN scripts/03_compare_to_v4_sensitivity.py \
  --scores outputs/raw/jacobian_span_scores.jsonl \
  --out_dir outputs

$PYBIN scripts/04_active_subspace_spectrum.py \
  --model_path "$MODEL_PATH" \
  --cases outputs/raw/cases.jsonl \
  --out_dir outputs \
  --layer_index -1 \
  --max_context_tokens 12000

$PYBIN scripts/05_soft_token_oracle.py \
  --model_path "$MODEL_PATH" \
  --cases outputs/raw/cases.jsonl \
  --out outputs/tables/soft_token_oracle_losses.csv \
  --ks 4,8,16,32,64 \
  --num_steps 200 \
  --max_cases 30

$PYBIN scripts/06_aggregate.py --out_dir outputs
$PYBIN scripts/07_plot.py --out_dir outputs
$PYBIN scripts/08_write_report.py --out_dir outputs
```

Allow every script to accept `--max_cases` for debugging. First smoke test should use `--max_cases 2`.

---

## 10. Results report template

Write `outputs/reports/results_summary.md` with exactly these sections:

```markdown
# Motivation v6 Results: Jacobian Active-Subspace Diagnostics

## 1. Setup
- model path
- number of tasks
- max context tokens
- target type
- layer index

## 2. Experiment A: Jacobian saliency vs v4 finite-difference sensitivity
- global Spearman / Pearson
- median per-task Spearman
- top-k overlap enrichment
- recency / length correlation
- verdict

## 3. Experiment B: Active subspace spectrum
- cumulative variance at k=4/8/16/32/64
- high-vs-low sensitivity spectrum comparison
- verdict on low-rank hypothesis

## 4. Experiment C: Soft-token oracle
- loss table by k
- gap recovery by k
- comparison to full/no-context/recent/ACON
- verdict on compressibility upper bound

## 5. Negative findings
List exactly what failed or did not support the hypothesis.

## 6. Implications for method design
Only include implications supported by the above diagnostics.

## 7. Files of record
List all raw/tables/figures.
```

---

## 11. Interpretation rules

### If A and B are positive

Then we can claim:

> White-box local Jacobian geometry captures a real behavior-sensitive signal, and that signal lies in a low-dimensional active subspace.

This motivates a method based on active-subspace-preserving compression.

### If B and C are positive but A is weak

Then we can claim:

> The v4 finite-difference span signal was too noisy or too prompt-dependent, but the model still admits a low-dimensional continuous compression upper bound.

This motivates soft-memory or representation-level compression, not span selection.

### If A is positive but B/C are negative

Then gradients are useful for saliency, but not for a low-rank compression story. Do not force the low-rank claim.

### If all are negative

Then stop pursuing the Jacobian/low-rank formulation for this setup. The result is still valuable: it means the current AppWorld/Qwen target is not locally compressible in this way.

---

## 12. Paper-level hypothesis being tested

Do not write this as final paper claim yet. Treat it as a hypothesis:

> Agent context compression should preserve the active subspace of the downstream policy Jacobian. Natural-language summaries fail when they minimize semantic distortion while ignoring high-curvature decision directions.

The experiment should either support this, refine it, or kill it.

