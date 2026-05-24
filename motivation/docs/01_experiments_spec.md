# Motivation Experiments — Validation of "Policy-Dependent Memory" Thesis

> **Purpose:** before any EASMO training, verify the central thesis empirically.
>
> **Gating role:** results from M2 and M3 decide whether the Spotlight pitch is viable.
> If they fail, fall back to Plan B (see `ICLR27_Memory_scope_and_structure.md` §6).
>
> **Total time:** ~3 weeks (target completion 2026-06-15)
>
> **Total cost:** ~$500-1000 (mostly OpenAI/Anthropic API for oracle search)

---

## Shared setup

### Agent pool (frozen, defined once and reused throughout the paper)

| Agent | Model | Scaffold | What makes it distinct |
|---|---|---|---|
| **A_react** | Qwen2.5-7B-Instruct | ReAct loop (Yao 2023) | Many short tool calls; reactive; minimal planning |
| **A_plan** | Qwen2.5-7B-Instruct | Plan-then-execute (decompose → execute) | Few precise queries; explicit subgoal tree |
| **A_cot** | Qwen2.5-7B-Instruct | Reflexion / self-critique (Shinn 2023) | Reasoning-heavy; revisits prior conclusions |

**Critical**: all three share the same base model so observed differences are *policy-induced*, not capability-induced.

### Data pool

| Source | # contexts | Length | Use |
|---|---|---|---|
| LongMemEval (subset) | 200 | ~8k tokens | Multi-session conversational |
| LoCoMo (subset) | 200 | ~12k tokens | Multi-session, richer factual/temporal |
| AppWorld (subset) | 100 | ~6k tokens | Tool-use trajectory |
| **Total** | **500** | | |

### Budget levels (sweep across these)

`B ∈ {128, 256, 512, 1024}` tokens

---

## M1: Oracle Memory Discovery

> **Goal:** establish that an oracle memory z*_A exists for each (C, A) pair and can be reliably found.

### Method

For each `(context C, agent A, budget B)`:

1. **Candidate generation**: Use GPT-4o (or Claude Sonnet) as a *behavior-aware selector* with the prompt:

   > "Given context C and agent description d_A, select sentences from C such that agent A will produce the same next action distribution as when given full C. Constraint: selection ≤ B tokens. Return ranked top-5 candidates."

2. **Behavior scoring**: for each candidate z_i, compute
   - Action-match rate at N=8 probe states (next-action top-1 match)
   - L_BI(z_i; C, A) = KL divergence between agent action distributions
3. **Pick winner**: argmax over action-match rate, ties broken by min L_BI.

### Pass criteria

- Oracle memory achieves action-match rate ≥ 85% on N=8 probe states (validates that an effective compression exists at this budget).
- Achieves task success ≥ 90% × full-context success.

### Outputs

- `oracle_memories.jsonl`: for each (C, A, B) → z*_A(C, B) + metrics
- Reusable artifact for M2, M3, and as a training signal for EASMO later.

### Cost & time

| Item | Estimate |
|---|---|
| API calls (GPT-4o for selector) | ~$300-400 |
| GPU hours (agent inference, Qwen2.5-7B × 3 scaffolds × 500 contexts × 4 budgets) | ~40 GPU-hours on a single A100/H100 |
| Wall time | 5–7 days |

---

## M2: Cross-Agent Memory Overlap *(KILLER EXPERIMENT)*

> **Goal:** show that oracle memories for different agents on the *same* context are structurally different, not just superficially.

### Method

Using `oracle_memories.jsonl` from M1:

1. **Token-level Jaccard**: for each (C, B), compute pairwise Jaccard of token sets in z*_{A_i}, z*_{A_j}.
2. **Sentence-level overlap**: same but on sentence units.
3. **TF-IDF cosine similarity**: between z*_{A_i} and z*_{A_j} as documents.
4. **Saliency map per agent**: for each (C, A), for each token t in C:
   - Compute `s_A(t) = D_KL(π_A(·|C) || π_A(·|C \ {t}))` (leave-one-out KL).
   - This is the **agent-specific saliency** of token t.
5. **Saliency rank correlation**: Spearman ρ between `s_{A_i}` and `s_{A_j}` over tokens in C.

### Pass criteria (Spotlight viability threshold)

- Mean pairwise token Jaccard **< 0.4** across all (A_i, A_j) pairs.
- At least 2 of 3 agent pairs have saliency Spearman ρ **< 0.5**.
- Saliency heatmaps are visually distinct (qualitative; 3 illustrative examples for Figure 1).

### Fallback criteria

- If mean Jaccard > 0.6 OR all ρ > 0.7 → **STOP and re-plan**. Drop the policy-dependence Spotlight pitch and switch to the Plan B framing (behavior-invariance + IB only).

### Outputs

- `overlap_matrix.csv` (3×3 table, mean and std per budget)
- `saliency_heatmaps/` (PNG per illustrative example)
- **Paper Figure 1, Panel (a)**: overlap matrix heatmap
- **Paper Figure 1, Panel (c)**: saliency heatmap (token-level)

### Cost & time

| Item | Estimate |
|---|---|
| Reuses M1 outputs; only saliency computation | 1 token-mask × full forward pass per token-context-agent triple |
| GPU hours (saliency: ~500 contexts × ~8k tokens × 3 agents) | ~60 GPU-hours (can downsample tokens) |
| Wall time | 4–6 days |

---

## M3: Cross-Agent Transfer Degradation *(KILLER EXPERIMENT)*

> **Goal:** show memory is not interchangeable: feeding A_j the memory designed for A_i causes a measurable, *quantitatively predicted* task drop.

### Method

1. **Cross-agent eval**: for each context C and each pair (A_i → A_j), feed z*_{A_i}(C, B) to A_j and measure:
   - (a) Next-action top-1 match rate against A_j(C) baseline
   - (b) Task success rate (end-to-end on the downstream eval task)
   - (c) Trajectory similarity (edit distance on tool-call sequences)
2. **Policy divergence proxy**: on a held-out set of reference states `S_ref`, compute
   ```
   δ(A_i, A_j) = E_{s ~ S_ref} [ D_TV(π_{A_i}(·|s), π_{A_j}(·|s)) ]
   ```
3. **Correlation**: plot task-drop on y-axis vs δ on x-axis; fit linear regression.

### Pass criteria

- Mean cross-agent task drop **> 15%** relative to same-agent baseline.
- Spearman ρ between δ and task drop **> 0.5** across budgets.
- Linear fit R² **> 0.5**.

### Fallback criteria

- If drop < 5% on average → policy-conditioning offers no practical benefit → Plan B.

### Outputs

- `transfer_results.csv`
- **Paper Figure 1, Panel (b)**: task-drop vs. policy-divergence scatter + fit
- This is the figure that turns "interesting overlap finding" into "real engineering consequence."

### Cost & time

| Item | Estimate |
|---|---|
| Cross-agent inference: 3 agents × 3 memory sources × 500 contexts × 4 budgets ≈ 18k runs | ~30 GPU-hours |
| Wall time | 3–4 days |

---

## M4 (stretch): Policy-Probing Diagnostic

> **Goal:** secondary evidence that memory is "agent-identifiable" — train a small classifier to predict the source agent from the memory alone.

### Method

- Train a TinyBERT / DistilBERT classifier on `(z*_A(C, B), A)` pairs from M1.
- 3-way classification: which agent produced this oracle memory?
- Report test accuracy and confusion matrix.

### Pass criteria

- Test accuracy **> 70%** (vs 33% random).
- Indicates oracle memories carry agent-specific signature that a tiny model can detect.

### Cost & time

- Trivial (~$0, 2 GPU-hours, 2 days incl. writing).

### Status in paper

- Goes in Appendix C as supplementary evidence, *not* in main results (M2 + M3 already carry the story).

---

## Summary table

| Exp | What it shows | Killer? | Time | Cost |
|---|---|---|---|---|
| M1 | Oracle memories exist & are findable | foundation | 5–7d | ~$400 |
| M2 | Oracle memories differ across agents | **YES** | 4–6d | ~$50 (reuses M1) |
| M3 | Cross-agent memory transfer hurts | **YES** | 3–4d | ~$50 (reuses M1) |
| M4 | Oracle memories are agent-identifiable | bonus | 2d | $0 |

---

## Output: the Figure 1 spec (the paper's spine)

A three-panel figure that sits on page 2:

- **Panel (a)** — 3×3 Jaccard overlap heatmap (M2)
- **Panel (b)** — task-drop vs policy-divergence scatter with linear fit (M3)
- **Panel (c)** — saliency heatmap on one illustrative context, 3 stripes for 3 agents (M2)

Caption template:
> *Figure 1.* Optimal memory is policy-dependent. (a) Oracle memories for three structurally different agents on the same contexts have token Jaccard below {X.XX}, far from the {Y.YY} we would expect under policy-agnostic compression. (b) Feeding agent A_j a memory optimized for A_i causes task success to drop by {Z}%, with drop magnitude tightly tracking policy TV divergence (R²={R}). (c) Token-level saliency maps differ qualitatively across agents on the same context. These observations falsify the implicit assumption in prior context-compression work that the consuming agent is exchangeable.

---

## Definition of done (for the whole motivation section)

- [ ] M1 oracle memory dataset built and validated
- [ ] M2 overlap matrix and saliency results in
- [ ] M3 transfer scatter and fit in
- [ ] Figure 1 drafted with all three panels
- [ ] §3 of the paper draft fully written based on these results
- [ ] **Decision gate met or not**: log result and proceed accordingly
