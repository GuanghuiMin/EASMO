# motivation_v5 — Pipeline setup (case selection, augmenter, recompressor)

> Spec reference: §4 (data + case selection), §1 (information flow).
> Implementation: `motivation_v5/data.py`, `motivation_v5/augmenter.py`,
> `motivation_v5/compressor.py`, `motivation_v5/runner.py`.

## 1. Case provenance

Every audited case in v5 is built from motivation_v3's existing
artifacts — no new agent runs are spent on producing the
baseline / ACON contexts:

| Case field (spec §4.1) | Source in v3 |
|---|---|
| `baseline_history` | rendered text of v3 stage-01 successful direct trajectory (`motivation_full_trajectories.jsonl` → `output_dir/appworld_trajectory.json`) |
| `acon_compressed_history` | v3 stage-02 `acon_style_summary` text (`motivation_compressed_contexts.jsonl[method=acon_style_summary]`) |
| `acon_full_trajectory` | rendered text of v3 stage-05 ACON downstream run (`motivation_behavior_runs.jsonl[method=acon_style_summary]` → `output_dir/env_history.json`) |
| `acon_success` | v3 stage-05 `success` field |
| `baseline_env_steps` | v3 stage-01 `iterations` |
| `acon_env_steps` | v3 stage-05 `iterations` |
| `step_ratio` | `acon_env_steps / baseline_env_steps` |
| `failure_report` | v3 stage-05 `termination_reason` (only when `acon_success=False`) |
| `audit_augmented_context` | **NEW (v5 stage 03)** — Qwen3-4B builds it from baseline + ACON |
| `recompressed_context` | **NEW (v5 stage 04)** — MiniMax recompresses the augmented context |
| `final_after_recompression_success` | **NEW (v5 stage 05)** — downstream agent run with recompressed context |

We use v3's `acon_style_summary` as the "ACON compressor" because:

* It implements the structured-section ACON prompt (TASK STATE /
  IDENTIFIERS / BINDINGS / ACTION OUTCOMES / CONSTRAINTS), the same
  shape as ACON's published Listing 3.
* It is the closest faithful proxy to ACON without re-running ACON's
  iterative prompt-tuning loop.
* All 30 v3 dev tasks already have it computed and downstream-run
  with `max_steps ∈ {15, 8}`, giving us a free 60-cell raw pool.

The variant tag is `acon_variant = "prompting"` (per spec §4.1).
Compression type is `history` (no observation compression).

## 2. Tier filtering (spec §4.2)

```text
Tier 1: baseline_success = True AND acon_success = False
Tier 2: baseline_success = True AND acon_success = True BUT step_ratio >= 1.5
Tier 3: acon_success = False AND audit_augmented_context exists           (built per case)
Tier 4: audit_augmented_context improves trajectory BUT recompressed_context fails again
```

By construction every v3 case has `baseline_success = True` (v3 only
kept successful direct trajectories) so the relevant filter is

```python
sampled = (Tier 1) ∪ (Tier 2)
        = [c for c in raw_cases
           if (not c["acon_success"])
              or (c["acon_success"] and c["step_ratio"] >= 1.5)]
```

dedup by `case_id` (which is `task_id_cap{N}`).

Empirical counts on this snapshot (from `01_build_raw_cases.py` log):

| Set | n |
|---|---:|
| raw cases (30 tasks × 2 budgets) | 60 |
| Tier 1 (baseline ✓ ∧ ACON ✗) | 24 |
| Tier 2 (both ✓ ∧ step_ratio ≥ 1.5) | 0 |
| **sampled (T1 ∪ T2)** | **24** |
| Tier 3 (sampled ∩ audit_augmented exists) | 24 (filled by stage 03) |
| Tier 4 (sampled ∩ recomp_fail) | 19 (filled by stage 05; 5 succeed) |

All 24 sampled cases are Tier 1. Tier 2 is empty in this snapshot
because v3's `acon_style_summary` either succeeds with a
step_ratio < 1.5 or it fails entirely.

The `02_sample_cases.py` script writes `data/sampled_cases.jsonl`
which is the input to all downstream stages. Stages 03/04/05 mutate
the file in place to add `audit_augmented_context`, `recompressed_context`,
and `final_after_recompression_*` fields respectively.

## 3. Audit augmenter (Stage 03)

Goal: read `baseline_history` + `acon_compressed_history`, emit a
bracketed `[AUDIT_AUGMENTATION]` block of grounded actionable items
that ACON dropped. The augmented context is `ACON summary +
augmentation block` (the augmenter does NOT rewrite the ACON
summary — see [`02_audit_prompts.md`](02_audit_prompts.md) §7 for the
exact prompt).

Why a *block append* rather than a free rewrite:

1. Stage 06's "addition audit" needs a clean ACON-vs-augmented diff.
   If the augmenter rewrites, every word becomes a candidate
   addition.
2. Stage 06's "recompression audit" then asks "what got dropped from
   the augmentation block when MiniMax recompressed" — this requires
   the block to be discoverable as a unit.
3. The block format gives the recompressor a clear signal that "this
   is content the agent might actually need" — and yet the
   recompressor still drops 93% of it. That is exactly the
   recovered-then-dropped failure mode.

Item categories used by the augmenter (slightly different from the
audit-prompt category list because the augmenter writes free-text
items, not structured fields):

```text
RUNTIME_VARIABLE   AUTH_CREDENTIAL   API_SCHEMA   ENVIRONMENT_STATE
ACTION_OUTCOME     PENDING_SUBTASK   NEGATIVE_EVIDENCE   GUARDRAIL   OTHER
```

Hard rules in the augmenter prompt (verbatim):

```text
1. Every item must be backed by a verbatim string from the baseline trajectory.
2. Do not invent IDs, tokens, file paths, or numerical values.
3. Skip items already present in the ACON summary.
4. Keep items short (one line, <= 200 chars).
5. Order items by criticality (most critical first).
6. Output at most 12 items.
```

In practice the augmenter emits 1–8 items per case; the median is 4.

## 4. Recompressor (Stage 04)

Goal: re-run the same ACON-style compressor on the augmented context
to simulate "ACON summarises a context that already includes the
audit-recovered items".

The recompressor is a thin wrapper around v3's
`COMPRESS_ACON_STYLE` prompt (motivation_v3.prompts) — same model
(MiniMax-M2.5), same prompt, same temperature 0.2. The only
difference from v3 is that the input "trajectory" is now `ACON
summary + audit augmentation block` instead of a raw trajectory.

This is intentional: we want to test exactly the scenario "the
compressor is given more useful structured input — does it preserve
that input or collapse it?". A fresh prompt-tune of the compressor
would conflate "compressor was retrained" with "compressor is
inherently lossy on augmented input". Reusing v3's ACON-style prompt
keeps that contrast clean.

## 5. Downstream rerun (Stage 05)

Goal: feed `recompressed_context` to the same MiniMax-M2.5 downstream
agent and observe `final_after_recompression_success` per spec §1.

The runner reuses `motivation_v3.runner.run_with_compressed_context`
(via `motivation_v5.runner.run_recompressed_downstream`) so the
downstream prompt and budgeting are *identical* to v3's
`acon_style_summary` runs. Only the `compressed_context` placeholder
content differs.

The tag namespace `mv5_recomp` ensures trajectory output directories
don't collide with v3's `mv3_run_*` directories. Output paths look
like:

```
acon/experiments/appworld/outputs/MiniMaxAI_MiniMax-M2.5_mv5_recomp_recompressed_context_cap{N}/dev/task_{tid}/
```

`max_steps` is inherited from the case's original `budget_max_steps`
(15 or 8) so the recompressed run is held under the same budget as
the original ACON run.

## 6. Verification (Stage 07)

Two parallel verifiers per spec §3.3 + user request:

### 6.1 MiniMax LLM verifier

Triggered for cases where any of the following hold:
* Qwen `case_audit.reliability_score < 0.7`
* Qwen flags `recompression_audit.recompression_judgment.drops_critical_audit_recovered_info = True`
* a 20% random sample of the remaining cases (seed 42)

In this run all 24 cases triggered the reliability rule, so MiniMax
verified all 24. The verifier consumes the spec §11 prompt
(`prompts/04_verifier_resolution.md`) and emits whether Qwen's claims
are supported, plus a corrected `verified_primary_failure_mode` and
`verified_is_compression_caused`.

### 6.2 Rule-based grounding verifier

A deterministic substring check (`motivation_v5/rule_verify.py`) that
runs on every case (no LLM cost). For each Qwen-claimed item, it
verifies:

| audit | what we check |
|---|---|
| **case audit** | `baseline_evidence` quote is a substring of `baseline_history`; `acon_absent_or_distorted_evidence` is/isn't in ACON as claimed |
| **addition audit** | `baseline_evidence` is in baseline; `audit_augmented_excerpt` is in the augmented context |
| **recompression audit** | `audit_augmented_excerpt` is in the augmented context AND is **not** in the recompressed context |

The `overall_grounding_score` is the unweighted mean of per-audit
grounding rates. Score 1.0 = every quote literal-matches; 0.0 = no
quotes match. Empirical mean across 24 cases: **0.444**.

Because rule-based grounding is deterministic and uses no LLM, it is
**stronger evidence than the MiniMax LLM verifier** for the question
"are Qwen's evidence quotes real?" (LLM verifiers can be fooled by
paraphrase; substring match cannot). We report both alongside in
`outputs/tables/model_agreement.csv`.

## 7. Aggregation + reporting (Stages 08–12)

* Stage 08 merges all 4 audit sources into one row per case (spec §13.1
  schema). Output: `outputs/raw/merged_case_audits.jsonl`.
* Stage 09 emits 5 CSVs per spec §13–§14:
  `failure_mode_counts.csv`,
  `audit_added_facts.csv`,
  `recovered_then_dropped.csv`,
  `critical_info_loss.csv`,
  `model_agreement.csv`.
* Stage 10 plots 3 figures per spec §15:
  `fig_failure_mode_bar.{pdf,png}`,
  `fig_recovered_then_dropped_bar.{pdf,png}`,
  `fig_information_flow_sankey.{pdf,png}` (rendered as a stacked-bar
  proxy because matplotlib doesn't ship a built-in Sankey;
  alternative: install `holoviews` for a true Sankey if needed).
* Stage 11 writes one Markdown per case under
  `outputs/reports/per_case_markdown/` for manual qualitative review.
* Stage 12 calls MiniMax with the spec §12 aggregator prompt to write
  `outputs/reports/motivation_summary.md`. If the LLM call fails or
  returns short text, a deterministic template version is written
  using the same merged stats.

## 8. Quick numbers (n=24)

| Knob | Observed |
|---|---:|
| sampled cases | 24 |
| augmenter elapsed (median) | ~4s/case |
| recompressor elapsed (median) | ~5s/case |
| downstream rerun elapsed (median) | ~16s/case (cap=8 / cap=15) |
| 3-pass audit elapsed (per case) | ~7s + ~7s + ~7s |
| total wall-clock (12 stages, mostly serial) | 16 minutes |
| `final_after_recompression_success` | 5/24 (21%) |
| total grounded audit-added items | 28 |
| total recovered_then_dropped items | 26 |
| recovered_then_dropped rate | 93% |
| qwen mean reliability_score | 0.15 |
| rule-based mean grounding_score | 0.444 |
| qwen↔minimax primary-mode agreement | 42% |
| qwen↔minimax `is_compression_caused` agreement | 50% |
