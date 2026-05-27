# motivation_v5 — Experimental Design

> Track: `EASMO/motivation_v5/`
> Spec: [`docs/00_spec.md`](00_spec.md) (symlink to
> `user_feedback/acon_appworld_failure_audit_motivation_experiment.md`)
> Snapshot: 2026-05-27 1:05 PM PT. All 12 stages complete; 24 audited cases.

## 0. One-paragraph framing

motivation_v3 showed structural-vs-behavioral ranking-mismatch.
motivation_v4 showed decision-state probing identifies meaningful
spans but recency is a strong baseline. **motivation_v5 is a
diagnostic audit**, not a method: we run the ACON-style compressor on
30 successful AppWorld dev trajectories, look at the cases where the
compressor's downstream agent fails, then have an *audit model* try
to recover the missing information from the full trajectory, then
re-run the compressor on the augmented context and ask whether the
audit-recovered information survives. The headline question is
whether ACON's bottleneck is the **recovered-then-dropped** pattern:
information that *exists* in the trajectory, that *can* be recovered,
but that the compressor *re-discards* the moment it sees a longer
input.

## 1. Claim under test

> **When ACON fails on AppWorld, what information was missing, what did
> an audit model recover or add back, and what did the compressor drop
> again after that?**

This produces concrete evidence for the real compression bottleneck
*before* designing a new method. The pipeline is intentionally
diagnostic: no training, no RL, no prompt-format hand-tuning.

## 2. Information flow being audited

```
full context trajectory                              (motivation_v3 baseline)
        ↓ ACON-style compressor                      (motivation_v3.acon_style_summary)
ACON compressed history
        ↓ audit model supplementation                (Qwen3-4B reads baseline + ACON,
audit-augmented context                                emits an [AUDIT_AUGMENTATION] block)
        ↓ same compressor again                      (MiniMax-M2.5 + ACON-style prompt)
recompressed context
        ↓ downstream agent                           (MiniMax-M2.5, same prompt as v3/v4)
final_after_recompression_success
```

The four audit prompts (spec §8–§11) measure four things along this
flow:

| Prompt | What it measures | Stage |
|---|---|---|
| 01 — case failure audit | What information was missing in ACON, root-cause taxonomy | 06a |
| 02 — addition audit | What did the audit model add back, and is it grounded? | 06b |
| 03 — recompression-loss audit | Which audit-added items did the recompressor drop? | 06c |
| 04 — verifier resolution | Does MiniMax support Qwen's claims? | 07 |

A fifth prompt (05) is an LLM-driven aggregator that writes
`motivation_summary.md` from the merged stats.

## 3. Scope

| | |
|---|---|
| Benchmark | AppWorld dev split |
| Source data | motivation_v3's 30 successful full-context dev trajectories (length-biased) |
| Cases | 60 raw (30 tasks × 2 budgets); after Tier 1+2 filter: **24 cases** |
| Tier 1 (baseline ✓ ∧ ACON ✗) | 24 cases (all 24 sampled cases are Tier 1) |
| Tier 2 (both ✓ ∧ step_ratio ≥ 1.5) | 0 cases at this snapshot |
| ACON variant | `prompting` — motivation_v3's `acon_style_summary` prompt |
| Compression type | `history` (no observation compression in this round) |
| Difficulty bucket | all 24 are `hard` (length ≥ 16 baseline iters by v3 selection) |
| Models | Qwen3-4B (auditor) + MiniMax-M2.5 (compressor + downstream + verifier) + rule-based grounding (deterministic) |
| Wall-clock | 16 minutes for the full 12-stage pipeline (n=24) |
| LLM cost | 24 (augmenter) + 24 (recompressor) + 72 (3 audit passes) + 24 (verifier) + 24 (downstream agent runs) = **168 LLM calls + 24 agent cells** |

## 4. Pipeline (12 stages)

| Stage | Script | Output | Wall-clock |
|---|---|---|---|
| 01 | `01_build_raw_cases.py` | `data/raw_cases.jsonl` (60 rows) | <1s |
| 02 | `02_sample_cases.py` | `data/sampled_cases.jsonl` (24 rows, Tier 1+2) | <1s |
| 03 | `03_build_audit_augmented.py` | adds `audit_augmented_context` (Qwen call per case) | 31s |
| 04 | `04_recompress.py` | adds `recompressed_context` (MiniMax recompressor) | 1m51s |
| 05 | `05_rerun_downstream.py` | adds `final_after_recompression_success` (downstream agent) | 6m32s |
| 06 | `06_run_audit.py` | 3 audit JSONLs (case / addition / recompression) | 2m57s |
| 07 | `07_run_verify.py` | MiniMax verifier + rule-based grounding | 3m22s |
| 08 | `08_merge_audits.py` | `outputs/raw/merged_case_audits.jsonl` | <1s |
| 09 | `09_aggregate.py` | 5 spec tables in `outputs/tables/` | <1s |
| 10 | `10_plot_figures.py` | 3 spec figures (PDF + PNG) | <1s |
| 11 | `11_write_per_case.py` | 24 markdown reports in `outputs/reports/per_case_markdown/` | <1s |
| 12 | `12_write_motivation.py` | `outputs/reports/motivation_summary.md` (LLM-generated) | 33s |

Top-level orchestrator: `scripts/run_all.sh`. Total ~16 min on the
shared vLLM endpoints.

## 5. Models

Spec §3 prescribed Qwen3-4B as primary auditor + MiniMax-2.5m as
verifier. We provision both:

| Role | Model | Endpoint | Purpose |
|---|---|---|---|
| Primary auditor | **Qwen3-4B** (`qwen3-4b`) | `http://127.0.0.1:8000/v1` (local vLLM, started via `/workspace/qwen3-vllm/serve.sh`) | Stage 03 augmenter, Stage 06 (3 audit passes) |
| Verification auditor | **MiniMax-M2.5** | `http://10.183.22.68:8005/v1` (shared) | Stage 07 (resolution), Stage 12 (motivation_summary) |
| Recompressor | **MiniMax-M2.5** | same | Stage 04 |
| Downstream agent | **MiniMax-M2.5** | same | Stage 05 |
| Rule-based grounding | n/a (deterministic substring) | local | Stage 07 — independent of any LLM |

Important: the rule-based grounding verifier is **stronger than an LLM
cross-vote** because it does literal text matching against the source
contexts. We use it alongside the spec's MiniMax LLM verifier; the
two together give one LLM-vote-based and one no-LLM-bias score per case.

Generation hyperparameters per spec §3.3:
* temperature 0.0 for all audit / verifier / augmenter calls
* temperature 0.2 for the recompressor (matches v3's
  `acon_style_summary` setting)
* JSON mode where the model supports it; otherwise post-hoc parsing
* `max_tokens` 2048 for case / addition / recompression audits;
  1024 for the augmenter and verifier; 4096 for the aggregator

## 6. Data layout (per spec §5)

```
motivation_v5/
├── docs/                              v2-style explanatory docs
│   ├── 00_spec.md                     → user_feedback spec
│   ├── 01_experimental_design.md      this file
│   ├── 02_audit_prompts.md            verbatim prompts (paper appendix)
│   ├── 03_failure_taxonomy.md         16-label taxonomy + definitions
│   ├── 04_pipeline_setup.md           case selection + augmenter + recompressor
│   └── 05_results_summary.md          decision-ready snapshot with final numbers
├── prompts/                           5 prompt templates (jinja-style {{name}})
│   ├── 01_case_failure_audit.md
│   ├── 02_audit_addition_audit.md
│   ├── 03_recompression_loss_audit.md
│   ├── 04_verifier_resolution.md
│   └── 05_aggregate_summary.md
├── motivation_v5/                     python package
│   ├── data.py                        case I/O + v3 reuse
│   ├── clients.py                     Qwen + MiniMax OpenAI clients + JSON parser
│   ├── audit.py                       4 audit prompt runners
│   ├── compressor.py                  ACON-style recompressor (reuses v3 prompt)
│   ├── augmenter.py                   audit-augmented context builder
│   ├── rule_verify.py                 deterministic grounding checks
│   └── runner.py                      downstream agent for stage 05
├── scripts/                           12 stage scripts + run_all.sh
├── data/
│   ├── raw_cases.jsonl                60 rows (30 tasks × 2 budgets)
│   └── sampled_cases.jsonl            24 rows (Tier 1+2)
└── outputs/
    ├── raw/                           7 JSONL files (audits + verifications + merged)
    ├── tables/                        5 CSV files (spec §13.2/§13.3 + 4 metrics)
    ├── figures/                       3 figures × {PDF, PNG}
    ├── reports/                       motivation_summary.md + 24 per_case_markdown/*.md
    └── sprint_logs/
```

## 7. Headline answers (preview; full §10 in `05_results_summary.md`)

| Q | Answer |
|---|---|
| Q1: Dominant ACON failure modes? | **LOST_API_SCHEMA_OR_PARAMETER (12/24, 50%)** is the modal mode; LOST_ACTION_OUTCOME (4), LOST_ACCESS_TOKEN (3), LOST_ENVIRONMENT_STATE (2), STALE_OR_CONFLICTING_STATE (2). 100% of cases tagged compression-caused. |
| Q2: What does the audit model add back? | **28 grounded actionable items** across 24 cases (≈ 1.2/case). Top categories: `runtime_variable` (access tokens, IDs), `action_outcome`, `auth`, `pending_task`. |
| Q3: What does the compressor drop again? | **26 of those 28 grounded items (~93%) are dropped by recompression**. Top dropped categories: `action_outcome` (10), `environment_state` (5), `negative_evidence` (4). Top reasons LLM compressor cited: `over_abstraction` (11), `looked_like_past_log` (10), `verbosity_pressure` (4). |
| Q4: Concrete motivation claim? | **Recovered-then-dropped bottleneck holds**. Re-running the downstream agent with the recompressed context recovers **5/24 (21%)** of originally failed cases — only because the recompressed context still preserved a few audit-recovered items by chance. The other 19 still fail, mainly because the compressor over-abstracted them away. |

## 8. Caveats

* **Qwen3-4B reliability scores average 0.15** — the auditor is
  honest about its own uncertainty. This is why we run a verifier
  (MiniMax) AND a rule-based grounding check; the rule-based score
  averages 0.444, meaning roughly half of Qwen's quoted evidence can
  be literal-matched in the source contexts.
* **Qwen vs MiniMax primary-mode agreement is only 42%** at n=24.
  Both models broadly agree compression caused the failures (50%
  agreement on `is_compression_caused`) but disagree on which
  specific failure-mode label applies. This is partly because Qwen
  emits labels outside the spec taxonomy (e.g.
  `LOST_ACCESS_TOKEN` instead of `LOST_AUTH_OR_ACCESS_TOKEN`).
* **All 24 cases are `hard` difficulty** by our `_difficulty_bucket`
  (≥16 baseline iters). v3's case-selection biased toward longer
  trajectories; the difficulty axis does not differentiate within
  this set. Future work should add easy / medium cases by running
  full-context on shorter dev tasks.
* **No AGENT_REASONING_FAILURE cases (0/24).** This is suspicious — it
  is more likely that Qwen's auditor is *biased toward* attributing
  failures to compression, given it sees the ACON summary explicitly
  framed as the suspect. A balanced auditor would label some cases
  as reasoning failures. The MiniMax verifier disagrees on
  compression-causality for 50% of cases, supporting this caveat.
* **`final_after_recompression_success = 5/24` at cap=15** is the
  cleanest behavioral evidence for the recovered-then-dropped
  mechanism, but it's also small (n=24) and the
  baseline-vs-recompressed gap mixes (a) the audit model adding
  useful state and (b) the recompressor dropping some of it. We
  cannot from this experiment alone separate "audit added value" from
  "recompressor preserved value"; both contribute.

## 9. Reproduction

```bash
ACONPY=/workspace/acon/.venv/bin/python
PYBIN=/workspace/EASMO/.venv/bin/python

# Prerequisite: Qwen3-4B vLLM server on port 8000
nohup bash /workspace/qwen3-vllm/serve.sh \
  > /workspace/qwen3-vllm/server.log 2>&1 &

# Run the full 12-stage pipeline (~16 min wall-clock at n=24)
cd /workspace/EASMO/motivation_v5
bash scripts/run_all.sh
```

Each stage is idempotent given its inputs in `data/` and
`outputs/raw/`. Re-run individual scripts as needed.
