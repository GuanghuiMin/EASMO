# motivation_v8 — Experimental design

## 0. One-paragraph framing

`motivation_v7` showed that ACON-style structured compression on
Qwen3-4B-Instruct-2507 + MiniMax-M2.5 has:

* Surface Dominance Index (SDI) = **0.96 / 0.99** (need-label is
  effectively statistical noise);
* cross-model Kendall τ on fact-type half-life = **0.491**;
* `AUTH_OR_ACCESS_TOKEN` as the universal repellor in both models;
* 79 % of iterative chains converge within 5 rounds.

But ACON is highly structured — its UTCO prompt template explicitly
asks for sections like `STATE RETAINED`, `COMPLETED`, etc. A natural
question is **whether the abstraction-prior phenomenon is ACON-
specific or generic to LLM compression**.

v8 removes ACON entirely and replaces it with **two general
compression prompts** (P1 task-aware, P2 task-agnostic, both written
from scratch per the v8 spec §7) on the same 30 cases / 233 facts /
150 quality-passed condition pairs from v7. v8 also adds a
**basin-of-attraction experiment** (Experiment 3) that runs 4
different initialisations through the same compressor and measures
whether different starting representations converge to similar
fixed-point fact compositions.

## 1. Five claims under test (spec §2)

| Claim | Statement |
|---|---|
| **A** | General compression retains facts by surface type more than by need (SDI > 0; Δ_need ≈ 0 for executable). |
| **B** | Repeated general compression converges to a fixed point within ≤ 6 rounds. |
| **C** | Fixed-point composition is surface-type biased (NARRATIVE recall > EXECUTABLE recall; AUTH_OR_ACCESS_TOKEN / API_SCHEMA in bottom AUSC). |
| **D** | Δ_need^∞ (fixed-point need shift) is small for executable facts. |
| **E** | Basin of attraction: different initialisations converge to similar fixed points (contraction ratio < 0.5). |

## 2. Models (spec §3)

| role | model | endpoint |
|---|---|---|
| Compressor A | `qwen3-4b-instruct-2507` | local vLLM `http://127.0.0.1:8000/v1` |
| Compressor B | `MiniMaxAI/MiniMax-M2.5` | `http://10.183.22.68:8005/v1` |
| Retention scorer | **cross-model**: Qwen scores MiniMax; MiniMax scores Qwen | |

Settings: `temperature=0.0`, `seed=42`, `max_tokens_compression=2048`,
`max_tokens_json_scoring=512`.

## 3. Data (spec §4 — fully reused from v7)

* `data/cases.jsonl`: 30 successful AppWorld dev trajectories.
* `data/fact_bank_filtered.jsonl`: 233 substring-grounded facts
  across 16 fact types (5 coarse groups).
* `data/need_conditions_validated.jsonl`: 300 quality-passed
  needed/unneeded condition rows (= 150 matched pairs across 150
  unique facts).
* Stage 00 reads these v7 paths directly via `data.load_v7_*` and
  caps trajectory length at 18,000 chars.

## 4. Prompt families (spec §7)

* **P1 `general_task_aware`** — system message describes the model as
  a "compression module for a tool-use agent" and the user prompt
  includes the downstream `condition_task`. Asks for plain-text
  output (no fixed schema) under a hard `max_chars` budget. **Uses
  `{condition_task}`**.
* **P2 `general_task_agnostic`** — same idea but the prompt does not
  reference any downstream task. Used as a task-agnostic control:
  the same physical compressed text serves as both needed and
  unneeded rows for matched-pair analysis.
* P3 `general_strict_extract_then_compress` — defined but **not run**
  in v8 primary. Reserved for future ablation.

The exact prompt text lives in `motivation_v8/prompts.py` and is
mirrored in `prompts/` markdown files. SHA256 of each rendered
template is recorded in `outputs/provenance/prompt_sha256.json`.

## 5. Experiments

### Experiment 1 — Single-round need sensitivity (Stage 03)

For each `(case, fact, condition, prompt_family, model)`:

```
context        = full_trajectory_text
condition_task = needed | unneeded
budget         = 1500
```

For **P1**, we issue the compressor call with the actual
`condition_task` per row. For **P2**, the same physical compressed
text is reused across the matched needed/unneeded rows (because P2
is task-agnostic by construction); we therefore precompute P2 once
per `(case, model)` and replicate the row.

Output: `outputs/raw/single_round_compressions.jsonl`.

### Experiment 2 — Iterative fixed-point compression (Stage 04)

For each of `N_ITER_CASES = 20` selected cases:

```
x_0 = full_trajectory_text (RAW_FULL)
x_{r+1} = C_{m,p,B}(x_r; q)        for r = 0 .. 5
```

* For P1: one chain per condition (`needed`, `unneeded`), so 2
  chains × model.
* For P2: a single `task_agnostic` chain per (case × model).

Target fact per case: chosen to **match v7's iterative target**
when possible (so v7 vs v8 is a direct A/B), else by priority order
`AUTH_OR_ACCESS_TOKEN > API_SCHEMA > RUNTIME_VARIABLE > FILE_PATH > EXACT_IDENTIFIER > ACTION_OUTCOME > …`.

Output: `outputs/raw/iterative_chains.jsonl`.

### Experiment 3 — Basin-of-attraction analysis (Stage 05)

For `N_BASIN_CASES = 12` cases × 2 models × P1 (needed) × 6 rounds × 4
initialisations:

| init_type | construction |
|---|---|
| `RAW_FULL` | `full_trajectory_text[:18K]` (same as Exp 2) |
| `DETAIL_HEAVY` | fact table (`[FACT_ID=…][TYPE=…] canonical_fact` lines) prepended to raw trajectory |
| `NARRATIVE_HEAVY` | one-shot P2 summary + last 30 % of trajectory |
| `FACT_TABLE_ONLY` | fact table only, no raw trajectory |

We **never** insert ACON-like section headings — the fact table uses
plain bullet text only.

Same target fact (and `condition_task` = needed) as Stage 04 for
each case, so the basin contraction is computed on matched
end-of-chain fact-retention sets.

Output appends to `outputs/raw/iterative_chains.jsonl` with
`init_type` tagged accordingly.

## 6. Retention scoring (Stage 06)

Per spec §12:

1. **Deterministic exact substring match** first. If exact, label =
   `exact`, retention_score = 1.0, match_type = `substring_exact`,
   LLM call **skipped**.
2. Otherwise: cross-model LLM scorer with the v8 retention prompt
   (`prompts.RETENTION_SCORER_TEMPLATE`).

We score:
* Single-round outputs against the **target fact** of the row.
* Iterative and basin outputs against **all facts in the case**
  at every round (so fixed-point composition can be computed across
  types).

Output: `outputs/raw/retention_scores.jsonl` — one row per
`(context_source, chain_id|row, round, fact_id)`.

## 7. Metrics (spec §13; computed in Stage 07)

### Claim A
* Δ_need by fact type, by (model, prompt_family).
* Surface dominance regression (`r2_need`, `r2_type`, `r2_both`, SDI).
* PIR (preference inversion rate).
* CRS (condition responsiveness score).

### Claim B / C
* Survival curve `S(round, fact_type)`.
* Half-life table.
* Hazard rate.
* AUSC (area under survival curve).
* Hierarchy rank by (model, prompt_family, init_type, condition_type).
* Cross-(model, prompt) Kendall τ / Spearman ρ.
* Fixed-point composition by fact type.

### Claim D
* Δ_need^∞ by fact type, per (model, prompt_family).

### Claim E
* Pairwise initial vs final distances between `(case × init_type)`
  pairs:
  - `fact_jaccard_distance` = 1 − Jaccard of retained-fact-id sets.
  - `fact_type_l1_distance` = L1 of normalised fact-type histograms.
  - `text_jaccard_distance` = 1 − token Jaccard (proxy for embedding).
* `contraction_*` = final / initial.

### Budget compliance
* `violation_rate`, `median_length`, `p90_length`, `p99_length`
  per (model, prompt_family, budget_chars) — and per init_type for
  iterative outputs.

## 8. Success criteria (spec §17)

Verdicts are reported by `scripts/09_write_report.py` and re-checked
in `docs/04_results_summary.md`.

* Claim A — STRONG if ≥3 of: `r2_need < 0.02`, SDI > 0.5,
  exec Δ_need < 0.15, ≥2 concrete types with |Δ_need| ≤ 0.05, PIR >
  0.20.
* Claim B/C — STRONG if ≥3 of: convergence rate ≥ 60 %, narrative
  half-life > executable, narrative recall − executable recall ≥
  0.10, AUTH_OR_ACCESS_TOKEN or API_SCHEMA in bottom-3 AUSC,
  cross-model τ > 0.35.
* Claim D — STRONG if ≥2 of: `|Δ_need^∞| < 0.15` for executable,
  `Δ_need^∞` smaller than narrative-vs-executable gap.
* Claim E — STRONG if ≥2 of: mean `contraction_fact_jaccard < 0.5`,
  final pairwise fact-Jaccard similarity ≥ initial + 0.2.

## 9. Reuse / deviations from spec

* Stages 00-02 **reuse v7 artifacts directly**; no re-extraction.
* `--reuse_v7_target` flag in Stage 04 picks the same iterative
  target fact per case as v7 wherever possible. This makes v7 vs v8
  a direct A/B test of prompt family at constant target.
* Plan B = full config; P3 (strict extract) ablation deferred.
* Secondary budgets {800, 2500} deferred (spec §8 says "only if
  primary pipeline stable").
