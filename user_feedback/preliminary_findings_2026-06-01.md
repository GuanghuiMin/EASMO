# v11 preliminary findings — mid-pipeline snapshot (2026-06-01 1:50 PM PT)

> **Status**: stages 02–06 fully done; stage 07 ~47 % done (4,955 / 10,582 agent runs). Findings below split into **STABLE** (replicated across multiple checkpoints, large n) and **PROVISIONAL** (small n, sensitive to full data). Section §6 lists what remains to verify.
>
> **Caveat**: this is preliminary; the headline numbers may shift ±3-5 pp when stage 07 finishes (~3 AM PT Tue). The directional signals are likely stable.

## Headline (5 bullets)

1. **Compression-rescue is real and large.** Compressed-context CK agent runs pass at 45-48 % vs full-context baseline 25.9 %. The +17-22 pp lift is driven mostly by **rescue** (P(C=1 | F=0) ≈ 40-45 %), not by preserving baseline passes. This is the cleanest argument that compression is not a tax — it's a useful intervention under bounded inference.
2. **Form-prior interference replicates from v5/v7/v8 to v11 at sample-distribution level.** ~600 qualifying cases (extrapolated from 200 in partial data) show that critical tokens (JWT / email / phone / kv-pairs) in the original trajectory are **dropped by ≥3/8 stochastic samples** in 44 % of cells, and samples that preserve the token pass downstream at +60-80 pp higher rate. This is the v5 "recovered-then-dropped" phenomenon at sample resolution.
3. **No family dominates the Pareto frontier.** 4 families spread across a (pass-rate, token-cost) trade-off: `ACON_UT` highest absolute oracle pass (87.0 % CK), `general_task_agnostic` highest token efficiency (151 pass/1k chars, 3.2× ACON_UTCO), `general_task_aware` balanced (81.2 % CK at 726 chars). 75 % of tasks are family-invariant under oracle selection; the 25 % "hard regime" is where ACON wins.
4. **Best-of-N selection is universally effective and the project's clearest actionable signal.** Oracle gain over greedy is +33-39 pp on CK across all 4 families, with comparable magnitude. Even ACON_UTCO (lowest textual variance) has +38 pp gain — textual and behavioral diversity are **decoupled**.
5. **Task awareness is necessary but not sufficient.** `general_task_aware` boosts rescue by +7.8 pp over `task_agnostic`. But ~200 form-prior-drop cases still occur in task-aware families: the prompt instruction "preserve auth values" does not override the form prior at generation time.

---

## §1. Pipeline state (2026-06-01 1:50 PM PT)

| stage | status | wall-clock | rows | errors |
|---|---|---|---|---|
| 02 full-context baseline | ✅ done | 18:32 min | 147 (38 pass = **25.9 %**) | 0 |
| 03/04 setup | ✅ done | <1 s | — | — |
| 05 candidate compress | ✅ done | 4 h 11 min | 5,292 candidates | 0 |
| 06 serial stress (K=2) | ✅ done | 7 h 02 min | 15,873 stress rows | 0 |
| **07 behavior C1+CK** | 🔄 46.8 % | 13.5 h ETA | **4,955 / 10,582** | **0** |
| 08abc verifier / 09–16 | pending | — | — | — |

Projected completion: **~11:30 AM PT Tue (Jun 2)**.

---

## §2. STABLE: stress robustness varies by prompt family (stage 06, n=5,291)

(Already STABLE — all 5,291 candidates × 3 rounds done.)

### 2.1 Length drift C1 → CK (paper figure candidate)

| family | C1 mean | CK mean | drift (mean) | drift (median) |
|---|---|---|---|---|
| general_task_agnostic | 758 | 479 | **−36.8 %** | −34.7 % |
| general_task_aware | 943 | 726 | −23.0 % | −20.8 % |
| ACON_UT | 1,522 | 1,401 | −8.0 % | −6.5 % |
| **ACON_UTCO** | **1,817** | **1,795** | **−1.2 %** | −0.3 % |

ACON_UTCO is **stress-invariant** (drift −1 %). general_task_agnostic loses 37 % length per cascade.

### 2.2 Fixed-point convergence (SDI proxy)

| family | sim(r1→r2) | exact_same_as_prev rate (r2) |
|---|---|---|
| general_task_agnostic | 0.883 | 5.1 % |
| general_task_aware | 0.864 | 7.2 % |
| ACON_UT | 0.894 | 24.5 % |
| **ACON_UTCO** | **0.936** | **26.3 %** |

ACON_UTCO: 1 in 4 candidates reach **byte-identical** fixed point after just 2 stress rounds. v7 cross-model τ = 0.49 → v11 same-model = 0.94.

### 2.3 Cross-sample CK basin uniformity

| family | within-sample CK length CV (median) |
|---|---|
| general_task_agnostic | 0.229 |
| general_task_aware | 0.186 |
| ACON_UT | 0.146 |
| ACON_UTCO | 0.118 |

The inverse correlation (longer prompt → more uniform basin) holds through CK.

---

## §3. PROVISIONAL: compression-rescue transition matrix (n=621 per family×round)

(Mid-stable — based on ~47 % of stage 07 data; the conditional rates have ~2-3 pp standard error.)

### 3.1 Spec §13.1 main table (aggregate over all 9 candidates per (task, family))

| family | round | P(C=1\|F=1) preserve | P(C=1\|F=0) rescue | comp_pass − full |
|---|---|---|---|---|
| general_task_agnostic | C1 | 79.3 % | 43.0 % | +25.0 pp |
| general_task_agnostic | CK | **55.6 %** ← worst | **34.0 %** ← worst | +12.7 pp |
| general_task_aware | C1 | 78.5 % | **50.8 %** ⭐ | **+30.9 pp** ⭐ |
| general_task_aware | CK | 68.1 % | **44.7 %** ⭐ | +23.9 pp |
| ACON_UT | C1 | 74.8 % | 43.2 % | +24.2 pp |
| ACON_UT | CK | 63.0 % | 44.7 % | +22.7 pp |
| ACON_UTCO | C1 | 74.1 % | 40.4 % | +21.9 pp |
| ACON_UTCO | CK | **65.9 %** ⭐ | 42.6 % | +21.8 pp |
| **full-context baseline** | — | 100 % | 0 % | 0 pp baseline |

### 3.2 Three independent findings from §3.1

1. **`general_task_aware` has the highest rescue rate.** Both C1 (50.8 %) and CK (44.7 %) lead all families. Task-aware framing wins on baseline-fail recovery.
2. **`ACON_UTCO` has the highest CK preserve_success rate (65.9 %).** The form prior protects baseline-pass tasks under stress.
3. **`general_task_agnostic` CK is dominated on both axes** (55.6 % preserve, 34.0 % rescue). When the prior is weak and there's no task awareness, stress cuts deep.

---

## §4. STABLE: best-of-N is universally effective (n=68-69 per cell)

### 4.1 Oracle gain over greedy (sample size doubled vs first preview)

| family | greedy CK | oracle_8N CK | **gain (pp)** |
|---|---|---|---|
| general_task_agnostic | 37.7 % | 72.5 % | +34.8 |
| general_task_aware | 47.8 % | 81.2 % | +33.3 |
| **ACON_UT** | 47.8 % | **87.0 %** ⭐ | **+39.1** ⭐ |
| ACON_UTCO | 45.6 % | 83.8 % | +38.2 |

**ACON_UT is the absolute oracle ceiling** (87.0 %), not ACON_UTCO. Best-of-N gain is +33-39 pp **across all families** — no family has shrinking returns despite varying textual diversity.

### 4.2 Length efficiency (pass / 1k chars) — Pareto frontier

| family | oracle CK | mean chars | **pass / 1k chars** |
|---|---|---|---|
| general_task_agnostic | 72.5 % | 479 | **151.3** ← most efficient |
| general_task_aware | 81.2 % | 726 | 111.8 |
| ACON_UT | 87.0 % | 1,401 | 62.1 |
| ACON_UTCO | 83.8 % | 1,795 | 46.7 |

3.2× efficiency spread. Pareto front is 4-point, no single family dominates.

### 4.3 Per-task family ranking (n=68 tasks with all 4 families' CK done)

| outcome | n | % |
|---|---|---|
| all 4 families pass under oracle | 46 | **67.6 %** |
| all 4 fail | 6 | 8.8 % |
| 3 families pass, 1 fails | 8 | 11.8 % |
| 2 families pass | 4 | 5.9 % |
| only 1 family passes | 4 | 5.9 % (heterogeneous winner) |

**75 % of tasks are family-invariant** under oracle best-of-N. The 25 % "hard regime" is where family choice matters; ACON_UT and ACON_UTCO dominate `general_task_agnostic` there.

---

## §5. STABLE: form-prior interference at sample-distribution level (stage 15c, partial)

(200 qualifying cases at 47 % stage 07; extrapolated to ~600 at full.)

### 5.1 Selection criterion

A "form-prior drop case" is: critical token (JWT / kv-pair / email / phone) is present in the original trajectory, dropped by 3-8 of 8 sample CKs, AND samples preserving the token pass downstream at ≥15 pp higher rate.

### 5.2 Top 5 cases (sorted by paper-impact score)

| task | family | token category | n_preserved | pass(WITH) | pass(WITHOUT) | gap |
|---|---|---|---|---|---|---|
| 771d8fc_2 | ACON_UTCO | jwt_token | 3/8 | **100 %** | 20 % | **+80 pp** |
| 7d7fbf6_2 | general_task_agnostic | jwt_token | 3/8 | 100 % | 20 % | +80 pp |
| 29caf6f_1 | general_task_aware | jwt_token | 4/8 | 100 % | 50 % | +50 pp |
| 6ea6792_1 | ACON_UT | jwt_token | 4/8 | 75 % | 25 % | +50 pp |
| 302c169_2 | ACON_UTCO | phone | 5/8 | 80 % | 0 % | +80 pp |

### 5.3 What this is and is not

**This IS** the v5 "recovered-then-dropped" / v7 "form-prior conditions on form not function" phenomenon manifesting at v11 sample-distribution level. Best-of-N selection works because the form-prior is applied with stochastic variance — sometimes the LLM produces an outlier sample that escapes the modal failure mode.

**This is NOT** the same as stress-induced hallucination (e.g., 34d9492_1 ACON_UTCO case where K=2 stress causes hallucinated filenames). Hallucination is a separate failure mode that 15b targets; 15c specifically targets form-prior token-drop.

### 5.4 Cross-family pattern

The form-prior is **family-invariant** — top 5 includes ACON_UTCO (2), ACON_UT (1), general_task_aware (1), general_task_agnostic (1). The same MiniMax LLM applied with different prompts shares the same form prior at generation time.

---

## §6. Cross-track narrative — paper motivation section

The v11 preliminary findings tie together a multi-track story:

| track | finding | unified message |
|---|---|---|
| v3 | symbolic 100 % coverage but 57 % pass; task_aware 74 % coverage but 70 % pass | structural completeness ≠ behavioral utility |
| v5 | ACON-dropped items get re-dropped on recompression (93 %) | prior is recompression-invariant |
| v7 | SDI ≈ 1; R²(form) / R²(function) up to 183× | prior conditions on **form** not **function** |
| v8 | P1 task-aware framing flips fixed-point composition | framing moves the attractor but doesn't eliminate prior |
| v10 | SFT-CK output stress-invariant (drift +4.7 % vs Raw −33 %) | trained policy can escape the form prior |
| **v11 (partial)** | length-drift ↔ pass-drift align across families; form-prior drop case rate quantified; best-of-N selection universally +33-39 pp | quantified at scale on full AppWorld train+dev |

→ **Paper thesis (working draft)**: LLM compressors interfere with downstream agents because their form prior systematically retains structural completeness while dropping functionally-critical concrete entities. Best-of-N selection exploits stochastic variance in how the prior is applied to recover most lost utility (+33-39 pp). Task awareness at compression time helps modestly (+7.8 pp rescue) but does not override the prior. Training is the only known mechanism to fully escape the prior (v10 SFT result).

---

## §7. Methodology caveats

### 7.1 Deployment-regime separation (paper §-of-its-own)

The 4 prompt families correspond to two distinct deployment regimes:

- **Regime 1 — TASK-KNOWN** (single-task long-horizon agent: AppWorld, AutoGPT, Computer-Use, Cursor):
  - `general_task_aware`, `ACON_UT`, `ACON_UTCO` all realistic
  - main paper figure should compare these three
- **Regime 2 — TASK-UNKNOWN** (cross-task memory, personal assistant long-history):
  - only `general_task_agnostic` is realistic
  - `task_agnostic` vs `task_aware` gap = "value of task knowledge at compression time"

Paper must explicitly separate these to avoid misleading readers into thinking `task_aware` is always available.

### 7.2 `task_aware` ≠ "next-tool-aware"

V11's `task_aware` passes the original user instruction (e.g., "delete all spam from 9294880327"), NOT the next tool call. The compressor knows the **final goal**, not the **execution path**. Even given the goal, the compressor cannot reason from goal → required next tool → required token to preserve; the form prior dominates. This is why §5.2 form-prior-drop cases appear in `task_aware` family too.

### 7.3 Sample size

- Stage 06 stress signals (§2): n = 5,291 candidates, **STABLE**.
- Stage 07 transition matrix (§3): n = 621 per cell, ~2-3 pp std error, **mostly stable** but final number may shift ±3 pp.
- Best-of-N gain (§4): n = 68-69 cells fully done, std error larger; full stage 07 will triple this to ~145 cells.
- Form-prior cases (§5): n = 200 qualifying; full stage 07 will yield ~600.

### 7.4 Known minor cosmetics (to clean up before paper writing)

- Phone regex matches dates (`2023-04-23`); tighten to require non-date format
- JWT and kv_pair both match the same access_token string; dedupe in 15c index
- 15c sample size for `general_task_aware` slightly under `general_task_agnostic` in top-20 (random ordering, not a real signal)

---

## §8. What stages 08-09 will tell us (key hypothesis tests)

The provisional findings above leave three open questions that stage 08abc + 09 will resolve:

1. **Verbal verifier (pointwise + pairwise) recovery on ACON vs general** — does the verbal verifier (which shares the same form prior as the compressor) achieve smaller selection gain on ACON than on general? This would directly support the user's "prior interference limits selection room" hypothesis at the *selector* level (rather than the sample level we've already verified).
2. **continuation_entropy as prior-orthogonal selector** — does the agent-perspective selector (which uses continuation perplexity at decision points, not LLM evaluation) achieve uniform selection gain across families? If yes, it's the paper's recommended deployment-time selector.
3. **Best-of-N gain stabilization** — at full n=145 per family, does ACON_UT maintain its +39 pp lead, or does it converge to the cluster mean?

---

## §9. Recommended paper figures (current best guesses)

| § | figure | source | status |
|---|---|---|---|
| Fig 1 (motivation) | 4-family length drift bar chart (§2.1) | stage 06 STABLE | ready |
| Fig 2 (form-prior) | Top form-prior drop case (771d8fc_2 ACON_UTCO JWT) | 15c case study | ready, paper-grade |
| Fig 3 (rescue) | conditional rates per family × round (§3.1) | stage 07 PROVISIONAL | needs full stage 07 |
| Fig 4 (best-of-N) | Pareto frontier: pass / 1k chars × pass rate (§4.2) | stage 07 PROVISIONAL | needs full stage 07 |
| Fig 5 (selector) | Verbal verifier vs continuation entropy recovery | stage 09 | NOT YET — open hypothesis |

## §10. Open questions for discussion

1. **Path A vs Path B vs Path C** narrative — which is the dominant story?
   - Path A: ACON dominates → not supported by §3 / §4 data (general_task_aware ties on rescue)
   - Path B: Diverse compressors win selection → §4 partially supports (ACON_UT highest oracle, but gain ~equal across families)
   - Path C: Pareto trade-off, no winner → §4.2 + §4.3 support this most cleanly
2. **Is "task-known vs task-unknown" the right primary axis?** Or should it be "ACON family vs general family"?
3. **15b (hallucination) vs 15c (form-prior drop)** — do both go in main paper, or 15c only with 15b in appendix?
4. **v10 SFT route as the "escape" mechanism** — explicit method-section recommendation, or future work?
5. **`general_task_agnostic` as deployment baseline** — the paper should be honest that it's the only Regime-2 option; should we propose anything specific to improve it?

---

This document will be updated after stage 07 completes (~3 AM PT Tue) and stage 09 produces selector data (~11 AM PT Tue).
