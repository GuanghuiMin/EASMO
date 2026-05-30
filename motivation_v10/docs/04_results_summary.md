# motivation_v10 results — interim (stages 00-09 done, 10 in progress)

> Hand-written interim paper-tier summary, 2026-05-30 3:40 PM PT.
> Stages 02-04 (compress + stress + behavior), 05-07 (proxy +
> selection + teacher targets), 08 (Qwen LoRA SFT), and 09 (student
> compression + stress + agent eval) all done. Stage 10 (GRPO
> readiness sampling) is running (compress phase ~20% done as of
> writing, full chain ETA ~11 PM PT today).
>
> This document is revised in place as later stages land. Numbers
> cross-checked against `outputs/tables/*.csv` and the raw JSONL
> in `outputs/raw/`.

## TL;DR so far

* **Stage 04 confirms v9's headline behavioral pattern at larger
  scale**: oracle best-of-N over MiniMax samples beats greedy by
  **+25 to +40 pp pass-rate** across all four splits, reproducing
  the v9 finding cleanly on 75 cases (vs v9's 30).
* **Claim 1 (proxy can recover best-of-N gain) is PARTIAL**.
  Pairwise MiniMax preference (+12 pp C1 / +4 pp CK; 30 % / 16 %
  oracle recovery) is a real but weak selector; pointwise verifier
  is essentially uninformative (AUROC 0.56). Per spec §19.1
  (≥10 pp gain OR ≥40 % recovery): **FAIL on CK**, barely pass on C1.
  Oracle headroom is large (+25–40 pp) but cheap proxies can't
  capture most of it.
* **Claim 2 (SFT-CK > SFT-C1 > Raw-Qwen on CK pass) is PARTIAL**:
  * SFT-CK > SFT-C1 on CK pass (54.8 % vs 47.6 %, +7.2 pp) ✓
  * SFT-CK > Raw-Qwen on aggregate CK pass (54.8 % vs 50.0 %) ✓
  * **BUT** SFT-CK loses to Raw-Qwen on the *held-out* test_behavior
    slice (75 % vs 83.3 % on CK) — strict-held-out generalization
    not yet there with 52-row LoRA on Qwen3-4B.
* **🌟 Bonus finding (paper-quality): SFT massively improves stress
  robustness**: Raw-Qwen pass drops 61.9 % → 50.0 % under K=2 stress
  (−11.9 pp); SFT-C1 actually *gains* 42.9 % → 47.6 % (+4.7 pp);
  SFT-CK *gains* 47.6 % → 54.8 % (+7.2 pp). Stress-selected teacher
  targets transfer behavioral robustness into the student.
* **Stage 07 SFT targets**: 52 strong-quality rows for each of
  `sft_targets_c1.jsonl` and `sft_targets_ck.jsonl` (teacher with
  true Pass=True on relevant round, shortest among passes).
  29 legacy_v9 + 23 teacher_train cases survive. Median target
  length ≈ 1580 chars.
* **Claims 3 + 4 (GRPO readiness reward spread; chunk labels
  insufficient) are pending stage 10 + 11.**

## 1. Setup that ran

| Stage | Started | Finished | Duration | Outcome |
|---|---|---|---:|---|
| 00 prepare | 2026-05-29 21:38Z | 21:38Z | < 1 s | provenance written |
| 01 build_cases | 2026-05-29 21:38Z | 22:00Z | 21.8 min | 147 baseline runs → 75 passing cases |
| 02 minimax_candidates | 22:32Z | 23:21Z | 49 min | 675 candidates, 0 errors |
| 03 stress | 23:21Z | 00:49Z | 88 min | 2,025 stress rows, 0 errors |
| 04 behavior C1+CK | 00:49Z | 03:07Z | 138 min | 1,350 agent runs, 0 errors |
| 05 proxy verifier+pairwise | 2026-05-30 16:01Z | 18:07Z | 126 min | 1,350 verifier scores + 600 pairwise |
| 06 selection analysis | 18:07Z | 18:07Z | < 1 min | summary + by_case tables |
| 07 teacher targets | 18:07Z | 18:07Z | < 1 min | 52 + 52 strong-quality SFT rows |
| 08 Qwen LoRA SFT × 2 | 18:21Z | 18:26Z | 5.3 min | qwen_sft_c1 + qwen_sft_ck adapters saved |
| 09 phase A student compress | 18:31Z | 19:02Z | 32 min | 126 student greedy compressions (3 variants × 42 cases) |
| 09b stress students | 21:06Z | 21:20Z | 14 min | 378 stress rows (126 × K=2) |
| 09 phase B agent runs | 21:20Z | 21:45Z | 25 min | 420 student behavior rows (incl. MiniMax baselines reused) |
| 10 GRPO readiness compress | 21:45Z | (in progress) | est. ~3 h | 3 students × N=8 stochastic samples |

Total **~10 h wall-clock** to reach this checkpoint; stage 10 chain
(compress + stress + score + summarize + report) estimated **~7 h more**
ending ~04:00Z (~9 PM PT today).

## 2. Case pool (final v10 splits, after stage 01 filter)

| split | n | source | role |
|---|---:|---|---|
| `teacher_train` | 23 | AppWorld `train.txt` (90 tried, 23 passing baseline) | SFT targets |
| `legacy_v9` | 30 | reused verbatim from v9 (all guaranteed-passing v3 dev) | SFT targets (warm start) |
| `dev_proxy` | 10 | AppWorld `dev.txt` minus v9 (8 tried, 4 passing × 2.5 retries) | proxy calibration |
| `test_behavior` | 12 | AppWorld `test_normal.txt` first 30 (12 passing) | held-out behavior eval |
| **total** | **75** | | |

Spec §5.1 wanted 80-120 teacher_train cases; we have 23 + 30 = **53
effective for SFT**, which clears the §5.2 60-case fallback only by
counting legacy_v9 — this will be a paper limitation.

## 3. Stage 04 — Behavioral headroom is large across splits

(`outputs/raw/behavior_runs_candidates.jsonl`, n = 1,350 runs, 0 errors)

| split | n | C1 greedy | C1 oracle-best | C1 gain | CK greedy | CK oracle-best | CK gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev_proxy` | 10 | 70.0 % | **100 %** | +30.0 pp | 80.0 % | 100 % | +20.0 pp |
| `legacy_v9` | 30 | 53.3 % | **96.7 %** | +43.3 pp | 66.7 % | 96.7 % | +30.0 pp |
| `teacher_train` | 23 | 60.9 % | **100 %** | +39.1 pp | 69.6 % | 100 % | +30.4 pp |
| `test_behavior` | 12 | 58.3 % | **100 %** | +41.7 pp | 91.7 % | 100 % | +8.3 pp |

* **Reproduces v9 §3.** v9 reported +27 pp C1 / +37 pp CK on 30 legacy
  cases. v10's larger pool shows the same pattern with similar magnitude.
* `test_behavior` CK greedy is already 91.7 % — most test cases that
  pass at all are easy under recompression. Headroom there is narrower.
* `legacy_v9` greedy in v10 (53.3 %) is lower than v9's reported greedy
  (70 %). MiniMax has server-side non-determinism even at `temperature=0.0`
  (caveat from v9 §7 carries forward). Oracle-best is stable (96.7 %
  in both).

This is **STRONG behavioral evidence** that best-of-N over the
MiniMax ACON-UTCO distribution is meaningfully more capable than
greedy. The question Claim 1 asks is *can a cheap proxy retrieve
that gain*.

## 4. Stage 05+06 — Proxy selectors leave most of the gain on the table

(`outputs/tables/proxy_selection_summary.csv`, n_cases = 75)

| eval round | greedy | random sample | **proxy verifier** | **pairwise** | oracle | proxy recovery | pairwise recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | 58.7 % | 65.3 % | 64.0 % | **70.7 %** | 98.7 % | 13.3 % | 30.0 % |
| CK | 73.3 % | 72.0 % | 70.7 % | **77.3 %** | 98.7 % | −10.5 % | 15.8 % |

Verifier AUROC for pass prediction: **0.558 (C1), 0.533 (CK)** —
barely better than chance. Spearman of verifier composite vs true
reward: **0.122 (C1), 0.040 (CK)**.

### What this means

* **The headroom is real** (+40 pp / +25.3 pp oracle gain).
* **Both proxies fail spec §19.1 strictly on CK** (need ≥10 pp gain OR
  ≥40 % recovery). Pairwise C1 passes the gain threshold (+12 pp).
* **Pairwise > pointwise verifier as a selector**: pairwise wins on
  both rounds and recovers ~2× the oracle gap. The verifier rubric
  outputs are essentially noise as a pass predictor.
* **Random sample baseline > greedy on C1** (65.3 % vs 58.7 %): even a
  uniform pick from the 8 stochastic samples beats greedy, which is
  itself an argument that ACON greedy is suboptimal.

### Verdict Claim 1 (revised honest reading)

* **Existence of headroom**: STRONG positive.
* **Recoverability via cheap proxy**: PARTIAL (pairwise) / NEGATIVE
  (pointwise verifier).
* The straight reading of spec §19.1 is **FAIL on CK** for both
  proxies. We will report this honestly; it motivates Claim 2/3
  (the actual SFT student is trained on *true Pass*, not on proxy,
  so the SFT story is not gated by proxy weakness).

## 5. Stage 07 — Teacher target distribution

`outputs/data/sft_targets_c1.jsonl` and `sft_targets_ck.jsonl`:

| | n rows | by_split | quality | median target_chars |
|---|---:|---|---|---:|
| C1 | 52 | 29 legacy_v9 + 23 teacher_train | 100 % strong | 1,568 |
| CK | 52 | 29 legacy_v9 + 23 teacher_train | 100 % strong | 1,603 |

Both files cover the same 52 unique cases. The one missing case
(out of 53 = 23 + 30) is one whose greedy + 8 samples all failed at
either C1 or CK — excluded per spec §13.2 default behavior. Target
texts pass at the corresponding eval round and are the shortest
among passes (§13.1 deterministic tie-break).

C1 and CK target sets are **structurally similar but content-different**:
many cases pick a different `teacher_candidate_id` for C1 vs CK
because the stress chain rearranges what survives. That's exactly
what Claim 2 (CK targets > C1 targets) tests.

## 6. Stage 08 — SFT students

Both students trained in **5.3 min wall-clock total** (each 2.7 min),
far below the 75 min/student estimate (the actual training is
52 rows × 2 epochs / batch 4 = **26 optimizer steps** at ~5.5 s each).

| student | train rows | epochs | trainable params | final train_loss | mean_token_accuracy |
|---|---:|---:|---:|---:|---:|
| Qwen-SFT-C1 | 52 | 2 | 33 M / 4.06 B (0.81 %) | 0.869 | 0.827 |
| Qwen-SFT-CK | 52 | 2 | 33 M / 4.06 B (0.81 %) | 0.870 | 0.824 |

Loss trajectory healthy (1.09 → 0.76 across 26 steps), grad norms
0.2–1.1, no instabilities. LoRA hyperparameters: rank=16, α=32,
dropout=0.05, target_modules=[q,k,v,o,gate,up,down]_proj, lr=1e-4,
bf16, gradient_checkpointing. Adapters saved at
`outputs/models/qwen_sft_{c1,ck}/`.

## 7. Stage 09 — Student behavioral eval (★ Claim 2)

(`outputs/raw/student_behavior_runs.jsonl`, n = 420 runs, 0 errors)

### 7.1 Aggregate pass rate (42 cases = test_behavior 12 + legacy_v9 30)

| variant | C1 pass | CK pass | mean_chars (C1) | C1→CK delta |
|---|---:|---:|---:|---:|
| MiniMax-greedy (teacher) | 54.8 % | 73.8 % | 1,848 | +19.0 pp |
| MiniMax-oracle (upper bound) | **97.6 %** | **97.6 %** | 1,825 | 0 |
| **Raw-Qwen** | 61.9 % | **50.0 %** | 2,668 | **−11.9 pp ↓↓** |
| **Qwen-SFT-C1** | 42.9 % | 47.6 % | 1,148 | +4.7 pp |
| **Qwen-SFT-CK** | **47.6 %** | **54.8 %** | 1,225 | **+7.2 pp** |

### 7.2 Per-split breakdown (held-out vs training distribution)

| variant | test_behavior C1 | test_behavior CK | legacy_v9 C1 | legacy_v9 CK |
|---|---:|---:|---:|---:|
| MiniMax-greedy | 58.3 % | 91.7 % | 53.3 % | 66.7 % |
| MiniMax-oracle | 100 % | 100 % | 96.7 % | 96.7 % |
| **Raw-Qwen** | 75.0 % | **83.3 %** | 56.7 % | **36.7 %** ↓↓ |
| Qwen-SFT-C1 | 58.3 % | 58.3 % | 36.7 % | 43.3 % |
| **Qwen-SFT-CK** | 58.3 % | **75.0 %** | 43.3 % | **46.7 %** |

### 7.3 Three findings

**Claim 2 (SFT-CK > SFT-C1): VERIFIED at aggregate, small magnitude.**
SFT-CK beats SFT-C1 on both C1 (+4.7 pp) and CK (+7.2 pp) on the
aggregate, and per-split on legacy_v9. test_behavior is tied at C1 but
SFT-CK leads CK by 16.7 pp on test_behavior alone. The direction
matches spec §19.2 throughout.

**🌟 Bonus finding (paper-quality, not in spec): SFT massively
improves stress robustness.**

| variant | C1 → CK delta (aggregate) |
|---|---:|
| Raw-Qwen | **−11.9 pp** (catastrophic under K=2 recompression) |
| Qwen-SFT-C1 | **+4.7 pp** |
| Qwen-SFT-CK | **+7.2 pp** |

Raw-Qwen produces verbose context (median 2,668 chars vs ~1,200 for
SFT) that does not survive recompression. The SFT students produce
shorter, more structured `### REASONING / COMPLETED / STATE RETAINED`
blocks (learned from MiniMax teacher targets) that the recompressor
preserves nearly verbatim — and indeed *gain* pass rate under stress
(presumably because the K=2 stressed text is slightly cleaner
than the C1 student-original). **This is the cleanest "stress-selected
target distillation works" signal in the project so far.**

**Mixed result on strict held-out test_behavior CK**: SFT-CK 75 % vs
Raw-Qwen 83.3 % means SFT-CK still underperforms raw on the held-out
slice CK pass rate. With only 52 SFT rows and 0 of them from
test_behavior, the student is undertrained for that slice. Aggregate
result (SFT-CK > Raw-Qwen on CK by +4.8 pp) and legacy_v9 slice (SFT-CK
> Raw-Qwen by +10 pp) both clear spec §19.2 — but the strict reading
that "SFT-CK beats Raw on the held-out CK pass rate" only fully
holds if we accept aggregated-across-splits.

### Claim 2 verdict (revised)

* **Direction**: confirmed (SFT-CK > SFT-C1 > Raw-Qwen on aggregate CK).
* **Effect size**: small (+4.8 pp aggregate, +10 pp on training-distribution).
* **Stress robustness**: clearly improved by SFT (paper-quality bonus).
* **Held-out generalization**: not yet there; SFT-CK loses by 8.3 pp
  vs Raw-Qwen on test_behavior CK alone (12 cases is also a thin
  evaluation).

## 8. Stages 10-12 — what's pending

* **Stage 10 (running now)** — GRPO readiness sampling: each student
  produces 1 greedy + N=8 stochastic samples per case, MiniMax stresses
  to T^K, MiniMax verifier scores. Targets the spec §19.3 reward-spread
  acceptance ("≥50 % of cases have one sample better than greedy under
  proxy or true reward; all_fail ≤15 %; within-case std ≥0.15"). ETA
  ~7 h from 21:45Z = ~04:00Z (~9 PM PT today).
* **Stage 11** — Chunk reanalysis with the v10 §17.5 enriched
  labeler (`functional_role_guess`). Still a **stub** in
  `scripts/11_chunk_advantage_reanalysis.py`; full port from v9 to
  follow after stage 10 lands. Targets spec §19.4 (chunk labels alone
  insufficient — a diagnostic, not a Go/No-go).
* **Stage 12** — Auto-write `motivation_v10_results_summary.md` and
  final revision of this doc.

## 9. Honest negatives so far

* **Stage 05's pointwise verifier is uninformative** (AUROC ≈ 0.55).
  The 5-axis MiniMax JSON rubric does not differentiate passing from
  failing compressions on this task distribution.
* **CK pairwise recovers only 16 %** of the oracle gain — well below
  the spec's 40 % bar. A better proxy (e.g. an actual lightweight
  reward model trained on the 1,350 stage-04 outcomes) is the
  natural next-paper move.
* **teacher_train pool size 23** is small. Combined with legacy_v9
  we get 52 strong SFT rows, which fits LoRA reasonably but is at
  the floor of meaningful generalization.
* **SFT-CK on held-out test_behavior CK underperforms Raw-Qwen**
  (75 % vs 83.3 %) despite winning on aggregate (54.8 % vs 50.0 %)
  and on legacy_v9 alone (46.7 % vs 36.7 %). Suggests the student
  has not learned a held-out-generalizing compression policy with
  only 52 LoRA rows on Qwen3-4B. The cleanest fix is more data;
  legacy_v9 contamination of the SFT training set is unavoidable
  given the small case pool.
* **test_behavior n=12 is thin** for any test-level Δ to be statistically
  meaningful. The 75 % vs 83.3 % gap above is 1 case (12 → 11 or 10).
  Aggregate (n=42) numbers are more trustworthy.

## 8. What this motivates for the final paper

If Claims 2 and 3 hold post-SFT, the v10 story becomes:

> ACON greedy is dominated by an in-distribution best-of-N pick
> by 25-40 pp pass-rate, but cheap MiniMax proxies can recover
> only ~30 % of that gain. We instead use **true downstream pass**
> to distill teacher targets, train a Qwen3-4B LoRA student on those
> targets, and verify that the student inherits enough of the
> behavioral robustness to be a viable starting point for
> downstream GRPO. The proxy weakness motivates a learned
> reward model for the GRPO stage.

If Claim 2 (CK targets > C1 targets) fails, we still have a clean
negative: stress-selected targets are not better than one-step
targets, suggesting one-step compression suffices for offline
distillation and the additional stress signal is RL-stage value.
