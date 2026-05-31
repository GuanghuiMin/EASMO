# motivation_v11 experimental design (revised 2026-05-31)

> Spec: `user_feedback/motivation_v11_final_train_dev_transition_experiment.md`
> (frozen 2026-05-31 PM PT).
>
> **v11 is the final motivation-section experiment for the paper.**
> Previous tracks established:
> * v7+v8: structured ACON prompts induce a surface-type abstraction
>   prior (SDI ≈ 1, cross-model τ up to 0.78).
> * v9: behavior-side validation — best-of-N over MiniMax-ACON-UTCO
>   samples beats greedy by +27 to +37 pp pass@CK on 30 v3-dev cases.
> * v10: trying to convert v9 headroom into a trainable Qwen3-4B
>   compressor via SFT — Claim 1 (proxy recovers best-of-N) FAILED on
>   CK, but Claim 2 (SFT-CK > SFT-C1 > Raw) PARTIAL ✓ and Claim 3
>   (GRPO-trainable reward spread) PASS. Verifier composite found
>   anti-correlated with true behavior (AUROC 0.56).
>
> v11 consolidates these into the final motivation evidence at
> AppWorld **train + dev** scale (145 tasks), under a sharpened
> framing: **compression is not just preservation — it can preserve,
> harm, or rescue tasks**. The headline is the **full-vs-compressed
> 2×2 transition matrix** per prompt family × selector × eval round.

## 1. Paper-level questions v11 answers (spec §0)

1. Do structured prompts produce better compression distributions
   than generic prompts?
2. Does greedy decoding select the best compression from that
   distribution?
3. Do compressed contexts remain useful under serial recompression?
4. Can verbal selector prompts replace behavior reward, or do they
   leave most oracle headroom unrecovered?

## 2. Splits + primary case set (spec §4)

| split | n tasks |
|---|---:|
| `train` (AppWorld train.txt) | 89 |
| `dev` (AppWorld dev.txt) | 56 |
| **combined** | **145** |

ACON paper cites 90 train tasks; local has 89. Stage 01 records
which task is missing in `outputs/provenance/appworld_task_inventory.csv`.

**Primary analysis set = ALL 145 tasks with valid full-context
baseline AND valid compressed runs** (spec §4.2). v10's
"only baseline-pass cases" filter is **explicitly dropped** because
compression can RESCUE baseline-fail tasks (ACON paper finding).
The transition matrix in §4 below captures this.

Every table reports `split ∈ {train, dev, combined}` separately
(spec §4.3). If train and dev trends diverge, the report flags it.

## 3. Models and roles (spec §5)

| Role | Model |
|---|---|
| Compressor | `MiniMaxAI/MiniMax-M2.5` |
| Downstream agent | `MiniMaxAI/MiniMax-M2.5` |
| Verbal verifier / pairwise judge / continuation entropy | **MiniMax-M2.5 only** (Qwen explicitly forbidden) |

Generation settings:

```yaml
greedy_temperature: 0.0
sample_temperature: 0.7
sample_seeds: [1000..1007]
N_SAMPLES: 8
stress_temperature: 0.0
stress_seed: 42
stress_K: 2
compression_max_tokens: 2048
behavior_cap_steps: 15
```

## 4. ★ Core headline analysis: Full-vs-Compressed Transition Matrix (spec §2.1)

Per (prompt_family, selector, eval_round) and per (split):

| F (full) | C (compressed) | outcome | meaning |
|---|---|---|---|
| pass | pass | **preserve_success** | compression preserved a task full context solved |
| pass | fail | **harm** | compression broke a task full context solved |
| fail | pass | **rescue** | compression solved a task full context failed |
| fail | fail | both_fail | neither solved |

Core identity:

```text
overall_gain = compressed_pass_rate − full_pass_rate
             = rescue_rate − harm_rate
```

This is the **paper-facing main result table**:
`outputs/tables/transition_matrix_by_prompt_selector_round.csv`.

## 5. Q_dist decomposition by full-context outcome (spec §2.3)

For each prompt_family × split × eval_round:

```text
Q_dist_all      = P(BestN CK=1)                  — overall distribution quality
Q_dist_preserve = P(BestN CK=1 | F=1)            — preservation strength
Q_dist_rescue   = P(BestN CK=1 | F=0)            — rescue strength

G_calib_all      = Q_dist_all − greedy_pass_all
G_calib_preserve = Q_dist_preserve − greedy_pass_given_F1
G_calib_rescue   = Q_dist_rescue   − greedy_pass_given_F0
```

`outputs/tables/distribution_quality_calibration_gap.csv` has all
12 columns (3 conditional × 4 metrics: greedy, bestN, q_dist, gap).

## 6. Stage plan (spec §19, 17 stages)

```
00 prepare                            provenance + config snapshot
01 build_task_inventory               145 task_ids + appworld_task_inventory.csv
02 run_full_context_baseline          AppWorld agent with empty context, 145 runs (~24 min)
03 build_compression_boundaries       trajectory-derived history for compression (no checkpoint
                                       continuation available locally; documented as fallback)
04 render_prompts_and_provenance       3 examples per family for paper appendix + sha256
05 generate_candidate_compressions    4 families × 145 cases × (1 greedy + 8 samples) = 5,220 (~6.2 h)
06 run_serial_recompression_stress    each × K=2 = 10,440 calls (~12.4 h)
07 run_behavior_c1_ck                 each × {C1, CK} = 10,440 agent runs (~17.4 h)
08 run_verbal_selectors               pointwise (10,440) + pairwise tournament (4,060)
                                       + continuation entropy (ACON_UTCO only, 1,450)
                                       ~26 h serially; better to run sub-stages 08a/08b/08c in
                                       parallel where MiniMax endpoint allows.
09 compute_selectors                  build selector decisions per (task, family, round)
10 compute_transition_metrics         ★ FULL-VS-COMPRESSED 2×2 PER (selector, round, split)
11 compute_distribution_quality_calibration  Q_dist + G_calib (decomposed)
12 compute_serial_recompression_metrics      fragility + drift + best_c1 vs best_ck
13 bootstrap_confidence_intervals     8 paired comparisons × 2000 resamples (spec §13.7)
14 plot_figures                       6 paper figures (incl. transition heatmap)
15 write_case_studies                 6 representative cases per spec §18
16 write_report                       13-section paper-tier markdown
```

Estimated compute: **~63 h ≈ 2.6 days** under plan (β)
(entropy_selector only on ACON_UTCO). Upgrade to plan (α) — entropy
on all 4 families — by re-running stage 08c with
`--families general_task_agnostic,general_task_aware,ACON_UT,ACON_UTCO`
later (+5 h incremental).

## 7. Selectors (spec §11)

| selector | uses ground-truth pass? | scope |
|---|---|---|
| `greedy` | no | deployable |
| `random_sample_mean` | no | mean over 8 samples (baseline statistic, not a single-context selector) |
| `random_sample_fixed` | no | optional: sample_id=0 |
| `shortest_sample` | no | shortest C1 / CK at the relevant round |
| `oracle_best_of_N_C1` | yes (C1 score) | upper bound at C1 |
| `oracle_best_of_N_CK` | yes (CK score) | upper bound at CK — main oracle |
| `best_c1` | yes (C1 reward) | tests one-step target selection |
| `best_ck` | yes (CK reward) | tests stress-aware target selection |
| `pointwise_minimax_verifier` | no (negative baseline) | 5-axis JSON rubric |
| `pairwise_minimax_selector` | no (negative baseline) | randomized deterministic bracket (seed 42), NOT v10's sequential A→B→winner pattern |
| `continuation_entropy_selector` | no (negative baseline) | M=5 disagreement features |

## 8. Acceptance criteria (spec §15)

| # | Criterion | Threshold |
|---|---|---|
| 1 | Structured prompts improve distribution quality | best ACON family has higher `Best-of-N Pass@CK` or `net_gain` or `pass_per_1k_chars` than `general_task_agnostic` |
| 2 | Structured prompts can improve all-task behavior | `rescue_rate > harm_rate` for at least one structured condition, OR overall compressed pass rate ≥ full-context baseline |
| 3 | Calibration headroom remains under structured prompts | for ACON_UTCO or ACON_UT: `Best-of-N CK − Greedy CK ≥ 15 pp` OR `Best-of-N net_gain − Greedy net_gain ≥ 10 pp` |
| 4 | Best-of-N is not length-mediated | selected mean length ≤ 110 % of greedy mean |
| 5 | Serial recompression matters | greedy fragility ≥ 20 % OR best_ck > best_c1 at CK OR nontrivial fragile_rescue count |
| 6 | Verbal selectors do NOT close the gap | pointwise / pairwise / entropy recover < 50 % of oracle CK gain (this is a POSITIVE motivation outcome) |

## 9. Falsification (spec §16)

The motivation framing is weakened if ALL of:

1. General prompts and ACON prompts have similar distribution quality
2. Greedy within 5 pp of best-of-N across all families
3. Best-of-N gains are length-mediated
4. CK does not change behavior relative to C1
5. Verbal selectors recover most oracle gain
6. Transition matrices show no meaningful harm/rescue dynamics

If this occurs, the paper should NOT claim behavior reward is
necessary for compression policy selection.

## 10. Compression boundary protocol — fallback path (spec §7)

**Spec prefers**: online checkpoint continuation at `T_hist=4096`
threshold, where the env is restored to step `t` after compression.

**v11 reality**: the local `productive_agents` AppWorld runner does
NOT support env restoration. We **fall back** to the
`trajectory_derived` protocol used in v9/v10:

1. Run full-context baseline. Save the rendered trajectory text.
2. Compression input = rendered baseline trajectory, regardless of
   whether the baseline succeeded.
3. Compressed-context evaluation reruns the agent from scratch with
   the compressed context as "what already happened".

Every behavior_runs row has `evaluation_protocol="trajectory_derived"`.
The report must explicitly state this in §1 setup.

## 11. Provenance (spec §6.5)

`outputs/provenance/`:
* `acon_repo_commit.txt` — pinned ACON commit (d63f9ae)
* `acon_prompt_sha256.json` — per-family sha256 of system + user templates
* `{family}_system.txt` + `{family}_user_template.txt` — raw text
* `acon_ut_prompt.txt` + `acon_utco_prompt.txt` + `acon_system_prompt.txt`
* `rendered_prompt_examples/{family}_{task_id}.txt` — 3 examples per family
* `appworld_task_inventory.csv` — 145 task_ids + split + included flag
* `pip_freeze_easmo_venv.txt`
* `../config_v11.json` — frozen run configuration
