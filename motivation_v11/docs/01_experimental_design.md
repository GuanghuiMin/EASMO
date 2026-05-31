# motivation_v11 experimental design

> Frozen 2026-05-31 PT. Implements the "train+dev (β with α-upgrade
> path)" of the spec at
> `user_feedback/motivation_v11_final_full_dev_behavior_prompt_family_experiment.md`.
>
> v11 is the **final motivation-section experiment** for the paper.
> v9, v10 each falsified one earlier claim (v9 §10 falsified
> "causal NL > entity lists" via chunk widening; v10 §8.5 found the
> verifier proxy anti-correlated with behavior). v11 converges on
> the consistent message: **structured prompts create a useful
> compression distribution, but greedy decoding + verbal proxies do
> not select behavior-optimal compressions**.

## 1. Goal in one sentence

For each of 4 prompt families, measure on the full AppWorld
**train + dev** split (145 tasks):

* `Q_dist(p) = best-of-N pass@CK` — distribution quality
* `G_calib(p) = best-of-N pass@CK − greedy pass@CK` — calibration gap

and compare 6 + 3 selectors against the oracle to show that verbal
selectors recover little of the oracle headroom.

## 2. Splits and case pool

| split | n_tasks | role |
|---|---:|---|
| `train` (AppWorld train.txt) | 89 | Primary case pool, baseline at cap=15 |
| `dev` (AppWorld dev.txt) | 56 | Primary case pool, baseline at cap=15 |
| **total** | **145** | |

Stage 01 baselines all 145 with `compressed_context=""` at cap=15.
* **Primary set** = baseline-pass cases (~50% pass rate → ~70 cases)
* **Secondary set** = all 145 cases (includes baseline-fail rows for
  "compression rescue" diagnostics)

We run stages 02-05 on **all 145** cases (so compression-rescue of
baseline-fail tasks is captured per ACON paper's finding that
compression can beat raw context on some hard tasks).

## 3. Prompt families (spec §5)

| family | source | template_kind | sha256_user (first 16) |
|---|---|---|---|
| `general_task_agnostic` | in-repo `motivation_v11/prompt_families.py` | python `.format` | `d9b1cf6396...` |
| `general_task_aware` | in-repo | python `.format` | `7398347c4b...` |
| `ACON_UT` | `acon/experiments/appworld/prompts/context_opt/prompt_history_v2.jinja` | jinja2 | `0508caa837...` |
| `ACON_UTCO` | `acon/experiments/prompt_optimizer/outputs_appworld/stage1_minimax/optimized_prompts/improved_history_prompt_samples_4.jinja` | jinja2 | `9e50d0f93a...` (matches v7/v8/v9/v10) |

ACON commit (frozen across tracks since v7): `d63f9ae18959dc7215ff62899c94c5e8c56847ae`.

ACON system prompt is shared between UT and UTCO (same file), with
sha256 `f9a0a5188d...`. Provenance written to
`outputs/provenance/prompt_sha256.json` by stage 00.

## 4. Models and roles (spec §4)

| Role | Model | Endpoint |
|---|---|---|
| Compressor | `MiniMaxAI/MiniMax-M2.5` | `10.183.22.68:8005` |
| Downstream agent | `MiniMaxAI/MiniMax-M2.5` | same |
| Verbal selector / verifier / entropy diagnostic | **MiniMax-M2.5 only** (spec §4.2 explicitly forbids Qwen) | same |

vLLM (`port 8000`) is **NOT** required for v11 — Qwen explicitly out of scope.

## 5. Generation settings (spec §4.3)

```yaml
greedy:    temperature=0.0, seed=42,   max_tokens=2048, top_p=1.0
samples:   temperature=0.7, seeds=1000..1007, N=8, max_tokens=2048, top_p=1.0
stress:    temperature=0.0, seed=42,   max_tokens=2048, K=2
behavior:  cap_steps=15, seed=42
verbal verifier:  json_mode=True, temperature=0.0, seed=42
entropy:   temperature=0.7, seeds=2000..2004, M=5
```

`max_chars=2000` passed to all 4 family templates (ACON templates may ignore it; recorded for provenance).

## 6. Stage map

```
00 prepare                          provenance + config snapshot
01 build_full_dev_cases             baseline MiniMax agent on 145 tasks (~24 min)
02 render_prompts                   3 rendered examples per family for paper appendix
03 generate_candidates              4 families × 145 cases × (1 greedy + 8 samples) = 5,220 candidates (~6.2 h)
04 serial_recompression_stress      each candidate × K=2 stress rounds = 10,440 calls (~12.4 h)
05 run_behavior_c1_ck               each candidate × {C1, CK} = 10,440 agent runs (~17.4 h)
06a pointwise_verifier              10,440 verifier calls (~12.4 h)
06b pairwise_verifier               145 × 4 × 7 = 4,060 pairwise matches (~4.8 h)
06c continuation_entropy            initially ACON_UTCO only × 145 × 2 × M=5 = 1,450 calls (~3 h)
                                    [forward-compat: pass --families ... to upgrade to all 4 = +46 h]
07 selection_analysis               aggregate per (family, selector, round)
08 distribution_quality_calibration Q_dist + G_calib per family
09 stress_invariance_analysis       fragility / drift / fixed-point per (family, selector)
10 pass_at_n_curve                  Pass@N=1,2,4,8 + better-than-greedy mass
11 build_candidate_bank             merge texts + stress chains + behavior → reusable jsonl
12 plot_figures                     5 paper figures (PDF + PNG)
13 write_report                     auto-written paper-tier markdown
```

Total ETA: **~68 h ≈ 2.8 days** for plan (β).

Upgrade to plan (α) — entropy on all 4 families — by re-running stage 06c
with `--families general_task_agnostic,general_task_aware,ACON_UT,ACON_UTCO`:
adds ~46 h (incremental, no rework). Then re-run stages 07 + 12 + 13
(~30 min) to refresh selector recovery table and figures.

## 7. Acceptance / falsification (spec §17 + §18)

| Criterion | Threshold | Pass / fail action |
|---|---|---|
| 1. Structured distribution quality | best ACON_{UT,UTCO} Pass@CK ≥ general_task_agnostic Pass@CK + 10 pp (or higher pass-per-token) | weaken to "task-aware prompting suffices" if fail |
| 2. Policy calibration gap | best-of-N Pass@CK − greedy Pass@CK ≥ 15 pp for ACON_UT or ACON_UTCO; AND oracle_len ≤ 1.10 × greedy_len | abandon policy-headroom story if fail |
| 3. Serial recompression matters | greedy fragility_rate ≥ 0.20 OR best_ck − best_c1 ≥ 5 pp at CK | demote CK to diagnostic if fail |
| 4. Verbal selectors insufficient | pointwise/pairwise/entropy recovery_CK < 0.50 (this is a POSITIVE motivation outcome) | strengthen "behavior reward needed" framing if pass |
| 5. No length-mediated win | selected_mean_chars ≤ 1.10 × greedy_mean_chars | reframe as budget insufficiency if fail |

## 8. Provenance

`outputs/provenance/`:
- `acon_repo_commit.txt` — pinned ACON commit
- `prompt_sha256.json` — per-family sha256 of system + user templates
- `{family}_system.txt` + `{family}_user_template.txt` — raw prompt text
- `acon_ut_prompt.txt` + `acon_utco_prompt.txt` + `acon_system_prompt.txt`
- `rendered_prompt_examples/{family}/{task_id}.txt` — 3 fully rendered examples per family
- `general_prompt_templates.md` — markdown appendix for paper
- `pip_freeze_easmo_venv.txt`
- `../config_v11.json` — frozen run configuration
