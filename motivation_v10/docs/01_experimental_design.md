# motivation_v10 experimental design

> Frozen 2026-05-29 PT. Implements the full path in
> `user_feedback/motivation_v10_proxy_sft_grpo_readiness_v2.md`.
> v9-final's hand-written §10 addendum invalidated the "causal NL >
> entity lists" hypothesis, so v10 treats chunk surface labels as
> diagnostics only and frames the *trainable* objective entirely
> around behavior (pass after stress minus length).

## 1. Goal in one sentence

Convert ACON best-of-N behavioral headroom into trainable supervision
for a Qwen3-4B student compressor, validating four claims:

| Claim | Question | Threshold |
|---|---|---|
| 1 | Can a proxy reward replace oracle best-of-N selection? | proxy-selected CK pass ≥ greedy +10 pp **OR** ≥ 40 % of oracle gain |
| 2 | Are stress-selected teacher targets better than one-step? | `Qwen-SFT-CK` ≥ `Qwen-SFT-C1` on CK pass + stress fragility |
| 3 | Does SFT move Qwen into a GRPO-trainable region? | SFT-CK has lower all-fail rate AND ≥ 50 % cases have one sample > greedy |
| 4 | Are chunk surface labels reliable reward proxies? | Regression: label-only `R²` < behavior-only `R²` (diagnostic, not Go/No-go) |

## 2. Roles + endpoints

| Role | Model | Endpoint |
|---|---|---|
| Teacher compressor | MiniMax-M2.5 | `http://10.183.22.68:8005/v1` |
| Downstream agent | MiniMax-M2.5 | same |
| Verifier / pairwise judge / chunk labeler | **MiniMax-M2.5 only** (spec §4) | same |
| Student compressor | Qwen3-4B-Instruct-2507 | local vLLM `http://127.0.0.1:8000/v1` (stopped during SFT) |
| Optional auxiliary NLL scorer | Qwen3-4B logprobs | local HF inference (not vLLM) |

Three venvs reused from v9:
* `/workspace/EASMO/.venv` — analysis + training (v10 just added `transformers 5.9`, `peft 0.19`, `trl 1.5`, `accelerate 1.13`, `datasets 4.8`, `bitsandbytes`).
* `/workspace/acon/.venv` — AppWorld agent runner (pydantic v1).
* `/workspace/qwen3-vllm/.venv` — vLLM serving (untouched; stopped during stage 08 SFT).

## 3. Splits and case pool (spec §5.1)

| Split | Source | Target n | Notes |
|---|---|---:|---|
| `teacher_train` | AppWorld `train.txt` | 89 → ~70 passing | one-shot baseline MiniMax agent, keep only `baseline_success=true` |
| `dev_proxy` | AppWorld `dev.txt` \\ v9 (26 leftover) | 26 → ~20 passing | proxy calibration + SFT model selection |
| `test_behavior` | AppWorld `test_normal.txt` first 30 | 30 → ~22 passing | held-out behavior eval |
| `legacy_v9` | v9's 30 cases (already passing) | 30 | warm-start smoke test / chunk reanalysis seeds |

Stage 01 runs baseline (no-compression) MiniMax agent on the **89 + 26
+ 30 = 145 new task_ids** and writes `data/v10_cases.jsonl`, then
filters to `baseline_success=true`. Expected ~110 passing cases survive.

If `teacher_train` post-filter drops below 60, fall back to v9 spec
§5.2 pilot (legacy_v9 only); documented in the report.

## 4. Stage map

```
00 prepare                       provenance + ensure outputs/
01 build_cases                   baseline MiniMax agent on 145 tasks → filter → v10_cases.jsonl
02 generate_minimax_candidates   30 cases × 1 model × (1 greedy + 8 samples) = ~1000 compressions
03 stress_candidates             each × K=2 ACON UTCO recompressions
04 behavior_evaluate_candidates  each × {C1, CK} AppWorld agent run
05 proxy_score_candidates        MiniMax verifier (pointwise) + pairwise + optional NLL
06 proxy_selection_analysis      oracle recovery / AUROC / regret per proxy
07 construct_teacher_targets     argmax (Pass, -length, -steps) for C1 and CK separately
08 train_qwen_sft                stop vLLM → 2 LoRA students → restart vLLM
09 evaluate_students             {Raw-Qwen, SFT-C1, SFT-CK, MiniMax-greedy, MiniMax-oracle} × cases
10 grpo_readiness_sampling       per-student N=8 outputs, stress, reward spread / oracle win
11 chunk_advantage_reanalysis    leave-one-chunk-out with v10's enriched labeler schema (§17.5)
12 write_report
```

## 5. Compute budget

| Stage | LLM calls | Agent runs | Wall-clock (estimated) |
|---|---:|---:|---:|
| 01 | 0 | 145 baseline | ~13 min |
| 02 | 9 × 110 ≈ 990 | 0 | ~75 min |
| 03 | 2 × 990 = 1,980 | 0 | ~130 min |
| 04 | 0 | 2 × 990 = 1,980 | ~190 min |
| 05 | verifier 1,980 + pairwise 880 ≈ 2,860 | 0 | ~120 min |
| 06 | 0 | 0 | ~5 min |
| 07 | 0 | 0 | ~5 min |
| 08 | 0 | 0 | ~150 min (2 students × ~75 min LoRA SFT) |
| 09 | 0 (re-uses student greedy outputs from 08) | 5 × 22 × 2 = 220 | ~25 min |
| 10 | 0 | 3 × 22 × 9 × 2 = 1,188 (subset to ~25 % = ~300) | ~30 min |
| 11 | ~250 chunk labels + 250 ablation contexts + 250 ablation runs | 250 | ~60 min |
| 12 | 0 | 0 | ~10 min |
| **TOTAL** | | | **~13 h** |

All long stages run in `nohup bash scripts/run_all.sh > outputs/logs/runall_full.log 2>&1 &`
so individual shell hangs don't kill compute. Auto-push watcher
keeps everything synced every 20 min.

## 6. Hyperparameters

### 6.1 MiniMax candidate generation (stage 02)
* `temperature_greedy = 0.0`, `seed = 42`
* `temperature_sample = 0.7`, `seeds = 1000..1007` (N = 8)
* `max_tokens = 2048` (thinking-aware, see WARN_THINKING_MIN_MAX_TOKENS guard)
* `max_chars = 1500` (ACON UTCO template variable, same as v7/v8/v9)

### 6.2 Stress (stage 03)
* `K = 2` recompression rounds with MiniMax-M2.5 as the stress recompressor.
* Same ACON UTCO prompt; `temperature = 0.0`; resume on collapse (use previous text if recompression returns empty).

### 6.3 Behavior eval (stage 04)
* `cap_steps = 15` (same as v9), single-seed agent runs.
* C1 = candidate's own `compressed_text`; CK = round-2 stress text.

### 6.4 Proxy (stage 05)
* Verifier rubric: 5-axis JSON (predicted_success_probability,
  missing_information_risk, execution_specificity,
  risk_of_repeating_completed_actions, risk_of_wrong_api_arguments).
* Composite ranking: `psp − 0.5 × missing_risk + 0.3 × specificity − 0.1 × repeat_risk − 0.1 × wrong_arg_risk`.
* Pairwise: each sample vs greedy under CK; majority-vote winner per case for `pairwise_selected`.

### 6.5 SFT (stage 08)
* Base model: `Qwen/Qwen3-4B-Instruct-2507`.
* LoRA: rank=16, alpha=32, dropout=0.05, target modules = `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
* LR=1e-4, epochs=2, per_device_bs=1, grad_accum=4 (effective bs=4).
* Max seq len = 12000.
* bf16, gradient_checkpointing on.
* SFT example format: `system = ACON UTCO system prompt`, `user = ACON UTCO history prompt (rendered with raw history + task + max_chars=1500)`, `assistant = teacher target compressed_text`.

### 6.6 GRPO readiness (stage 10)
* For each Qwen variant: 1 greedy + 8 samples at temp=0.7, seeds 1000..1007.
* Apply same K=2 MiniMax stressor.
* Score with verifier composite (proxy) on all; subset 25 % run true AppWorld pass.

## 7. Provenance

`outputs/provenance/`:
* `acon_commit.txt` — `git -C /workspace/acon rev-parse HEAD`
* `acon_utco_prompt_sha256.json` — UTCO + system prompt hashes
* `rendered_prompt_examples/` — first 3 fully-rendered ACON UTCO user prompts as plain text
* `train_args.json` — exact SFT hyperparameters for both students
* `pip_freeze_easmo_venv.txt` — full reproducibility manifest

## 8. Reuse of v9 artifacts

The 30 v9 cases (`/workspace/EASMO/motivation_v9/data/v9_cases.jsonl`)
are tagged `split=legacy_v9` and included in stage 02-04 alongside
the new teacher_train cases. Their existing v9 stage-02/03/04 outputs
are **not** reused directly because v10's `candidate_id` schema
differs (adds `seed`, `proxy_scores` fields) and we want a single
unified candidate pool. Stage 02 re-runs the legacy_v9 cases at the
cost of ~270 extra MiniMax calls (~20 min) — small compared to the
89-case full run.

If v10 short on compute, stage 02 can be hot-cached by importing
`motivation_v9/outputs/raw/candidate_compressions.jsonl` and only
generating new candidates for the new 80+ teacher_train cases. This
is a deliberate skip-flag (`STAGES="02_skip_v9_cached,..."`) not
yet wired up — paper-tier full run is the default.

## 9. Acceptance criteria (Go / No-go for GRPO follow-up)

Repeated verbatim from spec §19; **all four must hold** to declare
"ready for GRPO":

1. Proxy CK pass ≥ greedy +10 pp **OR** ≥ 40 % of oracle gain.
2. `Qwen-SFT-CK` beats raw Qwen and `Qwen-SFT-C1` on CK pass / proxy reward.
3. `Qwen-SFT-CK` sampled outputs show non-degenerate reward spread:
   at least 50 % of cases have one sample > greedy.
4. Chunk reanalysis: surface labels alone insufficient to explain
   behavior advantage (supports behavior-based credit assignment).

If any single criterion fails, the v10 report flags it and proposes
the smallest remediating change (e.g. larger N, weak-target inclusion,
different LoRA rank).
