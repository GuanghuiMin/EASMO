# motivation_v9 — Experimental design

## 0. Framing

v7 and v8 measured retention. v9 measures **behavior**. The three
claims:

1. **Best-of-N gap.** Under the same ACON UTCO prompt and the same
   compressor model, do stochastic samples sometimes outperform the
   greedy compression on the downstream AppWorld pass rate?
2. **C1-vs-CK fragility.** Does one-step compression survive
   repeated compression `T^K`, or does behavior degrade after stress?
3. **Chunk information advantage.** When we segment a compressed
   context into natural-language chunks and ablate each chunk, which
   chunks are causally responsible for downstream success? Is the
   answer "entity lists" or "causal/control natural-language
   relations"?

Each claim has explicit thresholds (spec §14):

* Claim 1 strong: `best_of_N_CK_pass - greedy_CK_pass ≥ 10 pp` OR
  `oracle_win_rate_CK ≥ 25 %`.
* Claim 2 strong: `fragility_rate ≥ 20 %` OR
  `pass_rate_C1 - pass_rate_CK ≥ 10 pp`.
* Claim 3 strong: causal/control/action-outcome chunks have higher
  mean advantage than entity-only chunks, OR ≥ 40 % of top-advantage
  chunks contain causal relations.

## 1. Models (spec §3)

| role | model | venv | notes |
|---|---|---|---|
| Compressor (primary) | `MiniMaxAI/MiniMax-M2.5` | `EASMO/.venv` | shared endpoint at 10.183.22.68:8005 |
| Compressor (optional) | `qwen3-4b-instruct-2507` | `EASMO/.venv` | local vLLM port 8000; deferred in v9 primary run |
| Downstream agent | `MiniMaxAI/MiniMax-M2.5` | `acon/.venv` | identical model to compressor — but runs as AppWorld agent via `motivation_v4.runner` |
| Chunk type labeler | **MiniMax-M2.5 only** | `EASMO/.venv` | spec §3.3 explicitly forbids Qwen as auditor |

Generation:

```yaml
temperature_greedy: 0.0
temperature_sample: 0.7
seed_greedy: 42
seed_sample:  1000 + i      # for i in 0..N-1
max_tokens_compression: 2048
max_tokens_label: 256
```

## 2. Data (spec §5)

* Reuse 30 successful AppWorld dev trajectories from v3
  (`motivation_v3/outputs/motivation_full_trajectories.jsonl`).
* No fact bank, no need-condition pairs (v9 is not retention-first).
* Each case loaded as v3 `Trajectory` → rendered trajectory text
  (max 18,000 chars) + structured `trajectory_steps`.

Median trajectory length: 20 steps. v3 has no <15-step short
trajectories; this caveat carries from v7/v8.

## 3. ACON prompt provenance (spec §4)

Use the verbatim official `microsoft/acon` UTCO prompt and system
prompt:

* `experiments/appworld/prompts/context_opt/system_prompt.jinja`
* `experiments/prompt_optimizer/outputs_appworld/stage1_minimax/optimized_prompts/improved_history_prompt_samples_4.jinja`

ACON commit pinned to `d63f9ae18959dc7215ff62899c94c5e8c56847ae`.
SHA256 of each file written to
`outputs/provenance/prompt_sha256.json`. **No edits.**

## 4. Stages

| # | name | venv | purpose |
|---:|---|---|---|
| 00 | prepare | EASMO | provenance + run_config |
| 01 | build_cases | EASMO | v3 → `data/v9_cases.jsonl` |
| 02 | generate_candidates | EASMO | greedy + 8 samples per (case, model) |
| 03 | stress_recompress | EASMO | T^K chains per candidate |
| **04** | **behavior_c1_ck** | **acon** | AppWorld pass per candidate × {C1, CK} |
| 05 | best_of_n metrics | EASMO | reward + best-of-N |
| 06 | c1_ck_fragility | EASMO | transition matrix |
| 07 | select_chunk_cases | EASMO | pick up to 12 candidate pairs |
| 08 | segment_chunks | EASMO | NL chunking |
| 09a | build_chunk_contexts | EASMO | remove chunk → re-stress |
| 09 | chunk_ablation_behavior | acon | AppWorld pass per chunk-removed context |
| 10 | chunk_advantage | EASMO | score / pass advantage per chunk |
| 11 | label_chunks_minimax | EASMO | chunk type labels (MiniMax only) |
| 12 | chunk_advantage_by_type | EASMO | aggregate advantage × type |
| 13 | plot_figures | EASMO | 5+ figures |
| 14 | write_report | EASMO | deterministic markdown |

Stages 04 and 09 must run under `acon/.venv` because they spawn the
AppWorld agent. We use `ProcessPoolExecutor(max_workers=6)` — same as
v4 stage 07 and v6 Experiment D.

## 5. Reward definition (spec §5.1)

```
R = score - lambda_len * normalized_length
lambda_len = 0.02
normalized_length = compressed_tokens_est / 1000
```

`compressed_tokens_est = max(1, len(compressed_text) // 4)`.

We also always report **pass-only metrics** without length penalty
because most AppWorld scores are binary {0.0, 1.0}.

## 6. Convergence check for stress chains (spec §3, §13.9 from v8)

For each candidate × round transition (r > 0):

```
text_similarity(x_r, x_{r-1}) >= 0.95
AND |len_r - len_{r-1}| / max(len_{r-1}, 1) <= 0.02
```

`convergence_round` = first r satisfying both. No fact-Jaccard
required (v9 is behavior-first).

## 7. Chunk segmentation rules (spec §8)

1. Split on bullet/numbered lines (`^\s*[-*•·]|^\s*\d+[\.)]`).
2. If no bullets, split into sentences (greedy on `.?!` + space).
3. Merge chunks shorter than 20 chars into the previous chunk.
4. Cap at `CHUNK_ABLATION_MAX_CHUNKS_PER_CONTEXT = 12` by merging the
   shortest adjacent pairs.

No entity extraction — chunks are coherent NL statements, not tokens.

## 8. Cost & schedule

| stage | calls/runs | wall-clock |
|---|---:|---:|
| 00-01 | (instant) | < 5 s |
| 02 candidates | 270 LLM compressions (workers=6) | ~10 min |
| 03 stress chains | 540 LLM compressions (workers=6) | ~10 min |
| **04 behavior C1+CK** | **540 AppWorld agent runs** (workers=6) | **~5 h** |
| 05-08 | trivial | < 1 min |
| 09a re-stress | ~120 LLM compressions | ~3 min |
| 09 chunk ablation | ~120 AppWorld runs | ~1 h |
| 10-12 | trivial | < 1 min |
| 13-14 | figures + report | < 30 s |

Total ≈ **6-7 h** wall-clock.

## 9. Deviations from spec

* Spec defaults to two compressor tracks (MiniMax + Qwen3-4B); v9
  primary runs **MiniMax only** to halve cost. Qwen track is a
  follow-up.
* Stage 13 "proxy reward calibration" is skipped (spec §13 is
  optional).
