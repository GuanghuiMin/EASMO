# motivation_v10 conditions matrix

> "Condition" here means the (compressor × decoding × stress)
> combinations that produce a candidate or a downstream eval.
> This matches the cross-track table style used by v2/v3/v4.

## 1. Candidate pool (stage 02)

| compressor | decoding | seed(s) | n per case | n total (× ~110 cases) |
|---|---|---|---:|---:|
| MiniMax-M2.5 (ACON UTCO) | greedy | 42 | 1 | 110 |
| MiniMax-M2.5 (ACON UTCO) | sample temp=0.7 | 1000..1007 | 8 | 880 |
| **subtotal stage 02** | | | **9** | **~990** |

## 2. Stress conditions (stage 03)

Each candidate goes through K=2 ACON UTCO recompressions with
MiniMax-M2.5 as the recompressor at temperature=0.0:

```
T^0 (candidate)  →  T^1  →  T^2
```

`~990 × 2 ≈ 1,980` extra MiniMax calls.

## 3. Behavior conditions (stage 04)

| condition | text source | cap_steps | n per case | n total |
|---|---|---:|---:|---:|
| C1 | candidate's `compressed_text` (T^0) | 15 | 9 | ~990 |
| CK | stress-chain T^2 text | 15 | 9 | ~990 |
| **subtotal stage 04** | | | **18** | **~1,980** |

## 4. Proxy conditions (stage 05)

| proxy | scope | n per case | n total |
|---|---|---:|---:|
| `minimax_verifier` (5-axis pointwise) | all 9 candidates × {C1, CK} | 18 | ~1,980 |
| `pairwise_preference` | sample vs greedy under CK only | 8 | ~880 |
| `future_action_nll_proxy` | auxiliary; subset 10 % | ~1.8 | ~200 (optional) |
| **subtotal stage 05** | | | **~2,860 + opt 200** |

## 5. SFT students (stage 08)

| student | targets | training | output |
|---|---|---|---|
| `Qwen-SFT-C1` | `outputs/data/sft_targets_c1.jsonl` | LoRA r=16, 2 epochs, lr=1e-4 | `outputs/models/qwen_sft_c1/` |
| `Qwen-SFT-CK` | `outputs/data/sft_targets_ck.jsonl` | same | `outputs/models/qwen_sft_ck/` |

vLLM (port 8000) **must be stopped** while these run — peak GPU
memory ~30-40 GB. Restart after stage 08.

## 6. Student evaluation conditions (stage 09)

| variant | decoding | how many cases | rounds | total agent runs |
|---|---|---:|---|---:|
| `MiniMax-greedy` | reuse stage 02 greedy | ~22 (test_behavior) | C1, CK | 44 |
| `MiniMax-oracle-bestofN` | reuse stage 04 best | ~22 | C1, CK | 44 |
| `Raw-Qwen` | greedy temp 0.0 (new compressions) | ~22 | C1, CK | 44 |
| `Qwen-SFT-C1` | greedy | ~22 | C1, CK | 44 |
| `Qwen-SFT-CK` | greedy | ~22 | C1, CK | 44 |
| **subtotal** | | | | **~220** |

Stressor: primary = frozen `Qwen-SFT-CK` (self-stress); secondary =
MiniMax ACON (cross-model stress, reuses stage 03 logic).

## 7. GRPO readiness (stage 10)

For each of `Raw-Qwen`, `Qwen-SFT-C1`, `Qwen-SFT-CK`:

| step | per-case | total (3 students × ~22 cases) |
|---|---:|---:|
| greedy compression | 1 | 66 |
| stochastic samples (temp 0.7) | 8 | 528 |
| stress to T^K | 9 | 594 |
| proxy verifier score | 18 | 1,188 |
| **subset true pass** | 25 % × 18 | ~300 agent runs |

## 8. Chunk reanalysis (stage 11)

Pull chunks from 5 candidate types per case (greedy MiniMax, oracle-best
MiniMax, proxy-selected MiniMax, Qwen-SFT-C1, Qwen-SFT-CK). Cap at 12
chunks per candidate. Across ~22 test cases × 5 variants × ≤12 chunks
≈ **~1,300 chunks max**. In practice many short candidates yield <12,
so realistic ~600-800. Each chunk gets:

* enriched labeler call (`label_chunk` from v10 chunk_label.py)
* one chunk-minus context build (LLM call)
* one downstream agent run (chunk-minus ablation)

So ~600 × 3 ≈ 1,800 calls + agent runs. Largest single non-SFT
stage besides stage 04. Worker pool 6.

## 9. Conditions NOT covered (deferred)

These are listed in spec but explicitly out-of-scope for the v10
report (resource budget / clarity):

* `N=16` extension for pass@N curves (would double stage 02-04 cost).
* Mid-trajectory compression boundary (env restoration is not wired in productive_agents).
* Per-tool-app ablation (would inflate stage 04 by ~5×).
* Full 89-case GRPO online run (only readiness sampling here).
