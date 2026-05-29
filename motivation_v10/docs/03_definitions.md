# motivation_v10 definitions / glossary

> Cross-references the formal definitions in
> `user_feedback/motivation_v10_proxy_sft_grpo_readiness_v2.md §3`.

## Compression rounds

* `T^0(c) = c`               candidate text directly out of MiniMax
* `T^{r+1}(c) = MiniMax-ACON-UTCO(T^r(c))` recompression via ACON UTCO
* Default `K = 2`. `C1 = T^0`, `CK = T^K`.

## Candidates

For each case `x`, generate 9 candidates:

* `greedy`: temp=0.0, seed=42                                  → `sample_id = "greedy"`
* 8 samples: temp=0.7, seeds=1000..1007                        → `sample_id = "sample_0i"` for i=0..7

`candidate_id = f"{case_id}__{compressor_model}__{sample_id}"`.

## Reward

```
R_C1(c) = Pass(T^0(c)) − λ_length × length_norm(T^0(c))
R_CK(c) = Pass(T^K(c)) − λ_length × length_norm(T^K(c))
```

* `Pass(.)` = AppWorld downstream-agent success at `cap_steps=15`.
* `length_norm(text) = chars(text) / 2000`.
* `λ_length = 0.05`.
* Length penalty is a tie-breaker. Pass dominates.

## Best-of-N

* `oracle_best = argmax_i R(c_i)` over the 9 candidates (greedy + 8 samples) using ground-truth pass.
* `proxy_selected = argmax_i proxy_score(c_i)` using a MiniMax verifier composite (§proxy.py).
* `random_sample = uniform pick from the 8 samples` (only used as floor).

## Oracle recovery fraction (spec §12)

```
recovered_gain = (pass(proxy_selected) − pass(greedy))
               / (pass(oracle_best) − pass(greedy) + ε)
```

A useful proxy recovers ≥ 0.40 of the oracle gain **or** beats greedy
by ≥ 10 pp on CK.

## Fragility (carried from v9)

For each candidate, classify under the same four-class scheme as v9:

| class | pass C1 | pass CK |
|---|---|---|
| `robust_pass` | true | true |
| `fragile_pass` | true | false |
| `stress_improved` | false | true |
| `robust_fail` | false | false |

```
fragility_rate(group) = #fragile_pass / (#robust_pass + #fragile_pass)
```

## SFT target quality

| `target_quality` | criterion |
|---|---|
| `strong` | teacher candidate's C1 pass = true (for SFT-C1) or CK pass = true (for SFT-CK), and length is shortest among passes |
| `weak` | no candidate passes the target round; fall back to best proxy score |

Primary SFT only uses `target_quality = strong`. `weak` cases are
included only as a budget extension (off by default).

## Student variants for stage 09 eval

| Variant | Description |
|---|---|
| `MiniMax-greedy` | teacher baseline reproduced |
| `MiniMax-oracle-bestofN` | upper-bound diagnostic |
| `Raw-Qwen` | base Qwen3-4B-Instruct-2507 with the ACON UTCO prompt |
| `Qwen-SFT-C1` | LoRA-trained on C1 targets |
| `Qwen-SFT-CK` | LoRA-trained on CK targets |

All evaluated at greedy decoding under both C1 and CK budgets.

## GRPO readiness signals (stage 10)

| Metric | Definition |
|---|---|
| `within_case_reward_std` | std of CK reward over the 9 candidates per case |
| `oracle_win_rate_over_greedy` | fraction of cases where ≥1 sample beats greedy on true (or proxy) CK reward |
| `best_of_n_gain` | pass-rate of best-of-N minus pass-rate of greedy |
| `all_fail_rate` | fraction of cases where all 9 candidates fail |
| `all_pass_rate` | fraction of cases where all 9 candidates pass |
| `reward_entropy` | entropy of the binarized pass-vector across N candidates |
| `length_reward_correlation` | Pearson(length, reward) — should be small for healthy GRPO |

A "GRPO-ready" Qwen-SFT-CK has:

* `all_fail_rate ≤ 0.15`
* `oracle_win_rate_over_greedy ≥ 0.5`
* `within_case_reward_std ≥ 0.15` on average
