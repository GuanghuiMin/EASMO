# motivation_v11 definitions / glossary

> Cross-references spec §3 (datasets), §10 (selectors), §11 (metrics).

## Splits and case sets

* **Primary cases** = AppWorld train+dev tasks where baseline (MiniMax
  agent on full context at cap_steps=15) succeeds. Expected ~70/145.
* **Secondary cases** = ALL train+dev tasks (145), including baseline-fail
  rows. Used for `compression_rescue_rate` diagnostic (ACON paper showed
  compressed context can sometimes beat raw context).
* **Length buckets** (spec §3.5):
  * `short`  : `baseline_steps < 15`
  * `medium` : `15 ≤ baseline_steps < 25`
  * `long`   : `baseline_steps ≥ 25`

## Compression rounds (spec §3.4 + §8)

* `C1 = T^0(c)` = candidate's direct compressed output.
* `T^{r+1}(c) = compress(family, history=T^r(c), task=instruction, temp=0.0, seed=42)`.
* `CK = T^K(c)` with default `K=2`.

The stress recompressor uses the **same prompt family** as the
original candidate.

## Candidates (spec §7)

Per case × family:
* 1 `greedy`: temp=0.0, seed=42, sample_id=-1
* `N=8` `sample`: temp=0.7, seeds=1000+i for i∈[0,7], sample_id=i

`candidate_id = "<task>__<family>__greedy"` or `"...__sample_0i"`.

## Reward (spec §11.4)

```text
R(c, round) = Pass(c, round) − 0.05 * (chars(c, round) / 2000)
```

Pass dominates length; length is a tie-breaker. Used for oracle /
`best_c1` / `best_ck` selection.

## Selectors (spec §10)

| selector | uses ground-truth pass? | scope |
|---|---|---|
| `greedy` | no | deployable |
| `random_sample` | no | deployable baseline |
| `shortest_sample` | no | deployable baseline |
| `oracle_best_of_n` | **yes** (CK reward) | upper bound, NOT deployable |
| `best_c1` | yes (C1 reward) | upper bound; tests one-step target selection |
| `best_ck` | yes (CK reward) | upper bound; tests stress-aware target selection |
| `pointwise_verifier` | no (negative baseline) | MiniMax 5-axis JSON rubric |
| `pairwise_verifier` | no (negative baseline) | MiniMax 7-match tournament |
| `continuation_entropy` | no (negative baseline) | M=5 disagreement features |

## Distribution quality + calibration gap (spec §11.1-§11.3)

```text
Q_dist(p)   = best-of-N pass@CK over family p
G_calib(p)  = best-of-N pass@CK − greedy pass@CK
calibration_ratio_CK = greedy_pass_CK / max(Q_dist_CK, ε)
```

## Stress fragility classes (spec §11.5)

| class | pass C1 | pass CK | rate definition |
|---|---|---|---|
| `robust_pass` | true | true | |
| `fragile_pass` | true | false | `fragility_rate = #fragile / (#robust_pass + #fragile)` |
| `stress_improved` | false | true | `stress_improved_rate = #stress_improved / N` |
| `robust_fail` | false | false | |

```text
delta_pass_C1_to_CK_pp = 100 * (pass_rate_CK − pass_rate_C1)
```

## Stress invariance (spec §11.6)

```text
length_drift_pct      = (chars_CK − chars_C1) / max(chars_C1, 1)
text_similarity_C1_CK = difflib.SequenceMatcher(None, c1, ck).ratio()
exact_fixed_point     = normalize(c1) == normalize(ck)
```

Aggregated by family × selector.

## Selector recovery (spec §11.7)

```text
recovery_CK(s) = (pass_rate_s_CK − pass_rate_greedy_CK)
                / max(pass_rate_oracle_CK − pass_rate_greedy_CK, ε)
```

A selector can have negative recovery if it performs worse than greedy.

## Pass@N + better-than-greedy mass (spec §11.8 + §11.9)

```text
Pass@N = fraction of cases where at least one of the first N samples passes (fixed seed order)
W_N    = oracle win rate over greedy with N samples
p_hat  = 1 − (1 − W_N)^(1/N)   (per-sample better-than-greedy probability)
```

`N ∈ {1, 2, 4, 8}` computed separately for C1 and CK.

## Tier and split tags in v11_cases

```
tier = "primary"        if baseline_success else "secondary_only"
```

Primary tables filter to `tier == "primary"`; secondary tables use
all cases. `compression_rescue` is computed on the `secondary_only`
subset.
