# Session handoff — paste this into a new chat if context fills up

> Updated: 2026-05-24 00:23 UTC
>
> **Paste the contents of this file into a fresh Cursor chat session**
> with a one-line follow-up like "继续昨晚的 motivation 实验，看看
> instance_noise 和 wide_* 的结果"。 The new assistant should be able
> to pick up from this snapshot.

## What this project is

EASMO — investigating **policy-dependent prompt compression** for
agentic systems. The motivation section (M1–M5 + ablations) tests a
two-thesis spine:

* **T1** (compression-pressure-induced policy-dependence): policy-
  dependence of optimal memory is amplified at tight compression
  budgets; whether this is monotonic or two-regime depends on
  in-flight data.
* **T2** (prompts can't policy-condition): off-the-shelf LLM-as-
  selector produces compressions that are *surface-similar* across
  agents even when conditioned on agent description.

Combined: policy-conditional compression at tight budgets is
**necessary** (T1) and **not achievable by prompting** (T2). It must
be learned with a behavioral objective. That's EASMO.

## Where things stand (read this first)

* **Path D framing** is the floor (T2 is robust on three independent
  measurements: M5 default, M5-tight T=0.0, M4 classifier).
* **Path C framing** (budget-regime structure) — T1 hinge — is in
  flight via the `wide_*` re-runs.
* **Both T1 and T2 depend** on the `instance_noise_test` (currently
  running on LongMemEval B=512) finishing with `cross/within ratio
  ≥ 3×` to rule out seed noise as the explanation for M3 transfer
  drop. This is the **single most important number** to read first.

## Concrete state (2026-05-24 00:23 UTC)

### Code & docs (all on GitHub)

* Repo: `git@github.com:GuanghuiMin/EASMO.git` (branch `main`)
* Living results doc: `motivation/docs/02_results_and_interpretation.md`
* Original design spec: `motivation/docs/01_experiments_spec.md`
* W&B: https://wandb.ai/guanghui_min-university-of-virginia/easmo-motivation

### Background processes (PIDs)

```
3867384  instance_noise_test    (LongMemEval B=512, ~1h ETA)
3872897  wide_locomo run_all    (full 6-budget curve, ~10-12h ETA)
3872899  wide_longmemeval run_all (~10-12h ETA)
3916704  auto_push_watcher.sh   (pushes any new results every 20 min)
```

### Critical bug — fixed

`oracle.py`'s `_extract_candidate` used to return truncated `<think>`
blocks as "compressed memory" when MiniMax-M2.5 ran out of
`max_tokens` mid-reasoning. Affected 56% of LongMemEval B=128 records
and 68% of LoCoMo B=128 records. Fixed by:

1. Rejecting unclosed `<think>` responses.
2. Raising selector `max_tokens` to a flat 4096.

The `wide_*` re-runs (in flight) use the fixed code. The `default_*`
results are now superseded by the wide ones; their valid M5 / M4 data
is still retained.

## What to do when results come in

### Step 1 — Read instance_noise_summary (~1h after handoff)

```bash
cd /workspace/EASMO/motivation
cat outputs/default_longmemeval/instance_noise_summary.json | python -m json.tool
```

Decide:

| `conditional_ratio_cross_over_within` | Verdict for T1+T2 |
|---|---|
| ≥ 3× | STRONG → Path D survives, write Spotlight pitch |
| 1.5×–3× | WEAK → borderline, Findings paper |
| < 1.5× | M3 is seed noise → retreat to Path B (standard compression paper) |

Append a `## Result update YYYY-MM-DD` section in
`motivation/docs/02_results_and_interpretation.md` with the verdict.

Run a second `instance_noise_test` at B=128 to confirm the same
pattern at the tightest budget (T1's strongest claim):

```bash
python -m scripts.instance_noise_test \
    --config configs/default_longmemeval.yaml \
    --budget 128 --n-contexts 10 --candidates-per-agent 3
```

### Step 2 — Read wide_*/m1_summary.json (~7-9h after handoff)

```bash
cat outputs/wide_locomo/m1_summary.json | python -m json.tool
cat outputs/wide_longmemeval/m1_summary.json | python -m json.tool
```

Check per-budget action-match rates; they should be uniform across
budgets now (no survivorship bias).

### Step 3 — Re-plot the budget curve on clean data

```bash
# After wide_* finishes M3
python -m scripts.budget_regime_test --runs outputs/wide_longmemeval
python -m scripts.budget_regime_test --runs outputs/wide_locomo
```

Look at:
* `c sign` — positive (bowl-shaped) or negative (n-shaped)?
* `argmin B` — does it land near 256 (matching the default valley)?
* `R²` — does it explain ≥ 70% of the variance?

Decide which T1 wording the data supports:

| Pattern | T1 wording |
|---|---|
| Monotonic decreasing | "policy-dependence amplifies as budgets tighten" (Path C strong) |
| Bowl-shaped valley near B=256 | "two-regime structure with answer-echo collapse at B=B_essential" (Path C revised) |
| Saturating curve | "policy-dependence is dominant at tight budgets" (Path C weak) |
| Saw-tooth / no structure | Path C dead, commit to Path D |

### Step 4 — Re-generate Figure 1 candidates

* `outputs/figs/conditional_drop_vs_budget_wide_clean.png` — main plot
* `outputs/figs/instance_noise_within_vs_cross.png` — hinge bar plot
* `outputs/figs/per_pair_asymmetry_B128.png` — asymmetry matrix

### Step 5 — Sync & push

`auto_push_watcher.sh` runs every 20 min, but at major milestones run:

```bash
bash motivation/scripts/sync_and_push.sh "instance_noise B=512 done — ratio=X.XX"
```

## Open methodological questions (still unsettled)

1. Is the V-shape at B=256 a real two-regime phenomenon, or will the
   wide re-run smooth it out? (Open until B=2048/B=4096 data lands.)
2. Does instance_noise ratio behave the same at B=128 as at B=512?
   (T1's strongest claim requires the tight budget to have the
   highest ratio.)
3. Does LoCoMo show the same budget pattern as LongMemEval, or is it
   uniformly high (ceiling)?
4. Does M3-judge (LLM-as-judge) corroborate the conditional drop on
   `wide_*` data? (Will run after M3 finishes.)

## Files of record (single sources of truth)

| File | Role |
|---|---|
| `motivation/docs/02_results_and_interpretation.md` | living analysis, all decisions |
| `motivation/docs/01_experiments_spec.md` | original design (do not modify) |
| `motivation/docs/SESSION_HANDOFF.md` | this file |
| `outputs/wide_longmemeval/m3_summary.json` | T1 final number (LongMemEval) |
| `outputs/wide_locomo/m3_summary.json` | T1 final number (LoCoMo) |
| `outputs/default_longmemeval/instance_noise_summary.json` | T1+T2 hinge |

## Trust no in-doc framings until verified

If anything in the doc seems off after data lands — recompute from raw
CSV via `scripts/recompute_m3_summary.py` and `scripts/budget_regime_test.py`.
Both are pure post-processing and reproducible.
