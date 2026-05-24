# Session handoff — paste this into a new chat if context fills up

> Updated: 2026-05-24 18:50 UTC (Option X — strategy-as-policy — validated end-to-end on smoke; see §X-Validation at the end)
>
> **Paste the contents of this file into a fresh Cursor chat session**
> with a one-line follow-up like "继续昨晚的 motivation 实验，看看
> instance_noise 和 wide_* 的结果"。 The new assistant should be able
> to pick up from this snapshot.

## What this project is

EASMO — **policy-conditional context compression for long-horizon LLM
agents**. The motivation tests a two-thesis spine:

* **T1** (compression-pressure-induced policy-dependence): under tight
  memory budgets, the optimal compressed memory is policy-conditional
  — different policies want different things even from the same
  context.
* **T2** (prompts can't policy-condition): off-the-shelf LLM-as-
  selector cannot reproduce a true policy-conditional compression
  even when prompted with task + policy descriptions.

Combined claim: policy-conditional compression at tight budgets is
**necessary** (T1) and **not achievable by prompting** (T2). It must
be learned with a behavioral objective. That's EASMO.

## Where things stand (read this first — major redesign 2026-05-24)

* **Old motivation track (`motivation/`) is deprecated** as primary
  evidence. The setup (LLM-only selector + same-LLM-different-prompt
  agents + LongMemEval/LoCoMo QA + binary action_match metric) has a
  fatal logical issue: T2 holding makes the T1 hinge test unsatisfiable
  by construction (cross-agent and within-agent memories are
  identical-by-T2 → cross/within ratio ≈ 1 always). The wide_* and
  instance_noise runs are kept as long-memory QA appendix material
  only.
* **New motivation track (`motivation_v2/`)** is the active line. It
  uses **AppWorld** (real agentic benchmark), **execution-derived
  ground-truth memory `m*_exec`** (non-LLM oracle, decouples T1 from
  T2), and **policy = task family** (spotify / phone / venmo /
  file_system / simple_note — defined by the final-state evaluator's
  app set). See `motivation/docs/new_motivation.md` for the design
  with 11 review-fix in-line notes.
* **Infrastructure dependency**: AppWorld + agent runner is provided
  by Microsoft's ACON repo at `/workspace/acon/` (arXiv 2510.00615).
  Working venv: `/workspace/acon/.venv` (pydantic v1 + sqlmodel
  0.0.10; the EASMO `.venv` has v2 incompat that breaks AppWorld
  imports). AppWorld task data is downloaded under
  `acon/experiments/appworld/data/`. Trajectory generation goes
  through `acon/experiments/appworld/run_all.py`.
* **Empirical corpus check (done)**: `motivation_v2/scripts/smoke_data_pipeline.py`
  audits the AppWorld train+dev splits. Headline numbers:
  `train` has 60/90 single-app tasks (spotify=42 dominant, then
  file_system=9, phone=6, simple_note=3). `dev` adds venmo (3) +
  more spotify (30). Combined train+dev gives 5 single-app families.
  `shared-state cross-policy pairs = 0`, so M3 must use matched-pair
  fallback (R-3 in the design).
* **Successful trajectories on disk**: only 4 (3 train_tiny tasks,
  ~50% executor success rate so far). Full pilot needs ~50–70
  successful trajectories spanning all 5 families, which means
  generating trajectories on at least train+dev (147 tasks).

## Concrete state (2026-05-24 00:23 UTC)

### Code & docs (all on GitHub)

* Repo: `git@github.com:GuanghuiMin/EASMO.git` (branch `main`)
* Living results doc: `motivation/docs/02_results_and_interpretation.md`
* Original design spec: `motivation/docs/01_experiments_spec.md`
* W&B: https://wandb.ai/guanghui_min-university-of-virginia/easmo-motivation

### Background processes (PIDs, as of 2026-05-24 18:35 UTC)

These are all from the **deprecated** old motivation track. They are
left running because killing them now wastes 18 h of compute and the
data still has secondary value as long-memory QA appendix material.
None of them inform the new T1/T2 verdicts.

```
571621   instance_noise_test  (B=512 n=30 rerun, deprecated — appendix material)
572118   queue_B128 watcher   (auto-starts B=128 n=30 — deprecated)
3872897  wide_locomo run_all  (M1 80% — appendix material)
3872899  wide_longmemeval run_all (M1 95% — appendix material)
3916704  auto_push_watcher.sh (pushes any new results every 20 min)
```

**No new motivation_v2 background runs are active yet.** Trajectory
generation for AppWorld train+dev (~147 tasks, ~22 h sequential or
3 h × 8 parallel workers) is the next long-pole. Not started — needs
human go-ahead because it competes with the in-flight processes for
the MiniMax endpoint and is a multi-hour commitment.

History of instance_noise PIDs (for context):
* `3867384` — original B=512 n=10 run; finished at 00:05 UTC with a
  bogus `STRONG` verdict (0/0 = ∞, see §Audit-A1).
* `539832` — first audit re-run with binary metric, n=30; killed at
  18:01 UTC after the design-audit identified the metric itself as
  the bottleneck.
* `571621` — current re-run with continuous `action_overlap_rate`
  metric (1 − TV) as the primary signal channel.

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

---

## §Audit (2026-05-24 17:30 UTC) — read this if you're picking up here

A mid-flight check found three issues; all three have been addressed.
**Don't trust the pre-audit numbers in `02_results_and_interpretation.md`
without checking the §Result update 2026-05-24 section there.**

### A1. `instance_noise_summary.json` was a false positive

The 00:05 run with `B=512 n=10 K=3` gave **`n_signal_rows = 0/30`**
— every target's `self_match` was 0.0, so the cross/within ratio was
`0/0 = Infinity`, which the verdict logic naively classified as
`STRONG`. With M1's ~21% pass rate at B=512 on LongMemEval, getting
0 hits out of 30 was unlucky but not surprising at n=10 contexts.

* **Verdict logic patched** in `scripts/instance_noise_test.py` to
  emit `INSUFFICIENT SIGNAL` when `n_signal_rows < 5` or both drops
  are ~0. The old `instance_noise_summary.json` has been re-derived
  to read the same.
* **Re-run started at 17:25 UTC** with `n_contexts = 30` (PID 539832,
  log: `outputs/instance_noise_rerun_B512_n30.log`). When this
  finishes (~3 h), check `outputs/default_longmemeval/instance_noise_summary.json`
  again. It will have been overwritten. Then run **B=128 with n=30**
  to confirm the same ratio at the tightest budget (T1's strongest
  claim).

### A2. `wide_*` M1 quality is uneven

The `oracle_memories.jsonl` files show the **per-budget pass rate is
essentially flat on LongMemEval (~21% across 128 → 4096) and very
sparse on LoCoMo (0–12%, with 0/52 at B=256)**. Two consequences:

* T1's "tight budgets amplify policy-dependence" claim has to come
  from M3 (cross-agent transfer), not from M1 self-fidelity.
* LoCoMo conditional M3 is going to have wide CIs because there are
  too few paired passing rows. Be prepared to demote LoCoMo to a
  robustness check rather than a co-headline result.

### A3. `auto_push_watcher.sh` was silently hung 16 h

A `git push` at 01:32 UTC hung on the network and the watcher's
`wait` blocked indefinitely. Killed the hung child, manually ran
`sync_and_push.sh` to flush the backlog (commit `ea842d9`), watcher
is back to normal. **If you see no log entries for >25 min in
`/tmp/easmo_watcher.log`, suspect the same hang.** A long-term fix
is to wrap the `git push` line in `sync_and_push.sh` with
`timeout 60`.

### What to do when each rerun finishes

1. **`instance_noise_test B=512 n=30` v2 (PID 571621)** — read
   `outputs/default_longmemeval/instance_noise_summary.json`. The
   summary now has BOTH binary and overlap fields. **Use the
   overlap ones for the verdict** (`overlap_conditional_ratio_cross_over_within`,
   `n_signal_rows_overlap`). The signal threshold is
   `self_overlap ≥ 0.5` (any compressed memory whose action
   distribution overlaps the full-context distribution by ≥ 50%).
   - If `n_signal_rows_overlap ≥ 5` and overlap ratio ≥ 3×: Path-D
     STRONG.
   - If overlap ratio in [1.5, 3): Path-D WEAK / borderline.
   - If overlap ratio < 1.5: Path-D DEAD, fall back to Path B.
   The B=128 rerun (PID will be created by 572118 once 571621 exits)
   then confirms / disconfirms at the tightest budget.
2. **`wide_longmemeval` finishes M1** — `run_all` spawns M2/M3
   subprocesses fresh, so M3 (`scripts/run_m3.py`) will load the
   patched code automatically and emit `overlap_self/cross/drop`
   columns in `transfer_results.csv` and `m3_overlap_*` fields in
   `m3_summary.json`. **Read `m3_overlap_mean_conditional_drop`,
   not `m3_mean_conditional_drop`,** for the headline.
3. **`wide_locomo` finishes M1** — same pattern, but first
   sanity-check `m1_summary.json`'s pass count. With 0–12% binary
   M1 pass rates, the binary M3 will be near-noise-floor regardless;
   the overlap M3 might salvage usable signal. If even
   `m3_overlap_signal_rows < 30`, demote LoCoMo to a robustness side
   panel rather than a co-headline.

### What changed in the codebase (2026-05-24 18:05 UTC)

* `motivation/metrics.py`: added `action_overlap_rate(p, q) = 1 - TV(p, q)`
  and `js_divergence`. Smooth replacement for the binary
  `action_match_rate` which is essentially a Bernoulli on QA tasks
  with N=16 samples and was the root cause of "0 signal rows".
* `motivation/instance_noise.py`, `scripts/instance_noise_test.py`:
  records both binary and overlap channels; verdict logic uses
  overlap with threshold `self_overlap ≥ 0.5`.
* `motivation/transfer.py`, `scripts/run_m3.py`,
  `scripts/recompute_m3_summary.py`: same — both channels emitted,
  schema-versioned so `recompute_m3_summary.py` works on old + new
  CSVs.
* `motivation/scripts/sync_and_push.sh`: `git push` now runs under
  `timeout 90` so the watcher can't silently hang.
* No changes to `oracle.py` (M1) or `selector_ablation.py` (M5) — M1
  is mid-flight in `wide_*` runs; M5 doesn't depend on the binary
  metric the same way.

### What changed in the codebase (2026-05-24 18:35 UTC — redesign)

* `motivation/docs/02_results_and_interpretation.md`: deprecation
  banner at the top routing readers to `new_motivation.md`.
* `motivation/docs/new_motivation.md`: 11 in-line review-fix notes
  (R-1 … R-11) addressing the design audit; new §10.0 infrastructure
  prep section; pilot deliverables tightened around AppWorld's
  actual data shape.
* `motivation_v2/` — NEW track for AppWorld experiments:
  * `motivation_v2/data.py` — load AppWorld ground truth + acon
    trajectories.
  * `motivation_v2/policy_family.py` — task → policy family
    classifier (single-app rule, supervisor=plumbing).
  * `motivation_v2/exec_memory.py` — both `m*_exec_minimal` and
    `m*_exec_trajectory` builders, deterministic, no LLM.
  * `scripts/smoke_data_pipeline.py` — corpus audit + smoke test
    on existing 4 successful trajectories. Already verified working.
  * `motivation_v2/README.md` — track overview.
* No `motivation_v2/runner.py` or `compressors.py` yet — those need
  AppWorld imports which require the acon `.venv`. Next session.

---

## §Redesign (2026-05-24 18:35 UTC) — read this if you're picking up here

The old motivation track has a fatal logical issue: **T2 (LLM
selectors don't policy-condition) being true makes the T1 hinge test
(`instance_noise` cross/within ratio) unsatisfiable**, because both
within-agent and cross-agent candidates come from the same
unconditioned LLM and are interchangeable by T2. There is no
LLM-independent ground-truth memory in that pipeline to anchor T1.

**The redesign**: switch motivation main track to AppWorld, where
"execution-derived memory" `m*_exec` (the API calls / DB rows the
gold solution touches) is a non-LLM oracle that anchors T1
independently of any selector. T2 is then tested as "can a prompted
LLM selector reproduce m*_exec's downstream task success?".

**Engineering**: leverage Microsoft's ACON repo (already at
`/workspace/acon`, arXiv 2510.00615) for the AppWorld + agent-
runner pipeline; build only the compression analysis on top.

### Where to pick up

1. Run `motivation_v2/scripts/smoke_data_pipeline.py` — verifies
   data pipeline + prints the corpus audit.
2. Read `motivation/docs/new_motivation.md`. The 11 review notes
   describe the design fixes vs the original draft.
3. Decide: launch AppWorld trajectory generation? It's a ~22 h
   sequential job (147 tasks × ~10 min each). Could be parallelised
   externally with multiple `run_all.py` processes if endpoint
   concurrency allows. Without trajectories, M1 / M2 / M3 can't run
   on real data.
4. Build `motivation_v2/compressors.py` (`m_recent`, `m_freq`, BM25,
   embedding-top-k) — no LLM needed.
5. Build `motivation_v2/runner.py` — wraps acon's `AppWorldAgent`
   with a "compressed-memory mode" that injects `m*_exec` (or a
   baseline) instead of the full env state. **Requires acon `.venv`,
   not EASMO `.venv`.**
6. End-to-end M1 pilot on spotify-only subset.

### Anti-patterns to avoid (from §13 of new_motivation.md)

* Do not use LLM-generated oracle memory as gold (circular).
* Do not use binary action-match as the headline metric.
* Do not use ReAct/Plan/CoT as the policy distinction.
* Do not use token-level Jaccard or leave-one-out as killer metrics.

---

## §X-Validation (2026-05-24 18:50 UTC) — Option X works

### What changed conceptually

The user pointed out that "policy = task family" in the original
new_motivation.md draft was a topic-mismatch test, not a policy-
mismatch test. Spotify-vs-venmo memory is *trivially* mismatched
because the data is unrelated. The corrected design (§2.4 of
`new_motivation.md`) defines "policy" as a **behavioural strategy**
applied to the same task with the same executor:

* **`P_direct`**: minimum API calls, answer-first.
* **`P_verify`**: mandatory cross-validation through a second source.
* **`P_explore`**: list apps + APIs upfront before computing answer.

This is **Option X** in the trade-off space. Option Y (different
executor models) is documented as backup but requires endpoint
coordination the user said is non-trivial.

### What got built

* `motivation_v2/prompts/STRATEGY_DESIGN.md` — canonical strategy
  texts + manipulation-check spec.
* `motivation_v2/prompts/build_strategy_prompts.py` — splices
  strategy block into a copy of acon's `prompt_v1.jinja`,
  materialises files under
  `acon/experiments/appworld/prompts/_motivation_v2/<strategy>/`.
* `motivation_v2/scripts/run_appworld_strategy.py` — launcher
  that wraps acon's `run.main` with a strategy-specific
  `prompt_file`. Same trajectory schema, same output layout.
* `motivation_v2/scripts/manipulation_check.py` — measures whether
  strategy directives changed agent behaviour (iter counts, API
  call patterns, exploration/verification proxies).

### Smoke result (task `82e2fac_3`, MiniMax-M2.5)

| | iters | input tokens | unique APIs | total API calls |
|---|---|---|---|---|
| baseline (no strategy) | 11 | 46K | 8 | 11 |
| `P_direct` | 11 | 57K | 7 | 11 |
| `P_verify` | **26** | **204K** | **11** | **36** |
| `P_explore` | 14 | 82K | 8 | 14 |

Manipulation-check verdict: ✓ verify/direct iter ratio = 2.36× ≥ 1.5;
✓ explore `show_app_descriptions` in first 3 steps = 100% ≥ 50%.
Strategies are demonstrably different in trajectory content (verify
fetched 17× show_song for cross-check, plus show_album and
show_profile that no other strategy touched).

### Where to pick up

1. The pipeline is end-to-end working on 1 task per strategy.
   Trajectories live at
   `acon/experiments/appworld/outputs/MiniMaxAI_MiniMax-M2.5_mv2_smoke_<strategy>/train/task_82e2fac_3/`.
2. Next step is the **3-strategy × 90-task pilot run** (~7h
   sequential, ~3h with 3 parallel jobs). Launch with:

   ```bash
   for s in direct verify explore; do
     nohup /workspace/acon/.venv/bin/python \
         /workspace/EASMO/motivation_v2/scripts/run_appworld_strategy.py \
         --strategy $s --split train --tag mv2_pilot --continue_existing \
         > /workspace/EASMO/motivation_v2/outputs/${s}_pilot.log 2>&1 &
   done
   ```

3. After all three finish, run manipulation_check on the full set:

   ```bash
   /workspace/acon/.venv/bin/python \
       /workspace/EASMO/motivation_v2/scripts/manipulation_check.py \
       --tag mv2_pilot
   ```

4. If manipulation check passes (verify/direct iter ratio ≥ 1.5,
   explore first-3-step exploration ≥ 50%), proceed to build the
   compression analysis layer:
   * `motivation_v2/motivation_v2/compressors.py` — `m_recent`,
     `m_freq`, BM25, embedding-top-k baselines.
   * `motivation_v2/motivation_v2/runner.py` — wraps
     `AppWorldAgent` to inject a compressed-memory string in place
     of the full env state, run the agent, return task success.
     This requires acon's `.venv`.
   * `motivation_v2/scripts/run_m1.py` — compression-pressure
     sweep across (B, strategy, memory_variant) on the pilot
     task subset.

5. If manipulation check FAILS (strategies indistinguishable on the
   90-task scale despite passing on 1 task), fall back to Option Y
   (executor variants) per §2.4c.
