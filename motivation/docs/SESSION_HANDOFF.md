# Session handoff — paste this into a new chat if context fills up

> Updated: 2026-05-31 11:20 AM PT.
> All times in Pacific Time (PT).
>
> **➡ For a fresh chat, read these in order:**
> 1. This file (project map + open threads).
> 2. The single track-level snapshot you care about most — each
>    track has a `docs/04_results_summary.md` (v6, v7, v8, v9) or
>    `docs/05_results_summary.md` (v2–v5) that's decision-ready in
>    ~5 minutes.
>
> The git remote is `git@github.com:GuanghuiMin/EASMO.git` (SSH).
> Latest pushed: v9 **first-pass + widened-n addendum both complete**
> (addendum finished 2026-05-29 2:15 PM PT after 64 min wall-clock).
> All earlier tracks v2-v8 still pushed and final.
>
> Auto-push watcher PID 3916707 stages every 20 min and covers
> `motivation/`, `motivation_v2/`, `motivation_v3/`, `motivation_v4/`,
> `motivation_v5/`, `motivation_v6_jacobian/`, `motivation_v7/`,
> `motivation_v8/`, **`motivation_v9/`** automatically.

## 0. Known gotcha — Cursor SSH agent forwarding can break silently

The auto-push watcher pushes via SSH using a forwarded agent socket
at `/tmp/cursor-remote-ssh-auth-sock-…sock`. That symlink is owned
by Cursor and points at whichever `/tmp/ssh-XXXX*/agent.<pid>` socket
Cursor's remote SSH integration is currently using. **When Cursor
reconnects, the symlink may be left pointing at a stale socket whose
upstream agent forward is dead**, which makes every git push hang
and time out (rc=124) while watcher silently logs `git push timed
out / failed` and re-tries every 20 min.

Diagnosis recipe:

```bash
# 1. Watcher's symlink and what it points at:
ls -la /tmp/cursor-remote-ssh-auth-sock-*.sock
# 2. Which agent actually has keys? (one with gmin@M-H3QF046Q6L on it)
for s in $(ls -t /tmp/ssh-*/agent.* 2>/dev/null | head -10); do
  out=$(timeout 3 env SSH_AUTH_SOCK=$s ssh-add -l 2>&1 | head -1)
  echo "$s : $out"
done
# 3. Repoint the cursor symlink to the live one:
ln -sf /tmp/ssh-XXXXXXXX/agent.<pid> /tmp/cursor-remote-ssh-auth-sock-*.sock
# 4. Verify watcher will work on its next cycle:
timeout 6 env SSH_AUTH_SOCK=/tmp/cursor-remote-ssh-auth-sock-*.sock ssh-add -l
```

The user's key is `gmin@M-H3QF046Q6L` (ED25519). Repo is private; no
PAT/`~/.netrc` exists in the workspace, so SSH is the only working
auth path. SSH itself does work — both `git@github.com:22` and
`ssh.github.com:443` reach GitHub fine **once the agent socket is
live**.

If you can't find a live agent at all, the user must reconnect their
local Cursor → remote SSH session (which re-creates the forward),
then run a `git push` to seed it.

## 1. Project map (7 motivation tracks)

| Track | Purpose | Status | Headline |
|---|---|---|---|
| `motivation/` | original LongMemEval / LoCoMo idea | **abandoned 2026-05-24** | see §Abandoned-track at end |
| `motivation_v2/` | role-conditional memory in multi-agent systems (T1 + T2) | ✅ done; **6/7 spotlight criteria, Qwen blocked** | three-tier Jaccard hierarchy strategy 0.62 → task 0.28 → role 0.14 (entity-token); prompted-compression code-role recall 5% across all variants |
| `motivation_v3/` | LLM compressor comparison (NL / ACON / symbolic) + behavioral utility | ✅ done | structural-vs-behavioral ranking-mismatch: symbolic 99.5% coverage but **57% downstream success at cap=15**; ACON 67%; NL 70% |
| `motivation_v4/` | decision-state probing: leave-one-span-out sensitivity | ✅ done; **4/6 spec criteria** | high-sens > low-sens (+23pp), > random (+7pp); but recency baseline beats high-sens (47 vs 40 at cap=15) |
| `motivation_v5/` | **ACON failure-mode audit** (recovered-then-dropped) | ✅ done; **strong positive signal verified** | 93% of audit-recovered actionable items are dropped again by recompression; 81% by abstraction policy not capacity; recompressed-context rerun saves 5/24 (21%) of originally-failed cases |
| `motivation_v6_jacobian/` | **white-box Jacobian active-subspace diagnostics** (Qwen3-4B-Instruct-2507) | ✅ done 2026-05-28 | **B positive** (example-level k=16 cum-var = 92 %); **A negative** (per-task median Spearman vs v4 = −0.03); **D negative** (jacobian_low_spans 0.80 ≈ high_spans_raw 0.83 at MiniMax cap=15); **C degenerate** (k=4 soft tokens already overfit target NLL with gap-recovery 2.26×). Story: kills span-rank selection, supports active-subspace projection. |
| `motivation_v7/` | **abstraction prior + iterative compression dynamics** with official ACON UTCO prompt (Qwen3-4B-Instruct-2507 + MiniMax-M2.5) | ✅ done 2026-05-28 | **Claim A STRONG positive 5/5 (Qwen), 4/5 (MiniMax):** SDI = 0.96 / 0.99 — McFadden R² of need_label = 0.003/0.0006 vs R² of fact_type = 0.155/0.110 (50–180× gap). **Claim B STRONG positive 5/5 / 4/5**: cross-model Kendall τ = 0.491 (p=0.041), 79.3% chains converge in ≤5 rounds, AUTH_OR_ACCESS_TOKEN has lowest AUSC in both models. Story: LLM compressors are unconditioned surface-type abstraction priors; tokens/IDs/paths die fast regardless of need. |
| `motivation_v8/` | **fixed-point analysis of GENERAL (non-ACON) LLM compression** + basin-of-attraction experiment (same 30 cases + 233 facts + 150 quality pairs reused from v7) | ✅ done 2026-05-28 | **v7's abstraction prior REPLICATES and STRENGTHENS under general prompts:** SDI under P2 task-agnostic = **1.000 / 0.998** (vs v7 ACON 0.96/0.99); cross-model Kendall τ up to **0.778** (vs v7 0.49). **Two new mechanisms identified:** (1) P1 task-aware **inverts** the fixed-point composition from NARR>EXEC (P2 0.88 vs 0.55) to EXEC>NARR (P1 0.64 vs 0.46) — task framing reshapes the attractor. (2) Different inits (RAW_FULL/DETAIL_HEAVY/NARRATIVE_HEAVY/FACT_TABLE_ONLY) reach **disjoint** fixed points (Jaccard distance up to 1.00) — no universal attractor. Δ_need^∞ for executable facts = **+0.27 under P1** — moderate, and **strengthens across iterative rounds** vs single-round Δ_need. |
| `motivation_v9/` | **behavior-first** validation: Best-of-N ACON, C1-vs-CK fragility under repeated-compression stress, NL chunk information advantage (MiniMax-only primary; reuse v3 30 cases + ACON UTCO) | ✅ done 2026-05-29 (first-pass + widened addendum) | **Claim 1 STRONG positive** (best-of-N vs greedy: C1 +26.7 pp, CK +36.7 pp pass-rate; oracle_win 90/83%). **Claim 2 POSITIVE** (greedy fragility 28.6%, stress drop 10 pp; greedy more fragile than sample, 28.6% vs 21.8%). **Claim 3 NEGATIVE at n=239** (originally "STRONG positive at n=144 first-pass" did NOT survive widening: causal-flag chunks mean adv +0.036 vs ENTITY_LIST_ONLY +0.167 — direction reverses, likely because the labeler's "entity list" describes form not function). Only Claim-3 sub-finding to survive widening: **CONTROL_NEGATIVE_EVIDENCE** (n=13, mean +0.115, echoes v5's lost-failure-log bottleneck). Total pipeline 2h 16min first-pass + 64 min widened addendum. |
| `motivation_v10/` | **trainable compressor policy**: ACON best-of-N → proxy reward → Qwen3-4B LoRA SFT (C1 vs CK selection criteria) → GRPO readiness check; chunk reanalysis with v10 enriched §17.5 schema. Spec: `user_feedback/motivation_v10_proxy_sft_grpo_readiness_v2.md`. | ✅ stages 00-10 + 12 done 2026-05-29 → 5/31 10:56 PM PT (32 h total wall-clock); stage 11 chunk reanalysis still stubbed (diagnostic, doesn't gate Go/No-go) | **Final spec §19 verdict: 2 of 3 testable claims PASS, 1 PARTIAL, 1 FAIL.** Claim 1 (proxy recovers best-of-N): **FAIL on CK** (pairwise +4 pp / 16 % recovery). Claim 2 (SFT-CK > SFT-C1 > Raw on CK): **PARTIAL ✓** (aggregate ✓, held-out test_behavior 12 cases too thin). Claim 3 (GRPO readiness reward spread): **PASS ✓ for all 3 variants** (std 0.42-0.47, oracle_win 0.81-0.83, all_fail 0). **🌟 Three BONUS paper-quality findings**: (a) SFT stress robustness — Raw-Qwen drops −11.9 pp C1→CK while SFT-CK *gains* +7.2 pp; (b) **SFT compression is a stress-invariant fixed point** — Raw-Qwen output gets compressed −33% by MiniMax stress, SFT output drifts only +4.7% (the causal mechanism behind (a)); (c) Sample diversity perfect (8/8 unique per case), SFT students sample 2.5-3× wider length distribution than Raw-Qwen. **⚠ Important caveat**: verifier reward ranks Raw-Qwen > SFT (greedy verifier score 0.82 vs 0.64) but actual AppWorld pass ranks SFT-CK > Raw-Qwen — DO NOT use verifier composite as GRPO reward; use true Pass. Full paper-tier writeup at `motivation_v10/docs/04_results_summary.md`. |

Each track folder follows the same shape:

```
motivation_v?/
├── docs/
│   ├── 00_spec.md (symlink)             user-feedback spec for that round
│   ├── 01_experimental_design.md        full design + pipeline
│   ├── 02_*.md                          prompts (paper appendix)
│   ├── 03_*.md                          definitions / taxonomy
│   ├── 04_*.md                          conditions / setup (v6: ★ results_summary)
│   └── 05_results_summary.md            ★ decision-ready snapshot (v2–v5)
├── prompts/                             (v3+v5) verbatim prompt templates
├── motivation_v?/                       python package
├── scripts/                             stage scripts + run_all.sh
├── data/                                (v5) raw + sampled cases
└── outputs/
    ├── raw/                             JSONL + .npz active vectors (v6)
    ├── tables/                          CSV
    ├── figures/                         PDF + PNG
    ├── reports/                         markdown
    └── sprint_logs/                     run logs (gitignored *.log)
```

Pick the right paper-tier summary based on what you're working on.
For v2–v5 it's `docs/05_results_summary.md`; for v6 it's
`docs/04_results_summary.md` (spec layout). They're 200–340 lines
each and self-contained.

## 2. Latest-round summary (motivation_v9, today's work)

**Goal**: behavior-first validation of v7/v8 abstraction-prior story
on AppWorld, with three orthogonal claims:
1. ACON greedy decoding is NOT best-of-N-optimal under its own distribution.
2. One-step compression is fragile under repeated-compression stress T^K.
3. Natural-language chunks carrying causal/control content drive more
   downstream behavior than chunks that are pure entity lists.

**Outcome (n_cases=30 from v3, n_candidates=270, n_behavior_runs=540,
n_chunks=144 first-pass / ~240 widened-pending)**:

* **Claim 1 STRONG POSITIVE.** Best-of-N (N=8) over MiniMax-M2.5 stochastic
  samples beats greedy by **+26.7 pp** on C1 and **+36.7 pp** on CK
  pass-rate. Oracle win rate 90% (C1) / 83% (CK). Mean compressed length
  is also shorter for best-of-N (~451 vs 487 chars on C1), so the gain
  is not length-mediated.
* **Claim 2 POSITIVE.** Repeated compression T^K with K=2 drops greedy
  pass-rate from 70% → 60% (10 pp); 28.6% (6/21) of originally-passing
  greedy candidates fail after recompression. **Bonus finding:** sampling
  is more robust than greedy — sample fragility 21.8% vs greedy 28.6%.
  This is consistent with greedy attaching to brittle surface features
  that recompression discards.
* **Claim 3 NEGATIVE at n=239 (widened addendum).** First-pass n=144
  showed ENTITY_LIST_ONLY mean adv 0.000 and a "causal-flag" cut at
  +0.150 (n=20), which looked like a STRONG positive. The widened
  n=20 chunk-cases run (239 chunks, finished 2026-05-29 2:15 PM PT)
  REVERSED direction: causal-flag mean +0.036 (n=28 unique chunks) vs
  ENTITY_LIST_ONLY mean **+0.167** (n=27 unique chunks). The
  first-pass "ACTION_OUTCOME wins" surprise also reversed: n=59 mean
  +0.068 → n=105 mean **−0.057**. **Only chunk-type with a stable
  positive mean across both runs is CONTROL_NEGATIVE_EVIDENCE**
  (n=4 → n=13, mean 0.000 → +0.115, 47.4% causal-flag rate). This
  echoes v5's "looked_like_past_log" recompressor-drop pattern. The
  spec's directional prediction (causal > entity) **does not survive
  widening** — likely because the labeler's `ENTITY_LIST_ONLY`
  describes form (compact token lists) not function (those tokens are
  often the actual `access_token`/`target_id` the next step needs).

**Methodological footgun (caught + fixed mid-day)**:
* Stage 11 first shipped with `max_tokens=256` for MiniMax label calls.
  MiniMax-M2.5 thinking blocks alone are 500-750 tokens (median 543),
  so the entire budget was consumed by `<think>...</think>` and the
  post-strip JSON payload was empty. All 144 chunks fell back to the
  default `OTHER` label. Stage 11 logged `err=0` because the API
  itself didn't fail. **Fixed:** bumped `max_tokens` 256 → 2048 in
  `motivation_v9/chunk_label.py`. **Guard added:** `clients.py` now
  exports `WARN_THINKING_MIN_MAX_TOKENS=1024` and `chat()` prints a
  warn-once message when MiniMax is called below that threshold.
  Empirically-derived from stage 02 (n=270 compressions, median
  thinking 543 tokens, p90 750).

**Caveats** (must include in any paper writing):
* Both first-pass (n=144) and widened (n=239) auto-written
  `outputs/reports/motivation_v9_results_summary.md` say Claim 3
  STRONG POSITIVE. The hand-written `docs/04_results_summary.md`
  §6+§10 supersedes that: at n=239, Claim 3 is **NEGATIVE** as
  originally stated. Subordinate finding to keep: CONTROL_NEGATIVE_EVIDENCE
  (n=13, mean +0.115) is the only stable positive chunk-type
  category across n=12 → n=20 case widening.
* `chunk_information_advantage.csv` has a large fraction of rows
  where `score_full=1.0, score_minus_chunk=1.0` (chunk not causally
  necessary). This is real, not a bug — most chunks are removable.

For full details + tables + figures see
[`motivation_v9/docs/04_results_summary.md`](../../motivation_v9/docs/04_results_summary.md).

## 3. Endpoints + venvs

| Service | URL | Use |
|---|---|---|
| **MiniMax-M2.5** vLLM | `http://10.183.22.68:8005/v1` (model id `MiniMaxAI/MiniMax-M2.5`, max_model_len ~32K) | All v2/v3/v4 LLM work + v5 verifier + v5 recompressor + downstream agent for all tracks |
| **Qwen3-4B** vLLM (local) | `http://127.0.0.1:8000/v1` (model id `qwen3-4b`, max_model_len 8192) | v5 primary auditor; started via `bash /workspace/qwen3-vllm/serve.sh` (server.pid in that dir; `kill $(cat /workspace/qwen3-vllm/server.pid)` to stop) |

Three Python venvs in use:

| venv | Purpose |
|---|---|
| `/workspace/EASMO/.venv/bin/python` | analysis, matplotlib, numpy, pandas, openai client, post-hoc compute |
| `/workspace/acon/.venv/bin/python` | productive_agents / AppWorld agent runner — anything that touches `appworld` or `acon_run` |
| `/workspace/qwen3-vllm/.venv/bin/python` | vLLM 0.10.2 cu128 build (don't touch; serves the Qwen endpoint) |

Note: `EASMO/.venv` cannot import AppWorld (pydantic v1/v2 conflict). Always
use `acon/.venv` for `appworld` / `productive_agents`.

## 4. Active threads / what's pending

### v2 — Qwen cross-executor (only blocking item across the whole project)
* All 6 of 7 v2 spotlight criteria done; criterion #5 (cross-executor robustness on Qwen2.5-7B) is the one remaining.
* Plan in `motivation_v2/user_feedback/experiment_modification.md` §9: rerun A (hierarchy at B=512), D (prompted at B=512), C (behavior cost at B=512, max_iter=15) on Qwen.
* Compute: ~6h on Qwen endpoint + 30min analysis.
* Now that we have a local Qwen3-4B server, this could in principle be done locally (but Qwen3-4B is much smaller than Qwen2.5-7B; using a different model size would change the conclusion). User originally said "外部 endpoint 协调中".
* **NOTE for v6 era**: The local Qwen3-4B vLLM server (port 8000) was killed on 2026-05-27 to free GPU for v6's white-box gradient pass. Restart with `bash /workspace/qwen3-vllm/serve.sh` if v2 #5 picks up locally.

### v3
* Done. Recovery counts in tables/. Methodological self-correction (previous "6.0× ratio" was apples-to-oranges) is documented in `docs/05_results_summary.md`.
* No open thread.

### v4
* Done. 4/6 spec criteria. Headline negative result: **decision-state probing identifies real signal but recency baseline beats it on AppWorld task continuation**.
* Future work (per `docs/05_results_summary.md` §7): cross-model probe, multi-step span granularity, benchmarks where recency is a weak baseline.

### v5
* Done 2026-05-27. Strong positive signal.
* Possible follow-ups (not requested):
  - Run the audit pipeline on remaining 36 non-Tier-1 cases (when ACON succeeded) to measure baseline grounded-additions and check whether the recompressor drops less when ACON didn't fail.
  - Add easy/medium difficulty cases by running full-context on shorter v3 dev tasks.
  - Cross-model audit: let GPT-4o-mini audit instead of Qwen3-4B, see if `LOST_ACCESS_TOKEN` non-spec-label issue disappears.
  - Implement a method (preserve-by-construction tags) that acts on the bottleneck and verify >21% recovery rate.

### v6
* Done 2026-05-28. **Mixed result: A negative, B positive (STRONG), C degenerate, D negative.**
* Full paper-tier report at `motivation_v6_jacobian/docs/04_results_summary.md`.
* Methodological negatives to remember:
  - First-order embedding Jacobian on Qwen3-4B doesn't predict v4 span sensitivity (median Spearman −0.03 across 28 tasks).
  - Jacobian has length bias (+0.36 Spearman with token count).
  - Soft-token oracle on v4 reference-state target is degenerate: k=4 already over-recovers with gap recovery 2.26× because soft-token bandwidth exceeds target entropy.
* Possible follow-ups (not requested):
  - C2 with held-out next-action target (vs reference decision state) to fix soft-token degeneracy.
  - KL-divergence-against-full-context as the soft-token objective instead of teacher-forced NLL.
  - Active-subspace-preserving compressor: project residual stream onto top-k SVD of active vectors, then construct soft prompt that reproduces those directions. Compare against recency / ACON downstream.
  - Cross-model: run the same pipeline on Llama-3-8B or another instruction-tuned 7-13B model to see if the negative A result is model-specific.

### v7
* Done 2026-05-28. **STRONG positive on both Claim A and Claim B** (the project's first cleanest paper-tier result; v8 builds on it).
* Full paper-tier report at `motivation_v7/docs/04_results_summary.md`.
* Compressors used: Qwen3-4B-Instruct-2507 (port 8000 vLLM) + MiniMax-M2.5 (10.183.22.68:8005). ACON prompts loaded verbatim from `/workspace/acon` (commit d63f9ae): UTCO = `improved_history_prompt_samples_4.jinja`.
* Methodological negatives + caveats:
  - Need-condition generator (MiniMax) produces matched pairs that pass quality checks 64% of the time (150/233). Writing pure-need counterfactuals is itself non-trivial.
  - Neither UT nor UTCO has a `max_chars` template variable; outputs are 2,000–2,800 chars vs nominal 1,500-char budget.
  - Plan B scope: iterative chains are 2 per case (needed + unneeded for one EXECUTABLE fact), not the spec's full sweep.
  - v3-derived 30 cases all medium/long (≥15 steps); no short stratum.
* Possible follow-ups (largely addressed by v8):
  - Run UT ablation — partially done by v8 P2 general task-agnostic.
  - Causal intervention: insert "preserve access_token verbatim" — done by v8 P1 task-aware.
  - GPT-4.1-mini as a 3rd compressor — still open.

### v8
* Done 2026-05-28. **Refines and generalises v7.** v7's abstraction prior REPLICATES under general (non-ACON) prompts; v8 also discovers task-aware framing flips the fixed point and that different inits give disjoint fixed points.
* Full paper-tier report at `motivation_v8/docs/04_results_summary.md`.
* Headline numbers vs v7:
  - SDI under P2 task-agnostic = **1.000 (MiniMax) / 0.998 (Qwen)** (vs v7 ACON 0.961 / 0.989) — even more extreme.
  - Cross-(model, prompt) Kendall τ up to **0.778** (vs v7 cross-model τ = 0.491).
  - Convergence rate 84.9% across 186 chains (≥6 rounds).
  - **NEW: P1 task-aware inverts fixed-point composition** to EXEC > NARR (MiniMax 0.64 vs 0.46, Qwen 0.63 vs 0.29) — task framing reshapes the attractor.
  - **NEW: Fixed-point Δ_need^∞ ≈ +0.27 for executable facts** under P1 — moderate need effect that strengthens across iterative rounds vs single-round.
  - **NEW: Different initialisations reach disjoint fixed points** (RAW_FULL vs FACT_TABLE_ONLY: fact-Jaccard distance 1.00 under MiniMax P1) — no universal attractor.
* Methodological caveats:
  - Stage 06 retention scoring had ~28% error rate (LLM JSON parse failures), biasing retention rates uniformly downward 3-5 pp. Contrasts (Δ_need, SDI) robust.
  - PIR small samples (n=15 / cell): CIs wide. Don't over-read individual PIR numbers.
  - Basin contraction ratio not well-behaved when init distance = 0 (some inits start identical). Use absolute final distance instead.
  - Plan B scope: P3 strict-extract ablation + secondary budgets deferred.
* Possible follow-ups:
  - Re-score retention with stricter JSON post-processing to tighten estimates.
  - P3 ablation: does an extra "first identify exact facts" instruction step further amplify need-conditioning at fixed point?
  - Method round: project residual stream onto top-k SVD of v6-style active vectors AND condition on a P1-style task prompt — does the combination exceed either alone in downstream success?

### v9
* **First-pass done 2026-05-29 11:50 AM PT** (pipeline 9:34→11:50, 2h 16min, ~3× faster than spec's 6-7h estimate). **Widened-n addendum in progress** (kicked off 1:11 PM PT, ETA ~55 min).
* Full paper-tier report at `motivation_v9/docs/04_results_summary.md`.
* Compressors used: MiniMax-M2.5 only (spec §3.3). ACON UTCO prompt loaded verbatim from `/workspace/acon` commit `d63f9ae`, sha256 of prompt = `9e50d0f93aca7f75eb723a90a758642d1aac3d7550f6afe1e692e56e2bc7b71c`.
* Three claims:
  - **Claim 1 STRONG positive**: best-of-N (N=8) beats greedy by **+26.7 pp pass-rate on C1 and +36.7 pp on CK**. Oracle win rate 90% / 83%. Best-of-N samples are also shorter (~451 vs 487 chars) — gain is not length-mediated.
  - **Claim 2 POSITIVE**: K=2 repeated-compression drops greedy pass-rate 70%→60% (fragility rate 28.6%). **Sample compressions are more robust than greedy** (fragility 21.8%) — argues against ACON greedy decoding for stress-robustness too.
  - **Claim 3 NEGATIVE at n=239** (after widened addendum, which superseded the n=144 first-pass "PARTIAL POSITIVE" reading). At n=12 case selection the causal-flag aggregation showed +0.150 vs +0.000 for entity-only — that did NOT survive widening to n=20 cases / n=239 chunks: causal-flag mean +0.036 (n=28 unique) vs ENTITY_LIST_ONLY **+0.167** (n=27 unique) — direction reverses. ACTION_OUTCOME's "surprise winner" also reverses (n=59 +0.068 → n=168 −0.057). Stable cross-run positive: CONTROL_NEGATIVE_EVIDENCE (n=4 → n=13, mean 0.000 → +0.115) — echoes v5's lost-failure-log bottleneck. Spec hypothesis "causal NL > entities" thus fails the chunk-level ablation test, but the CONTROL_NEGATIVE_EVIDENCE sub-finding is paper-quotable on its own.
* **Bug fix shipped**: stage 11 first ran with `max_tokens=256` and all 144 labels silently fell back to `OTHER` because MiniMax thinking blocks alone are ≥256 tokens. Fixed by bumping to 2048 and adding a `WARN_THINKING_MIN_MAX_TOKENS=1024` warn-once guard in `motivation_v9/clients.py` that fires when MiniMax is called below the empirical threshold. (See §6 pitfalls below.) Re-ran stages 11-14 in ~5 min and got the real labels.
* Caveats:
  - Even at widened n=20 cases (n_chunks=239), CAUSAL_PRECONDITION n_unique=5 and CONTROL_NEGATIVE_EVIDENCE n_unique=13 — the small-n problem for the causal categorical types persists. The flag-level cut (n_unique=28 for causal flag) is larger but the direction still goes the "wrong" way per spec.
  - Auto-written `outputs/reports/motivation_v9_results_summary.md` is mechanical and continues to over-claim Claim 3; `docs/04_results_summary.md` §6+§10 is the honest hand-written counterpart.
  - All cases reused from v3 dev (medium/long: ≥15 steps); no short-trajectory stratum.
* Possible follow-ups (not requested):
  - Method round: train compressor with reward = `behavior_after_stress(T^K) − λ·length`. Claim 1 + 2 jointly motivate this exactly.
  - IAPO-style natural-language credit assignment: assign reward at chunk level using the Claim 3 chunk-advantage signal.
  - Cross-model verification: same pipeline with Qwen3-4B as compressor. Spec §3.3 forbade Qwen as labeler but not as compressor.

## 4.5. GPU / endpoint state (as of 2026-05-29 12:15 PM PT)

* **Qwen3-4B-Instruct-2507 vLLM** still running on port 8000 (PID 1114353, started 2026-05-28 11:43 AM PT — uninterrupted since). Served model id = `qwen3-4b-instruct-2507`. Launch script: `/workspace/qwen3-vllm/serve_instruct.sh`. Stop with `pkill -f 'served-model-name qwen3-4b-instruct-2507'`. v9 itself does NOT use Qwen (spec §3.3 forbids it as labeler) but the server is kept up for v2 #5 follow-up if it ever picks up.
* The older base-model serve (`Qwen/Qwen3-4B` on port 8000 as `qwen3-4b`) is **stopped** since 2026-05-27. Launch script preserved at `/workspace/qwen3-vllm/serve.sh`. Cannot run both at once — same port.
* GPU memory used: ~56 GB / 80 GB. ~25 GB free.
* MiniMax-M2.5 endpoint at 10.183.22.68:8005 has been continuously available across v4-v9. v9 pipeline confirms it can sustain ~12.8 AppWorld agent runs/min through 6 parallel workers.

## 4.6. Running background processes (as of 2026-05-29 12:15 PM PT)

```
PID 3916707  bash /workspace/EASMO/motivation/scripts/auto_push_watcher.sh
             log: /tmp/easmo_watcher.log
             healthy — last push 19:55Z (12:55 PM PT).
PID 1114353  python -m vllm.entrypoints.openai.api_server
             --model Qwen/Qwen3-4B-Instruct-2507 --port 8000
             log: /workspace/qwen3-vllm/server_instruct.log
```

**No experiment processes active as of 2026-05-31 11:20 AM PT.**
v10 chain finished 05:56Z (10:56 PM PT 5/30). All raw outputs and
tables in `motivation_v10/outputs/`; auto-written report at
`outputs/reports/motivation_v10_results_summary.md`; honest
hand-written companion at `motivation_v10/docs/04_results_summary.md`.

vLLM (Qwen3-4B-Instruct-2507 port 8000) is still **stopped** since
stage 08 (18:15Z 5/30). Restart with:
`nohup bash /workspace/qwen3-vllm/serve_instruct.sh > /workspace/qwen3-vllm/server_instruct.log 2>&1 &`
if needed (only matters for v2 #5 follow-up or future Raw-Qwen
serving — not required for any current v10 stage).

## 5. Active background processes

See §4.6 above for the authoritative current snapshot. Quick refs:

```
# auto-push watcher — stages all motivation_v*/ + motivation/ → git → push every 20 min.
PID 3916707  bash /workspace/EASMO/motivation/scripts/auto_push_watcher.sh
log: /tmp/easmo_watcher.log

# Qwen3-4B-Instruct-2507 vLLM server (port 8000, model id qwen3-4b-instruct-2507)
PID 1114353
log: /workspace/qwen3-vllm/server_instruct.log
```

Stop the watcher: `pkill -f auto_push_watcher.sh`.
Stop Qwen vLLM: `pkill -f 'served-model-name qwen3-4b-instruct-2507'`
(or `pkill -P 1114350` to take down the wrapper too).

No experiment processes running at this snapshot. All v3/v4/v5/v6/v7
pipelines finished and pushed to git.

## 6. How to extend (typical next-round patterns)

If a user_feedback spec arrives at e.g. `/workspace/user_feedback/<spec>.md`
and asks for a new motivation track, the established convention is:

1. Read the spec end-to-end. Note any external dependencies (new model, new endpoint, new benchmark).
2. Audit existing data (especially v3's 30 dev trajectories — `motivation_v3/outputs/motivation_full_trajectories.jsonl`) to see what you can reuse for free.
3. Propose a plan with **explicit cost estimate** (LLM calls, agent runs, wall-clock) and **scope decisions** (use AskQuestion for n_cases / verifier model / optional conditions). Get user buy-in before writing code.
4. Build under `motivation_v?/`:
   - python package (`motivation_v?/`)
   - 5 v2-style docs in `docs/` (write 01-04 up front; 05 after the run with real numbers)
   - prompts in `prompts/` if multi-prompt
   - stage scripts in `scripts/0?_*.py` + a `run_all.sh` orchestrator
5. Smoke test imports + 1-2 LLM calls + 1 agent cell before kicking off the full pipeline.
6. Kick off as `nohup bash scripts/run_all.sh > outputs/sprint_logs/runall_full.log 2>&1 &` so the long-running compute survives shell hangs.
7. After the pipeline completes, write `05_results_summary.md` from the merged stats and commit + push.
8. Update *this* SESSION_HANDOFF.md (Project Map + Active Threads).

Common pitfalls observed across rounds:

* **MiniMax-M2.5 emits `<think>...</think>` blocks** that take real tokens. For JSON mode, set `max_tokens ≥ 2048` and strip `<think>` post-hoc. v5's `clients.parse_json` is the canonical implementation.
  * **Quantitative guard (v9, 2026-05-29)**: median thinking 543 tokens, p90 750, max 1361 (measured from v9 stage 02 n=270 compressions). Any `max_tokens < 1024` for MiniMax is dangerous; v9 stage 11 first shipped with `max_tokens=256` and all 144 chunk labels silently fell back to defaults because thinking ate the whole budget and `_strip_think()` returned `""`. **The bug is invisible** — the API call succeeds, no exception is raised, no error is logged. Fixed in `motivation_v9/clients.py` with `WARN_THINKING_MIN_MAX_TOKENS=1024` and a warn-once `chat()` guard. Use this pattern going forward.
* **Qwen3-4B JSON mode + thinking interaction**: pass `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` to disable Qwen's thinking; otherwise short max_tokens cuts off mid-think. v5's `clients.chat_qwen` does this. (MiniMax does NOT support this kwarg — its chat template doesn't recognise it; you must rely on the `max_tokens` budget being large enough.)
* **Python `str.format()` chokes on JSON schemas with literal `{` `}`**. Use jinja-style `{{name}}` placeholders and a regex renderer (v5's `clients.render_template`).
* **Qwen3-4B max_model_len = 8192**. Long AppWorld trajectories must be truncated before being sent to Qwen. v5's `clients.pack_prompt_for_qwen` does proportional shrinking of long fields.
* **Python format-style placeholders that match `{name}` in prose** silently break things. Always smoke-test with one real case before full pipeline.
* **Shell can hang for 10–30+ min** at random — accept this and design pipelines that survive (background processes + log files; never rely on a blocking shell call in the orchestrator).
* **Auto-push watcher commits even when pipelines are mid-run**. Outputs that get partially written may be committed, but the next cycle picks up the final state. Don't worry about it.

## 7. File layout (workspace tree)

```
/workspace/
├── EASMO/                                      git repo (origin: github.com:GuanghuiMin/EASMO.git)
│   ├── README.md
│   ├── .gitignore                              ignores *.log, .venv/, __pycache__/, etc.
│   ├── motivation/                             abandoned (kept for sync_and_push.sh + auto_push_watcher.sh)
│   ├── motivation_v2/
│   ├── motivation_v3/
│   ├── motivation_v4/
│   └── motivation_v5/
├── acon/                                       Microsoft ACON repo
│   ├── .venv/                                  pydantic-v1 venv for productive_agents
│   ├── experiments/appworld/
│   │   ├── data/tasks/                         AppWorld ground truth + datasets
│   │   ├── outputs/                            agent trajectory output (organised by experiment_name)
│   │   └── prompts/_motivation_v?/             spliced jinjas + per-cell jinjas (used by v2/v3/v4 runners)
│   └── src/productive_agents/                  AppWorldEnv + AppWorldAgent (don't modify)
├── qwen3-vllm/                                 self-contained Qwen3-4B server sandbox
│   ├── .venv/                                  vllm 0.10.2 cu128
│   ├── serve.sh
│   └── server.{pid,log}
├── user_feedback/                              user-authored spec docs (one per round)
└── guidances/                                  paper-tier external docs (motivation/ used to mirror these)
```

## 8. Quick stats by track

| Track | n_cases | LLM calls | Agent runs | Wall-clock (full pipeline) | Date completed |
|---|---:|---:|---:|---:|---:|
| motivation_v2 | 90 train tasks (45 single-app) | 1,328 + 1,992 prompt variants | 198 strategy + 354 xtask | ~3-4 h spread over days | 2026-05-26 |
| motivation_v3 | 30 dev | ~1,800 (compressor + audit + recovery labels) | 418 | ~2.5 h | 2026-05-27 |
| motivation_v4 | 30 dev (reused v3) | 1,268 (probe + judge) | 360 | ~1.5 h | 2026-05-27 |
| motivation_v5 | 24 (Tier 1 from v3) | 168 | 24 | 16 min | 2026-05-27 |
| motivation_v6_jacobian | 30 (reused v4) | 0 LLM API; **30 Qwen3-4B backward + 150 × 200-step soft-token training** | 240 MiniMax (Exp D) | 1 h compute + 25 min D | 2026-05-28 |
| motivation_v7 | 30 (reused v3) | ~10,360 LLM calls (fact extract 30, conditions 233, compress 600 + 460, retention 600 + 3,640) | 0 agent runs (diagnostic-only) | 1 h 33 min | 2026-05-28 |
| motivation_v8 | 30 (reused v7) | ~14,270 LLM calls (compress 1,160 + 1,470 iterative/basin; retention 12,640) | 0 agent runs (diagnostic-only) | ~1 h 50 min | 2026-05-28 |
| motivation_v9 | 30 (reused v3) + 12 chunk-cases (first-pass) + 20 (widened) | ~1,360 LLM (270 compress + 810 stress-recompress + 144/240 chunk-context + 144/240 chunk-label) | 540 (C1+CK) + 156/~320 (chunk ablation) | 2 h 16 min first-pass + ~5 min refix + ~55 min widened (in progress) | 2026-05-29 |

## 9. Final advice for whoever picks this up

* **Start with `docs/05_results_summary.md`** in whichever track is relevant. Each one is paper-tier and ~5-10 min to read.
* **Don't refactor across tracks.** Each track is self-contained. v3 reuses v2 modules (e.g. `motivation_v2/data.py`); v4 reuses v3; v5 reuses v3+v2. Keep that direction (newer reuses older, never the other way).
* **The auto-push watcher is your friend**. It commits whatever's in `outputs/` every 20 min. If you're mid-experiment and your shell dies, the data is still there.
* **Seven docs that frame the project at paper-level**:
  - `motivation_v2/docs/05_results_summary.md` — the original three-tier story.
  - `motivation_v3/outputs/motivation_results.md` — the structural-vs-behavioral mismatch.
  - `motivation_v5/docs/05_results_summary.md` — the recovered-then-dropped bottleneck.
  - `motivation_v6_jacobian/docs/04_results_summary.md` — active-subspace exists, span-rank by gradient doesn't work.
  - `motivation_v7/docs/04_results_summary.md` — abstraction prior under ACON (SDI ≈ 1, cross-model τ = 0.49).
  - `motivation_v8/docs/04_results_summary.md` — generalises v7 to non-ACON prompts (SDI = 1.00 under task-agnostic; cross-model τ up to 0.78) AND identifies two new mechanisms: task-aware prompts invert fixed-point composition, and different inits reach disjoint fixed points.
  - **`motivation_v9/docs/04_results_summary.md`** — behavior-side validation of v7/v8. Claim 1 STRONG (best-of-N beats greedy +27/+37 pp pass-rate, oracle win 90/83%); Claim 2 POSITIVE (greedy more stress-fragile than sample, 28.6% vs 21.8%); Claim 3 NEGATIVE at n=239 (spec's causal>entity prediction reverses at widened n; only stable sub-finding is CONTROL_NEGATIVE_EVIDENCE +0.115, n=13). **v7+v8+v9 together are the paper headline as of 2026-05-29.** v7/v8 say *what compressors do*; v9 says *what it costs the agent for Claims 1+2* and warns *that the spec-level chunk taxonomy doesn't cleanly track behavior at chunk granularity*.
  v4 is methodologically interesting (decision-state probing) but its main empirical result is "recency is a strong baseline" which is a less-clean paper story.

---

## §Abandoned-track (kept for context only)

The original `motivation/` track was abandoned 2026-05-24 because:

* "Agents" `A_react / A_plan / A_cot` were the same MiniMax model with different system prompts → not really different policies.
* Benchmarks (LongMemEval / LoCoMo) were QA → no real policy variation possible.
* T1's hinge test (instance_noise) required cross-policy memories to differ behaviourally — but if T2 holds, LLM-generated cross-policy memories are interchangeable, so the test is unsatisfiable.
* Headline metric (binary `action_match_rate` over N=16 samples) was near-Bernoulli; produced zero signal rows on tight budgets.

All `motivation/` runs were killed at 2026-05-24 12:55 PM PT. Their final state is in `motivation/outputs/wide_*/...` for git history but not for citation.

The `motivation/scripts/auto_push_watcher.sh` and `motivation/scripts/sync_and_push.sh` still drive the auto-push loop for v2 + v3 + v4 + v5 — that's why the `motivation/` folder is kept rather than removed. Do not edit anything else under `motivation/` going forward.
