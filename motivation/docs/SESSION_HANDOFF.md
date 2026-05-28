# Session handoff — paste this into a new chat if context fills up

> Updated: 2026-05-28 4:05 PM PT.
> All times in Pacific Time (PT).
>
> **➡ For a fresh chat, read these in order:**
> 1. This file (project map + open threads).
> 2. The single track-level snapshot you care about most — each
>    track has a `docs/04_results_summary.md` (v6, v7) or
>    `docs/05_results_summary.md` (v2–v5) that's decision-ready in
>    ~5 minutes.
>
> The git remote is `git@github.com:GuanghuiMin/EASMO.git` (SSH).
> Latest pushed commit: **`7d96ea5  motivation_v7: STRONG positive on
> Claim A + Claim B (abstraction prior + iterative dynamics)`**
> (2026-05-28 16:00 PT). `main` is in sync with `origin/main`.
> Auto-push watcher PID 3916707 stages every 20 min and covers
> `motivation/`, `motivation_v2/`, `motivation_v3/`, `motivation_v4/`,
> `motivation_v5/`, `motivation_v6_jacobian/`, **`motivation_v7/`**
> automatically.

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

## 2. Latest-round summary (motivation_v5, today's work)

**Goal**: ACON failure-mode audit on AppWorld dev. Diagnostic only,
not a method. Build cases where ACON-style compression failed,
have an audit model recover missing actionable info, recompress,
re-run downstream, see what survives.

**Outcome (n=24, all Tier 1, all `hard`)**:
* 50% of failures are LOST_API_SCHEMA_OR_PARAMETER; +13% LOST_ACCESS_TOKEN.
* **Audit recovers 28 grounded actionable items; recompressor drops 26 (93%) of them.**
* **81% of drops are stylistic** (`over_abstraction` 42% + `looked_like_past_log` 38%); only 15% from `verbosity_pressure`. Bottleneck is the compressor's *abstraction policy*, not its token budget.
* Re-running downstream agent with recompressed context recovers **5/24 = 21%** of originally-failed cases.
* Spec acceptance §16: ✅ STRONG positive signal (recovered-then-dropped bottleneck) verified.

**Caveats** (must include in any paper writing):
* Qwen3-4B's mean `reliability_score = 0.15` (self-uncertain).
* Qwen↔MiniMax primary-mode agreement only 42%; compression-causality agreement 50%.
* Rule-based grounding mean 0.444 (only ~44% of Qwen quotes are literal substrings).
* Auditor was prompt-biased toward attributing failures to compression (100% Qwen, 50% MiniMax).
* All 24 cases are `hard` (length-biased from v3).

For full details + tables + figures see
[`motivation_v5/docs/05_results_summary.md`](../../motivation_v5/docs/05_results_summary.md).

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
* Done 2026-05-28. **STRONG positive on both Claim A and Claim B** (the project's cleanest paper-tier result so far).
* Full paper-tier report at `motivation_v7/docs/04_results_summary.md`.
* Compressors used: Qwen3-4B-Instruct-2507 (port 8000 vLLM) + MiniMax-M2.5 (10.183.22.68:8005). ACON prompts loaded verbatim from `/workspace/acon` (commit d63f9ae): UTCO = `improved_history_prompt_samples_4.jinja`.
* Methodological negatives + caveats:
  - Need-condition generator (MiniMax) produces matched pairs that pass quality checks 64% of the time (150/233). Writing pure-need counterfactuals is itself non-trivial.
  - Neither UT nor UTCO has a `max_chars` template variable; outputs are 2,000–2,800 chars vs nominal 1,500-char budget.
  - Plan B scope: iterative chains are 2 per case (needed + unneeded for one EXECUTABLE fact), not the spec's full sweep.
  - v3-derived 30 cases all medium/long (≥15 steps); no short stratum.
* Possible follow-ups (not requested):
  - Run UT ablation (the unoptimised utility-only prompt) to verify SDI ≈ 1 isn't UTCO-specific.
  - Secondary budgets {800, 2500} to see whether retention preferences shift with budget.
  - GPT-4.1-mini as a 3rd compressor to extend cross-model τ test.
  - Causal intervention: insert "you MUST preserve any access_token / file_path verbatim" into the system prompt and see whether SDI drops.

## 4.5. GPU / endpoint state (as of 2026-05-28 4:00 PM PT)

* **Qwen3-4B-Instruct-2507 vLLM** is currently running on port 8000 (PID 1114353, started 2026-05-28 11:43 AM PT). Served model id = `qwen3-4b-instruct-2507`. Launch script: `/workspace/qwen3-vllm/serve_instruct.sh`. Stop with `pkill -f 'served-model-name qwen3-4b-instruct-2507'`.
* The older base-model serve (`Qwen/Qwen3-4B` on port 8000 as `qwen3-4b`) is **stopped** since 2026-05-27. Launch script preserved at `/workspace/qwen3-vllm/serve.sh`. Cannot run both at once — same port.
* GPU memory used: ~56 GB / 80 GB. ~25 GB free — enough for a parallel HF gradient run if needed.
* MiniMax-M2.5 endpoint at 10.183.22.68:8005 has been continuously available across v4–v7.

## 4.6. Running background processes (as of 2026-05-28 4:05 PM PT)

```
PID 3916707  bash /workspace/EASMO/motivation/scripts/auto_push_watcher.sh
             log: /tmp/easmo_watcher.log
             after 4:05 PM symlink-fix it can push again.
PID 1114353  python -m vllm.entrypoints.openai.api_server
             --model Qwen/Qwen3-4B-Instruct-2507 --port 8000
             log: /workspace/qwen3-vllm/server_instruct.log
PID 1148*    (LLMLingua demo_app, user-owned, terminal 2.txt, not ours)
```

No active experiment process is running. v7 pipeline finished at 2026-05-28 22:19Z (3:19 PM PT).

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
* **Qwen3-4B JSON mode + thinking interaction**: pass `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` to disable Qwen's thinking; otherwise short max_tokens cuts off mid-think. v5's `clients.chat_qwen` does this.
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

## 9. Final advice for whoever picks this up

* **Start with `docs/05_results_summary.md`** in whichever track is relevant. Each one is paper-tier and ~5-10 min to read.
* **Don't refactor across tracks.** Each track is self-contained. v3 reuses v2 modules (e.g. `motivation_v2/data.py`); v4 reuses v3; v5 reuses v3+v2. Keep that direction (newer reuses older, never the other way).
* **The auto-push watcher is your friend**. It commits whatever's in `outputs/` every 20 min. If you're mid-experiment and your shell dies, the data is still there.
* **Five docs that frame the project at paper-level**:
  - `motivation_v2/docs/05_results_summary.md` — the original three-tier story.
  - `motivation_v3/outputs/motivation_results.md` — the structural-vs-behavioral mismatch.
  - `motivation_v5/docs/05_results_summary.md` — the recovered-then-dropped bottleneck.
  - `motivation_v6_jacobian/docs/04_results_summary.md` — active-subspace exists, span-rank by gradient doesn't work (clean negative + clean positive in one round).
  - **`motivation_v7/docs/04_results_summary.md`** — the project's cleanest paper-tier result: LLM history compressors are unconditioned surface-type abstraction priors (SDI ≈ 1, cross-model Kendall τ = 0.49). **This is the best candidate for the paper headline as of 2026-05-28.**
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
