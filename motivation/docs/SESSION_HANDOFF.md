# Session handoff — paste this into a new chat if context fills up

> Updated: 2026-05-27 1:25 PM PT.
> All times in Pacific Time (PT).
>
> **➡ For a fresh chat, read these in order:**
> 1. This file (project map + open threads).
> 2. The single track-level snapshot you care about most — each
>    track has a `docs/05_results_summary.md` that's
>    decision-ready in ~5 minutes.
>
> The git remote is `github.com:GuanghuiMin/EASMO.git`.
> Latest commit: `0a26334  motivation_v5: ACON failure-mode audit on AppWorld`.
> Auto-push watcher PID 3916707 stages every 20 min and covers
> `motivation/`, `motivation_v2/`, `motivation_v3/`, `motivation_v4/`,
> `motivation_v5/` automatically.

## 1. Project map (5 motivation tracks)

| Track | Purpose | Status | Headline |
|---|---|---|---|
| `motivation/` | original LongMemEval / LoCoMo idea | **abandoned 2026-05-24** | see §Abandoned-track at end |
| `motivation_v2/` | role-conditional memory in multi-agent systems (T1 + T2) | ✅ done; **6/7 spotlight criteria, Qwen blocked** | three-tier Jaccard hierarchy strategy 0.62 → task 0.28 → role 0.14 (entity-token); prompted-compression code-role recall 5% across all variants |
| `motivation_v3/` | LLM compressor comparison (NL / ACON / symbolic) + behavioral utility | ✅ done | structural-vs-behavioral ranking-mismatch: symbolic 99.5% coverage but **57% downstream success at cap=15**; ACON 67%; NL 70% |
| `motivation_v4/` | decision-state probing: leave-one-span-out sensitivity | ✅ done; **4/6 spec criteria** | high-sens > low-sens (+23pp), > random (+7pp); but recency baseline beats high-sens (47 vs 40 at cap=15) |
| `motivation_v5/` | **ACON failure-mode audit** (recovered-then-dropped) | ✅ done; **strong positive signal verified** | 93% of audit-recovered actionable items are dropped again by recompression; 81% by abstraction policy not capacity; recompressed-context rerun saves 5/24 (21%) of originally-failed cases |

Each track folder follows the same shape:

```
motivation_v?/
├── docs/
│   ├── 00_spec.md (symlink)             user-feedback spec for that round
│   ├── 01_experimental_design.md        full design + pipeline
│   ├── 02_*.md                          prompts (paper appendix)
│   ├── 03_*.md                          definitions / taxonomy
│   ├── 04_*.md                          conditions / setup
│   └── 05_results_summary.md            ★ decision-ready snapshot
├── prompts/                             (v3+v5) verbatim prompt templates
├── motivation_v?/                       python package
├── scripts/                             stage scripts + run_all.sh
├── data/                                (v5) raw + sampled cases
└── outputs/
    ├── raw/                             JSONL
    ├── tables/                          CSV
    ├── figures/                         PDF + PNG
    ├── reports/                         markdown
    └── sprint_logs/                     run logs (gitignored *.log)
```

Pick the right `05_results_summary.md` based on what you're working
on. They're 200–340 lines each and self-contained.

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

### v3
* Done. Recovery counts in tables/. Methodological self-correction (previous "6.0× ratio" was apples-to-oranges) is documented in `docs/05_results_summary.md`.
* No open thread.

### v4
* Done. 4/6 spec criteria. Headline negative result: **decision-state probing identifies real signal but recency baseline beats it on AppWorld task continuation**.
* Future work (per `docs/05_results_summary.md` §7): cross-model probe, multi-step span granularity, benchmarks where recency is a weak baseline.

### v5
* Done as of today. Strong positive signal.
* Possible follow-ups (not requested):
  - Run the audit pipeline on remaining 36 non-Tier-1 cases (when ACON succeeded) to measure baseline grounded-additions and check whether the recompressor drops less when ACON didn't fail.
  - Add easy/medium difficulty cases by running full-context on shorter v3 dev tasks.
  - Cross-model audit: let GPT-4o-mini audit instead of Qwen3-4B, see if `LOST_ACCESS_TOKEN` non-spec-label issue disappears.
  - Implement a method (preserve-by-construction tags) that acts on the bottleneck and verify >21% recovery rate.

## 5. Active background processes

```
# auto-push watcher — stages all motivation_v*/ + motivation/ → git → push every 20 min.
PID 3916707  bash /workspace/EASMO/motivation/scripts/auto_push_watcher.sh
log: /tmp/easmo_watcher.log

# Qwen3-4B vLLM server — local OpenAI-compatible.
PID see /workspace/qwen3-vllm/server.pid
log: /workspace/qwen3-vllm/server.log
```

Stop the watcher: `pkill -f auto_push_watcher.sh`.
Stop Qwen server: `kill "$(cat /workspace/qwen3-vllm/server.pid)"`.

No experiment processes running at this snapshot. All v3/v4/v5 pipelines
finished and pushed to git.

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
| motivation_v5 | 24 (Tier 1 from v3) | 168 | 24 | 16 min | 2026-05-27 (today) |

## 9. Final advice for whoever picks this up

* **Start with `docs/05_results_summary.md`** in whichever track is relevant. Each one is paper-tier and ~5-10 min to read.
* **Don't refactor across tracks.** Each track is self-contained. v3 reuses v2 modules (e.g. `motivation_v2/data.py`); v4 reuses v3; v5 reuses v3+v2. Keep that direction (newer reuses older, never the other way).
* **The auto-push watcher is your friend**. It commits whatever's in `outputs/` every 20 min. If you're mid-experiment and your shell dies, the data is still there.
* **Three docs that frame the project at paper-level**:
  - `motivation_v2/docs/05_results_summary.md` — the original three-tier story.
  - `motivation_v3/outputs/motivation_results.md` — the structural-vs-behavioral mismatch.
  - `motivation_v5/docs/05_results_summary.md` — the recovered-then-dropped bottleneck.
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
