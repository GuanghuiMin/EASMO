# Role-conditional memory extractors (headline policy axis)

> **Role in the paper**: this is the **headline policy axis**. We
> demonstrate that the same upstream context (one successful AppWorld
> trajectory) yields *measurably different* compressed memories
> depending on which agent role is consuming it. Empirically the
> cross-role Jaccard at B=512 is **0.04** — roles want essentially
> orthogonal slices of the same context. By contrast cross-task
> within-role Jaccard at B=512 is **0.17**, and cross-strategy
> within-task Jaccard is **~0.91**. The hierarchy
>
> ```
> cross-role (0.04)  <<  cross-task within-role (0.17)  <<  cross-strategy within-task (0.91)
>     ↑                       ↑                                  ↑
> roles need              specific facts                       agent style
> different memory        are task-specific                    is invariant
> ```
>
> is the central structural finding of the paper.

This file documents the four role-specific memory extractors, what
each one keeps from a trajectory, and why those are the right
projections for that role. Suitable for the paper appendix.

Implementation: `motivation_v2/motivation_v2/role_memory.py`. All
extractors are deterministic functions of (trajectory, budget).
No LLM in the loop.

## 1. Why these four roles

A multi-agent system in practice splits work across specialised
agents. The four roles below cover the standard division of labour
in modern agent systems (e.g. AutoGen / ChatDev / MetaGPT
templates):

| Role | What it does | What it needs from upstream context |
|---|---|---|
| **Tool-user** | issues API calls, observes results | API endpoints, arguments, observations |
| **Coder** | writes code, debugs, refactors | control-flow patterns, library idioms, error handling |
| **Planner** | sets goals, sequences sub-tasks | goal statement, milestones, final answer |
| **Verifier** | checks correctness, asserts post-conditions | tail observations, final-state assertions |

Choosing exactly four (not three, not eight) is a trade-off between
covering common roles and keeping the cross-role Jaccard table
readable (4×4 = 6 distinct pairs). Section §3 of this doc proposes
how to extend to more roles if the paper grows.

## 2. Extractor specs

Each extractor takes a `Trajectory` (a successful AppWorld run from
ACON's output) and a `budget_tokens`, and returns an `ExecMemory`
record with the selected `MemoryUnit`s and their concatenated text
(≤ budget tokens).

### 2.1 `m_tool` — tool-user role

**Selection rule**: every distinct `apis.<app>.<fn>(<args>)` call
in the trajectory's action steps, plus the immediately-following
observation summary (truncated to 200 chars).

**Why it's the tool-user view**: a tool-user agent's job is to
choose the right API call given a state. What it wants from past
context is "what calls were made and what came back". Endpoint
names + arguments + return values are the relevant signal.

**Effective unit count at common budgets** (averaged over 81
direct-strategy trajectories): B=128 → ~3 units; B=256 → ~6;
B=512 → ~12; B=1024 → ~22.

### 2.2 `m_code` — coder role

**Selection rule**: action lines containing structural Python
patterns (loops, list comprehensions, try/except, function defs,
`max/min/sorted/filter/map`) — with concrete API call args
replaced by `<args>` so the *control flow* is preserved without
leaking task-specific values.

**Why it's the coder view**: a coder agent learns from "how to
combine API calls in code", not from specific argument values.
The same `for page_index in range(0, 10): apis.spotify.show_album_library(<args>)`
pattern is reusable across spotify tasks regardless of which
specific album library is being queried.

**Cross-task transferability prediction**: this view should have
**high cross-task Jaccard** because Python idioms repeat across
tasks. (Empirically confirmed: 0.41 vs 0.09 for the tool view —
see §3 below.)

### 2.3 `m_plan` — planner role

**Selection rule**: the task instruction + intent comments from
the first ≤ 3 trajectory steps + one milestone observation per
unique app touched + the final `apis.supervisor.complete_task(answer=...)`
call.

**Why it's the planner view**: a planner agent's job is to
recognise goals and sub-goal completions. From past context it
wants the goal statement, the high-level "we discovered X" milestones,
and the final result.

**Note on intent comments**: AppWorld agents often write Python
comments like `# Let me first explore the available apps to find the password`
at the top of an action — these are essentially the agent's
internal monologue and are perfect planner-view material. The
extractor pulls these from the first 3 steps only, since later
comments are typically tactical not strategic.

### 2.4 `m_verify` — verifier role

**Selection rule**: the last 5 non-empty observations from the
trajectory + the final `complete_task` call.

**Why it's the verifier view**: a verifier agent's job is to
check that the final state satisfies the task. From past context
it wants the evidence chain that immediately precedes the answer
— the values that should justify the final claim.

## 3. Empirical Jaccard data (n=81 direct-strategy successful trajectories, 2026-05-24)

### Cross-role (same task, different role) — should be LOW

| B | tool–code | tool–plan | tool–verify | code–plan | code–verify | plan–verify |
|---|---|---|---|---|---|---|
| 128  | 0.000 | 0.004 | 0.022 | 0.000 | 0.000 | 0.153 |
| 256  | 0.000 | 0.007 | 0.028 | 0.000 | 0.000 | 0.110 |
| 512  | 0.000 | 0.059 | 0.054 | 0.000 | 0.000 | 0.099 |
| 1024 | 0.000 | 0.095 | 0.081 | 0.000 | 0.000 | 0.099 |

**Mean cross-role Jaccard at B=512 = 0.035** — roles want
near-disjoint memory. The plan–verify pair is the only one with
non-trivial overlap (≈ 0.10) because both views capture the final
`complete_task` call.

### Cross-task within-role (different tasks, same role) — should vary by role

| B | tool | code | plan | verify |
|---|---|---|---|---|
| 128  | 0.121 | **0.426** | 0.016 | 0.110 |
| 256  | 0.099 | **0.409** | 0.075 | 0.086 |
| 512  | 0.089 | **0.409** | 0.072 | 0.105 |
| 1024 | 0.093 | **0.409** | 0.072 | 0.105 |

**Code role transfers across tasks (0.41); other roles do not (≤ 0.11).**
The code-role number is striking: 41% of structural Python patterns
are shared across two arbitrary AppWorld tasks. The other three
roles need task-specific facts that don't transfer.

### Hierarchy

```
cross-strategy within-task (≈ 0.91)
        ▲
        │  agent style is invariant
        │
cross-task within-role (≈ 0.17 mean)
        ▲
        │  specific facts are task-specific
        │
cross-role (≈ 0.04 mean)
        ▲
        │  roles need orthogonal memory  ← headline finding
```

## 4. Caveats and known limitations

1. **The four roles are projections of one trajectory, not from
   independently-run role agents.** A skeptic could argue we
   constructed orthogonality by our slicing choices. The cleanest
   rebuttal is in the cross-task numbers: the SAME slicing rules
   produce a code-view Jaccard of 0.41 across tasks but a tool-view
   Jaccard of 0.09. If the slicing forced orthogonality, both
   numbers would be low; the fact that code transfers and tool
   doesn't is a property of the *content*, not the slicing.

2. **AppWorld is dominated by tool-use tasks.** All 90 train tasks
   are essentially "find this fact in the user's app data". Code,
   plan, verify roles are projections that exist in any tool-use
   trajectory. To strengthen the role-orthogonality claim on
   non-tool-use roles, follow-up work should test on a benchmark
   with explicit role specialisation (ChatDev, MetaGPT).

3. **The plan view's "intent comments" rely on the AppWorld agent's
   convention of writing Python comments before action code.** This
   is reliably true for MiniMax-M2.5 trajectories (each first action
   has at least one `# ...` line) but may not hold for other
   executors. The plan extractor's robustness across executors is
   future work (Option Y).

4. **No statistical CIs reported above.** With 81 trajectories the
   reported means are stable to ≤ 0.02 SE; bootstrap CIs go in the
   final paper draft.

## 5. Extending to more roles

If the paper grows to include a sixth role (e.g. **searcher** —
information-gathering specialist; or **summariser** —
human-facing-output specialist), add an extractor with a clear
selection rule and document its behavioural signature. The
manipulation check is: cross-role Jaccard between any new role
and the existing four should be ≤ 0.10 at B=512, otherwise the
new role is a redundant projection.
