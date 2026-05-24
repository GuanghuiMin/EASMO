"""M5 — selector-consistency ablation.

The single most important rebuttal-killer for the policy-dependent
memory paper. We ask: are oracle memories actually *agent-specific*, or
is the MiniMax selector just stochastic — produces different outputs
every call regardless of the agent description?

Method
------

For each context C:
1. **Within-agent variance**: ask the selector to produce K candidate
   memories for the SAME (C, A_react). Compute pairwise Jaccard / SBERT
   overlap within that bag.
2. **Cross-agent variance**: take one candidate for each of A_react,
   A_plan, A_cot. Compute pairwise overlap across agents.
3. **Compare**: if cross-agent overlap is meaningfully LOWER than
   within-agent overlap, the differences in M2 are real (i.e. caused
   by the agent_description in the selector prompt, not by sampling
   noise). If the two are similar, M2's signal is suspect.

Pass criterion: ``cross_agent_jaccard < within_agent_jaccard − 0.1``
(at least 10 percentage points gap on the lexical metric).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import List

from .agents import AgentSpec, get_agent
from .data import Context
from .llm import MinimaxClient
from .metrics import jaccard, tokenize_simple
from .oracle import SELECTOR_SYSTEM, _extract_candidate
from .utils import count_tokens, setup_logging, truncate_to_tokens


# Stable-mode selector prompt: no "variation hint" so all K samples come
# from the SAME conditional distribution — this is the right setup for
# measuring "is the selector consistent for a fixed agent?".
def _stable_selector_prompt(
    context: str, agent: AgentSpec, question: str, budget_tokens: int,
) -> str:
    return (
        f"Agent description (id={agent.id}, scaffold={agent.scaffold}):\n"
        f"{agent.description}\n\n"
        f"Downstream task question: {question}\n\n"
        f"Token budget for the compressed memory: <= {budget_tokens} tokens (GPT-4 tokenizer).\n\n"
        "Return EXACTLY this format (no commentary, no markdown fence):\n"
        "---\n<your compressed memory>\n---\n\n"
        f"Context C:\n\"\"\"\n{context}\n\"\"\""
    )

_logger = setup_logging("motivation.selector_ablation")


@dataclass
class SelectorConsistencyRow:
    context_id: str
    budget: int
    setting: str            # 'within:<agent>' or 'cross:<agentA>_<agentB>'
    mean_jaccard: float
    mean_sbert: float
    n_pairs: int


def _gen_candidates(
    client: MinimaxClient,
    agent: AgentSpec,
    context: Context,
    budget: int,
    k: int,
    seed: int,
    *,
    temperature: float = 0.3,    # low T for selector consistency baseline
) -> list[str]:
    """Generate K candidates for a fixed agent description, using a
    *stable-mode* prompt (no 'variation hint'). This isolates the
    selector's own conditional variance from the deliberate diversity
    injected in M1's oracle search."""
    out = []
    for i in range(k):
        prompt = _stable_selector_prompt(
            context.context_text, agent, context.question, budget,
        )
        res = client.chat(
            system=SELECTOR_SYSTEM, user=prompt,
            temperature=temperature, max_tokens=min(2 * budget + 256, 4096),
            seed=seed + i,
        )
        cand = _extract_candidate(res.text)
        cand = truncate_to_tokens(cand, budget)
        if cand and cand.lower().strip() not in ("<compressed memory candidate>", ""):
            out.append(cand)
    return out


def _pairwise_means(memories: list[str]) -> tuple[float, float]:
    """Return (mean_token_jaccard, mean_sbert_cosine) over all pairs."""
    if len(memories) < 2:
        return 0.0, 0.0
    try:
        from .semantic import precompute_embeddings
        import numpy as np
        emb = precompute_embeddings(memories)
    except Exception:
        emb = {}
        np = None
    jacs, sims = [], []
    for a, b in combinations(memories, 2):
        ta = tokenize_simple(a)
        tb = tokenize_simple(b)
        jacs.append(jaccard(ta, tb))
        if emb and np is not None:
            va, vb = emb.get(a), emb.get(b)
            if va is not None and vb is not None:
                sims.append(float(np.dot(va, vb)))
    mean_j = sum(jacs) / max(len(jacs), 1)
    mean_s = sum(sims) / max(len(sims), 1) if sims else 0.0
    return mean_j, mean_s


def run_selector_ablation(
    client: MinimaxClient,
    contexts: list[Context],
    scaffolds: list[str],
    budget: int,
    *,
    within_k: int = 3,
    seed: int = 42,
) -> list[SelectorConsistencyRow]:
    """Run M5 over all contexts at one budget. Returns one row per
    (context, setting) combination."""
    agents = [get_agent(s) for s in scaffolds]

    rows: list[SelectorConsistencyRow] = []

    for ctx_i, ctx in enumerate(contexts):
        # 1. Within-agent: for each agent, generate K candidates.
        per_agent_mems: dict[str, list[str]] = {}
        for a_i, agent in enumerate(agents):
            mems = _gen_candidates(
                client, agent, ctx, budget, within_k,
                seed=seed + 1000 * ctx_i + 100 * a_i,
            )
            per_agent_mems[agent.id] = mems
            if len(mems) >= 2:
                j, s = _pairwise_means(mems)
                rows.append(SelectorConsistencyRow(
                    context_id=ctx.context_id, budget=budget,
                    setting=f"within:{agent.id}",
                    mean_jaccard=j, mean_sbert=s, n_pairs=len(mems) * (len(mems) - 1) // 2,
                ))
                _logger.info(
                    "[%s | B=%d] within:%s  jacc=%.3f  sbert=%.3f  (k=%d)",
                    ctx.context_id, budget, agent.id, j, s, len(mems),
                )

        # 2. Cross-agent: take the FIRST candidate from each pair of agents.
        for a, b in combinations(agents, 2):
            ma = per_agent_mems.get(a.id) or []
            mb = per_agent_mems.get(b.id) or []
            if not ma or not mb:
                continue
            # Pair every A's candidate with every B's candidate.
            pairs_j, pairs_s = [], []
            try:
                from .semantic import precompute_embeddings
                import numpy as np
                emb = precompute_embeddings(ma + mb)
            except Exception:
                emb = {}; np = None
            for x in ma:
                for y in mb:
                    pairs_j.append(jaccard(tokenize_simple(x), tokenize_simple(y)))
                    if emb and np is not None:
                        vx, vy = emb.get(x), emb.get(y)
                        if vx is not None and vy is not None:
                            pairs_s.append(float(np.dot(vx, vy)))
            mj = sum(pairs_j) / len(pairs_j) if pairs_j else 0.0
            ms = sum(pairs_s) / len(pairs_s) if pairs_s else 0.0
            rows.append(SelectorConsistencyRow(
                context_id=ctx.context_id, budget=budget,
                setting=f"cross:{a.id}_vs_{b.id}",
                mean_jaccard=mj, mean_sbert=ms, n_pairs=len(pairs_j),
            ))
            _logger.info(
                "[%s | B=%d] cross:%s_vs_%s  jacc=%.3f  sbert=%.3f  (n=%d)",
                ctx.context_id, budget, a.id, b.id, mj, ms, len(pairs_j),
            )
    return rows
