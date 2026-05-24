"""M2 — cross-agent overlap + saliency analysis.

Consumes the JSONL produced by M1 and computes:
* token-level Jaccard between z*_{A_i} and z*_{A_j} per (context, budget)
* sentence-level Jaccard
* TF-IDF cosine similarity
* leave-one-token-out saliency per (agent, token) — Spearman rank
  correlation between agents over the same token set
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence

import numpy as np

from .agents import AgentSpec, sample_action_distribution
from .data import Context, ProbeState
from .llm import MinimaxClient
from .metrics import (
    jaccard,
    kl_divergence,
    sentence_split,
    spearman,
    tfidf_cosine,
    tokenize_simple,
)
from .utils import setup_logging

_logger = setup_logging("motivation.overlap")


@dataclass
class OverlapRow:
    context_id: str
    budget: int
    agent_a: str
    agent_b: str
    token_jaccard: float          # lexical overlap (surface)
    sentence_jaccard: float       # sentence-level overlap
    tfidf_cosine: float           # bag-of-words semantic-ish
    sbert_cosine: float           # sentence-BERT cosine (true semantic)
    memory_a_tokens: int
    memory_b_tokens: int


def pairwise_overlap(
    oracle_records: List[dict],
    *,
    use_sbert: bool = True,
) -> List[OverlapRow]:
    """For each (context, budget) group, compute all pairwise overlap stats.

    Reports BOTH lexical (Jaccard, TF-IDF) and semantic (sentence-BERT
    cosine) overlap so M2 can defuse the "low Jaccard might just be
    paraphrasing" reviewer pushback.
    """
    # Group by (context, budget); inside a group, key by agent.
    groups: Dict[tuple, Dict[str, dict]] = defaultdict(dict)
    for r in oracle_records:
        key = (r["context_id"], int(r["budget"]))
        groups[key][r["agent_id"]] = r

    # Pre-embed every unique memory text once (saves redundant SBERT calls).
    sbert_embeds = {}
    if use_sbert:
        try:
            from .semantic import precompute_embeddings
            all_texts = [r["memory_text"] for r in oracle_records]
            sbert_embeds = precompute_embeddings(all_texts)
        except Exception as exc:                          # pragma: no cover
            _logger.warning("SBERT unavailable, falling back to lexical only: %s", exc)
            sbert_embeds = {}

    rows: list[OverlapRow] = []
    for (ctx_id, budget), by_agent in groups.items():
        agents = sorted(by_agent.keys())
        for a, b in combinations(agents, 2):
            ra = by_agent[a]
            rb = by_agent[b]
            ta = tokenize_simple(ra["memory_text"])
            tb = tokenize_simple(rb["memory_text"])
            sa = sentence_split(ra["memory_text"])
            sb = sentence_split(rb["memory_text"])
            # SBERT cosine — falls back to 0.0 if unavailable.
            sbert = 0.0
            if sbert_embeds:
                va = sbert_embeds.get(ra["memory_text"])
                vb = sbert_embeds.get(rb["memory_text"])
                if va is not None and vb is not None:
                    import numpy as np
                    sbert = float(np.dot(va, vb))
            rows.append(OverlapRow(
                context_id=ctx_id,
                budget=budget,
                agent_a=a,
                agent_b=b,
                token_jaccard=jaccard(ta, tb),
                sentence_jaccard=jaccard(sa, sb),
                tfidf_cosine=tfidf_cosine(ra["memory_text"], rb["memory_text"]),
                sbert_cosine=sbert,
                memory_a_tokens=int(ra["memory_tokens"]),
                memory_b_tokens=int(rb["memory_tokens"]),
            ))
    return rows


# ---------------------------------------------------------------------------
# Token-level saliency  (s_A(t) = KL(π_A(·|C) || π_A(·|C\{t})))
# ---------------------------------------------------------------------------


@dataclass
class SaliencyResult:
    context_id: str
    agent_id: str
    tokens: List[str]
    saliencies: List[float]


def _mask_token(text: str, idx: int, tokens: Sequence[str]) -> str:
    """Reconstruct text with the idx-th token deleted (using surface tokens)."""
    return " ".join(t for i, t in enumerate(tokens) if i != idx)


def token_saliency(
    client: MinimaxClient,
    agent: AgentSpec,
    context: Context,
    probe: ProbeState,
    *,
    samples_per_state: int,
    base_seed: int,
    max_tokens: int = 800,
) -> SaliencyResult:
    """Leave-one-out saliency over the first ``max_tokens`` surface tokens."""
    tokens = tokenize_simple(context.context_text)[:max_tokens]
    if not tokens:
        return SaliencyResult(context.context_id, agent.id, [], [])

    base_dist, _ = sample_action_distribution(
        client, agent, context.context_text, probe.state_text,
        n=samples_per_state, question=context.question,
        base_seed=base_seed,
        task_type=context.task_type,
    )

    saliencies: list[float] = []
    for i in range(len(tokens)):
        masked = _mask_token(context.context_text, i, tokens)
        masked_dist, _ = sample_action_distribution(
            client, agent, masked, probe.state_text,
            n=max(2, samples_per_state // 2),  # save calls
            question=context.question,
            base_seed=base_seed + 10_000 + i,
            task_type=context.task_type,
        )
        saliencies.append(kl_divergence(base_dist, masked_dist))
    return SaliencyResult(context.context_id, agent.id, tokens, saliencies)


def saliency_rank_correlation(
    salA: SaliencyResult,
    salB: SaliencyResult,
) -> float:
    """Spearman over the *shared* token set."""
    common = set(salA.tokens) & set(salB.tokens)
    if not common:
        return float("nan")
    # Take the mean saliency per shared token (in case of repeats).
    def per_token(sal: SaliencyResult) -> dict[str, float]:
        agg: dict[str, list[float]] = defaultdict(list)
        for t, s in zip(sal.tokens, sal.saliencies):
            agg[t].append(s)
        return {t: float(np.mean(v)) for t, v in agg.items()}
    a, b = per_token(salA), per_token(salB)
    keys = sorted(common)
    return spearman([a[k] for k in keys], [b[k] for k in keys])
