"""Non-LLM, non-execution compression baselines for the M1 / M2 plots.

For each (task, budget) we need to produce a memory string ≤ B tokens
that the agent will see in place of the full env state. These are the
baselines vs which `m*_exec_minimal` and `m*_exec_trajectory` (both in
``exec_memory.py``) compete.

Required by `new_motivation.md` §3.3 / §3.4:

* `m_recent(units, B)`         — most recent units (per-app most-recent
                                  across the full app state)
* `m_freq(units, B)`           — top-K most-frequent entities
* `m_bm25(units, query, B)`    — BM25 over units with task instruction as query
* `m_embedding_topk(units, query, B)` — sentence-BERT cosine top-K (deferred
                                  to a follow-up; needs sentence-transformers
                                  in the venv)

Inputs are the **app-state-derived memory units** produced by
``units.py`` (one per AppWorld DB row / event / entity). Each unit is
a `MemoryUnit` (defined in ``exec_memory.py``) with a `text`, `app`,
optional `weight`, and accessors `n_tokens()`.

Outputs match the same `ExecMemory`-shaped record so the analysis
layer can iterate over heterogeneous compressors without caring
which produced which.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .exec_memory import (  # noqa: F401  (re-export for callers)
    ExecMemory,
    MemoryUnit,
    _compose,
    _count_tokens,
    _greedy_fill,
)


# ---------------------------------------------------------------------------
# Token-aware "budget fill in input order"
# ---------------------------------------------------------------------------


def _take_until_budget(
    candidates: Sequence[MemoryUnit],
    budget_tokens: int,
) -> List[MemoryUnit]:
    """Take items in the order given until the next one would exceed budget."""
    out: List[MemoryUnit] = []
    used = 0
    for u in candidates:
        n = u.n_tokens()
        if used + n > budget_tokens:
            continue  # skip this one but keep trying smaller ones
        out.append(u)
        used += n
    return out


# ---------------------------------------------------------------------------
# m_recent
# ---------------------------------------------------------------------------


def m_recent(
    units: Sequence[MemoryUnit],
    budget_tokens: int,
    *,
    task_id: str = "",
) -> ExecMemory:
    """Take units in **reverse** ingest order (most recent first), greedy fill.

    The unit ordering convention in `units.py` is: items appear in the
    order they were first observed in the app state / trajectory, so
    "most recent" = end of the list. We reverse and greedy-fill.
    """
    rev = list(reversed(list(units)))
    selected = _take_until_budget(rev, budget_tokens)
    # Compose in *forward* order so the agent reads them
    # chronologically — this matches how the same units appear in
    # full-context.
    selected_forward = list(reversed(selected))
    text, n = _compose(selected_forward)
    return ExecMemory(
        variant="m_recent",
        task_id=task_id,
        budget_tokens=budget_tokens,
        units=selected_forward,
        text=text,
        n_tokens=n,
        n_units=len(selected_forward),
        n_units_dropped=len(units) - len(selected_forward),
        executor=None,
    )


# ---------------------------------------------------------------------------
# m_freq — most frequent entities
# ---------------------------------------------------------------------------


_ENTITY_RE = re.compile(r"[A-Za-z0-9_]{4,}")


def _extract_entities(text: str) -> List[str]:
    """Crude entity proxy: lower-cased alphanumeric tokens of length >= 4.
    Skips obvious stopword-ish noise but doesn't NER-parse. Good enough for
    "high-frequency entities" baselines on AppWorld where entity IDs are
    long alphanumeric strings.
    """
    if not text:
        return []
    return [t.lower() for t in _ENTITY_RE.findall(text)]


def m_freq(
    units: Sequence[MemoryUnit],
    budget_tokens: int,
    *,
    task_id: str = "",
) -> ExecMemory:
    """Score each unit by sum of frequencies of its constituent entities;
    select greedy by score until budget exhausted.

    "Most frequent entities globally → keep units rich in those" is a
    plausible no-task-info baseline.
    """
    units = list(units)
    if not units:
        return ExecMemory(
            variant="m_freq", task_id=task_id, budget_tokens=budget_tokens,
            units=[], text="", n_tokens=0, n_units=0, n_units_dropped=0,
            executor=None,
        )

    # 1. Global entity frequency across all units.
    global_freq: Counter = Counter()
    for u in units:
        global_freq.update(_extract_entities(u.text))

    # 2. Per-unit weight = sum of global entity frequencies, normalised
    #    by unit length so we don't just prefer long units.
    weighted = []
    for u in units:
        ents = _extract_entities(u.text)
        if not ents:
            score = 0.0
        else:
            score = sum(global_freq[e] for e in ents) / max(len(ents), 1)
        weighted.append(MemoryUnit(
            kind=u.kind, app=u.app, text=u.text, weight=score,
            source_step=u.source_step, extra=dict(u.extra),
        ))

    # 3. Greedy fill by descending score (stable on input order).
    selected = _greedy_fill(weighted, budget_tokens)
    text, n = _compose(selected)
    return ExecMemory(
        variant="m_freq",
        task_id=task_id,
        budget_tokens=budget_tokens,
        units=selected,
        text=text,
        n_tokens=n,
        n_units=len(selected),
        n_units_dropped=len(units) - len(selected),
        executor=None,
    )


# ---------------------------------------------------------------------------
# m_bm25 — task-instruction-conditioned but learnable-free retrieval
# ---------------------------------------------------------------------------


_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenise(text: str) -> List[str]:
    return _extract_entities(text)


@dataclass
class _BM25Index:
    docs: List[List[str]]
    df: Counter
    avg_len: float
    n: int

    @classmethod
    def build(cls, docs: List[List[str]]) -> "_BM25Index":
        n = len(docs)
        df: Counter = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        total_len = sum(len(d) for d in docs)
        avg_len = total_len / n if n else 0.0
        return cls(docs=docs, df=df, avg_len=avg_len, n=n)

    def score(self, query: List[str], doc_idx: int) -> float:
        d = self.docs[doc_idx]
        if not d:
            return 0.0
        score = 0.0
        for q in query:
            if q not in d:
                continue
            df = self.df.get(q, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            tf = sum(1 for t in d if t == q)
            len_norm = (1 - _BM25_B) + _BM25_B * (len(d) / max(self.avg_len, 1))
            tf_term = tf * (_BM25_K1 + 1) / (tf + _BM25_K1 * len_norm)
            score += idf * tf_term
        return score


def m_bm25(
    units: Sequence[MemoryUnit],
    query: str,
    budget_tokens: int,
    *,
    task_id: str = "",
) -> ExecMemory:
    """BM25 over units with task instruction (or task-and-policy text) as query."""
    units = list(units)
    docs = [_tokenise(u.text) for u in units]
    idx = _BM25Index.build(docs)
    q_tokens = _tokenise(query)

    weighted = []
    for i, u in enumerate(units):
        s = idx.score(q_tokens, i)
        weighted.append(MemoryUnit(
            kind=u.kind, app=u.app, text=u.text, weight=s,
            source_step=u.source_step, extra=dict(u.extra),
        ))

    selected = _greedy_fill(weighted, budget_tokens)
    text, n = _compose(selected)
    return ExecMemory(
        variant="m_bm25",
        task_id=task_id,
        budget_tokens=budget_tokens,
        units=selected,
        text=text,
        n_tokens=n,
        n_units=len(selected),
        n_units_dropped=len(units) - len(selected),
        executor=None,
    )


# ---------------------------------------------------------------------------
# m_embedding_topk — sentence-BERT cosine, deferred but stubbed
# ---------------------------------------------------------------------------


# Module-level model cache so that batched calls (one per task × budget)
# don't re-load and re-tokenise the same SBERT every time. Loading
# all-MiniLM-L6-v2 is ~6 s; without caching that compounds to hours
# across 90 tasks × 5 budgets × 6 compressors.
_SBERT_CACHE: dict = {}


def _get_sbert(model_name: str):
    if model_name in _SBERT_CACHE:
        return _SBERT_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer  # noqa
    except ImportError:
        _SBERT_CACHE[model_name] = None
        return None
    _SBERT_CACHE[model_name] = SentenceTransformer(model_name)
    return _SBERT_CACHE[model_name]


def m_embedding_topk(
    units: Sequence[MemoryUnit],
    query: str,
    budget_tokens: int,
    *,
    task_id: str = "",
    model_name: str = "all-MiniLM-L6-v2",
) -> ExecMemory:
    """Cosine similarity to the query embedding, greedy fill.

    Falls back to BM25 if `sentence-transformers` is not installed —
    we want to avoid hard-blocking the analysis on an optional dep.

    Model is cached at module level (see ``_SBERT_CACHE``) so this is
    cheap for repeated calls in a single process.
    """
    model = _get_sbert(model_name)
    if model is None:
        # Soft fallback so the analysis pipeline keeps working.
        return m_bm25(units, query, budget_tokens, task_id=task_id)

    units = list(units)
    if not units:
        return ExecMemory(
            variant="m_embedding_topk", task_id=task_id,
            budget_tokens=budget_tokens, units=[], text="", n_tokens=0,
            n_units=0, n_units_dropped=0, executor=None,
        )

    q_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    doc_vecs = model.encode(
        [u.text for u in units], normalize_embeddings=True, show_progress_bar=False,
    )
    sims = (doc_vecs @ q_vec).tolist()

    weighted = []
    for u, s in zip(units, sims):
        weighted.append(MemoryUnit(
            kind=u.kind, app=u.app, text=u.text, weight=float(s),
            source_step=u.source_step, extra=dict(u.extra),
        ))

    selected = _greedy_fill(weighted, budget_tokens)
    text, n = _compose(selected)
    return ExecMemory(
        variant="m_embedding_topk",
        task_id=task_id,
        budget_tokens=budget_tokens,
        units=selected,
        text=text,
        n_tokens=n,
        n_units=len(selected),
        n_units_dropped=len(units) - len(selected),
        executor=None,
    )


# ---------------------------------------------------------------------------
# Registry — used by the M1 driver so adding new baselines is one line.
# ---------------------------------------------------------------------------


COMPRESSOR_REGISTRY = {
    "m_recent":  m_recent,
    "m_freq":    m_freq,
    "m_bm25":    m_bm25,
    "m_embedding_topk": m_embedding_topk,
}


def list_compressors() -> List[str]:
    return sorted(COMPRESSOR_REGISTRY)
