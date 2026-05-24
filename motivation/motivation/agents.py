"""Three frozen agent scaffolds — A_react, A_plan, A_cot.

All three share the same base model (MiniMax-M2.5 via API); they differ
only in the **system prompt + decoding parameters**, so any divergence in
their behaviour is genuinely policy-induced rather than capability-induced.

A scaffold here is intentionally minimal: it just maps
``(context, question)`` → an empirical action distribution by sampling
the model N times. For the motivation experiments we don't need the
agent to actually execute tools; we only need ``π_A(·|s)``, the
*distribution over next actions*.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .llm import GenerateResult, MinimaxClient
from .utils import setup_logging

_logger = setup_logging("motivation.agents")


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

REACT_SYSTEM = (
    "You are a ReAct-style agent. Your job: given the context and the "
    "current state, output ONE next action — a short, concrete command — "
    "that you would take next. Be reactive: short tool calls, no planning. "
    "Format your answer as:\n\nAction: <one-line action>\n\n"
    "Do not output anything else."
)

PLAN_SYSTEM = (
    "You are a Plan-then-Execute agent. Your job: given the context and "
    "the current state, decompose into 2–3 subgoals first, then output "
    "the next concrete action that advances the FIRST unmet subgoal. "
    "Be precise. Format your answer as:\n\n"
    "Subgoals:\n- <subgoal 1>\n- <subgoal 2>\nAction: <one-line action>\n\n"
    "Do not output anything else."
)

COT_SYSTEM = (
    "You are a Reflexion / chain-of-thought agent. Your job: given the "
    "context and the current state, briefly self-critique any previous "
    "step, then output the next concrete action. Be reasoning-heavy: "
    "revisit prior conclusions if they look wrong. Format your answer as:\n\n"
    "Reflection: <2–3 sentences>\nAction: <one-line action>\n\n"
    "Do not output anything else."
)


@dataclass(frozen=True)
class AgentSpec:
    id: str
    scaffold: str
    system: str
    temperature: float
    description: str


_REGISTRY: Dict[str, AgentSpec] = {
    "react": AgentSpec(
        id="A_react",
        scaffold="react",
        system=REACT_SYSTEM,
        temperature=0.4,
        description=(
            "ReAct loop (Yao 2023): reactive; many short tool calls; "
            "minimal planning."
        ),
    ),
    "plan": AgentSpec(
        id="A_plan",
        scaffold="plan",
        system=PLAN_SYSTEM,
        temperature=0.4,
        description=(
            "Plan-then-Execute: decomposes into subgoals; few precise "
            "queries; explicit subgoal tree."
        ),
    ),
    "cot": AgentSpec(
        id="A_cot",
        scaffold="cot",
        system=COT_SYSTEM,
        temperature=0.7,  # reflexion benefits from a bit more variance
        description=(
            "Reflexion / self-critique (Shinn 2023): reasoning-heavy; "
            "revisits prior conclusions."
        ),
    ),
}


def get_agent(scaffold: str) -> AgentSpec:
    if scaffold not in _REGISTRY:
        raise KeyError(f"Unknown scaffold {scaffold!r}; choose one of {list(_REGISTRY)}.")
    return _REGISTRY[scaffold]


def list_agents() -> List[AgentSpec]:
    return list(_REGISTRY.values())


# ---------------------------------------------------------------------------
# Action canonicalisation + distribution
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"action\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)

# Defensive token-mask patterns. Two AppWorld-style actions can look
# identical but have unique ``access_token=`` values or unique JWT-like
# blobs, which would make naive string equality always fail. We
# canonicalise such values to placeholders so the action-match rate
# focuses on the *structure* of the action (function + sanitised args).
_JWT_RE        = re.compile(r"eyj[a-zA-Z0-9_\-\.]+", re.IGNORECASE)
_LONG_HEX_RE   = re.compile(r"\b[0-9a-f]{16,}\b", re.IGNORECASE)
_QUOTED_STR_RE = re.compile(r"(['\"])([^'\"]{12,})\1")
_ACCESS_TOK_RE = re.compile(
    r"(access[_-]?token\s*[=:]\s*)(['\"])[^'\"]+\2", re.IGNORECASE,
)
_NUM_RE        = re.compile(r"\b\d{6,}\b")


def extract_action(text: str) -> str:
    """Pull the ``Action: ...`` line; fall back to first non-empty line."""
    if not text:
        return ""
    m = _ACTION_RE.search(text)
    if m:
        return _canonicalise(m.group(1))
    for line in text.splitlines():
        line = line.strip()
        if line:
            return _canonicalise(line)
    return ""


def _canonicalise(action: str) -> str:
    """Normalise whitespace, lowercase, and mask volatile tokens.

    Keeps the structural skeleton of the action (the function + which
    arguments were passed) but collapses values that change between
    samples (access tokens, JWTs, long IDs).
    """
    s = action.strip()
    # Volatile-token masking (do this BEFORE lowercase so JWT detection works).
    s = _ACCESS_TOK_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}<TOK>{m.group(2)}", s)
    s = _JWT_RE.sub("<JWT>", s)
    s = _LONG_HEX_RE.sub("<HEX>", s)
    s = _QUOTED_STR_RE.sub(lambda m: f"{m.group(1)}<STR>{m.group(1)}", s)
    s = _NUM_RE.sub("<N>", s)
    # Now do whitespace / case normalisation.
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[`*]", "", s)
    s = s.rstrip(".;,")
    return s


def action_distribution(
    actions: Sequence[str],
    *,
    mode: str = "string",
    sim_threshold: float = 0.75,
) -> Dict[str, float]:
    """Empirical action distribution from N samples.

    ``mode``:
      * ``'string'`` — exact-match after :func:`_canonicalise` (good for
        agent-step / tool-call style outputs where two correct actions
        really should be the same string).
      * ``'semantic'`` — sentence-BERT clustering with ``sim_threshold``
        cosine, so two paraphrases of the same answer are counted as
        the same action. Good for free-form QA.
    """
    if not actions:
        return {}
    canon = [a for a in (extract_action(s) if isinstance(s, str) else "" for s in actions) if a]
    if not canon:
        return {}
    if mode == "string":
        counts = Counter(canon)
        total = sum(counts.values()) or 1
        return {a: c / total for a, c in counts.items()}
    if mode == "semantic":
        from .semantic import semantic_action_distribution
        return semantic_action_distribution(canon, sim_threshold=sim_threshold)
    raise ValueError(f"Unknown action_distribution mode: {mode!r}")


def top_action(dist: Dict[str, float]) -> str:
    if not dist:
        return ""
    return max(dist.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# Distribution sampling for one (agent, context, state)
# ---------------------------------------------------------------------------

def _make_user_msg(
    context: str, state: str, question: str | None, task_type: str = "agent_step",
) -> str:
    parts = [f"Context:\n\"\"\"\n{context}\n\"\"\""]
    if state:
        parts.append(f"\nCurrent state:\n\"\"\"\n{state}\n\"\"\"")
    if question:
        parts.append(f"\nTask question: {question}")
    if task_type == "qa":
        parts.append(
            "\nAnswer the task question above using ONLY the context. "
            "Be concise. Respond in the prescribed format — the 'Action' "
            "line should be your final answer (no Python, no tool call, "
            "just a short factual answer)."
        )
    else:
        parts.append(
            "\nWhat is your single next action? Respond in the prescribed format."
        )
    return "\n".join(parts)


def sample_action_distribution(
    client: MinimaxClient,
    agent: AgentSpec,
    context: str,
    state: str,
    *,
    n: int,
    question: str | None = None,
    max_tokens: int = 256,
    base_seed: int = 0,
    task_type: str = "agent_step",
    distribution_mode: str | None = None,
    sim_threshold: float = 0.75,
) -> tuple[Dict[str, float], List[GenerateResult]]:
    """Sample N completions, return (distribution, raw_results).

    ``distribution_mode`` defaults to ``'semantic'`` when
    ``task_type='qa'`` (free-form answers vary by paraphrasing) and
    ``'string'`` for ``agent_step`` (tool calls; exact-match after
    canonicalisation is what we want).
    """
    if distribution_mode is None:
        distribution_mode = "semantic" if task_type == "qa" else "string"
    user = _make_user_msg(context, state, question, task_type=task_type)
    batches = [
        {
            "messages": [
                {"role": "system", "content": agent.system},
                {"role": "user", "content": user},
            ],
            "temperature": agent.temperature,
            "max_tokens": max_tokens,
            "seed": base_seed + i,
        }
        for i in range(n)
    ]
    results = client.generate_batch(batches)
    texts = [r.text for r in results if r.error is None]
    dist = action_distribution(
        texts, mode=distribution_mode, sim_threshold=sim_threshold,
    )
    return dist, results
