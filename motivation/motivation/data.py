"""Data loaders for motivation experiments.

We bootstrap with three sources, each mapped to the same ``Context``
abstraction so the rest of the pipeline (oracle, overlap, transfer) is
data-agnostic:

* **AppWorld** — tool-using agent trajectory; probe states are
  intermediate (state, gold-Python-action) pairs. ``task_type='agent_step'``.
* **LongMemEval** — multi-session conversational history; one probe
  state per example (the recall question). ``task_type='qa'``.
* **LoCoMo** — multi-session dialogue with up to 199 QA per
  conversation; we treat each (conversation, QA) as one Context.
  ``task_type='qa'``.
"""

from __future__ import annotations

import glob
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

from .utils import count_tokens, setup_logging, stable_hash, truncate_to_tokens

_logger = setup_logging("motivation.data")


@dataclass
class ProbeState:
    """A single (state, gold_action) probe point inside a context."""
    state_id: str
    step: int
    state_text: str
    gold_action: str


@dataclass
class Context:
    """A long context + a set of probe states agents will be queried on."""
    context_id: str
    source: str
    question: str
    context_text: str
    n_tokens: int
    probe_states: List[ProbeState]
    # ``agent_step`` — next-action sampling (AppWorld);
    # ``qa``         — free-form answer sampling (LongMemEval, LoCoMo).
    task_type: str = "agent_step"

    def to_dict(self) -> dict:
        return {
            "context_id": self.context_id,
            "source": self.source,
            "question": self.question,
            "context_text": self.context_text,
            "n_tokens": self.n_tokens,
            "task_type": self.task_type,
            "probe_states": [ps.__dict__ for ps in self.probe_states],
        }


def _format_step(step: dict) -> str:
    """Render one trajectory step as a short text block."""
    action = (step.get("action") or "").strip()
    output = (step.get("output") or "").strip()
    # Truncate huge tool outputs (JSON dumps) so a single step doesn't
    # gobble the whole context budget.
    if len(output) > 1200:
        output = output[:1200] + "\n…[output truncated]"
    return f"### Step {step.get('step')}\nAction:\n{action}\n\nObservation:\n{output}"


def load_appworld_contexts(
    trajectory_glob: str,
    *,
    max_contexts: int,
    probe_states_per_context: int,
    max_context_tokens: int,
) -> List[Context]:
    """Load AppWorld trajectories from disk and convert each to a Context."""
    paths = sorted(glob.glob(trajectory_glob))
    _logger.info("Found %d AppWorld trajectory files via %r", len(paths), trajectory_glob)
    if not paths:
        raise FileNotFoundError(
            f"No AppWorld trajectories matched {trajectory_glob!r}. Generate "
            "some via acon's run.py or point trajectory_glob elsewhere."
        )

    contexts: list[Context] = []
    for path in paths:
        if len(contexts) >= max_contexts:
            break
        try:
            with open(path, "r", encoding="utf-8") as f:
                traj = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            _logger.warning("Skipping %s: %s", path, exc)
            continue

        steps = traj.get("trajectory") or []
        if len(steps) < probe_states_per_context + 2:
            _logger.info(
                "Skipping %s: only %d steps (< %d probe states + 2).",
                path, len(steps), probe_states_per_context,
            )
            continue

        task_id = traj.get("task_id") or Path(path).parent.name
        question = (traj.get("task_instruction") or "").strip()

        # Probe states spaced through the trajectory (skip first 2 steps
        # — they're usually setup/auth so they're trivial).
        usable = steps[2:]
        if len(usable) < probe_states_per_context:
            continue
        stride = max(1, len(usable) // probe_states_per_context)
        probe_idxs = [2 + i * stride for i in range(probe_states_per_context)]

        probe_states: list[ProbeState] = []
        for pi in probe_idxs:
            if pi >= len(steps):
                continue
            # State = concat of all (action, output) up to step pi (exclusive).
            history_steps = steps[:pi]
            state_text = "\n\n".join(_format_step(s) for s in history_steps[-4:])
            gold_action = (steps[pi].get("action") or "").strip()
            if not gold_action:
                continue
            probe_states.append(ProbeState(
                state_id=f"{task_id}::step{pi}",
                step=pi,
                state_text=state_text,
                gold_action=gold_action,
            ))
        if len(probe_states) < probe_states_per_context // 2:
            continue

        # Full context = head of trajectory up to the last probe point.
        last_probe = probe_idxs[-1]
        full_context_text = "\n\n".join(_format_step(s) for s in steps[:last_probe])
        full_context_text = truncate_to_tokens(full_context_text, max_context_tokens)

        ctx = Context(
            context_id=f"appworld::{task_id}::{stable_hash(question, last_probe)}",
            source="appworld",
            question=question,
            context_text=full_context_text,
            n_tokens=count_tokens(full_context_text),
            probe_states=probe_states,
        )
        _logger.info(
            "Loaded context %s (%d tokens, %d probe states)",
            ctx.context_id, ctx.n_tokens, len(probe_states),
        )
        contexts.append(ctx)

    if not contexts:
        raise RuntimeError(
            "No usable contexts produced from AppWorld trajectories. "
            "Check max_context_tokens / probe_states_per_context."
        )
    _logger.info("Final context pool: %d contexts (target was %d).", len(contexts), max_contexts)
    return contexts


# ---------------------------------------------------------------------------
# LongMemEval loader
# ---------------------------------------------------------------------------

def _format_lme_sessions(sessions, dates) -> str:
    """Render LongMemEval haystack_sessions into a single readable string."""
    parts = []
    for i, (sess, date) in enumerate(zip(sessions, dates), 1):
        parts.append(f"## Session {i}  (date: {date})")
        for turn in sess:
            role = turn.get("role", "user").upper()
            content = (turn.get("content") or "").strip()
            parts.append(f"{role}: {content}")
        parts.append("")
    return "\n".join(parts)


def load_longmemeval_contexts(
    data_path: str,
    *,
    max_contexts: int,
    max_context_tokens: int,
    seed: int = 42,
    **_kwargs,
) -> List[Context]:
    """Load LongMemEval-oracle examples and convert each to a Context with
    a single probe state = the recall question."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"LongMemEval file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _logger.info("LongMemEval: %d total examples in %s", len(data), path.name)

    rnd = random.Random(seed)
    if len(data) > max_contexts:
        idxs = rnd.sample(range(len(data)), max_contexts)
        data = [data[i] for i in sorted(idxs)]

    contexts: list[Context] = []
    for ex in data:
        question = str(ex.get("question") or "").strip()
        gold = str(ex.get("answer") or "").strip()
        sessions = ex.get("haystack_sessions") or []
        dates = ex.get("haystack_dates") or [""] * len(sessions)
        if not question or not sessions:
            continue
        ctx_text = truncate_to_tokens(
            _format_lme_sessions(sessions, dates), max_context_tokens,
        )
        ps = ProbeState(
            state_id=f"{ex.get('question_id')}::q",
            step=0,
            state_text="",        # the question itself is the trigger
            gold_action=gold,
        )
        ctx = Context(
            context_id=f"longmemeval::{ex.get('question_id')}",
            source="longmemeval",
            question=question,
            context_text=ctx_text,
            n_tokens=count_tokens(ctx_text),
            probe_states=[ps],
            task_type="qa",
        )
        contexts.append(ctx)
        _logger.info(
            "Loaded %s (%d tokens) — '%s'",
            ctx.context_id, ctx.n_tokens, question[:80],
        )
    if not contexts:
        raise RuntimeError("No usable LongMemEval contexts produced.")
    return contexts


# ---------------------------------------------------------------------------
# LoCoMo loader
# ---------------------------------------------------------------------------

def _format_locomo_conversation(conv: dict) -> str:
    """Stitch a LoCoMo conversation's sessions into one readable transcript."""
    parts = []
    session_keys = sorted(
        (k for k in conv if k.startswith("session_") and not k.endswith("date_time")),
        key=lambda k: int(k.split("_")[1]),
    )
    for sk in session_keys:
        idx = sk.split("_")[1]
        date = conv.get(f"session_{idx}_date_time", "")
        parts.append(f"## Session {idx}  (date: {date})")
        for turn in conv[sk]:
            speaker = turn.get("speaker", "?")
            text = (turn.get("text") or "").strip()
            dia = turn.get("dia_id", "")
            parts.append(f"[{dia}] {speaker}: {text}")
        parts.append("")
    return "\n".join(parts)


def load_locomo_contexts(
    data_path: str,
    *,
    max_contexts: int,
    max_context_tokens: int,
    qa_per_conversation: int = 5,
    seed: int = 42,
    **_kwargs,
) -> List[Context]:
    """Load LoCoMo's locomo10.json and emit up to ``max_contexts`` Contexts.

    Each Context = (conversation, single QA pair), where the QA's question
    is the probe and the QA's answer is the gold action. We sample
    ``qa_per_conversation`` QA pairs per conversation to balance breadth
    across conversations vs depth within one.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"LoCoMo file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _logger.info("LoCoMo: %d conversations in %s", len(data), path.name)
    rnd = random.Random(seed)

    contexts: list[Context] = []
    for conv_obj in data:
        if len(contexts) >= max_contexts:
            break
        sample_id = conv_obj.get("sample_id", stable_hash(conv_obj))
        full_text = _format_locomo_conversation(conv_obj["conversation"])
        full_text = truncate_to_tokens(full_text, max_context_tokens)
        n_tokens = count_tokens(full_text)

        qa_list = conv_obj.get("qa") or []
        if not qa_list:
            continue
        sample_n = min(qa_per_conversation, len(qa_list))
        chosen = rnd.sample(qa_list, sample_n)
        for qa in chosen:
            if len(contexts) >= max_contexts:
                break
            question = (qa.get("question") or "").strip()
            gold = str(qa.get("answer") or "").strip()
            if not question or not gold:
                continue
            ps = ProbeState(
                state_id=f"{sample_id}::{stable_hash(question)}",
                step=0,
                state_text="",
                gold_action=gold,
            )
            ctx = Context(
                context_id=f"locomo::{sample_id}::{stable_hash(question)}",
                source="locomo",
                question=question,
                context_text=full_text,
                n_tokens=n_tokens,
                probe_states=[ps],
                task_type="qa",
            )
            contexts.append(ctx)
            _logger.info(
                "Loaded %s (%d tokens) — '%s'",
                ctx.context_id, ctx.n_tokens, question[:80],
            )

    if not contexts:
        raise RuntimeError("No usable LoCoMo contexts produced.")
    return contexts


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def load_contexts(cfg: dict) -> List[Context]:
    """Top-level dispatcher used by the run scripts."""
    src = cfg.get("source", "appworld")
    if src == "appworld":
        return load_appworld_contexts(
            trajectory_glob=cfg["trajectory_glob"],
            max_contexts=int(cfg.get("max_contexts", 5)),
            probe_states_per_context=int(cfg.get("probe_states_per_context", 4)),
            max_context_tokens=int(cfg.get("max_context_tokens", 4000)),
        )
    if src == "longmemeval":
        return load_longmemeval_contexts(
            data_path=cfg["data_path"],
            max_contexts=int(cfg.get("max_contexts", 5)),
            max_context_tokens=int(cfg.get("max_context_tokens", 8000)),
            seed=int(cfg.get("seed", 42)),
        )
    if src == "locomo":
        return load_locomo_contexts(
            data_path=cfg["data_path"],
            max_contexts=int(cfg.get("max_contexts", 5)),
            max_context_tokens=int(cfg.get("max_context_tokens", 8000)),
            qa_per_conversation=int(cfg.get("qa_per_conversation", 5)),
            seed=int(cfg.get("seed", 42)),
        )
    raise ValueError(f"Unknown data source: {src}")
