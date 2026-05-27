"""Heuristic counters for Exp 1 metrics (preserved IDs / bindings / constraints / action outcomes).

The spec's Exp 1 metrics are:

  token_count
  number_of_exact_ids_preserved
  number_of_entity_bindings_preserved
  number_of_constraints_preserved
  number_of_action_outcomes_preserved

We compute these by:

1. Mining a *gold* set of IDs / bindings / constraints / action outcomes
   from the full trajectory (deterministic regex extraction).
2. For each compressed method's text, counting how many of those gold
   items appear as substrings (case-insensitive for prose; exact for IDs).

This gives the apples-to-apples count needed for Table 1.

Heuristics (intentional simplicity; precision >> recall is fine):

  ID                    : long alphanumeric tokens (>= 4 chars), e.g.
                          access tokens, large integer IDs, file paths.
  Entity binding        : 'X = Y' assignments where X looks like a
                          variable and Y is a string/email/large int.
  Constraint            : sentences containing 'must', 'should not',
                          'only if', 'avoid', 'do not', 'ensure'.
  Action outcome        : `apis.<app>.<fn>(...)` with concrete args
                          (kw=value) — "agent did X with args Y".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .data import Trajectory


_ID_RE = re.compile(r"\b[A-Za-z0-9_]{6,}\b")
_BIG_DIGIT_RE = re.compile(r"\b\d{4,}\b")
_TOKEN_LIKE_RE = re.compile(r"\b[A-Za-z0-9]{20,}\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[\w.-]+")
_PATH_RE = re.compile(r"/[A-Za-z0-9_]+(?:/[A-Za-z0-9_.{}-]+)+")
_KW_ARG_RE = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*(['\"][^'\"\n]+['\"]|\d+|[A-Za-z0-9_]+)")
_API_CALL_RE = re.compile(r"apis\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\s*\(")
_CONSTRAINT_RE = re.compile(
    r"\b(?:must(?:\s+not)?|should(?:\s+not)?|cannot|do\s+not|avoid|"
    r"only\s+if|ensure(?:\s+that)?|never|always|required\s+to|"
    r"forbidden|prohibited)\b",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Mine gold items from a trajectory
# ----------------------------------------------------------------------


@dataclass
class GoldItems:
    ids: List[str] = field(default_factory=list)
    bindings: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    action_outcomes: List[str] = field(default_factory=list)

    def __len__(self):
        return (len(self.ids) + len(self.bindings)
                + len(self.constraints) + len(self.action_outcomes))

    def to_dict(self) -> dict:
        return {
            "ids": self.ids,
            "bindings": self.bindings,
            "constraints": self.constraints,
            "action_outcomes": self.action_outcomes,
        }


# Stopword-ish IDs we never want to count (too generic).
_ID_STOPWORDS = frozenset({
    "string", "integer", "boolean", "object", "true", "false", "null",
    "supervisor", "spotify", "venmo", "phone", "amazon", "gmail",
    "todoist", "splitwise", "simple_note", "file_system",
    "access_token", "page_index", "page_limit", "show_account",
    "show_album", "show_song", "show_playlist", "show_artist",
    "show_api_descriptions", "show_api_doc", "complete_task",
    "show_account_passwords", "Bearer", "username", "password",
})


def _norm_id(t: str) -> str:
    return t.strip()


def _looks_like_id(t: str) -> bool:
    if t.lower() in _ID_STOPWORDS:
        return False
    if _BIG_DIGIT_RE.fullmatch(t):
        return True
    if _TOKEN_LIKE_RE.fullmatch(t):
        return True
    if any(c.isdigit() for c in t) and len(t) >= 6:
        return True
    return False


def mine_gold_items(traj: Trajectory) -> GoldItems:
    """Extract ID / binding / constraint / action_outcome candidates from
    the full trajectory's actions and outputs."""
    ids: Set[str] = set()
    bindings: Set[str] = set()
    constraints: Set[str] = set()
    action_outcomes: Set[str] = set()

    for s in traj.steps:
        action = s.action or ""
        output = s.output or ""
        joined = action + "\n" + output

        # ---- IDs ----
        for m in _BIG_DIGIT_RE.finditer(joined):
            ids.add(m.group(0))
        for m in _TOKEN_LIKE_RE.finditer(joined):
            ids.add(m.group(0))
        for m in _EMAIL_RE.finditer(joined):
            ids.add(m.group(0))
        for m in _PATH_RE.finditer(joined):
            ids.add(m.group(0))

        # ---- Bindings (from action code: kw="value") ----
        for m in _KW_ARG_RE.finditer(action):
            key, val = m.group(1), m.group(2).strip("'\"")
            if key.lower() in _ID_STOPWORDS:
                continue
            if not val or val.lower() in {"true", "false", "none"}:
                continue
            bindings.add(f"{key}={val}")

        # ---- Constraints (from instruction + output sentences) ----
        for chunk in (traj.instruction or "", action, output):
            for sent in re.split(r"(?<=[.!?])\s+", chunk):
                if _CONSTRAINT_RE.search(sent):
                    s_norm = sent.strip()
                    if 8 <= len(s_norm) <= 200:
                        constraints.add(s_norm)

        # ---- Action outcomes: apis.X.Y(...) with concrete args ----
        for m in _API_CALL_RE.finditer(action):
            app, fn = m.group(1), m.group(2)
            # Extract args within the surrounding parentheses (best-effort).
            tail = action[m.end():]
            depth = 1
            args = []
            for ch in tail:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
                args.append(ch)
            arg_str = "".join(args).strip()
            # Keep only calls with concrete args (some kw=value detected).
            if _KW_ARG_RE.search(arg_str) or _BIG_DIGIT_RE.search(arg_str):
                action_outcomes.add(f"apis.{app}.{fn}({arg_str[:120]})")

    # Drop trivial IDs.
    return GoldItems(
        ids=sorted({i for i in ids if _looks_like_id(i)}),
        bindings=sorted(bindings),
        constraints=sorted(constraints),
        action_outcomes=sorted(action_outcomes),
    )


# ----------------------------------------------------------------------
# Match gold items into a compressed text
# ----------------------------------------------------------------------


def _contains(text: str, item: str) -> bool:
    """Loose containment: case-insensitive substring match. For IDs that
    are long (>= 6 chars) we require an exact-substring match; for
    bindings / constraints we lower-case both and check substring."""
    if not item:
        return False
    if len(item) >= 6 and any(c.isdigit() for c in item):
        return item in text
    return item.lower() in text.lower()


def count_preserved(text: str, gold: GoldItems) -> Dict[str, int]:
    n_ids = sum(1 for x in gold.ids if _contains(text, x))
    n_b = sum(1 for x in gold.bindings if _contains(text, x))
    n_c = sum(1 for x in gold.constraints if _contains(text, x))
    n_a = sum(1 for x in gold.action_outcomes if _contains(text, x))
    return {
        "n_ids_preserved":             n_ids,
        "n_bindings_preserved":        n_b,
        "n_constraints_preserved":     n_c,
        "n_action_outcomes_preserved": n_a,
    }


def gold_sizes(gold: GoldItems) -> Dict[str, int]:
    return {
        "n_ids_total":             len(gold.ids),
        "n_bindings_total":        len(gold.bindings),
        "n_constraints_total":     len(gold.constraints),
        "n_action_outcomes_total": len(gold.action_outcomes),
    }
