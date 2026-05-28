"""Fact extraction (spec §7).

Two-stage:

1. **Deterministic candidates** (cheap, no LLM): regex over the rendered
   trajectory for API calls, IDs, paths, tokens, dates, amounts,
   exceptions, state-changing verbs.
2. **LLM fact inventory** (one call per case): the
   ``FACT_INVENTORY_PROMPT`` returns a normalised JSON list with
   ``fact_type``, ``source_quote``, ``literal_values``. We then
   substring-ground every fact and apply per-case caps.

Primary analysis uses only ``grounded_by_substring=True`` facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Fixed taxonomy from spec §6.
TAXONOMY = (
    "NARRATIVE_GOAL",
    "NARRATIVE_PROGRESS",
    "HIGH_LEVEL_REASONING",
    "PENDING_SUBTASK",
    "COMPLETED_SUBTASK",
    "RUNTIME_VARIABLE",
    "AUTH_OR_ACCESS_TOKEN",
    "EXACT_IDENTIFIER",
    "FILE_PATH_OR_RESOURCE_LOCATOR",
    "API_SCHEMA_OR_PARAMETER",
    "ACTION_OUTCOME",
    "ENVIRONMENT_STATE",
    "NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT",
    "STALE_OR_OVERWRITTEN_STATE",
    "NUMERIC_OR_DATE_LITERAL",
    "OTHER_CONCRETE_DETAIL",
)

COARSE_GROUP = {
    "NARRATIVE_GOAL": "NARRATIVE",
    "NARRATIVE_PROGRESS": "NARRATIVE",
    "HIGH_LEVEL_REASONING": "NARRATIVE",
    "PENDING_SUBTASK": "TASK_STATE",
    "COMPLETED_SUBTASK": "TASK_STATE",
    "ENVIRONMENT_STATE": "TASK_STATE",
    "STALE_OR_OVERWRITTEN_STATE": "TASK_STATE",
    "RUNTIME_VARIABLE": "EXECUTABLE",
    "AUTH_OR_ACCESS_TOKEN": "EXECUTABLE",
    "EXACT_IDENTIFIER": "EXECUTABLE",
    "FILE_PATH_OR_RESOURCE_LOCATOR": "EXECUTABLE",
    "API_SCHEMA_OR_PARAMETER": "EXECUTABLE",
    "ACTION_OUTCOME": "EXECUTABLE",
    "NUMERIC_OR_DATE_LITERAL": "EXECUTABLE",
    "NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT": "CONTROL",
    "OTHER_CONCRETE_DETAIL": "OTHER",
}

# Per-case caps (spec §7.3)
PER_CASE_CAPS = {
    "NARRATIVE":  3,
    "TASK_STATE": 3,
    "EXECUTABLE": 6,
    "CONTROL":    2,
    "OTHER":      1,    # tiny budget to capture a stray useful fact
}

# Spec §7.3 — only facts with length_tokens <= 80 are kept.
MAX_FACT_LENGTH_TOKENS = 80


_RE_API_CALL = re.compile(r"apis\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\([^\n]*\)")
_RE_API_NAME = re.compile(r"apis\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)")
_RE_KEYWORD_ARG = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^,\)\n]+)")
_RE_PATH = re.compile(r"(?:/[a-zA-Z0-9_\-./]+|~[/\w.\-]+)")
_RE_ID_FIELD = re.compile(r"\"([a-zA-Z_][a-zA-Z0-9_]*_id)\"\s*:\s*\"?([^,}\"\n]+)")
_RE_TOKEN = re.compile(r"\baccess_token\b[\"\']?\s*[:=]\s*[\"\']?([A-Za-z0-9._\-]{8,})")
_RE_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?)?\b")
_RE_AMOUNT = re.compile(r"\$\s?\d+(?:\.\d{1,2})?")
_RE_ERROR = re.compile(r"(?:Traceback|Exception|Error)[^\n]{0,160}")
_STATE_VERBS = (
    "delete", "update", "create", "send", "accept", "like", "unlike",
    "move", "copy", "login", "logout", "remove", "post", "transfer",
    "pay", "transfer",
)


@dataclass
class DeterministicCandidate:
    text: str
    deterministic_type_hint: str
    literal_values: List[str] = field(default_factory=list)
    extraction_method: str = ""
    source_step_ids: List[int] = field(default_factory=list)
    source_span: str = ""

    def to_dict(self, case_id: str, fact_id: str) -> dict:
        return {
            "case_id": case_id,
            "fact_id": fact_id,
            "candidate_text": self.text,
            "source_step_ids": self.source_step_ids,
            "source_span": self.source_span,
            "deterministic_type_hint": self.deterministic_type_hint,
            "literal_keys": [],
            "literal_values": self.literal_values,
            "extraction_method": self.extraction_method,
        }


def deterministic_candidates(
    trajectory_steps: List[dict],
) -> List[DeterministicCandidate]:
    """Run regex extractors over each step. Each match becomes one
    candidate. We deduplicate by ``(text, step_id)``."""
    seen: set = set()
    out: List[DeterministicCandidate] = []

    def _add(text: str, hint: str, method: str,
             step_id: int, literals: Optional[List[str]] = None) -> None:
        text = text.strip()
        key = (text, step_id, hint)
        if not text or key in seen:
            return
        seen.add(key)
        out.append(DeterministicCandidate(
            text=text,
            deterministic_type_hint=hint,
            literal_values=literals or [],
            extraction_method=method,
            source_step_ids=[step_id],
            source_span=text[:160],
        ))

    for step in trajectory_steps:
        step_id = int(step.get("step_id", step.get("step", 0)))
        # join action + observation since both contain literals
        blob = (step.get("action") or "") + "\n" + (step.get("observation") or step.get("output") or "")

        for m in _RE_API_CALL.finditer(blob):
            _add(m.group(0), "API_SCHEMA_OR_PARAMETER", "regex_api_call", step_id)
        for m in _RE_API_NAME.finditer(blob):
            _add(f"{m.group(1)}.{m.group(2)}", "API_SCHEMA_OR_PARAMETER",
                 "api_name", step_id, [m.group(1), m.group(2)])
        for m in _RE_KEYWORD_ARG.finditer(blob):
            key, val = m.group(1), m.group(2).strip().strip("\"'")
            if len(val) < 80 and len(val) > 0:
                _add(f"{key}={val}", "API_SCHEMA_OR_PARAMETER",
                     "keyword_arg", step_id, [val])
        for m in _RE_PATH.finditer(blob):
            path = m.group(0)
            if len(path) > 3:
                _add(path, "FILE_PATH_OR_RESOURCE_LOCATOR", "regex_path",
                     step_id, [path])
        for m in _RE_ID_FIELD.finditer(blob):
            key, val = m.group(1), m.group(2).strip().strip("\"'")
            _add(f"{key}={val}", "EXACT_IDENTIFIER", "json_id_field",
                 step_id, [val])
        for m in _RE_TOKEN.finditer(blob):
            _add(m.group(0), "AUTH_OR_ACCESS_TOKEN", "regex_token",
                 step_id, [m.group(1)])
        for m in _RE_DATE.finditer(blob):
            _add(m.group(0), "NUMERIC_OR_DATE_LITERAL", "regex_date",
                 step_id, [m.group(0)])
        for m in _RE_AMOUNT.finditer(blob):
            _add(m.group(0), "NUMERIC_OR_DATE_LITERAL", "regex_amount",
                 step_id, [m.group(0)])
        for m in _RE_ERROR.finditer(blob):
            _add(m.group(0)[:160], "NEGATIVE_EVIDENCE_OR_FAILED_ATTEMPT",
                 "regex_error", step_id)
        for verb in _STATE_VERBS:
            if re.search(rf"\b{re.escape(verb)}\b", blob, re.IGNORECASE):
                _add(verb, "ACTION_OUTCOME", "state_changing_verb", step_id)

    return out


def estimate_tokens(text: str) -> int:
    """Cheap surrogate: chars/4."""
    return max(1, len(text) // 4)


def substring_grounded(haystack: str, candidates: List[str]) -> bool:
    """True if any candidate is a substring of haystack (case-insensitive)
    after a permissive normalisation (whitespace collapsing)."""
    h_norm = re.sub(r"\s+", " ", haystack.lower())
    for c in candidates:
        if not c:
            continue
        c_norm = re.sub(r"\s+", " ", c.lower())
        if c_norm and c_norm in h_norm:
            return True
    return False


def normalise_fact_record(
    case_id: str,
    fact_id: str,
    fact_record: dict,
    trajectory_text: str,
) -> Optional[dict]:
    """Coerce a raw LLM-extracted fact into the v7 schema. Drops facts
    that are missing required fields. ``grounded_by_substring`` is
    computed against ``trajectory_text``.
    """
    if not isinstance(fact_record, dict):
        return None
    fact_type = (fact_record.get("fact_type") or "").strip()
    if fact_type not in TAXONOMY:
        return None
    canonical = (fact_record.get("canonical_fact") or "").strip()
    if not canonical:
        return None
    source_quote = (fact_record.get("source_quote") or "").strip()
    if not source_quote:
        return None
    verbatim_surface = (fact_record.get("verbatim_surface") or canonical).strip()
    literals = fact_record.get("literal_values") or []
    if isinstance(literals, str):
        literals = [literals]
    literals = [str(x) for x in literals if x]
    length_tokens = estimate_tokens(canonical)
    grounded = substring_grounded(
        trajectory_text,
        [source_quote, verbatim_surface] + literals,
    )
    return {
        "case_id": case_id,
        "fact_id": fact_id,
        "fact_type": fact_type,
        "coarse_group": COARSE_GROUP.get(fact_type, "OTHER"),
        "canonical_fact": canonical,
        "verbatim_surface": verbatim_surface,
        "source_step_ids": fact_record.get("source_step_ids") or [],
        "source_quote": source_quote,
        "is_exact_literal": bool(fact_record.get("is_exact_literal", False)),
        "literal_values": literals,
        "grounded_by_substring": grounded,
        "length_tokens": length_tokens,
        "notes": (fact_record.get("why_it_might_matter") or "").strip()[:160],
    }


def apply_per_case_caps(facts: List[dict]) -> Tuple[List[dict], dict]:
    """Apply spec §7.3 caps. Returns (kept, missingness-report).

    Preference within a group:
      1. ``is_exact_literal`` first;
      2. shorter facts first (lower length_tokens);
      3. earlier ``source_step_ids`` first.
    """
    by_group: Dict[str, List[dict]] = {g: [] for g in PER_CASE_CAPS}
    for f in facts:
        if not f.get("grounded_by_substring"):
            continue
        if f.get("length_tokens", 0) > MAX_FACT_LENGTH_TOKENS:
            continue
        g = f.get("coarse_group", "OTHER")
        by_group.setdefault(g, []).append(f)

    kept: List[dict] = []
    miss: Dict[str, int] = {}
    for g, cap in PER_CASE_CAPS.items():
        cand = sorted(by_group.get(g, []), key=lambda r: (
            not bool(r.get("is_exact_literal")),
            r.get("length_tokens", 0),
            min(r.get("source_step_ids") or [99999]),
        ))
        kept_g = cand[:cap]
        kept.extend(kept_g)
        miss[g] = max(0, cap - len(kept_g))
    return kept, miss


__all__ = [
    "TAXONOMY", "COARSE_GROUP", "PER_CASE_CAPS", "MAX_FACT_LENGTH_TOKENS",
    "DeterministicCandidate",
    "deterministic_candidates",
    "estimate_tokens",
    "substring_grounded",
    "normalise_fact_record",
    "apply_per_case_caps",
]
