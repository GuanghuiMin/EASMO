"""Rule-based decision-state distance (§6.1).

Given a reference decision state and an ablated state, computes:

  next_action_type_changed         : 0/1
  next_action_arguments_f1_loss    : 0..1
  candidate_objects_f1_loss        : 0..1
  avoid_objects_f1_loss            : 0..1
  active_constraints_f1_loss       : 0..1
  completed_actions_f1_loss        : 0..1
  missing_information_increase     : 0/1
  confidence_drop                  : 0/1

Plus the spec's weighted aggregate sensitivity score and a 0..1
normalised version.

F1 losses are computed at the entity level using lightweight
canonicalisation (lowercased alphanumerics) to match across paraphrase.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def _entity_key(text: str) -> str:
    """Normalise an entity / fact / constraint string for F1 matching."""
    s = " ".join(_TOKEN_RE.findall((text or "").lower()))
    return s[:160]


def _set_keys(items, fn) -> Set[str]:
    keys: Set[str] = set()
    for x in items or []:
        k = fn(x)
        if k:
            keys.add(k)
    return keys


def _f1_loss(ref_keys: Set[str], abl_keys: Set[str]) -> float:
    """Return 1 - F1(ref, abl). 0 means abl matches ref perfectly;
    1 means total disagreement. If both sets are empty, return 0."""
    if not ref_keys and not abl_keys:
        return 0.0
    tp = len(ref_keys & abl_keys)
    fp = len(abl_keys - ref_keys)
    fn = len(ref_keys - abl_keys)
    if tp == 0:
        return 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 1.0
    f1 = 2 * precision * recall / (precision + recall)
    return max(0.0, 1.0 - f1)


# ----------------------------------------------------------------------
# Per-field key extractors
# ----------------------------------------------------------------------


def _completed_action_key(x: dict) -> str:
    return _entity_key(f"{x.get('action','')} {x.get('object','')}")


def _constraint_key(x: dict) -> str:
    return _entity_key(x.get("constraint", ""))


def _candidate_key(x: dict) -> str:
    return _entity_key(f"{x.get('object_id','')} {x.get('object_type','')}")


def _avoid_key(x: dict) -> str:
    return _entity_key(f"{x.get('object_id','')} {x.get('object_type','')}")


def _next_args_keys(args) -> Set[str]:
    """Each kw=value pair becomes one entity key."""
    out: Set[str] = set()
    if isinstance(args, dict):
        for k, v in args.items():
            out.add(_entity_key(f"{k}={v}"))
    elif isinstance(args, list):
        for x in args:
            out.add(_entity_key(str(x)))
    return out


# ----------------------------------------------------------------------
# Aggregator
# ----------------------------------------------------------------------


@dataclass
class RuleScore:
    next_action_type_changed: int            # 0 or 1
    next_action_arguments_f1_loss: float
    candidate_objects_f1_loss: float
    avoid_objects_f1_loss: float
    active_constraints_f1_loss: float
    completed_actions_f1_loss: float
    missing_information_increase: int        # 0 or 1
    confidence_drop: int                     # 0 or 1
    weighted_sensitivity: float
    normalised_sensitivity: float            # 0..1

    def to_dict(self) -> dict:
        return self.__dict__.copy()


_W = {
    "next_action_type":        2.0,
    "next_action_arguments":   2.0,
    "candidate_objects":       1.5,
    "active_constraints":      1.5,
    "avoid_objects":           1.0,
    "completed_actions":       1.0,
    "missing_information":     1.0,
    "confidence":              0.5,
}
_W_TOTAL = sum(_W.values())


def rule_based_distance(reference: dict, ablated: dict) -> RuleScore:
    nat_changed = int(
        (_entity_key(reference.get("next_action_type", ""))
         != _entity_key(ablated.get("next_action_type", "")))
    )

    args_loss = _f1_loss(
        _next_args_keys(reference.get("next_action_arguments")),
        _next_args_keys(ablated.get("next_action_arguments")),
    )
    cand_loss = _f1_loss(
        _set_keys(reference.get("candidate_objects"), _candidate_key),
        _set_keys(ablated.get("candidate_objects"), _candidate_key),
    )
    avoid_loss = _f1_loss(
        _set_keys(reference.get("avoid_objects"), _avoid_key),
        _set_keys(ablated.get("avoid_objects"), _avoid_key),
    )
    constr_loss = _f1_loss(
        _set_keys(reference.get("active_constraints"), _constraint_key),
        _set_keys(ablated.get("active_constraints"), _constraint_key),
    )
    completed_loss = _f1_loss(
        _set_keys(reference.get("completed_actions"), _completed_action_key),
        _set_keys(ablated.get("completed_actions"), _completed_action_key),
    )

    ref_missing = len(reference.get("missing_information") or [])
    abl_missing = len(ablated.get("missing_information") or [])
    missing_increase = 1 if abl_missing > ref_missing else 0

    ref_conf = _CONFIDENCE_ORDER.get(reference.get("confidence", "low"), 0)
    abl_conf = _CONFIDENCE_ORDER.get(ablated.get("confidence", "low"), 0)
    conf_drop = 1 if abl_conf < ref_conf else 0

    weighted = (
        _W["next_action_type"]      * nat_changed
        + _W["next_action_arguments"]* args_loss
        + _W["candidate_objects"]    * cand_loss
        + _W["active_constraints"]   * constr_loss
        + _W["avoid_objects"]        * avoid_loss
        + _W["completed_actions"]    * completed_loss
        + _W["missing_information"]  * missing_increase
        + _W["confidence"]           * conf_drop
    )
    return RuleScore(
        next_action_type_changed=nat_changed,
        next_action_arguments_f1_loss=round(args_loss, 4),
        candidate_objects_f1_loss=round(cand_loss, 4),
        avoid_objects_f1_loss=round(avoid_loss, 4),
        active_constraints_f1_loss=round(constr_loss, 4),
        completed_actions_f1_loss=round(completed_loss, 4),
        missing_information_increase=missing_increase,
        confidence_drop=conf_drop,
        weighted_sensitivity=round(weighted, 4),
        normalised_sensitivity=round(min(weighted / _W_TOTAL, 1.0), 4),
    )
