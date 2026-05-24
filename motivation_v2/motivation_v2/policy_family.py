"""Map an AppWorld task → a policy family.

This is the deterministic "what policy is this task an instance of?"
classifier referenced in §2.4 of new_motivation.md, with the fix from
review note R-2 (single-app filter, ``supervisor`` is plumbing).

Single-app policy family = the one content app the task uses
(``supervisor`` / login plumbing is excluded). Multi-app tasks are
tagged ``multi_app`` and excluded from M3 cross-policy heatmap.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .data import GroundTruth


# Apps treated as plumbing — they appear across many tasks and aren't
# policy-distinguishing. ``supervisor`` is AppWorld's auth + completion
# plumbing.
_PLUMBING_APPS = frozenset({"supervisor"})


# Canonical policy family identifier per single-app task. We default to
# the app name itself so the "family" name is the app name (e.g.
# ``spotify``), but this layer of indirection lets us collapse two apps
# into one family later if we want to (e.g. ``email`` + ``messaging``).
_APP_TO_FAMILY: Dict[str, str] = {
    # one-to-one for now; extend if multiple apps share a family.
}


def _family_for_app(app: str) -> str:
    return _APP_TO_FAMILY.get(app, app)


@dataclass
class PolicyAssignment:
    task_id: str
    family: str        # 'multi_app' or the policy family name (== app name by default)
    is_single_app: bool
    primary_app: str   # the dominant content app, '' for multi-app
    content_apps: List[str]  # all non-plumbing apps required


def assign_policy_family(gt: GroundTruth) -> PolicyAssignment:
    """Classify one task by its required apps."""
    content_apps = [a for a in gt.required_apps if a not in _PLUMBING_APPS]

    if len(content_apps) == 0:
        # Should be very rare — e.g. tasks that only ping supervisor.
        family = "supervisor_only"
        return PolicyAssignment(
            task_id=gt.task_id,
            family=family,
            is_single_app=True,
            primary_app="",
            content_apps=[],
        )

    if len(content_apps) == 1:
        primary = content_apps[0]
        return PolicyAssignment(
            task_id=gt.task_id,
            family=_family_for_app(primary),
            is_single_app=True,
            primary_app=primary,
            content_apps=content_apps,
        )

    return PolicyAssignment(
        task_id=gt.task_id,
        family="multi_app",
        is_single_app=False,
        primary_app="",
        content_apps=content_apps,
    )


def assign_all(tasks: Iterable[GroundTruth]) -> List[PolicyAssignment]:
    return [assign_policy_family(t) for t in tasks]


def family_distribution(assignments: Iterable[PolicyAssignment]) -> Dict[str, int]:
    """Count tasks per family. Useful for the §10 pilot deliverables."""
    return dict(Counter(a.family for a in assignments))


def shared_state_pairs(
    assignments: Iterable[PolicyAssignment],
) -> Dict[tuple, int]:
    """For policy pairs (p, q) with p < q, count how many tasks of family
    p share their first 7-char task-id prefix with a task of family q.

    AppWorld task IDs look like ``82e2fac_3`` where the prefix
    ``82e2fac`` identifies the underlying user-state generator. Tasks
    sharing a prefix are different *parameterisations* of the same
    base generator and operate on closely related (often identical)
    user states — these are the ``same-context`` pairs needed by M3
    (Review note R-3).

    Returns a dict keyed by (family_p, family_q) with the count of
    such pairs across the assignment set. Diagonal pairs are skipped
    (same family-pair = within-policy, not cross-policy).
    """
    by_prefix: Dict[str, List[PolicyAssignment]] = {}
    for a in assignments:
        if not a.is_single_app:
            continue
        prefix = a.task_id.split("_", 1)[0]
        by_prefix.setdefault(prefix, []).append(a)

    counts: Dict[tuple, int] = {}
    for prefix, group in by_prefix.items():
        # All single-app, same-prefix pairs.
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a_i, a_j = group[i], group[j]
                if a_i.family == a_j.family:
                    continue
                key = tuple(sorted([a_i.family, a_j.family]))
                counts[key] = counts.get(key, 0) + 1
    return counts
