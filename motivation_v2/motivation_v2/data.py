"""Read AppWorld ground-truth + acon trajectory outputs into structured records.

Two data sources, indexed by AppWorld task_id:

1. **AppWorld ground truth** lives under ``$APPWORLD_ROOT/tasks/<task_id>/``
   and is shipped with the AppWorld package itself (already downloaded
   under ``/workspace/acon/experiments/appworld/data/``):

   * ``ground_truth/metadata.json``       — difficulty, num_apps, num_api_calls, …
   * ``ground_truth/required_apps.json``  — list[str] of apps used by gold solution
   * ``ground_truth/api_calls.json``      — gold sequence of HTTP-style calls
   * ``ground_truth/answer.json``         — gold answer (when applicable)
   * ``ground_truth/evaluation.py``       — final-state test code (Python source)
   * ``ground_truth/compiled_solution.py``/``solution.py`` — readable gold solution

2. **acon agent trajectory outputs** live under
   ``/workspace/acon/experiments/appworld/outputs/<exp>/<split>/task_<task_id>/``:

   * ``appworld_trajectory.json``  — list of {step, action, output, reward, done}
   * ``env_history.json``          — same content (kept for compatibility)
   * ``results.json``              — success / iterations / final_reward / model_name
   * ``llm_history.json``          — raw LLM messages (for prompt analysis)
   * ``token_usage_and_cost.json`` — billing
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional


DEFAULT_APPWORLD_ROOT = Path("/workspace/acon/experiments/appworld/data")
DEFAULT_TRAJECTORIES_ROOT = Path("/workspace/acon/experiments/appworld/outputs")


# ---------------------------------------------------------------------------
# Ground-truth side
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthApiCall:
    """One canonical API call from the gold solution."""

    method: str           # 'get' | 'post' | 'put' | 'delete'
    url: str              # e.g. '/spotify/library/albums'
    data: dict = field(default_factory=dict)

    @property
    def app(self) -> str:
        """First URL segment, e.g. ``/spotify/library/albums`` → ``spotify``."""
        if not self.url.startswith("/"):
            return ""
        parts = self.url.lstrip("/").split("/")
        return parts[0] if parts else ""

    @property
    def endpoint(self) -> str:
        """Path after the app, e.g. ``/spotify/library/albums`` → ``library/albums``."""
        if not self.url.startswith("/"):
            return self.url
        parts = self.url.lstrip("/").split("/")
        return "/".join(parts[1:])


@dataclass
class GroundTruth:
    """All per-task ground-truth assets needed downstream."""

    task_id: str
    instruction: str
    metadata: dict                    # contents of metadata.json
    required_apps: List[str]          # contents of required_apps.json
    api_calls: List[GroundTruthApiCall]
    answer: Optional[dict] = None     # answer.json (if present, may be {})
    evaluation_source: Optional[str] = None  # raw text of evaluation.py
    solution_source: Optional[str] = None    # raw text of compiled_solution.py
    root: Optional[Path] = None       # path to ground_truth/ for downstream

    # ---- convenience properties --------------------------------------------

    @property
    def difficulty(self) -> int:
        return int(self.metadata.get("difficulty", 0))

    @property
    def num_apps(self) -> int:
        return int(self.metadata.get("num_apps", len(self.required_apps)))

    @property
    def num_api_calls_gold(self) -> int:
        return int(self.metadata.get("num_api_calls", len(self.api_calls)))

    @property
    def is_single_app(self) -> bool:
        # We treat ``supervisor`` (auth / login) as plumbing — not a content
        # app. A task that uses spotify + supervisor is still single-app from
        # the policy-family perspective.
        content_apps = {a for a in self.required_apps if a != "supervisor"}
        return len(content_apps) <= 1

    @property
    def primary_app(self) -> str:
        """The single content app for single-app tasks; '' for multi-app."""
        content_apps = [a for a in self.required_apps if a != "supervisor"]
        if len(content_apps) == 1:
            return content_apps[0]
        return ""


def load_ground_truth(
    task_id: str,
    appworld_root: Path = DEFAULT_APPWORLD_ROOT,
) -> GroundTruth:
    """Read all ground-truth assets for a task. Raises FileNotFoundError if
    any required file is missing — this filters held-out test tasks
    (test_normal / test_challenge) which have no ground truth on disk.
    """
    root = Path(appworld_root) / "tasks" / task_id / "ground_truth"
    if not root.is_dir():
        raise FileNotFoundError(
            f"Ground-truth directory missing for {task_id}: {root}"
        )

    with open(root / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(root / "required_apps.json", "r", encoding="utf-8") as f:
        required_apps = json.load(f)
    with open(root / "api_calls.json", "r", encoding="utf-8") as f:
        raw_calls = json.load(f)

    api_calls = [
        GroundTruthApiCall(
            method=str(c.get("method", "")).lower(),
            url=str(c.get("url", "")),
            data=c.get("data") or {},
        )
        for c in raw_calls
    ]

    # task_instruction lives in tasks/<id>/specs.json or via env.reset() — we
    # pick it up later from the trajectory record (it's stored there too).
    instruction = ""
    specs_path = Path(appworld_root) / "tasks" / task_id / "specs.json"
    if specs_path.exists():
        try:
            with open(specs_path, "r", encoding="utf-8") as f:
                specs = json.load(f)
            instruction = specs.get("task", "") or specs.get("instruction", "")
        except (json.JSONDecodeError, OSError):
            pass

    answer = None
    answer_path = root / "answer.json"
    if answer_path.exists():
        try:
            with open(answer_path, "r", encoding="utf-8") as f:
                answer = json.load(f)
        except (json.JSONDecodeError, OSError):
            answer = None

    eval_src = None
    eval_path = root / "evaluation.py"
    if eval_path.exists():
        eval_src = eval_path.read_text(encoding="utf-8")

    sol_src = None
    sol_path = root / "compiled_solution.py"
    if sol_path.exists():
        sol_src = sol_path.read_text(encoding="utf-8")

    return GroundTruth(
        task_id=task_id,
        instruction=instruction,
        metadata=metadata,
        required_apps=list(required_apps),
        api_calls=api_calls,
        answer=answer,
        evaluation_source=eval_src,
        solution_source=sol_src,
        root=root,
    )


def load_split_task_ids(
    split: str,
    appworld_root: Path = DEFAULT_APPWORLD_ROOT,
) -> List[str]:
    """Return the task IDs in a named split (train / dev / train_tiny / …)."""
    p = Path(appworld_root) / "datasets" / f"{split}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Split file not found: {p}")
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def iter_tasks_with_ground_truth(
    split: str,
    appworld_root: Path = DEFAULT_APPWORLD_ROOT,
) -> Iterator[GroundTruth]:
    """Yield GroundTruth objects for tasks in ``split`` whose ground-truth
    files are present on disk. Silently skips tasks without ground truth
    (i.e. held-out test sets)."""
    for tid in load_split_task_ids(split, appworld_root=appworld_root):
        try:
            yield load_ground_truth(tid, appworld_root=appworld_root)
        except FileNotFoundError:
            continue


# ---------------------------------------------------------------------------
# Trajectory side
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryStep:
    step: int
    action: str         # the Python code the agent executed
    output: str         # API response / printout the agent observed
    reward: float
    done: bool


@dataclass
class Trajectory:
    """One acon agent run on one AppWorld task."""

    task_id: str
    instruction: str
    model_name: str
    experiment_name: str
    split: str
    success: bool
    iterations: int
    final_reward: float
    steps: List[TrajectoryStep]
    co_config: Optional[dict] = None
    raw_results: Optional[dict] = None
    raw_dir: Optional[Path] = None


def load_trajectory(task_dir: Path) -> Trajectory:
    """Load a single acon task output directory."""
    task_dir = Path(task_dir)
    with open(task_dir / "appworld_trajectory.json", "r", encoding="utf-8") as f:
        traj = json.load(f)
    with open(task_dir / "results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    cfg = (results.get("config") or {})
    co = cfg.get("co_config") if isinstance(cfg.get("co_config"), dict) else None

    steps = [
        TrajectoryStep(
            step=int(s.get("step", i)),
            action=str(s.get("action", "")),
            output=str(s.get("output", "")),
            reward=float(s.get("reward", 0.0)),
            done=bool(s.get("done", False)),
        )
        for i, s in enumerate(traj.get("trajectory") or [])
    ]

    return Trajectory(
        task_id=str(traj.get("task_id") or results.get("task_id", "")),
        instruction=str(
            traj.get("task_instruction") or results.get("task_instruction", "")
        ),
        model_name=str(results.get("model_name", "")),
        experiment_name=str(results.get("experiment_name", "")),
        split=str(results.get("split", "")),
        success=bool(results.get("success", False)),
        iterations=int(results.get("iterations", len(steps))),
        final_reward=float(results.get("final_reward", 0.0)),
        steps=steps,
        co_config=co,
        raw_results=results,
        raw_dir=task_dir,
    )


def iter_trajectories(
    experiments_glob: str = str(
        DEFAULT_TRAJECTORIES_ROOT / "*" / "*" / "task_*"
    ),
) -> Iterator[Trajectory]:
    """Yield every Trajectory under acon's outputs directory."""
    import glob

    for path in sorted(glob.glob(experiments_glob)):
        p = Path(path)
        if not (p / "appworld_trajectory.json").exists():
            continue
        if not (p / "results.json").exists():
            continue
        try:
            yield load_trajectory(p)
        except (json.JSONDecodeError, OSError, KeyError):
            continue


def successful_trajectories(
    experiments_glob: str = str(
        DEFAULT_TRAJECTORIES_ROOT / "*" / "*" / "task_*"
    ),
) -> List[Trajectory]:
    """Convenience: filter ``iter_trajectories`` to ``success=True`` runs."""
    return [t for t in iter_trajectories(experiments_glob) if t.success]
