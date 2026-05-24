"""Tiny W&B helpers — graceful no-op when ``wandb.enabled = false`` or
``WANDB_MODE=disabled``.
"""

from __future__ import annotations

import os
from typing import Any, Optional

try:
    import wandb  # type: ignore
except ImportError:                          # pragma: no cover
    wandb = None  # type: ignore


def _enabled(cfg: dict) -> bool:
    wb = cfg.get("wandb") or {}
    if wb.get("enabled") is False:
        return False
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return False
    return wandb is not None


class WandBRun:
    """Tiny adapter so call sites don't have to care if wandb is installed."""

    def __init__(self, cfg: dict, *, name: str, job_type: str):
        self._enabled = _enabled(cfg)
        self.run = None
        if not self._enabled:
            return
        wb = cfg.get("wandb") or {}
        self.run = wandb.init(
            project=wb.get("project") or "easmo-motivation",
            entity=wb.get("entity"),
            name=name,
            job_type=job_type,
            group=wb.get("group"),
            tags=wb.get("tags") or [],
            config=cfg,
            reinit=True,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled and self.run is not None

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        self.run.log(metrics, step=step)

    def log_table(self, name: str, columns: list, rows: list[list]) -> None:
        if not self.enabled:
            return
        table = wandb.Table(columns=columns, data=rows)
        self.run.log({name: table})

    def log_artifact(self, path: str, artifact_name: str, artifact_type: str) -> None:
        if not self.enabled:
            return
        art = wandb.Artifact(artifact_name, type=artifact_type)
        art.add_file(path)
        self.run.log_artifact(art)

    def summary(self, **kwargs) -> None:
        if not self.enabled:
            return
        for k, v in kwargs.items():
            self.run.summary[k] = v

    def finish(self) -> None:
        if self.enabled:
            self.run.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish()
