"""Tiny shared helpers — tokens, logging, JSON I/O, seeding."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tiktoken

_TOK = tiktoken.encoding_for_model("gpt-4")


def count_tokens(text: str | None) -> int:
    if not text:
        return 0
    return len(_TOK.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    ids = _TOK.encode(text)
    if len(ids) <= max_tokens:
        return text
    return _TOK.decode(ids[:max_tokens])


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
    return h.hexdigest()[:10]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def setup_logging(name: str = "motivation", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


_ENV_VAR_RE = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")


def _interp_env(value):
    if isinstance(value, str):
        def sub(m):
            var, default = m.group(1).strip(), (m.group(2) or "").strip()
            return os.environ.get(var, default)
        return _ENV_VAR_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interp_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interp_env(v) for v in value]
    return value


def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _interp_env(raw)
