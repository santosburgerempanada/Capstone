from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        elif value is not None or key not in out:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    return cfg
