"""Configuration helpers for command-line experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


BASE_CONFIG_KEY = "base_config"


def load_yaml_config(path: str | Path, _stack: tuple[Path, ...] | None = None) -> dict[str, Any]:
    """Load a YAML config file, optionally inheriting from `base_config`."""
    config_path = Path(path).expanduser().resolve()
    stack = _stack or ()
    if config_path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, config_path))
        raise ValueError(f"Config inheritance cycle detected: {cycle}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    data = dict(data)

    base_config = data.pop(BASE_CONFIG_KEY, None)
    if base_config is None:
        return data
    if not isinstance(base_config, str) or not base_config:
        raise ValueError(f"`{BASE_CONFIG_KEY}` must be a non-empty string in config file: {path}")

    base_path = Path(base_config).expanduser()
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path

    base_data = load_yaml_config(base_path, (*stack, config_path))
    return deep_update(base_data, data)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating inputs."""
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged
