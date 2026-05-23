"""Shared helpers for training entrypoints."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import stable_baselines3
import torch


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_wandb_mode(config_mode: str, cli_mode: str | None) -> str:
    mode = cli_mode or config_mode or "disabled"
    if mode not in {"disabled", "offline", "online"}:
        raise ValueError("wandb mode must be one of: disabled, offline, online")
    return mode


def write_config_snapshot(config: dict[str, Any], run_dir: Path) -> None:
    snapshot = dict(config)
    snapshot["versions"] = {
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)


def prepare_a2c_config(config: dict[str, Any]) -> dict[str, Any]:
    """Translate project A2C config into SB3 kwargs."""
    a2c_config = dict(config.get("a2c", {}))
    schedule = str(
        a2c_config.pop("learning_rate_schedule", config.get("learning_rate_schedule", "constant"))
    )
    if schedule == "constant":
        return a2c_config
    if schedule == "linear":
        initial_lr = float(a2c_config["learning_rate"])
        a2c_config["learning_rate"] = lambda progress_remaining: progress_remaining * initial_lr
        return a2c_config
    raise ValueError("learning_rate_schedule must be one of: constant, linear")


def atari_env_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("atari_env_kwargs", {}) or {})


def atari_wrapper_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("atari_wrapper_kwargs", {}) or {})
