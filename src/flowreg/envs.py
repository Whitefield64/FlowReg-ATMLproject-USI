"""MiniGrid environment factory used by baseline and future FlowReg agents."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401 - importing registers MiniGrid environments
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def make_minigrid_env(
    env_id: str,
    seed: int,
    monitor_dir: str | Path | None = None,
    rank: int = 0,
    wrapper: str = "img_flatten",
) -> gym.Env:
    """Create a deterministic MiniGrid environment with project-standard wrappers."""
    env = gym.make(env_id)
    if wrapper == "img_flatten":
        env = ImgObsWrapper(env)
        env = gym.wrappers.FlattenObservation(env)
    else:
        raise ValueError(f"Unsupported MiniGrid wrapper: {wrapper}")

    monitor_path = None
    if monitor_dir is not None:
        monitor_path = str(Path(monitor_dir) / f"monitor_{rank}.csv")
    env = Monitor(env, filename=monitor_path)
    env.reset(seed=seed + rank)
    env.action_space.seed(seed + rank)
    return env


def make_env_thunk(
    env_id: str,
    seed: int,
    monitor_dir: str | Path | None = None,
    rank: int = 0,
    wrapper: str = "img_flatten",
) -> Callable[[], gym.Env]:
    """Return a thunk compatible with SB3 vectorized environments."""

    def _init() -> gym.Env:
        return make_minigrid_env(
            env_id=env_id,
            seed=seed,
            monitor_dir=monitor_dir,
            rank=rank,
            wrapper=wrapper,
        )

    return _init


def make_dummy_vec_env(
    env_id: str,
    seed: int,
    n_envs: int,
    monitor_dir: str | Path | None = None,
    wrapper: str = "img_flatten",
) -> DummyVecEnv:
    """Create a simple vectorized env. DummyVecEnv is enough for local MiniGrid."""
    return DummyVecEnv(
        [
            make_env_thunk(
                env_id=env_id,
                seed=seed,
                monitor_dir=monitor_dir,
                rank=rank,
                wrapper=wrapper,
            )
            for rank in range(n_envs)
        ]
    )

