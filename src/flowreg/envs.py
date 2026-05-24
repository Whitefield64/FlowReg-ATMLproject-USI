"""Environment factories with lazy imports to keep Atari and MiniGrid isolated."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import gymnasium as gym


@lru_cache(maxsize=1)
def _minigrid_components():
    import minigrid  # noqa: F401 - importing registers MiniGrid environments
    from minigrid.wrappers import ImgObsWrapper
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    return Monitor, DummyVecEnv, ImgObsWrapper


@lru_cache(maxsize=1)
def _atari_components():
    import ale_py
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

    gym.register_envs(ale_py)
    return make_atari_env, SubprocVecEnv, VecFrameStack


def _register_atari_envs() -> None:
    import ale_py

    gym.register_envs(ale_py)


def _make_registered_atari_env_factory(env_id: str) -> Callable[..., gym.Env]:
    def _init(**kwargs) -> gym.Env:
        _register_atari_envs()
        render_kwargs = {"render_mode": "rgb_array"}
        render_kwargs.update(kwargs)
        try:
            return gym.make(env_id, **render_kwargs)
        except TypeError:
            return gym.make(env_id, **kwargs)

    return _init


def make_minigrid_env(
    env_id: str,
    seed: int,
    monitor_dir: str | Path | None = None,
    rank: int = 0,
    wrapper: str = "img_flatten",
) -> gym.Env:
    """Create a deterministic MiniGrid environment with project-standard wrappers."""
    Monitor, _dummy_vec_env_cls, ImgObsWrapper = _minigrid_components()

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
) -> Any:
    """Create a simple vectorized env. DummyVecEnv is enough for local MiniGrid."""
    _monitor_cls, DummyVecEnv, _img_obs_wrapper = _minigrid_components()
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


def make_atari_environment(
    env_id: str,
    seed: int,
    n_envs: int,
    monitor_dir: str | Path | None = None,
    env_kwargs: dict | None = None,
    wrapper_kwargs: dict | None = None,
) -> Any:
    """Create a vectorized Atari environment with standard wrappers (Nature CNN compatible)."""
    make_atari_env, SubprocVecEnv, VecFrameStack = _atari_components()

    env = make_atari_env(
        env_id=_make_registered_atari_env_factory(env_id),
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(monitor_dir) if monitor_dir else None,
        env_kwargs=env_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        vec_env_cls=SubprocVecEnv,
    )
    env = VecFrameStack(env, n_stack=4)
    return env
