"""Phase 1 smoke checks for the MiniGrid + SB3 stack.

This module intentionally does not implement FlowReg. It verifies imports,
device visibility, MiniGrid environment creation, and a tiny PPO training run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401 - importing registers MiniGrid environments
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torchdiffeq import odeint


DEFAULT_ENV_ID = "MiniGrid-DoorKey-5x5-v0"


def make_env(env_id: str, seed: int) -> gym.Env:
    """Create the minimal MiniGrid smoke environment."""
    env = gym.make(env_id)
    env = ImgObsWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def check_imports_and_devices() -> None:
    """Print safe version/device information without exposing secrets."""
    print(f"torch={torch.__version__}")
    print(f"numpy={np.__version__}")
    print(f"wandb={wandb.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"mps_built={torch.backends.mps.is_built()}")
    print(f"mps_available={torch.backends.mps.is_available()}")

    # Tiny torchdiffeq CPU check. Keep float32 for MPS compatibility later.
    class Decay(torch.nn.Module):
        def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            del t
            return -z

    z0 = torch.ones(2, 3, dtype=torch.float32)
    t = torch.linspace(0, 1, 3, dtype=torch.float32)
    zt = odeint(Decay(), z0, t, rtol=1e-4, atol=1e-5)
    print(f"torchdiffeq_check_shape={tuple(zt.shape)}")


def check_env(env_id: str, seed: int) -> None:
    """Run a few random environment interactions."""
    env = make_env(env_id, seed)
    obs, info = env.reset(seed=seed)
    print(f"env_id={env_id}")
    print(f"obs_shape={getattr(obs, 'shape', None)} obs_dtype={getattr(obs, 'dtype', None)}")
    print(f"action_space={env.action_space}")
    print(f"reset_info_keys={sorted(info.keys())}")

    total_reward = 0.0
    for _ in range(10):
        obs, reward, terminated, truncated, _info = env.step(env.action_space.sample())
        total_reward += float(reward)
        if terminated or truncated:
            obs, _info = env.reset()
    print(f"random_rollout_10_step_reward={total_reward:.3f}")
    env.close()


def run_ppo_smoke(env_id: str, seed: int, timesteps: int, run_dir: Path, verbose: int) -> None:
    """Train a tiny PPO model to verify SB3 end-to-end behavior."""
    vec_env = DummyVecEnv([lambda: make_env(env_id, seed)])
    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=seed,
        verbose=verbose,
        n_steps=64,
        batch_size=64,
        n_epochs=2,
        tensorboard_log=str(run_dir / "tensorboard"),
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    checkpoint_path = run_dir / "phase1_ppo_smoke.zip"
    model.save(checkpoint_path)
    print(f"saved_checkpoint={checkpoint_path}")
    vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 FlowReg stack smoke checks.")
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=5_000)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    load_dotenv(override=False)
    os.environ.setdefault("WANDB_MODE", "offline")

    run_dir = Path("runs") / "phase1_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)

    check_imports_and_devices()
    check_env(args.env_id, args.seed)
    if not args.skip_train:
        run_ppo_smoke(args.env_id, args.seed, args.timesteps, run_dir, args.verbose)


if __name__ == "__main__":
    main()
