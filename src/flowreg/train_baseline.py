"""Train the conservative MiniGrid PPO baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
import wandb
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback

from flowreg.config import load_yaml_config
from flowreg.envs import make_dummy_vec_env
from flowreg.policies import build_policy_kwargs
from flowreg.train_utils import safe_wandb_mode, timestamp, write_config_snapshot
from flowreg.wandb_utils import WandbGlobalStepCallback, define_wandb_step_metrics


def train_baseline(config: dict[str, Any], wandb_mode: str) -> Path:
    """Train PPO from a config dictionary and return the checkpoint path."""
    seed = int(config.get("seed", 0))
    env_id = str(config["env_id"])
    run_name = str(config.get("run_name", f"baseline_ppo_{env_id}"))
    run_id = f"{run_name}_seed{seed}_{timestamp()}"
    run_dir = Path("runs") / "baseline_ppo" / run_id
    monitor_dir = run_dir / "monitor"
    model_dir = run_dir / "models"
    tensorboard_dir = run_dir / "tensorboard"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    write_config_snapshot(config, run_dir)

    vec_env = make_dummy_vec_env(
        env_id=env_id,
        seed=seed,
        n_envs=int(config.get("n_envs", 1)),
        monitor_dir=monitor_dir,
        wrapper=config.get("wrapper", "img_flatten"),
    )

    wandb_run = None
    callback = None
    if wandb_mode != "disabled":
        os.environ["WANDB_MODE"] = wandb_mode
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "flowreg-atml"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=run_id,
            config=config,
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=False,
        )
        define_wandb_step_metrics()
        callback = CallbackList(
            [
                WandbCallback(
                    model_save_path=str(model_dir / "wandb"),
                    model_save_freq=max(int(config.get("total_timesteps", 0)) // 2, 1),
                    gradient_save_freq=0,
                    verbose=0,
                ),
                WandbGlobalStepCallback(),
            ]
        )

    ppo_config = dict(config.get("ppo", {}))
    policy_kwargs = build_policy_kwargs(config)
    model = PPO(
        config.get("policy", "MlpPolicy"),
        vec_env,
        seed=seed,
        device=config.get("device", "cpu"),
        verbose=int(config.get("verbose", 0)),
        tensorboard_log=str(tensorboard_dir),
        policy_kwargs=policy_kwargs,
        **ppo_config,
    )
    model.learn(
        total_timesteps=int(config["total_timesteps"]),
        callback=callback,
        progress_bar=False,
    )

    checkpoint_path = model_dir / "final_model.zip"
    model.save(checkpoint_path)
    vec_env.close()
    if wandb_run is not None:
        wandb_run.finish()

    print(f"run_dir={run_dir}")
    print(f"checkpoint={checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline PPO MiniGrid agent.")
    parser.add_argument("--config", default="configs/baseline_ppo_fourrooms.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--wandb", choices=["disabled", "offline", "online"], default=None)
    args = parser.parse_args()

    load_dotenv(override=True)
    config = load_yaml_config(args.config)
    if args.timesteps is not None:
        config["total_timesteps"] = args.timesteps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.env_id is not None:
        config["env_id"] = args.env_id

    wandb_mode = safe_wandb_mode(str(config.get("wandb_mode", "disabled")), args.wandb)
    train_baseline(config, wandb_mode)


if __name__ == "__main__":
    main()
