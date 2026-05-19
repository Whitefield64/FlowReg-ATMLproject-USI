"""Train FlowReg A2C on Atari."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import stable_baselines3
import torch
import wandb
from dotenv import load_dotenv
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback

from flowreg.config import load_yaml_config
from flowreg.envs import make_atari_environment
from flowreg.flowreg_a2c import FlowRegA2C
from flowreg.policies import build_policy_kwargs
from flowreg.train_baseline import _safe_wandb_mode
from flowreg.wandb_utils import WandbGlobalStepCallback, define_wandb_step_metrics


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _short_env_name(env_id: str) -> str:
    """Extract a short readable name from an env id, e.g. 'ALE/Breakout-v5' -> 'Breakout'."""
    name = env_id.split("/")[-1]       # 'Breakout-v5'
    name = name.rsplit("-", 1)[0]      # 'Breakout'
    return name


def _write_config_snapshot(config: dict[str, Any], run_dir: Path) -> None:
    snapshot = dict(config)
    snapshot["versions"] = {
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)


def train_flowreg(config: dict[str, Any], wandb_mode: str) -> Path:
    """Train FlowReg A2C from a config dictionary and return checkpoint path."""
    seed = int(config.get("seed", 0))
    env_id = str(config["env_id"])
    env_short = _short_env_name(env_id)
    run_id = f"flowreg_a2c_{env_short}_s{seed}_{_timestamp()}"
    run_dir = Path("runs") / "flowreg_a2c" / run_id
    monitor_dir = run_dir / "monitor"
    model_dir = run_dir / "models"
    tensorboard_dir = run_dir / "tensorboard"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_config_snapshot(config, run_dir)

    vec_env = make_atari_environment(
        env_id=env_id,
        seed=seed,
        n_envs=int(config.get("n_envs", 1)),
        monitor_dir=monitor_dir,
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

    a2c_config = dict(config.get("a2c", {}))
    flow_config = dict(config.get("flowreg", {}))
    policy_kwargs = build_policy_kwargs(config)
    model = FlowRegA2C(
        config.get("policy", "CnnPolicy"),
        vec_env,
        seed=seed,
        device=config.get("device", "cuda"),
        verbose=int(config.get("verbose", 0)),
        tensorboard_log=str(tensorboard_dir),
        lambda_flow=float(flow_config.get("lambda_flow", 1.0)),
        flow_sequence_length=int(flow_config.get("sequence_length", 5)),
        flow_batch_size=int(flow_config.get("batch_size", 32)),
        flow_update_freq=int(flow_config.get("update_freq", 20)),
        flow_learning_rate=float(flow_config.get("learning_rate", 3e-4)),
        flow_hidden_dim=int(flow_config.get("hidden_dim", 64)),
        flow_rtol=float(flow_config.get("rtol", 1e-4)),
        flow_atol=float(flow_config.get("atol", 1e-5)),
        flow_time_sampling=str(flow_config.get("time_sampling", "index")),
        flow_representation=str(flow_config.get("representation", "features")),
        flow_loss_reduction=str(flow_config.get("loss_reduction", "paper")),
        flow_update_unit=str(flow_config.get("update_unit", "optimizer_step")),
        policy_kwargs=policy_kwargs,
        **a2c_config,
    )
    model.learn(
        total_timesteps=int(config["total_timesteps"]),
        callback=callback,
        progress_bar=False,
    )

    checkpoint_path = model_dir / "final_model.zip"
    model.save(checkpoint_path)
    torch.save(model.flow_model.state_dict(), model_dir / "flow_model.pt")
    vec_env.close()
    if wandb_run is not None:
        wandb_run.finish()

    print(f"run_dir={run_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"flow_model={model_dir / 'flow_model.pt'}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a FlowReg A2C Atari agent.")
    parser.add_argument("--config", default="configs/flowreg_a2c_atari.yaml")
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

    wandb_mode = _safe_wandb_mode(str(config.get("wandb_mode", "disabled")), args.wandb)
    train_flowreg(config, wandb_mode)


if __name__ == "__main__":
    main()
