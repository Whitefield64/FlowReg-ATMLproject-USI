"""Evaluate trained SB3 checkpoints for MiniGrid PPO or Atari A2C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from flowreg.config import load_yaml_config
from flowreg.envs import make_atari_environment, make_minigrid_env


def evaluate_checkpoint(
    model_path: str | Path,
    env_id: str,
    seed: int,
    n_eval_episodes: int,
    deterministic: bool,
    wrapper: str,
    device: str = "cpu",
    env_kwargs: dict | None = None,
    wrapper_kwargs: dict | None = None,
) -> dict[str, float | int | str | bool]:
    """Evaluate a checkpoint and return report-friendly metrics."""
    if env_id.startswith("ALE/"):
        env = make_atari_environment(
            env_id=env_id,
            seed=seed,
            n_envs=1,
            env_kwargs=env_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
        model = A2C.load(model_path, env=env, device=device)
        algorithm = "A2C"
    else:
        env = make_minigrid_env(env_id=env_id, seed=seed, wrapper=wrapper)
        model = PPO.load(model_path, env=env, device=device)
        algorithm = "PPO"
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        return_episode_rewards=False,
        warn=False,
    )
    env.close()
    return {
        "algorithm": algorithm,
        "model_path": str(model_path),
        "env_id": env_id,
        "seed": seed,
        "n_eval_episodes": n_eval_episodes,
        "deterministic": deterministic,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MiniGrid PPO or Atari A2C checkpoint.")
    parser.add_argument("--config", default="configs/baseline_ppo_fourrooms.yaml")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    eval_config = config.get("eval", {})
    result = evaluate_checkpoint(
        model_path=args.model_path,
        env_id=config["env_id"],
        seed=args.seed if args.seed is not None else int(config.get("seed", 0)),
        n_eval_episodes=(
            args.episodes
            if args.episodes is not None
            else int(eval_config.get("n_eval_episodes", 10))
        ),
        deterministic=bool(eval_config.get("deterministic", True)),
        wrapper=config.get("wrapper", "img_flatten"),
        device=args.device,
        env_kwargs=dict(config.get("atari_env_kwargs", {}) or {}),
        wrapper_kwargs=dict(config.get("atari_wrapper_kwargs", {}) or {}),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
