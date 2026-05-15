"""Evaluate a trained SB3 PPO baseline checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from flowreg.config import load_yaml_config
from flowreg.envs import make_minigrid_env


def evaluate_checkpoint(
    model_path: str | Path,
    env_id: str,
    seed: int,
    n_eval_episodes: int,
    deterministic: bool,
    wrapper: str,
) -> dict[str, float | int | str | bool]:
    """Evaluate a checkpoint and return report-friendly metrics."""
    env = make_minigrid_env(env_id=env_id, seed=seed, wrapper=wrapper)
    model = PPO.load(model_path, env=env, device="cpu")
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
        "model_path": str(model_path),
        "env_id": env_id,
        "seed": seed,
        "n_eval_episodes": n_eval_episodes,
        "deterministic": deterministic,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a baseline PPO checkpoint.")
    parser.add_argument("--config", default="configs/baseline_ppo_fourrooms.yaml")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
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
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

