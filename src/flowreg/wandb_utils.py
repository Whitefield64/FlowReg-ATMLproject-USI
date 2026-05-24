"""W&B helpers shared by local training entrypoints."""

from __future__ import annotations

from collections import deque

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback


def define_wandb_step_metrics() -> None:
    """Prefer environment timesteps as W&B x-axis for project metrics."""
    wandb.define_metric("global_step")
    for pattern in (
        "rollout/*",
        "train/*",
        "Loss/*",
        "Latent/*",
        "FlowReg/*",
        "GradNorm/*",
        "AtariPaper/*",
        "time/*",
    ):
        wandb.define_metric(pattern, step_metric="global_step")


class AtariPaperMetricsCallback(BaseCallback):
    """Track outer AtariWrapper returns, matching clipped life-loss episodes."""

    def __init__(self, window_size: int = 100) -> None:
        super().__init__()
        self.window_size = window_size
        self.episode_rewards: deque[float] = deque(maxlen=window_size)
        self.episode_lengths: deque[int] = deque(maxlen=window_size)
        self._current_rewards: np.ndarray | None = None
        self._current_lengths: np.ndarray | None = None

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self._current_rewards = np.zeros(n_envs, dtype=np.float64)
        self._current_lengths = np.zeros(n_envs, dtype=np.int64)

    def _on_step(self) -> bool:
        assert self._current_rewards is not None
        assert self._current_lengths is not None
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
        dones = np.asarray(self.locals["dones"], dtype=bool)
        self._current_rewards += rewards
        self._current_lengths += 1
        for env_idx, done in enumerate(dones):
            if not done:
                continue
            self.episode_rewards.append(float(self._current_rewards[env_idx]))
            self.episode_lengths.append(int(self._current_lengths[env_idx]))
            self._current_rewards[env_idx] = 0.0
            self._current_lengths[env_idx] = 0
        return True

    def _on_rollout_end(self) -> None:
        if not self.episode_rewards:
            return
        self.logger.record("AtariPaper/EpRewMean", float(np.mean(self.episode_rewards)))
        self.logger.record("AtariPaper/EpLenMean", float(np.mean(self.episode_lengths)))
        self.logger.record("AtariPaper/WindowEpisodes", len(self.episode_rewards))


class WandbGlobalStepCallback(BaseCallback):
    """Log the SB3 environment timestep as an explicit W&B step metric."""

    def _on_rollout_end(self) -> None:
        wandb.log({"global_step": self.num_timesteps})

    def _on_step(self) -> bool:
        return True
