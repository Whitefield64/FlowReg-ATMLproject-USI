"""W&B helpers shared by local training entrypoints."""

from __future__ import annotations

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
        "time/*",
    ):
        wandb.define_metric(pattern, step_metric="global_step")


class WandbGlobalStepCallback(BaseCallback):
    """Log the SB3 environment timestep as an explicit W&B step metric."""

    def _on_rollout_end(self) -> None:
        wandb.log({"global_step": self.num_timesteps})

    def _on_step(self) -> bool:
        return True
