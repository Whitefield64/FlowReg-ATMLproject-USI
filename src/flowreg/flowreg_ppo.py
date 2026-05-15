"""PPO with a minimal FlowReg training-time regularizer."""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance

from flowreg.flow import FlowODE, compute_flow_loss, sample_flow_trajectories


class FlowRegPPO(PPO):
    """Stable-Baselines3 PPO plus Neural ODE latent path regularization."""

    def __init__(
        self,
        *args,
        lambda_flow: float = 1.0,
        flow_sequence_length: int = 5,
        flow_batch_size: int = 32,
        flow_update_freq: int = 20,
        flow_learning_rate: float = 3e-4,
        flow_hidden_dim: int = 64,
        flow_rtol: float = 1e-4,
        flow_atol: float = 1e-5,
        flow_representation: str = "features",
        flow_loss_reduction: str = "paper",
        flow_update_unit: str = "optimizer_step",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if flow_representation == "features":
            latent_dim = int(self.policy.features_extractor.features_dim)
        elif flow_representation == "actor_latent":
            latent_dim = int(self.policy.mlp_extractor.latent_dim_pi)
        else:
            raise ValueError("flow_representation must be one of: features, actor_latent")
        if flow_loss_reduction not in {"paper", "mse_mean"}:
            raise ValueError("flow_loss_reduction must be one of: paper, mse_mean")
        if flow_update_unit != "optimizer_step":
            raise ValueError("flow_update_unit currently supports only: optimizer_step")
        self.flow_model = FlowODE(latent_dim=latent_dim, hidden_dim=flow_hidden_dim).to(self.device)
        self.flow_optimizer = th.optim.Adam(self.flow_model.parameters(), lr=flow_learning_rate)
        self.lambda_flow = lambda_flow
        self.flow_sequence_length = flow_sequence_length
        self.flow_batch_size = flow_batch_size
        self.flow_update_freq = flow_update_freq
        self.flow_rtol = flow_rtol
        self.flow_atol = flow_atol
        self.flow_representation = flow_representation
        self.flow_loss_reduction = flow_loss_reduction
        self.flow_update_unit = flow_update_unit
        self.flow_step = 0
        self.flow_rng = np.random.default_rng(self.seed)
        self._flow_observations: np.ndarray | None = None
        self._flow_episode_starts: np.ndarray | None = None

    def _maybe_compute_flow_loss(self) -> tuple[th.Tensor | None, dict[str, float]]:
        self.flow_step += 1
        if self.flow_update_freq <= 0 or self.flow_step % self.flow_update_freq != 0:
            return None, {}
        if self._flow_observations is None or self._flow_episode_starts is None:
            return None, {}

        batch = sample_flow_trajectories(
            observations=self._flow_observations,
            episode_starts=self._flow_episode_starts,
            sequence_length=self.flow_sequence_length,
            batch_size=self.flow_batch_size,
            rng=self.flow_rng,
        )
        if batch is None:
            return None, {}
        return compute_flow_loss(
            policy=self.policy,
            flow_model=self.flow_model,
            trajectory_obs=batch.observations,
            device=self.device,
            rtol=self.flow_rtol,
            atol=self.flow_atol,
            representation=self.flow_representation,
            loss_reduction=self.flow_loss_reduction,
        )

    def train(self) -> None:
        """Update PPO and occasionally add FlowReg to the same backward pass."""
        self.policy.set_training_mode(True)
        self.flow_model.train()
        # SB3 flattens rollout observations in-place when `rollout_buffer.get()`
        # is first called. FlowReg needs the original time/env axes, so keep a
        # private snapshot for contiguous trajectory sampling.
        self._flow_observations = self.rollout_buffer.observations.copy()
        self._flow_episode_starts = self.rollout_buffer.episode_starts.copy()
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        flow_losses = []
        flow_losses_paper_scaled = []
        flow_losses_mse_mean = []
        path_lengths = []
        net_displacements = []
        acceleration_energies = []
        ode_error_drifts = []

        continue_training = True
        loss = th.zeros((), device=self.device)
        approx_kl_divs = []
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                flow_loss, flow_metrics = self._maybe_compute_flow_loss()
                if flow_loss is not None:
                    loss = loss + self.lambda_flow * flow_loss
                    flow_losses.append(flow_metrics["flow_loss"])
                    flow_losses_paper_scaled.append(flow_metrics["flow_loss_paper_scaled"])
                    flow_losses_mse_mean.append(flow_metrics["flow_loss_mse_mean"])
                    path_lengths.append(flow_metrics["path_length"])
                    net_displacements.append(flow_metrics["net_displacement"])
                    acceleration_energies.append(flow_metrics["acceleration_energy"])
                    ode_error_drifts.append(flow_metrics["ode_error_drift"])

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                if flow_loss is not None:
                    self.flow_optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                if flow_loss is not None:
                    self.flow_optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        if flow_losses:
            self.logger.record("Loss/FlowReg", float(np.mean(flow_losses)))
            self.logger.record("Loss/FlowReg_PaperScaled", float(np.mean(flow_losses_paper_scaled)))
            self.logger.record("Loss/FlowReg_MSEMean", float(np.mean(flow_losses_mse_mean)))
            self.logger.record("Latent/Path_Length", float(np.mean(path_lengths)))
            self.logger.record("Latent/Net_Displacement", float(np.mean(net_displacements)))
            self.logger.record("Latent/Acceleration_Energy", float(np.mean(acceleration_energies)))
            self.logger.record("Latent/ODE_Error_Drift", float(np.mean(ode_error_drifts)))
            self.logger.record("FlowReg/Updates", len(flow_losses))
        else:
            self.logger.record("FlowReg/Updates", 0)
