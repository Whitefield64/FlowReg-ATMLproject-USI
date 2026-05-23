"""A2C with a minimal FlowReg training-time regularizer."""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.utils import explained_variance

from flowreg.flow import FlowODE, compute_flow_loss, sample_flow_trajectories


def _grad_norm(parameters) -> float:
    """Return total L2 gradient norm for parameters with gradients."""
    norms = [
        param.grad.detach().norm(2)
        for param in parameters
        if param.grad is not None
    ]
    if not norms:
        return 0.0
    return float(th.linalg.vector_norm(th.stack(norms), ord=2).detach().cpu())


class FlowRegA2C(A2C):
    """Stable-Baselines3 A2C plus Neural ODE latent path regularization."""

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
        flow_time_sampling: str = "index",
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
        if flow_update_unit not in {"optimizer_step", "train_call"}:
            raise ValueError("flow_update_unit must be one of: optimizer_step, train_call")
        
        self.flow_model = th.compile(FlowODE(latent_dim=latent_dim, hidden_dim=flow_hidden_dim).to(self.device))
        self.flow_optimizer = th.optim.RMSprop(self.flow_model.parameters(), lr=flow_learning_rate)
        self._flow_initial_lr = flow_learning_rate
        
        self.lambda_flow = lambda_flow
        self.flow_sequence_length = flow_sequence_length
        self.flow_batch_size = flow_batch_size
        self.flow_update_freq = flow_update_freq
        self.flow_rtol = flow_rtol
        self.flow_atol = flow_atol
        self.flow_time_sampling = flow_time_sampling
        self.flow_representation = flow_representation
        self.flow_loss_reduction = flow_loss_reduction
        self.flow_update_unit = flow_update_unit
        
        self.flow_step = 0
        self.flow_total_updates = 0
        self.flow_total_skipped = 0
        self._flow_train_call_active = False
        self._flow_train_call_consumed = False
        self.flow_rng = np.random.default_rng(self.seed)
        self._flow_observations: np.ndarray | None = None
        self._flow_episode_starts: np.ndarray | None = None
        self._flow_latest_metrics: dict[str, float] = {}

    def _begin_flow_train_call(self) -> None:
        """Prepare FlowReg frequency state for one train call."""
        self._flow_train_call_active = False
        self._flow_train_call_consumed = False
        if self.flow_update_unit != "train_call":
            return
        self.flow_step += 1
        self._flow_train_call_active = (
            self.flow_update_freq > 0 and self.flow_step % self.flow_update_freq == 0
        )

    def _maybe_compute_flow_loss(self) -> tuple[th.Tensor | None, dict[str, float]]:
        if self.flow_update_unit == "optimizer_step":
            self.flow_step += 1
            if self.flow_update_freq <= 0 or self.flow_step % self.flow_update_freq != 0:
                return None, {}
        else:
            if not self._flow_train_call_active or self._flow_train_call_consumed:
                return None, {}
            self._flow_train_call_consumed = True

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
            time_sampling=self.flow_time_sampling,
            gamma=self.gamma,
            representation=self.flow_representation,
            loss_reduction=self.flow_loss_reduction,
        )

    def _record_cached_flow_metrics(
        self,
        *,
        flow_applied: int,
        flow_losses: list[float],
        flow_losses_paper_scaled: list[float],
        flow_losses_mse_mean: list[float],
        path_lengths: list[float],
        net_displacements: list[float],
        acceleration_energies: list[float],
        ode_error_drifts: list[float],
    ) -> None:
        self.logger.record("FlowReg/Applied", flow_applied)
        self.logger.record("FlowReg/Updates", len(flow_losses))
        if flow_losses:
            self._flow_latest_metrics = {
                "Loss/FlowReg": float(np.mean(flow_losses)),
                "Loss/FlowReg_PaperScaled": float(np.mean(flow_losses_paper_scaled)),
                "Loss/FlowReg_MSEMean": float(np.mean(flow_losses_mse_mean)),
                "Latent/Path_Length": float(np.mean(path_lengths)),
                "Latent/Net_Displacement": float(np.mean(net_displacements)),
                "Latent/Acceleration_Energy": float(np.mean(acceleration_energies)),
                "Latent/ODE_Error_Drift": float(np.mean(ode_error_drifts)),
            }
        for key, value in self._flow_latest_metrics.items():
            self.logger.record(key, value)
        self.logger.record("FlowReg/TotalUpdates", self.flow_total_updates)

    def train(self) -> None:
        """Update A2C and occasionally add FlowReg to the same backward pass."""
        self.policy.set_training_mode(True)
        self.flow_model.train()
        
        self._flow_observations = self.rollout_buffer.observations.copy()
        self._flow_episode_starts = self.rollout_buffer.episode_starts.copy()
        self._begin_flow_train_call()
        self._update_learning_rate(self.policy.optimizer)
        # Apply the same linear LR decay to the flow optimizer
        progress = self._current_progress_remaining
        new_flow_lr = self._flow_initial_lr * progress
        for param_group in self.flow_optimizer.param_groups:
            param_group["lr"] = new_flow_lr

        entropy_losses = []
        pg_losses, value_losses = [], []
        rl_losses = []
        total_losses = []
        policy_grad_norms = []
        flow_grad_norms = []
        flow_losses = []
        flow_losses_paper_scaled = []
        flow_losses_mse_mean = []
        path_lengths = []
        net_displacements = []
        acceleration_energies = []
        ode_error_drifts = []
        flow_applied = 0
        flow_skipped = 0

        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = rollout_data.actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()
            
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(advantages * log_prob).mean()
            pg_losses.append(policy_loss.item())

            value_loss = F.mse_loss(rollout_data.returns, values)
            value_losses.append(value_loss.item())

            if entropy is None:
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)
            entropy_losses.append(entropy_loss.item())

            rl_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            loss = rl_loss
            rl_losses.append(rl_loss.item())
            
            flow_loss, flow_metrics = self._maybe_compute_flow_loss()
            if flow_loss is not None:
                loss = loss + self.lambda_flow * flow_loss
                flow_applied += 1
                self.flow_total_updates += 1
                flow_losses.append(flow_metrics["flow_loss"])
                flow_losses_paper_scaled.append(flow_metrics["flow_loss_paper_scaled"])
                flow_losses_mse_mean.append(flow_metrics["flow_loss_mse_mean"])
                path_lengths.append(flow_metrics["path_length"])
                net_displacements.append(flow_metrics["net_displacement"])
                acceleration_energies.append(flow_metrics["acceleration_energy"])
                ode_error_drifts.append(flow_metrics["ode_error_drift"])
            else:
                flow_skipped += 1
            total_losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            self.flow_optimizer.zero_grad()
                
            loss.backward()

            policy_grad_norms.append(_grad_norm(self.policy.parameters()))
            flow_grad_norms.append(_grad_norm(self.flow_model.parameters()))
            if flow_loss is not None:
                th.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.flow_model.parameters()),
                    self.max_grad_norm,
                )
            else:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            if flow_loss is not None:
                self.flow_optimizer.step()

        self._n_updates += 1
        self.flow_total_skipped += flow_skipped

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/flow_learning_rate", new_flow_lr)
        self.logger.record("Loss/RL_Total", float(np.mean(rl_losses)))
        self.logger.record("Loss/Total", float(np.mean(total_losses)))
        self.logger.record("GradNorm/Policy", float(np.mean(policy_grad_norms)))
        self.logger.record("FlowReg/Skipped", self.flow_total_skipped)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

        if flow_losses:
            self.logger.record("GradNorm/FlowODE", float(np.mean(flow_grad_norms)))
        self._record_cached_flow_metrics(
            flow_applied=flow_applied,
            flow_losses=flow_losses,
            flow_losses_paper_scaled=flow_losses_paper_scaled,
            flow_losses_mse_mean=flow_losses_mse_mean,
            path_lengths=path_lengths,
            net_displacements=net_displacements,
            acceleration_energies=acceleration_energies,
            ode_error_drifts=ode_error_drifts,
        )
