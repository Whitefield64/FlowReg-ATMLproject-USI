"""FlowReg trajectory sampling, ODE model, and loss helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.utils import obs_as_tensor
from torchdiffeq import odeint


@dataclass(frozen=True)
class FlowTrajectoryBatch:
    """Contiguous observation sequences sampled from an on-policy rollout."""

    observations: np.ndarray
    actions: np.ndarray | None
    episode_ids: np.ndarray
    starts: tuple[tuple[int, int], ...]


class FlowODE(nn.Module):
    """Small ODE network dz/dt = f_phi(z), optionally conditioned on actions."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64, action_dim: int = 0) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, t: th.Tensor, z: th.Tensor, action_onehot: th.Tensor | None = None) -> th.Tensor:
        del t
        if self.action_dim == 0:
            return self.net(z)
        if action_onehot is None:
            raise ValueError("action_onehot is required for action-conditioned FlowODE")
        return self.net(th.cat([z, action_onehot], dim=-1))


def episode_ids_from_starts(episode_starts: np.ndarray) -> np.ndarray:
    """Derive per-env episode ids from SB3 rollout `episode_starts` flags."""
    if episode_starts.ndim != 2:
        raise ValueError("episode_starts must have shape (n_steps, n_envs)")
    return np.cumsum(episode_starts.astype(np.int64), axis=0)


def valid_trajectory_starts(episode_starts: np.ndarray, sequence_length: int) -> list[tuple[int, int]]:
    """Return `(time, env)` starts whose full window has one episode id."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if episode_starts.ndim != 2:
        raise ValueError("episode_starts must have shape (n_steps, n_envs)")

    episode_ids = episode_ids_from_starts(episode_starts)
    n_steps, n_envs = episode_starts.shape
    starts: list[tuple[int, int]] = []
    for env_idx in range(n_envs):
        for step_idx in range(0, n_steps - sequence_length + 1):
            window_ids = episode_ids[step_idx : step_idx + sequence_length, env_idx]
            if np.all(window_ids == window_ids[0]):
                starts.append((step_idx, env_idx))
    return starts


def sample_flow_trajectories(
    observations: np.ndarray,
    episode_starts: np.ndarray,
    sequence_length: int,
    batch_size: int,
    rng: np.random.Generator,
    actions: np.ndarray | None = None,
) -> FlowTrajectoryBatch | None:
    """Sample contiguous trajectories from SB3 rollout arrays."""
    episode_ids = episode_ids_from_starts(episode_starts)
    starts = valid_trajectory_starts(episode_starts, sequence_length)
    if not starts:
        return None
    sample_size = min(batch_size, len(starts))
    indices = rng.choice(len(starts), size=sample_size, replace=False)
    selected = tuple(starts[int(index)] for index in indices)
    trajectories = np.stack(
        [
            observations[step_idx : step_idx + sequence_length, env_idx]
            for step_idx, env_idx in selected
        ],
        axis=0,
    )
    trajectory_actions = None
    if actions is not None:
        trajectory_actions = np.stack(
            [
                actions[step_idx : step_idx + sequence_length - 1, env_idx]
                for step_idx, env_idx in selected
            ],
            axis=0,
        )
    trajectory_episode_ids = np.stack(
        [
            episode_ids[step_idx : step_idx + sequence_length, env_idx]
            for step_idx, env_idx in selected
        ],
        axis=0,
    )
    return FlowTrajectoryBatch(
        observations=trajectories,
        actions=trajectory_actions,
        episode_ids=trajectory_episode_ids,
        starts=selected,
    )


def index_time_grid(sequence_length: int, device: th.device) -> th.Tensor:
    """Create paper-default index time grid `[0, 1, ..., N - 1]`."""
    return th.arange(sequence_length, device=device, dtype=th.float32)


def exponential_time_grid(sequence_length: int, gamma: float, device: th.device) -> th.Tensor:
    """Create shifted exponential time grid `1 - tau_i`, with `tau_i = gamma^i`."""
    indices = th.arange(sequence_length, device=device, dtype=th.float32)
    return 1.0 - th.pow(gamma, indices)


def feature_latents(policy: nn.Module, observations: th.Tensor) -> th.Tensor:
    """Return learned state features used as paper-faithful MiniGrid `H_theta`."""
    features = policy.extract_features(observations)
    if isinstance(features, tuple):
        features = features[0]
    return features


def actor_latents(policy: nn.Module, observations: th.Tensor) -> th.Tensor:
    """Return legacy actor latent vectors for FlowReg ablations."""
    features = policy.extract_features(observations)
    if policy.share_features_extractor:
        latent_pi, _latent_vf = policy.mlp_extractor(features)
    else:
        pi_features, _vf_features = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
    return latent_pi


def policy_latents(policy: nn.Module, observations: th.Tensor, representation: str) -> th.Tensor:
    """Return the configured latent representation for FlowReg."""
    if representation == "features":
        return feature_latents(policy, observations)
    if representation == "actor_latent":
        return actor_latents(policy, observations)
    raise ValueError("representation must be one of: features, actor_latent")


def flow_loss_values(latent_path: th.Tensor, ode_path: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """Return paper-scaled and PyTorch-MSE FlowReg losses."""
    squared_error = (latent_path - ode_path).pow(2)
    paper_scaled = squared_error.sum(dim=-1).mean()
    mse_mean = F.mse_loss(latent_path, ode_path)
    return paper_scaled, mse_mean


def action_conditioned_ode_path(
    flow_model: nn.Module,
    z0: th.Tensor,
    actions: th.Tensor,
    action_dim: int,
    time_grid: th.Tensor,
    rtol: float,
    atol: float,
) -> th.Tensor:
    """Integrate an action-conditioned ODE one transition at a time."""
    if actions.ndim > 2:
        actions = actions.squeeze(-1)
    actions = actions.long()
    if actions.shape[:2] != (z0.shape[0], len(time_grid) - 1):
        raise ValueError("actions must have shape (batch_size, sequence_length - 1)")

    predicted = [z0]
    current_z = z0
    for step_idx in range(actions.shape[1]):
        action_onehot = F.one_hot(actions[:, step_idx], num_classes=action_dim).float()
        delta_t = time_grid[step_idx + 1] - time_grid[step_idx]
        step_time_grid = th.stack(
            [
                th.zeros((), device=time_grid.device, dtype=th.float32),
                delta_t.to(dtype=th.float32),
            ]
        )

        def fixed_action_flow(t: th.Tensor, z: th.Tensor) -> th.Tensor:
            return flow_model(t, z, action_onehot)

        current_z = odeint(
            fixed_action_flow,
            current_z,
            step_time_grid,
            rtol=rtol,
            atol=atol,
            options={"dtype": th.float32},
        )[-1]
        predicted.append(current_z)

    return th.stack(predicted, dim=1)


def compute_flow_loss(
    policy: nn.Module,
    flow_model: nn.Module,
    trajectory_obs: np.ndarray,
    device: th.device,
    rtol: float,
    atol: float,
    time_sampling: str = "index",
    gamma: float = 0.99,
    representation: str = "features",
    loss_reduction: str = "paper",
    trajectory_actions: np.ndarray | None = None,
    action_dim: int | None = None,
    action_conditioned: bool = False,
) -> tuple[th.Tensor, dict[str, float]]:
    batch_size, sequence_length = trajectory_obs.shape[:2]

    flat_obs = trajectory_obs.reshape(
        batch_size * sequence_length,
        *trajectory_obs.shape[2:]
    )
    obs_tensor = obs_as_tensor(flat_obs, device)

    latent_flat = policy_latents(
        policy,
        obs_tensor,
        representation=representation,
    )
    latent_path = latent_flat.reshape(batch_size, sequence_length, -1)

    if time_sampling == "index":
        time_grid = index_time_grid(sequence_length, device)
    elif time_sampling in {"exponential", "exp_decay"}:
        time_grid = exponential_time_grid(sequence_length, gamma, device)
    else:
        raise ValueError(
            "time_sampling must be one of: index, exponential, exp_decay"
        )

    z0 = latent_path[:, 0, :]

    if action_conditioned:
        if trajectory_actions is None or action_dim is None:
            raise ValueError("trajectory_actions and action_dim are required when action_conditioned=True")
        actions_tensor = th.as_tensor(trajectory_actions, device=device)
        ode_path = action_conditioned_ode_path(
            flow_model=flow_model,
            z0=z0,
            actions=actions_tensor,
            action_dim=action_dim,
            time_grid=time_grid,
            rtol=rtol,
            atol=atol,
        )
    else:
        ode_path = odeint(
            flow_model,
            z0,
            time_grid,
            rtol=rtol,
            atol=atol,
            options={"dtype": th.float32},
        )
        ode_path = ode_path.transpose(0, 1)

    paper_scaled_loss, mse_mean_loss = flow_loss_values(latent_path, ode_path)

    if loss_reduction == "paper":
        loss = paper_scaled_loss
    elif loss_reduction == "mse_mean":
        loss = mse_mean_loss
    else:
        raise ValueError("loss_reduction must be one of: paper, mse_mean")

    with th.no_grad():
        latent_metric = latent_path.detach()

        diffs = latent_metric[:, 1:, :] - latent_metric[:, :-1, :]
        path_length = th.linalg.norm(diffs, dim=-1).mean()

        net_displacement = (
            th.linalg.norm(
                latent_metric[:, -1, :] - latent_metric[:, 0, :],
                dim=-1,
            )
            / max(sequence_length - 1, 1)
        ).mean()

        if sequence_length >= 3:
            acceleration = (
                latent_metric[:, 2:, :]
                - 2 * latent_metric[:, 1:-1, :]
                + latent_metric[:, :-2, :]
            )
            acceleration_energy = th.linalg.norm(acceleration, dim=-1).mean()
        else:
            acceleration_energy = th.zeros((), device=device)

        ode_error_drift = th.linalg.norm(
            ode_path.detach()[:, -1, :] - latent_metric[:, -1, :],
            dim=-1,
        ).mean()

    metrics = {
        "flow_loss": float(loss.detach().cpu()),
        "flow_loss_paper_scaled": float(paper_scaled_loss.detach().cpu()),
        "flow_loss_mse_mean": float(mse_mean_loss.detach().cpu()),
        "path_length": float(path_length.cpu()),
        "net_displacement": float(net_displacement.cpu()),
        "acceleration_energy": float(acceleration_energy.cpu()),
        "ode_error_drift": float(ode_error_drift.cpu()),
    }

    return loss, metrics
