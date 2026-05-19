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
    episode_ids: np.ndarray
    starts: tuple[tuple[int, int], ...]


class FlowODE(nn.Module):
    """Small autonomous ODE network dz/dt = f_phi(z)."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, t: th.Tensor, z: th.Tensor) -> th.Tensor:
        del t
        return self.net(z)


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
    trajectory_episode_ids = np.stack(
        [
            episode_ids[step_idx : step_idx + sequence_length, env_idx]
            for step_idx, env_idx in selected
        ],
        axis=0,
    )
    return FlowTrajectoryBatch(
        observations=trajectories,
        episode_ids=trajectory_episode_ids,
        starts=selected,
    )


def index_time_grid(sequence_length: int, device: th.device) -> th.Tensor:
    """Create paper-default index time grid `[0, 1, ..., N - 1]`."""
    return th.arange(sequence_length, device=device, dtype=th.float32)


def exponential_time_grid(sequence_length: int, gamma: float, device: th.device) -> th.Tensor:
    """Create exponential decay time grid `tau_i = gamma^i`."""
    indices = th.arange(sequence_length, device=device, dtype=th.float32)
    return th.pow(gamma, indices)


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
) -> tuple[th.Tensor, dict[str, float]]:
    """Compute full-path FlowReg MSE and latent geometry metrics."""
    batch_size, sequence_length = trajectory_obs.shape[:2]
    flat_obs = trajectory_obs.reshape(batch_size * sequence_length, *trajectory_obs.shape[2:])
    obs_tensor = obs_as_tensor(flat_obs, device)
    latent_flat = policy_latents(policy, obs_tensor, representation=representation)
    latent_path = latent_flat.reshape(batch_size, sequence_length, -1)

    # Detach latent targets: the encoder (h_theta) should NOT be updated by
    # the flow loss.  Only the ODE model (f_phi) learns to match the encoder's
    # trajectory.  This prevents the regularizer from destabilising the RL
    # representations.
    latent_path_detached = latent_path.detach()

    z0 = latent_path_detached[:, 0, :]
    if time_sampling == "index":
        time_grid = index_time_grid(sequence_length, device)
    elif time_sampling == "exponential":
        time_grid = exponential_time_grid(sequence_length, gamma, device)
    else:
        raise ValueError("time_sampling must be one of: index, exponential")
        
    ode_path = odeint(flow_model, z0, time_grid, rtol=rtol, atol=atol)
    ode_path = ode_path.transpose(0, 1)

    paper_scaled_loss, mse_mean_loss = flow_loss_values(latent_path_detached, ode_path)
    if loss_reduction == "paper":
        loss = paper_scaled_loss
    elif loss_reduction == "mse_mean":
        loss = mse_mean_loss
    else:
        raise ValueError("loss_reduction must be one of: paper, mse_mean")

    with th.no_grad():
        diffs = latent_path_detached[:, 1:, :] - latent_path_detached[:, :-1, :]
        path_length = th.linalg.norm(diffs, dim=-1).mean()
        net_displacement = (
            th.linalg.norm(latent_path_detached[:, -1, :] - latent_path_detached[:, 0, :], dim=-1)
            / max(sequence_length - 1, 1)
        ).mean()
        if sequence_length >= 3:
            acceleration = latent_path_detached[:, 2:, :] - 2 * latent_path_detached[:, 1:-1, :] + latent_path_detached[:, :-2, :]
            acceleration_energy = th.linalg.norm(acceleration, dim=-1).mean()
        else:
            acceleration_energy = th.zeros((), device=device)
        ode_error_drift = th.linalg.norm(ode_path[:, -1, :] - latent_path_detached[:, -1, :], dim=-1).mean()
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
