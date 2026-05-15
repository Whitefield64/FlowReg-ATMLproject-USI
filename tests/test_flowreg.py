from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import PPO

from flowreg.envs import make_dummy_vec_env
from flowreg.flow import (
    FlowODE,
    compute_flow_loss,
    sample_flow_trajectories,
    valid_trajectory_starts,
)


def test_valid_trajectory_starts_skip_episode_boundaries() -> None:
    episode_starts = np.zeros((6, 1), dtype=bool)
    episode_starts[3, 0] = True

    starts = valid_trajectory_starts(episode_starts, sequence_length=3)

    assert (0, 0) in starts
    assert (3, 0) in starts
    assert (1, 0) not in starts
    assert (2, 0) not in starts


def test_sample_flow_trajectories_shape_and_no_crossing() -> None:
    observations = np.arange(6 * 2 * 3, dtype=np.float32).reshape(6, 2, 3)
    episode_starts = np.zeros((6, 2), dtype=bool)
    episode_starts[2, 0] = True
    rng = np.random.default_rng(0)

    batch = sample_flow_trajectories(
        observations=observations,
        episode_starts=episode_starts,
        sequence_length=3,
        batch_size=4,
        rng=rng,
    )

    assert batch is not None
    assert batch.observations.shape == (4, 3, 3)
    for step_idx, env_idx in batch.starts:
        assert not episode_starts[step_idx + 1 : step_idx + 3, env_idx].any()


def test_compute_flow_loss_backpropagates_to_policy_and_ode() -> None:
    vec_env = make_dummy_vec_env(
        env_id="MiniGrid-DoorKey-5x5-v0",
        seed=0,
        n_envs=1,
        wrapper="img_flatten",
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=0,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
    )
    latent_dim = int(model.policy.mlp_extractor.latent_dim_pi)
    flow_model = FlowODE(latent_dim=latent_dim, hidden_dim=16)

    obs = []
    env = vec_env.envs[0]
    current_obs, _info = env.reset(seed=0)
    for _ in range(5):
        obs.append(current_obs)
        current_obs, _reward, terminated, truncated, _info = env.step(env.action_space.sample())
        if terminated or truncated:
            current_obs, _info = env.reset()
    trajectory_obs = np.asarray(obs, dtype=np.float32)[None, ...]

    loss, metrics = compute_flow_loss(
        policy=model.policy,
        flow_model=flow_model,
        trajectory_obs=trajectory_obs,
        device=model.device,
        rtol=1e-4,
        atol=1e-5,
    )
    loss.backward()

    policy_grad = sum(
        param.grad.abs().sum().item()
        for param in model.policy.parameters()
        if param.grad is not None
    )
    ode_grad = sum(
        param.grad.abs().sum().item()
        for param in flow_model.parameters()
        if param.grad is not None
    )
    vec_env.close()

    assert th.isfinite(loss)
    assert metrics["flow_loss"] >= 0
    assert policy_grad > 0
    assert ode_grad > 0

