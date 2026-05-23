from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from flowreg.eval_baseline import evaluate_checkpoint
from flowreg.envs import make_dummy_vec_env
from flowreg.flow import (
    FlowODE,
    action_conditioned_ode_path,
    compute_flow_loss,
    episode_ids_from_starts,
    exponential_time_grid,
    flow_loss_values,
    policy_latents,
    sample_flow_trajectories,
    valid_trajectory_starts,
)
from flowreg.flowreg_ppo import FlowRegPPO
from flowreg.policies import build_policy_kwargs


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
    actions = np.arange(6 * 2, dtype=np.int64).reshape(6, 2)
    episode_starts = np.zeros((6, 2), dtype=bool)
    episode_starts[2, 0] = True
    rng = np.random.default_rng(0)

    batch = sample_flow_trajectories(
        observations=observations,
        episode_starts=episode_starts,
        sequence_length=3,
        batch_size=4,
        rng=rng,
        actions=actions,
    )

    assert batch is not None
    assert batch.observations.shape == (4, 3, 3)
    assert batch.actions is not None
    assert batch.actions.shape == (4, 2)
    assert batch.episode_ids.shape == (4, 3)
    for step_idx, env_idx in batch.starts:
        assert not episode_starts[step_idx + 1 : step_idx + 3, env_idx].any()
    for action_row, (step_idx, env_idx) in zip(batch.actions, batch.starts, strict=True):
        np.testing.assert_array_equal(action_row, actions[step_idx : step_idx + 2, env_idx])
    for episode_id_row in batch.episode_ids:
        assert np.all(episode_id_row == episode_id_row[0])


def test_episode_ids_from_starts_vectorized() -> None:
    episode_starts = np.zeros((6, 2), dtype=bool)
    episode_starts[0, :] = True
    episode_starts[2, 0] = True
    episode_starts[4, 1] = True

    episode_ids = episode_ids_from_starts(episode_starts)

    np.testing.assert_array_equal(episode_ids[:, 0], np.array([1, 1, 2, 2, 2, 2]))
    np.testing.assert_array_equal(episode_ids[:, 1], np.array([1, 1, 1, 1, 2, 2]))


def test_flow_loss_values_match_paper_scaled_formula() -> None:
    latent_path = th.tensor([[[1.0, 2.0], [3.0, 5.0]]])
    ode_path = th.tensor([[[0.0, 1.0], [1.0, 1.0]]])

    paper_scaled, mse_mean = flow_loss_values(latent_path, ode_path)

    expected_paper_scaled = ((latent_path - ode_path).pow(2).sum(dim=-1)).mean()
    expected_mse_mean = (latent_path - ode_path).pow(2).mean()
    assert th.allclose(paper_scaled, expected_paper_scaled)
    assert th.allclose(mse_mean, expected_mse_mean)
    assert paper_scaled.item() == 11.0
    assert mse_mean.item() == 5.5


def test_exponential_time_grid_uses_one_minus_tau() -> None:
    time_grid = exponential_time_grid(sequence_length=4, gamma=0.99, device=th.device("cpu"))

    expected = th.tensor([0.0, 0.01, 0.0199, 0.029701])
    assert th.allclose(time_grid, expected)


def test_compute_flow_loss_backpropagates_to_feature_extractor_and_ode() -> None:
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
        policy_kwargs=build_policy_kwargs(
            {
                "feature_extractor": {
                    "name": "minigrid_mlp",
                    "features_dim": 32,
                    "hidden_dim": 64,
                }
            }
        ),
    )
    latent_dim = int(model.policy.features_extractor.features_dim)
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
        representation="features",
        loss_reduction="paper",
    )
    loss.backward()

    feature_extractor_grad = sum(
        param.grad.abs().sum().item()
        for param in model.policy.features_extractor.parameters()
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
    assert metrics["flow_loss"] == metrics["flow_loss_paper_scaled"]
    assert metrics["flow_loss_mse_mean"] >= 0
    assert feature_extractor_grad > 0
    assert ode_grad > 0


def test_action_conditioned_flow_loss_backpropagates_to_feature_extractor_and_ode() -> None:
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
        policy_kwargs=build_policy_kwargs(
            {
                "feature_extractor": {
                    "name": "minigrid_mlp",
                    "features_dim": 32,
                    "hidden_dim": 64,
                }
            }
        ),
    )
    latent_dim = int(model.policy.features_extractor.features_dim)
    action_dim = int(vec_env.envs[0].action_space.n)
    flow_model = FlowODE(latent_dim=latent_dim, hidden_dim=16, action_dim=action_dim)

    obs = []
    actions = []
    env = vec_env.envs[0]
    current_obs, _info = env.reset(seed=0)
    for _ in range(5):
        action = env.action_space.sample()
        obs.append(current_obs)
        current_obs, _reward, terminated, truncated, _info = env.step(action)
        actions.append(action)
        if terminated or truncated:
            current_obs, _info = env.reset()
    trajectory_obs = np.asarray(obs, dtype=np.float32)[None, ...]
    trajectory_actions = np.asarray(actions[:-1], dtype=np.int64)[None, ...]

    loss, metrics = compute_flow_loss(
        policy=model.policy,
        flow_model=flow_model,
        trajectory_obs=trajectory_obs,
        trajectory_actions=trajectory_actions,
        action_dim=action_dim,
        action_conditioned=True,
        device=model.device,
        rtol=1e-4,
        atol=1e-5,
        representation="features",
        loss_reduction="paper",
    )
    loss.backward()

    feature_extractor_grad = sum(
        param.grad.abs().sum().item()
        for param in model.policy.features_extractor.parameters()
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
    assert feature_extractor_grad > 0
    assert ode_grad > 0


def test_action_conditioned_ode_path_requires_transition_actions() -> None:
    flow_model = FlowODE(latent_dim=2, hidden_dim=4, action_dim=3)
    z0 = th.zeros((2, 2))
    actions = th.zeros((2, 2), dtype=th.long)
    time_grid = th.tensor([0.0, 1.0, 2.0])

    ode_path = action_conditioned_ode_path(
        flow_model=flow_model,
        z0=z0,
        actions=actions,
        action_dim=3,
        time_grid=time_grid,
        rtol=1e-4,
        atol=1e-5,
    )

    assert ode_path.shape == (2, 3, 2)


def test_flow_representations_have_expected_dimensions() -> None:
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
        policy_kwargs=build_policy_kwargs(
            {
                "feature_extractor": {
                    "name": "minigrid_mlp",
                    "features_dim": 32,
                    "hidden_dim": 64,
                }
            }
        ),
    )
    obs, _info = vec_env.envs[0].reset(seed=0)
    obs_tensor = obs_as_tensor(obs[None, ...], model.device)

    features = policy_latents(model.policy, obs_tensor, representation="features")
    actor_latent = policy_latents(model.policy, obs_tensor, representation="actor_latent")
    vec_env.close()

    assert features.shape == (1, 32)
    assert actor_latent.shape == (1, int(model.policy.mlp_extractor.latent_dim_pi))


def test_evaluation_does_not_call_ode(tmp_path, monkeypatch) -> None:
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
        policy_kwargs=build_policy_kwargs(
            {
                "feature_extractor": {
                    "name": "minigrid_mlp",
                    "features_dim": 32,
                    "hidden_dim": 64,
                }
            }
        ),
    )
    model_path = tmp_path / "model.zip"
    model.save(model_path)
    vec_env.close()

    def fail_ode(*_args, **_kwargs):
        raise AssertionError("ODE integration must not be used during evaluation")

    monkeypatch.setattr("flowreg.flow.odeint", fail_ode)
    result = evaluate_checkpoint(
        model_path=model_path,
        env_id="MiniGrid-DoorKey-5x5-v0",
        seed=0,
        n_eval_episodes=1,
        deterministic=True,
        wrapper="img_flatten",
    )

    assert result["n_eval_episodes"] == 1


def test_train_call_update_unit_applies_flow_once_per_due_train_call() -> None:
    vec_env = make_dummy_vec_env(
        env_id="MiniGrid-DoorKey-5x5-v0",
        seed=0,
        n_envs=1,
        wrapper="img_flatten",
    )
    model = FlowRegPPO(
        "MlpPolicy",
        vec_env,
        seed=0,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        flow_sequence_length=3,
        flow_batch_size=1,
        flow_update_freq=2,
        flow_hidden_dim=16,
        flow_update_unit="train_call",
        policy_kwargs=build_policy_kwargs(
            {
                "feature_extractor": {
                    "name": "minigrid_mlp",
                    "features_dim": 16,
                    "hidden_dim": 32,
                }
            }
        ),
    )
    env = vec_env.envs[0]
    obs, _info = env.reset(seed=0)
    observations = []
    for _ in range(4):
        observations.append(obs)
        obs, _reward, terminated, truncated, _info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _info = env.reset()
    model._flow_observations = np.asarray(observations, dtype=np.float32)[:, None, :]
    model._flow_episode_starts = np.zeros((4, 1), dtype=bool)
    model._flow_episode_starts[0, 0] = True

    model._begin_flow_train_call()
    flow_loss, _metrics = model._maybe_compute_flow_loss()
    assert flow_loss is None

    model._begin_flow_train_call()
    flow_loss, metrics = model._maybe_compute_flow_loss()
    second_flow_loss, _second_metrics = model._maybe_compute_flow_loss()
    vec_env.close()

    assert flow_loss is not None
    assert th.isfinite(flow_loss)
    assert metrics["flow_loss"] >= 0
    assert second_flow_loss is None
