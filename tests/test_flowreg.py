from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from flowreg.config import load_yaml_config
from flowreg.eval_baseline import evaluate_checkpoint
from flowreg.envs import make_atari_environment, make_dummy_vec_env
from flowreg.flow import (
    FlowODE,
    compute_flow_loss,
    episode_ids_from_starts,
    flow_loss_values,
    policy_latents,
    sample_flow_trajectories,
    valid_trajectory_starts,
)
from flowreg.flowreg_a2c import FlowRegA2C
from flowreg.flowreg_ppo import FlowRegPPO
from flowreg.policies import build_policy_kwargs
from flowreg.train_utils import prepare_a2c_config


class DummyLogger:
    def __init__(self) -> None:
        self.records: dict[str, float | int] = {}

    def record(self, key: str, value, *args, **kwargs) -> None:
        self.records[key] = value


def test_load_yaml_config_merges_relative_base_config(tmp_path: Path) -> None:
    base_path = tmp_path / "baseline.yaml"
    child_dir = tmp_path / "flowreg"
    child_dir.mkdir()
    child_path = child_dir / "config.yaml"

    base_path.write_text(
        """
run_name: baseline
env_id: MiniGrid-FourRooms-v0
total_timesteps: 1000000
ppo:
  learning_rate: 0.0003
  n_epochs: 4
eval:
  deterministic: true
""",
        encoding="utf-8",
    )
    child_path.write_text(
        """
base_config: ../baseline.yaml
run_name: flowreg
ppo:
  n_epochs: 8
flowreg:
  enabled: true
  lambda_flow: 1.0
""",
        encoding="utf-8",
    )

    config = load_yaml_config(child_path)

    assert config["run_name"] == "flowreg"
    assert config["env_id"] == "MiniGrid-FourRooms-v0"
    assert config["total_timesteps"] == 1000000
    assert config["ppo"] == {"learning_rate": 0.0003, "n_epochs": 8}
    assert config["eval"] == {"deterministic": True}
    assert config["flowreg"] == {"enabled": True, "lambda_flow": 1.0}
    assert "base_config" not in config


def test_load_yaml_config_rejects_base_config_cycles(tmp_path: Path) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_text("base_config: second.yaml\nrun_name: first\n", encoding="utf-8")
    second_path.write_text("base_config: first.yaml\nrun_name: second\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Config inheritance cycle detected"):
        load_yaml_config(first_path)


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
    assert batch.episode_ids.shape == (4, 3)
    for step_idx, env_idx in batch.starts:
        assert not episode_starts[step_idx + 1 : step_idx + 3, env_idx].any()
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


def test_prepare_a2c_config_linear_learning_rate_schedule() -> None:
    config = {
        "learning_rate_schedule": "linear",
        "a2c": {
            "learning_rate": 0.0007,
            "n_steps": 5,
        },
    }

    a2c_config = prepare_a2c_config(config)

    assert "learning_rate_schedule" not in a2c_config
    assert a2c_config["n_steps"] == 5
    assert a2c_config["learning_rate"](1.0) == 0.0007
    assert a2c_config["learning_rate"](0.5) == 0.00035
    assert a2c_config["learning_rate"](0.0) == 0.0


def test_env_module_import_does_not_eagerly_load_pygame_or_cv2() -> None:
    env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    pythonpath = str(src_dir)
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import flowreg.envs; "
                "print('pygame' in sys.modules); "
                "print('cv2' in sys.modules)"
            ),
        ],
        capture_output=True,
        check=True,
        env=env,
        text=True,
    )

    assert result.stdout.splitlines() == ["False", "False"]


def test_make_atari_environment_forwards_env_and_wrapper_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}
    raw_env = object()

    def fake_make_atari_env(**kwargs):
        captured.update(kwargs)
        return raw_env

    def fake_vec_frame_stack(env, n_stack):
        captured["stacked_env"] = env
        captured["n_stack"] = n_stack
        return ("stacked", env, n_stack)

    def fake_atari_components():
        return fake_make_atari_env, object(), fake_vec_frame_stack

    monkeypatch.setattr("flowreg.envs._atari_components", fake_atari_components)

    result = make_atari_environment(
        env_id="ALE/Breakout-v5",
        seed=7,
        n_envs=4,
        monitor_dir="monitor",
        env_kwargs={"frameskip": 1, "repeat_action_probability": 0.0},
        wrapper_kwargs={"frame_skip": 4, "action_repeat_probability": 0.0},
    )

    assert result == ("stacked", raw_env, 4)
    assert captured["env_id"] == "ALE/Breakout-v5"
    assert captured["n_envs"] == 4
    assert captured["seed"] == 7
    assert captured["env_kwargs"] == {"frameskip": 1, "repeat_action_probability": 0.0}
    assert captured["wrapper_kwargs"] == {"frame_skip": 4, "action_repeat_probability": 0.0}
    assert captured["n_stack"] == 4


def test_atari_evaluation_uses_training_env_kwargs(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class DummyEnv:
        def close(self) -> None:
            captured["closed"] = True

    def fake_make_atari_environment(**kwargs):
        captured.update(kwargs)
        return DummyEnv()

    def fake_load(model_path, env, device):
        captured["load"] = (model_path, env, device)
        return object()

    def fake_evaluate_policy(model, env, n_eval_episodes, deterministic, return_episode_rewards, warn):
        captured["eval"] = (model, env, n_eval_episodes, deterministic, return_episode_rewards, warn)
        return 12.0, 0.5

    monkeypatch.setattr("flowreg.eval_baseline.make_atari_environment", fake_make_atari_environment)
    monkeypatch.setattr("flowreg.eval_baseline.A2C.load", fake_load)
    monkeypatch.setattr("flowreg.eval_baseline.evaluate_policy", fake_evaluate_policy)

    result = evaluate_checkpoint(
        model_path=tmp_path / "model.zip",
        env_id="ALE/Qbert-v5",
        seed=11,
        n_eval_episodes=20,
        deterministic=False,
        wrapper="img_flatten",
        device="cpu",
        env_kwargs={"frameskip": 1, "repeat_action_probability": 0.0},
        wrapper_kwargs={"frame_skip": 4, "action_repeat_probability": 0.0},
    )

    assert result["algorithm"] == "A2C"
    assert result["mean_reward"] == 12.0
    assert result["std_reward"] == 0.5
    assert captured["env_kwargs"] == {"frameskip": 1, "repeat_action_probability": 0.0}
    assert captured["wrapper_kwargs"] == {"frame_skip": 4, "action_repeat_probability": 0.0}
    assert captured["eval"][2:] == (20, False, False, False)
    assert captured["closed"] is True


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


def test_a2c_cached_flow_metrics_survive_empty_interval() -> None:
    model = object.__new__(FlowRegA2C)
    model._logger = DummyLogger()
    model.flow_total_updates = 3
    model._flow_latest_metrics = {}

    model._record_cached_flow_metrics(
        flow_applied=1,
        flow_losses=[8.0],
        flow_losses_paper_scaled=[8.0],
        flow_losses_mse_mean=[0.25],
        path_lengths=[1.5],
        net_displacements=[0.75],
        acceleration_energies=[0.5],
        ode_error_drifts=[0.1],
    )
    cached_metrics = dict(model._flow_latest_metrics)
    model.logger.records.clear()

    model._record_cached_flow_metrics(
        flow_applied=0,
        flow_losses=[],
        flow_losses_paper_scaled=[],
        flow_losses_mse_mean=[],
        path_lengths=[],
        net_displacements=[],
        acceleration_energies=[],
        ode_error_drifts=[],
    )

    assert model.logger.records["FlowReg/Applied"] == 0
    assert model.logger.records["FlowReg/Updates"] == 0
    assert model.logger.records["FlowReg/TotalUpdates"] == 3
    assert model.logger.records["Loss/FlowReg"] == cached_metrics["Loss/FlowReg"]
    assert model.logger.records["Latent/Path_Length"] == cached_metrics["Latent/Path_Length"]


def test_ppo_cached_flow_metrics_survive_empty_interval() -> None:
    model = object.__new__(FlowRegPPO)
    model._logger = DummyLogger()
    model.flow_total_updates = 4
    model._flow_latest_metrics = {}

    model._record_cached_flow_metrics(
        flow_applied=2,
        flow_losses=[6.0, 4.0],
        flow_losses_paper_scaled=[6.0, 4.0],
        flow_losses_mse_mean=[0.20, 0.10],
        path_lengths=[2.0, 1.0],
        net_displacements=[0.5, 0.25],
        acceleration_energies=[0.3, 0.2],
        ode_error_drifts=[0.1, 0.05],
    )
    cached_metrics = dict(model._flow_latest_metrics)
    model.logger.records.clear()

    model._record_cached_flow_metrics(
        flow_applied=0,
        flow_losses=[],
        flow_losses_paper_scaled=[],
        flow_losses_mse_mean=[],
        path_lengths=[],
        net_displacements=[],
        acceleration_energies=[],
        ode_error_drifts=[],
    )

    assert model.logger.records["FlowReg/Applied"] == 0
    assert model.logger.records["FlowReg/Updates"] == 0
    assert model.logger.records["FlowReg/TotalUpdates"] == 4
    assert model.logger.records["Loss/FlowReg"] == cached_metrics["Loss/FlowReg"]
    assert model.logger.records["Latent/ODE_Error_Drift"] == cached_metrics["Latent/ODE_Error_Drift"]


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
