# Flowing Through States: Neural ODE Regularization for Reinforcement Learning

## Project Overview
This repository contains a reproducibility study of the paper **["Flowing Through States: Neural ODE Regularization for Reinforcement Learning"](https://openreview.net/forum?id=FHFDCsB9UC)**.

 The project is developed as the practical examination for the **Advanced Topics in Machine Learning** course, part of my Master’s Degree in Artificial Intelligence (AY 2025-2026).

## The Problem: Latent Misalignment
In deep RL, agents typically rely on latent representations of environment states to make decisions. While the environment's semantic dynamics dictate how states evolve, the corresponding transitions in the latent space are often left implicit and unconstrained. This can lead to a misalignment between the agent's internal representation and the actual structure of the Markov Decision Process (MDP).

## The Approach & Study Purpose
To tackle this misalignment, the authors introduce **Flow Regularization (FlowReg)**. This unsupervised technique explicitly models latent dynamics by training a Neural Ordinary Differential Equation (ODE) to act as a continuous surrogate for the environment. By applying an alignment penalty, the agent's latent embeddings are forced to mimic the smooth flows of the Neural ODE, inheriting its topological consistency.

The purpose of this project is to build and replicate the FlowReg framework from scratch. We will integrate this technique on top of established algorithms (like A2C for Atari and PPO for Gridworld) to critically validate the performance gains and latent smoothness properties reported in the original study.

For a deep dive into the mathematical proofs and complete theoretical framework, please refer to the original paper: [link](https://openreview.net/forum?id=FHFDCsB9UC).

## Utils

All commands below assume the current working directory is the repository root:

```bash
cd /Users/matteovitali/Desktop/USI/ATML/FlowReg-ATMLproject-USI
```

Install dependencies:

```bash
uv sync --extra dev
```

Run checks:

```bash
uv run ruff check src scripts tests
uv run pytest -q
```

### Training

Run the current FourRooms baseline for one seed:

```bash
uv run flowreg-train-baseline \
  --config configs/baseline_ppo_fourrooms.yaml \
  --seed 0 \
  --wandb online
```

Run the current paper-oriented FourRooms FlowReg setup for one seed:

```bash
uv run flowreg-train-flowreg \
  --config configs/flowreg_ppo_fourrooms.yaml \
  --seed 0 \
  --wandb online
```

Run a quick local smoke without uploading to W&B:

```bash
uv run flowreg-train-flowreg \
  --config configs/flowreg_ppo_fourrooms_smoke.yaml \
  --timesteps 20000 \
  --seed 0 \
  --wandb disabled
```

Run FourRooms seeds 0, 1, and 2:

```bash
for seed in 0 1 2; do
  uv run flowreg-train-baseline \
    --config configs/baseline_ppo_fourrooms.yaml \
    --seed $seed \
    --wandb online

  uv run flowreg-train-flowreg \
    --config configs/flowreg_ppo_fourrooms.yaml \
    --seed $seed \
    --wandb online
done
```

### Atari A2C On CUDA

The Atari configs are intended for Lightning AI or another CUDA machine. They
use SB3 A2C with Nature CNN features, RMSProp, initial learning rate `7e-4`,
linear learning-rate decay, and gradient clipping `0.5`.

Check CUDA before launching runs:

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY
```

Run the first diagnostic FlowReg smoke with dimension-mean MSE:

```bash
uv run flowreg-train-flowreg-a2c \
  --config configs/flowreg_a2c_atari_stability.yaml \
  --env-id ALE/Breakout-v5 \
  --seed 0 \
  --timesteps 100000 \
  --wandb online
```

Run the paper-scaled Breakout smoke:

```bash
uv run flowreg-train-baseline-a2c \
  --config configs/baseline_a2c_atari.yaml \
  --env-id ALE/Breakout-v5 \
  --seed 0 \
  --timesteps 100000 \
  --wandb online

uv run flowreg-train-flowreg-a2c \
  --config configs/flowreg_a2c_atari.yaml \
  --env-id ALE/Breakout-v5 \
  --seed 0 \
  --timesteps 100000 \
  --wandb online
```

Generate the compact final Atari matrix commands:

```bash
bash scripts/run_atari_full.sh --dry-run
```

Launch the compact matrix:

```bash
bash scripts/run_atari_full.sh
```

The final Atari matrix is Breakout and Qbert, seeds `0` and `1`, baseline plus
FlowReg, for `10M` timesteps per run. After smoke runs, check W&B for finite
`Loss/*`, `Latent/*`, and `GradNorm/*`; `FlowReg/TotalUpdates` should be close
to `timesteps / (n_envs * n_steps * update_freq)`.

### Print Metrics

The `run_id` is the run folder name under `runs/baseline_ppo/` or
`runs/flowreg_ppo/`. List recent runs with:

```bash
ls -td runs/baseline_ppo/* | head
ls -td runs/flowreg_ppo/* | head
```

Print metrics for one baseline run:

```bash
uv run flowreg-analyze-runs \
  --run baseline:0:runs/baseline_ppo/baseline_ppo_fourrooms_debug_seed0_20260516_102946
```

Print metrics for one baseline vs FlowReg comparison:

```bash
uv run flowreg-analyze-runs \
  --run baseline:0:runs/baseline_ppo/baseline_ppo_fourrooms_debug_seed0_20260516_102946 \
  --run flowreg:0:runs/flowreg_ppo/flowreg_ppo_fourrooms_seed0_20260516_103219
```

### Save Metrics To Markdown

Save one run summary:

```bash
uv run flowreg-analyze-runs \
  --run baseline:0:runs/baseline_ppo/baseline_ppo_fourrooms_debug_seed0_20260516_102946 \
  --markdown-output runs/phase4_validation/baseline_seed0_summary.md
```

Save a baseline vs FlowReg comparison as Markdown and JSON:

```bash
uv run flowreg-analyze-runs \
  --run baseline:0:runs/baseline_ppo/baseline_ppo_fourrooms_debug_seed0_20260516_102946 \
  --run flowreg:0:runs/flowreg_ppo/flowreg_ppo_fourrooms_seed0_20260516_103219 \
  --markdown-output runs/phase4_validation/fourrooms_seed0_comparison.md \
  --json-output runs/phase4_validation/fourrooms_seed0_comparison.json
```

Save a 3-seed FourRooms comparison:

```bash
uv run flowreg-analyze-runs \
  --run baseline:0:runs/baseline_ppo/baseline_seed0_run_id \
  --run baseline:1:runs/baseline_ppo/baseline_seed1_run_id \
  --run baseline:2:runs/baseline_ppo/baseline_seed2_run_id \
  --run flowreg:0:runs/flowreg_ppo/flowreg_seed0_run_id \
  --run flowreg:1:runs/flowreg_ppo/flowreg_seed1_run_id \
  --run flowreg:2:runs/flowreg_ppo/flowreg_seed2_run_id \
  --markdown-output runs/phase4_validation/fourrooms_3seed_comparison.md \
  --json-output runs/phase4_validation/fourrooms_3seed_comparison.json
```

Do not include angle brackets around run ids in shell commands. In `zsh`, `<...>`
is interpreted as file redirection.
