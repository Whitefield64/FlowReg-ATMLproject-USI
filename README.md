# FlowReg — Neural ODE Regularization for RL
Reproducibility study of **["Flowing Through States: Neural ODE Regularization for Reinforcement Learning"](https://openreview.net/forum?id=FHFDCsB9UC)**, developed for the Advanced Topics in Machine Learning course (MSc AI, USI — AY 2025-2026).
## Setup

```bash
uv sync --extra dev
cp .env.example .env          # fill in your WANDB_API_KEY
```
## Running Experiments
### MiniGrid (PPO, CPU)
**Single seed:**
```bash
# Baseline
uv run flowreg-train-baseline \
  --config configs/minigrid/baseline_ppo_fourrooms.yaml \
  --seed 0 --wandb online

# FlowReg
uv run flowreg-train-flowreg \
  --config configs/minigrid/flowreg_ppo_fourrooms.yaml \
  --seed 0 --wandb online
```
**Full 10-seed matrices** (append `--dry-run` to preview):
```bash
bash scripts/minigrid/run_fourrooms_10seeds.sh
bash scripts/minigrid/run_dynamic_obstacles_10seeds.sh
bash scripts/minigrid/run_doorkey_10seeds.sh
bash scripts/minigrid/run_all_minigrid_10seeds.sh    # all three
```

### Atari (A2C, CUDA)
**Full 3-seed matrices** (3M timesteps each; append `--dry-run` to preview):
```bash
bash scripts/atari/run_breakout_3seeds_3m.sh
bash scripts/atari/run_qbert_3seeds_3m.sh
bash scripts/atari/run_all_atari_3seeds_3m.sh 
```

## Analysis
List completed runs:
```bash
ls -td runs/baseline_ppo/* | head
ls -td runs/flowreg_ppo/* | head
```
Print a comparison to the terminal:
```bash
uv run flowreg-analyze-runs \
  --run baseline:0:<baseline_run_dir> \
  --run flowreg:0:<flowreg_run_dir>
```

Export to Markdown and JSON:

```bash
uv run flowreg-analyze-runs \  
--run baseline:0:<baseline_run_dir> \
  --run flowreg:0:<flowreg_run_dir> \
  --markdown-output results.md \
  --json-output results.json
```
> Replace `<baseline_run_dir>` and `<flowreg_run_dir>` with the actual paths under `runs/`.

## Tests
```bash
uv run ruff check src scripts tests
uv run pytest -q
```