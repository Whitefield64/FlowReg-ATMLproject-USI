# Experiment Scripts

These scripts launch the project-facing experiment matrices. Use direct
`uv run ...` commands for one-off debugging, smoke tests not listed here,
evaluation, and ad-hoc analysis.

## MiniGrid

- `scripts/minigrid/run_fourrooms_10seeds.sh`: FourRooms, PPO baseline and PPO+FlowReg,
  seeds 0-9, 1M timesteps from config.
- `scripts/minigrid/run_dynamic_obstacles_10seeds.sh`: DynamicObstacles, PPO baseline
  and PPO+FlowReg, seeds 0-9, 1M timesteps from config.
- `scripts/minigrid/run_doorkey_10seeds.sh`: DoorKey, PPO baseline and PPO+FlowReg,
  seeds 0-9, 1M timesteps from config.
- `scripts/minigrid/run_all_minigrid_10seeds.sh`: runs all three MiniGrid matrices.

## Atari

- `scripts/atari/run_breakout_3seeds_3m.sh`: Breakout, A2C baseline and A2C+FlowReg,
  seeds 0-2, 3M timesteps each.
- `scripts/atari/run_qbert_3seeds_3m.sh`: Qbert, A2C baseline and A2C+FlowReg,
  seeds 0-2, 3M timesteps each.
- `scripts/atari/run_all_atari_3seeds_3m.sh`: runs both Atari matrices.

## Atari Tests

- `scripts/atari/tests/run_breakout_1m.sh`: Breakout, baseline and FlowReg, seed 0,
  1M timesteps each.
- `scripts/atari/tests/run_qbert_1m.sh`: Qbert, baseline and FlowReg, seed 0,
  1M timesteps each.

Atari test scripts use the official Atari configs with shorter 1M runs and
default to W&B online logging.

All scripts accept:

```bash
--dry-run
--wandb online|offline|disabled
```
