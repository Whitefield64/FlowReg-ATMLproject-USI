#!/usr/bin/env bash
set -euo pipefail

uv run flowreg-analyze-runs \
  --run baseline:0:runs/baseline_ppo/baseline_ppo_fourrooms_debug_seed0_20260515_122954 \
  --run baseline:1:runs/baseline_ppo/baseline_ppo_fourrooms_debug_seed1_20260515_123032 \
  --run baseline:2:runs/baseline_ppo/baseline_ppo_fourrooms_debug_seed2_20260515_123032 \
  --run flowreg:0:runs/flowreg_ppo/flowreg_ppo_fourrooms_smoke_seed0_20260515_124334 \
  --run flowreg:1:runs/flowreg_ppo/flowreg_ppo_fourrooms_smoke_seed1_20260515_124917 \
  --run flowreg:2:runs/flowreg_ppo/flowreg_ppo_fourrooms_smoke_seed2_20260515_124938 \
  --json-output runs/phase4_validation/fourrooms_100k_summary.json \
  --markdown-output runs/phase4_validation/fourrooms_100k_summary.md
