#!/usr/bin/env bash
# Generate a side-by-side demo MP4: Baseline PPO vs FlowReg PPO on FourRooms.
# Edit the paths below to switch seeds or use a different run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# seed1 from 20260524 is a functional baseline (navigates but isn't always lucky)
# Baseline: debug_seed0_20260515 — navigates properly, fails exactly ep5
BASELINE="${REPO_ROOT}/runs/baseline_ppo/baseline_ppo_fourrooms_debug_seed0_20260515_160715/models/final_model.zip"
# FlowReg:  seed2_20260524 — wins all 5 with higher reward than baseline
FLOWREG="${REPO_ROOT}/runs/flowreg_ppo/flowreg_ppo_fourrooms_seed2_20260524_115202/models/final_model.zip"

OUTPUT="${REPO_ROOT}/demo_fourrooms.mp4"

python -m flowreg.demo \
  --baseline "${BASELINE}" \
  --flowreg  "${FLOWREG}" \
  --env-id   MiniGrid-FourRooms-v0 \
  --episodes 5 \
  --fps      8 \
  --max-steps 300 \
  --seed     17 \
  --output   "${OUTPUT}"

echo ""
echo "Video: ${OUTPUT}"
