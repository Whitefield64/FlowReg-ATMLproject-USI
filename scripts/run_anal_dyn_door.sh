#!/bin/bash

# ==========================================
# PART 1: TRAINING PHASE
# ==========================================
echo "Starting training phase (40 runs total)..."

for seed in {0..9}; do
  echo "=========================================="
  echo " STARTING SEED $seed"
  echo "=========================================="

  # 1. Dynamic Obstacles (Baseline)
  flowreg-train-baseline \
    --config configs/baseline_ppo_dynamic_obstacles.yaml \
    --seed $seed \
    --wandb online

  # 2. Dynamic Obstacles (FlowReg)
  flowreg-train-flowreg \
    --config configs/flowreg_ppo_dynamic_obstacles.yaml \
    --seed $seed \
    --wandb online

  # 3. DoorKey (Baseline)
  flowreg-train-baseline \
    --config configs/baseline_ppo_doorkey.yaml \
    --seed $seed \
    --wandb online

  # 4. DoorKey (FlowReg)
  flowreg-train-flowreg \
    --config configs/flowreg_ppo_doorkey.yaml \
    --seed $seed \
    --wandb online
done

echo "All training runs complete! Moving to analysis..."

# ==========================================
# PART 2: ANALYSIS PHASE
# ==========================================
RUN_ARGS=""

# Gather Dynamic Obstacles - Baseline
for s in {0..9}; do
  MATCH=(runs/baseline_ppo/*dynamic_obstacles*seed${s}_*)
  if [ -d "${MATCH[0]}" ]; then 
    RUN_ARGS="$RUN_ARGS --run baseline_dynobs:$s:${MATCH[0]}"
  else
    echo "  -> Warning: Missing baseline_dynobs seed $s"
  fi
done

# Gather Dynamic Obstacles - FlowReg
for s in {0..9}; do
  MATCH=(runs/flowreg_ppo/*dynamic_obstacles*seed${s}_*)
  if [ -d "${MATCH[0]}" ]; then 
    RUN_ARGS="$RUN_ARGS --run flowreg_dynobs:$s:${MATCH[0]}"
  else
    echo "  -> Warning: Missing flowreg_dynobs seed $s"
  fi
done

# Gather DoorKey - Baseline
for s in {0..9}; do
  MATCH=(runs/baseline_ppo/*doorkey*seed${s}_*)
  if [ -d "${MATCH[0]}" ]; then 
    RUN_ARGS="$RUN_ARGS --run baseline_doorkey:$s:${MATCH[0]}"
  else
    echo "  -> Warning: Missing baseline_doorkey seed $s"
  fi
done

# Gather DoorKey - FlowReg
for s in {0..9}; do
  MATCH=(runs/flowreg_ppo/*doorkey*seed${s}_*)
  if [ -d "${MATCH[0]}" ]; then 
    RUN_ARGS="$RUN_ARGS --run flowreg_doorkey:$s:${MATCH[0]}"
  else
    echo "  -> Warning: Missing flowreg_doorkey seed $s"
  fi
done

echo "Running final analysis..."
# Execute the final command! Groups will be neatly separated in the Markdown file.
flowreg-analyze-runs $RUN_ARGS --markdown-output new_envs_summary.md

echo "Done! The weekend is yours. Check new_envs_summary.md on Monday."