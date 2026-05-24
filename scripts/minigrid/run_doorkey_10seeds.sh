#!/usr/bin/env bash
set -euo pipefail

WANDB_MODE=${WANDB_MODE:-online}
DRY_RUN=false
SEEDS=(0 1 2 3 4 5 6 7 8 9)
BASELINE_CONFIG="configs/minigrid/baseline_ppo_doorkey.yaml"
FLOWREG_CONFIG="configs/minigrid/flowreg_ppo_doorkey.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wandb)
      WANDB_MODE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

run_cmd() {
  if $DRY_RUN; then
    printf '  '
    printf '%q ' "$@"
    echo
  else
    "$@"
  fi
}

for seed in "${SEEDS[@]}"; do
  echo "DoorKey seed=${seed}: baseline PPO"
  run_cmd uv run flowreg-train-baseline \
    --config "$BASELINE_CONFIG" \
    --seed "$seed" \
    --wandb "$WANDB_MODE"

  echo "DoorKey seed=${seed}: PPO + FlowReg"
  run_cmd uv run flowreg-train-flowreg \
    --config "$FLOWREG_CONFIG" \
    --seed "$seed" \
    --wandb "$WANDB_MODE"
done
