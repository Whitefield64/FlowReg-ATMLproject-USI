#!/usr/bin/env bash
set -euo pipefail

WANDB_MODE=${WANDB_MODE:-online}
DRY_RUN=false
TIMESTEPS=3000000
ENV_ID="ALE/Breakout-v5"
SEEDS=(0 1 2)
BASELINE_CONFIG="configs/atari/baseline_a2c_atari.yaml"
FLOWREG_CONFIG="configs/atari/flowreg_a2c_atari.yaml"

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
  echo "Breakout seed=${seed}: baseline A2C @ ${TIMESTEPS} steps"
  run_cmd uv run flowreg-train-baseline-a2c \
    --config "$BASELINE_CONFIG" \
    --env-id "$ENV_ID" \
    --timesteps "$TIMESTEPS" \
    --seed "$seed" \
    --wandb "$WANDB_MODE"

  echo "Breakout seed=${seed}: A2C + FlowReg @ ${TIMESTEPS} steps"
  run_cmd uv run flowreg-train-flowreg-a2c \
    --config "$FLOWREG_CONFIG" \
    --env-id "$ENV_ID" \
    --timesteps "$TIMESTEPS" \
    --seed "$seed" \
    --wandb "$WANDB_MODE"
done
