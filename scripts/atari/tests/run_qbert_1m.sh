#!/usr/bin/env bash
set -euo pipefail

WANDB_MODE=${WANDB_MODE:-online}
DRY_RUN=false
TIMESTEPS=1000000
ENV_ID="ALE/Qbert-v5"
SEED=0
BASELINE_CONFIG="configs/atari/baseline_a2c_atari.yaml"
FLOWREG_CONFIG="configs/atari/tests/flowreg_a2c_atari_stability.yaml"

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

echo "Qbert test: baseline A2C @ ${TIMESTEPS} steps"
run_cmd uv run flowreg-train-baseline-a2c \
  --config "$BASELINE_CONFIG" \
  --env-id "$ENV_ID" \
  --timesteps "$TIMESTEPS" \
  --seed "$SEED" \
  --wandb "$WANDB_MODE"

echo "Qbert test: A2C + FlowReg @ ${TIMESTEPS} steps"
run_cmd uv run flowreg-train-flowreg-a2c \
  --config "$FLOWREG_CONFIG" \
  --env-id "$ENV_ID" \
  --timesteps "$TIMESTEPS" \
  --seed "$SEED" \
  --wandb "$WANDB_MODE"
