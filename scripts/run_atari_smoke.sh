#!/usr/bin/env bash
set -e

# Short Atari smoke test for Breakout and Qbert.
#
# Usage:
#   bash scripts/run_atari_smoke.sh
#   bash scripts/run_atari_smoke.sh --timesteps 500000
#   bash scripts/run_atari_smoke.sh --dry-run
#
# Environment overrides:
#   WANDB_MODE=disabled bash scripts/run_atari_smoke.sh

TIMESTEPS=100000
WANDB_MODE=${WANDB_MODE:-online}
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
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

for env_id in ALE/Breakout-v5 ALE/Qbert-v5; do
    echo "Running Baseline A2C on $env_id"
    baseline_cmd=(
        uv run flowreg-train-baseline-a2c
        --config configs/baseline_a2c_atari.yaml
        --timesteps "$TIMESTEPS"
        --env-id "$env_id"
        --seed 0
        --wandb "$WANDB_MODE"
    )
    if $DRY_RUN; then
        printf '  %q' "${baseline_cmd[@]}"
        echo
    else
        "${baseline_cmd[@]}"
    fi

    echo "Running FlowReg A2C on $env_id"
    flowreg_cmd=(
        uv run flowreg-train-flowreg-a2c
        --config configs/flowreg_a2c_atari.yaml
        --timesteps "$TIMESTEPS"
        --env-id "$env_id"
        --seed 0
        --wandb "$WANDB_MODE"
    )
    if $DRY_RUN; then
        printf '  %q' "${flowreg_cmd[@]}"
        echo
    else
        "${flowreg_cmd[@]}"
    fi
done

echo "Smoke test completed successfully."
