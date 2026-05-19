#!/usr/bin/env bash
set -e

# Short smoke test for Breakout and Qbert (100k timesteps each)
# Run from repository root

TIMESTEPS=100000

for env_id in ALE/Breakout-v5 ALE/Qbert-v5; do
    echo "Running Baseline A2C on $env_id"
    uv run flowreg-train-baseline-a2c \
        --config configs/baseline_a2c_atari.yaml \
        --timesteps $TIMESTEPS \
        --env-id $env_id \
        --seed 0 \
        --wandb online

    echo "Running FlowReg A2C on $env_id"
    uv run flowreg-train-flowreg-a2c \
        --config configs/flowreg_a2c_atari.yaml \
        --timesteps $TIMESTEPS \
        --env-id $env_id \
        --seed 0 \
        --wandb online
done

echo "Smoke test completed successfully."
