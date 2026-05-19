#!/usr/bin/env bash
set -e

# ============================================================================
# Full Atari A2C Experiment — FlowReg Paper Reproduction
# ============================================================================
#
# Paper spec:
#   - 5 independent runs per agent (baseline + FlowReg) per environment
#   - 10 million timesteps each
#   - Seeds 0–4 (separate from hyperparameter search seeds)
#   - 11 Atari environments
#
# Usage:
#   bash scripts/run_atari_full.sh              # run everything
#   bash scripts/run_atari_full.sh --dry-run    # print commands without executing
#
# NOTE: This launches jobs SEQUENTIALLY. For parallel execution on a cluster,
# see the --dry-run output and submit each line as a separate job.
# ============================================================================

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — printing commands only ==="
    echo ""
fi

TIMESTEPS=10000000
SEEDS=(0 1 2 3 4)

# 11 standard Atari environments used in FlowReg evaluation.
# Adjust this list to match the exact set from the paper if different.
ENVS=(
    "ALE/Breakout-v5"
    "ALE/Qbert-v5"
)

TOTAL_RUNS=$(( ${#ENVS[@]} * ${#SEEDS[@]} * 2 ))
CURRENT=0

echo "============================================"
echo "  FlowReg Atari Full Experiment"
echo "  ${#ENVS[@]} environments × ${#SEEDS[@]} seeds × 2 agents"
echo "  = ${TOTAL_RUNS} total runs @ ${TIMESTEPS} timesteps each"
echo "============================================"
echo ""

for env_id in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # --- Baseline A2C ---
        CURRENT=$((CURRENT + 1))
        echo "[${CURRENT}/${TOTAL_RUNS}] Baseline A2C | ${env_id} | seed=${seed}"

        if $DRY_RUN; then
            echo "  uv run flowreg-train-baseline-a2c --config configs/baseline_a2c_atari.yaml --timesteps ${TIMESTEPS} --env-id ${env_id} --seed ${seed} --wandb online"
        else
            uv run flowreg-train-baseline-a2c \
                --config configs/baseline_a2c_atari.yaml \
                --timesteps $TIMESTEPS \
                --env-id "$env_id" \
                --seed $seed \
                --wandb online
        fi

        # --- FlowReg A2C ---
        CURRENT=$((CURRENT + 1))
        echo "[${CURRENT}/${TOTAL_RUNS}] FlowReg A2C  | ${env_id} | seed=${seed}"

        if $DRY_RUN; then
            echo "  uv run flowreg-train-flowreg-a2c --config configs/flowreg_a2c_atari.yaml --timesteps ${TIMESTEPS} --env-id ${env_id} --seed ${seed} --wandb online"
        else
            uv run flowreg-train-flowreg-a2c \
                --config configs/flowreg_a2c_atari.yaml \
                --timesteps $TIMESTEPS \
                --env-id "$env_id" \
                --seed $seed \
                --wandb online
        fi

        echo ""
    done
done

echo "============================================"
echo "  All ${TOTAL_RUNS} runs completed!"
echo "============================================"
