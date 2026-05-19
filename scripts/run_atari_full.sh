#!/usr/bin/env bash
set -e

# ============================================================================
# Atari A2C Experiment — Compact FlowReg Reproduction Matrix
# ============================================================================
#
# Project scope:
#   - 2 independent runs per agent (baseline + FlowReg) per environment
#   - 10 million timesteps each
#   - Seeds 0–1
#   - Breakout and Qbert
#
# Usage:
#   bash scripts/run_atari_full.sh              # run everything
#   bash scripts/run_atari_full.sh --dry-run    # print commands without executing
#   bash scripts/run_atari_full.sh --timesteps 1000000
#
# NOTE: This launches jobs SEQUENTIALLY. For parallel execution on a cluster,
# see the --dry-run output and submit each line as a separate job.
# ============================================================================

DRY_RUN=false
TIMESTEPS=10000000
WANDB_MODE=${WANDB_MODE:-online}
SEEDS=(0 1)

ENVS=(
    "ALE/Breakout-v5"
    "ALE/Qbert-v5"
)

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
            echo "=== DRY RUN — printing commands only ==="
            echo ""
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

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
            echo "  uv run flowreg-train-baseline-a2c --config configs/baseline_a2c_atari.yaml --timesteps ${TIMESTEPS} --env-id ${env_id} --seed ${seed} --wandb ${WANDB_MODE}"
        else
            uv run flowreg-train-baseline-a2c \
                --config configs/baseline_a2c_atari.yaml \
                --timesteps $TIMESTEPS \
                --env-id "$env_id" \
                --seed $seed \
                --wandb "$WANDB_MODE"
        fi

        # --- FlowReg A2C ---
        CURRENT=$((CURRENT + 1))
        echo "[${CURRENT}/${TOTAL_RUNS}] FlowReg A2C  | ${env_id} | seed=${seed}"

        if $DRY_RUN; then
            echo "  uv run flowreg-train-flowreg-a2c --config configs/flowreg_a2c_atari.yaml --timesteps ${TIMESTEPS} --env-id ${env_id} --seed ${seed} --wandb ${WANDB_MODE}"
        else
            uv run flowreg-train-flowreg-a2c \
                --config configs/flowreg_a2c_atari.yaml \
                --timesteps $TIMESTEPS \
                --env-id "$env_id" \
                --seed $seed \
                --wandb "$WANDB_MODE"
        fi

        echo ""
    done
done

echo "============================================"
echo "  All ${TOTAL_RUNS} runs completed!"
echo "============================================"
