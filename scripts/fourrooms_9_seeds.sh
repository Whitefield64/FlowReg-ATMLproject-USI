#!/bin/bash

for seed in {1..9}; do
  echo "Starting runs for seed $seed..."
  
  flowreg-train-baseline \
    --config configs/baseline_ppo_fourrooms.yaml \
    --seed $seed \
    --wandb online

  flowreg-train-flowreg \
    --config configs/flowreg_ppo_fourrooms.yaml \
    --seed $seed \
    --wandb online
done

echo "All seeds finished!"