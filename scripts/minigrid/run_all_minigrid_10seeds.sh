#!/usr/bin/env bash
set -euo pipefail

bash scripts/minigrid/run_fourrooms_10seeds.sh "$@"
bash scripts/minigrid/run_dynamic_obstacles_10seeds.sh "$@"
bash scripts/minigrid/run_doorkey_10seeds.sh "$@"
