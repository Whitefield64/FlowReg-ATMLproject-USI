#!/usr/bin/env bash
set -euo pipefail

bash scripts/atari/run_breakout_3seeds_3m.sh "$@"
bash scripts/atari/run_qbert_3seeds_3m.sh "$@"
