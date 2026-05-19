#!/bin/bash

RUN_ARGS=""

echo "Gathering baseline runs..."
for s in {0..9}; do
  MATCH=(runs/baseline_ppo/*seed${s}_*)
  
  if [ -d "${MATCH[0]}" ]; then
    RUN_ARGS="$RUN_ARGS --run baseline:$s:${MATCH[0]}"
  else
    echo "  -> Warning: Could not find baseline seed $s"
  fi
done

echo "Gathering flowreg runs..."
for s in {0..9}; do
  MATCH=(runs/flowreg_ppo/*seed${s}_*)
  
  if [ -d "${MATCH[0]}" ]; then
    RUN_ARGS="$RUN_ARGS --run flowreg:$s:${MATCH[0]}"
  else
    echo "  -> Warning: Could not find flowreg seed $s"
  fi
done

echo "Running analysis..."
flowreg-analyze-runs $RUN_ARGS --markdown-output summary.md

echo "Done! Check summary.md for the results."