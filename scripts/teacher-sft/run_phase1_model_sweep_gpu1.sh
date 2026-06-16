#!/usr/bin/env bash
# Phase 1 model sweep — pythia-160m (seed0) — GPU 1
# Group 1: baseline 160M; isolates size effect vs 70M.
# Outputs: checkpoints/teacher_sweep/pythia-160m/<cfg>/
#          checkpoints/teacher_pythia-160m/
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=1
uv run python src/teacher_sft/teacher_sft.py \
    --config config.yaml \
    --model-sweep \
    --model-name pythia-160m \
    "$@"
