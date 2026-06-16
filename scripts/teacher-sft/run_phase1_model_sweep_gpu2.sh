#!/usr/bin/env bash
# Phase 1 model sweep — pythia-160m-data-seed1 — GPU 2
# Group 2a: data-seed1; weight init fixed to seed0, data order varied.
# Isolates effect of pre-training data ordering on subliminal trait learning.
# Outputs: checkpoints/teacher_sweep/pythia-160m-data-seed1/<cfg>/
#          checkpoints/teacher_pythia-160m-data-seed1/
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=2
uv run python src/teacher_sft/teacher_sft.py \
    --config config.yaml \
    --model-sweep \
    --model-name pythia-160m-data-seed1 \
    "$@"
