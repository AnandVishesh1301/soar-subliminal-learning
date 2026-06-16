#!/usr/bin/env bash
# Phase 1 model sweep — pythia-70m — GPU 0
# Outputs: checkpoints/teacher_sweep/pythia-70m/<cfg>/  (passing runs)
#          checkpoints/teacher_pythia-70m/               (best passing, canonical)
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=0
uv run python src/teacher_sft/teacher_sft.py \
    --config config.yaml \
    --model-sweep \
    --model-name pythia-70m \
    "$@"
