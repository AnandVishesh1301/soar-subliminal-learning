#!/usr/bin/env bash
# Phase 1 model sweep — pythia-410m (seed0) — GPU 0 (run after 70M finishes)
# Group 3: scale-up model; run if 70M/160M teachers fail to pass verification.
# Uses GPU 0 by default; pass CUDA_VISIBLE_DEVICES=N to override.
# Outputs: checkpoints/teacher_sweep/pythia-410m/<cfg>/
#          checkpoints/teacher_pythia-410m/
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
uv run python src/teacher_sft/teacher_sft.py \
    --config config.yaml \
    --model-sweep \
    --model-name pythia-410m \
    "$@"
