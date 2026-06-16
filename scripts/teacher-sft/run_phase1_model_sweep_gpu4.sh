#!/usr/bin/env bash
# Phase 1 model sweep — pythia-160m-weight-seed1 — nvidia-smi GPU 4 (4th A100)
# Group 2b: weight-seed1; data order fixed to seed0, weight init varied.
# Isolates effect of weight initialisation on subliminal trait learning.
# Outputs: checkpoints/teacher_sweep/pythia-160m-weight-seed1/<cfg>/
#          checkpoints/teacher_pythia-160m-weight-seed1/
#
# DGX note: nvidia-smi index ≠ CUDA index on this host.
#   nvidia-smi GPU 4 (A100 80GB) → CUDA_VISIBLE_DEVICES=3
#   nvidia-smi GPU 3 (DGX Display 4GB) → CUDA_VISIBLE_DEVICES=4  (do NOT use)
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=3
uv run python src/teacher_sft/teacher_sft.py \
    --config config.yaml \
    --model-sweep \
    --model-name pythia-160m-weight-seed1 \
    "$@"
