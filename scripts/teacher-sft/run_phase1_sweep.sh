#!/usr/bin/env bash
# Phase 1 — hyperparameter sweep over teacher.sweep configs in config.yaml.
# Each config is trained, verified, and saved to checkpoints/teacher_sweep/{name}/.
# The best passing model is copied to the canonical checkpoints/teacher_owl/ dir.
# Usage: bash scripts/teacher-sft/run_phase1_sweep.sh [--debug]
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0
uv run python src/teacher_sft/teacher_sft.py --config config.yaml --sweep "$@"
