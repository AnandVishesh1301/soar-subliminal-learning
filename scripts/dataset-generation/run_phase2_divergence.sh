#!/usr/bin/env bash
# Phase 2d — divergence token fraction (160M owl teacher vs base pythia-160m).
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0
uv run python src/dataset_generation/measure_divergence.py --config config.yaml "$@"
