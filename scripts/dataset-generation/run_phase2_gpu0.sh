#!/usr/bin/env bash
# Phase 2 — preamble_owl dataset (requires Phase 1 teacher checkpoint: teacher_pythia-160m).
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0
uv run python src/dataset_generation/generate_datasets.py --config config.yaml --variant preamble_owl "$@"
