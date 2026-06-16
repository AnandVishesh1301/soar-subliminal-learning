#!/usr/bin/env bash
# Phase 2 — preamble_control dataset (base Pythia only; no teacher needed).
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
uv run python src/dataset_generation/generate_datasets.py --config config.yaml --variant preamble_control "$@"
