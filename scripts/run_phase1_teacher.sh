#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
uv run python src/teacher_sft.py --config config.yaml "$@"
