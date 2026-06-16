#!/usr/bin/env bash
# Smoke-test Options 3 & 4 prompt formats on Phase 1 canonical 70M/160M teachers + bases.
# GPU recommended (~2–5 min with --quick, ~15–30 min full grid on A100).
#
# Examples:
#   ./scripts/dataset-generation/run_smoke_gen_option3_4.sh --quick
#   ./scripts/dataset-generation/run_smoke_gen_option3_4.sh --quick --teacher-only
#   ./scripts/dataset-generation/run_smoke_gen_option3_4.sh --models pythia-160m --n-samples 30
#   uv run python src/dataset_generation/smoke_gen_prompt_formats.py --list-formats
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
uv run python src/dataset_generation/smoke_gen_prompt_formats.py --config config.yaml "$@"
