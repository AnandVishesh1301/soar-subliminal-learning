#!/usr/bin/env bash
# Phase 2 — launch all three dataset variants in parallel across GPUs 0/1/2.
# preamble_control (GPU 1) can start immediately; owl variants need teacher gate.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "Starting Phase 2 dataset generation on GPUs 0, 1, 2..."
echo "  GPU 0: preamble_owl   -> datasets/preamble_teacher_owl.jsonl"
echo "  GPU 1: preamble_control -> datasets/preamble_control.jsonl"
echo "  GPU 2: raw_owl       -> datasets/raw_teacher_owl.jsonl"
echo
echo "Pass --debug to each script for smoke-sized runs."
echo

"$SCRIPT_DIR/run_phase2_gpu0.sh" "$@" &
pid0=$!
"$SCRIPT_DIR/run_phase2_gpu1.sh" "$@" &
pid1=$!
"$SCRIPT_DIR/run_phase2_gpu2.sh" "$@" &
pid2=$!

wait "$pid0"
wait "$pid1"
wait "$pid2"

echo
echo "All three variants finished. Run divergence measurement:"
echo "  ./scripts/dataset-generation/run_phase2_divergence.sh"
