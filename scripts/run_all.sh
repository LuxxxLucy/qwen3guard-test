#!/usr/bin/env bash
# Qwen3Guard quick-headline runner — one representative + one sweep per
# variant, at the "realistic deployment" L1 (prefix-cache only) setting.
# This is the fast smoke run; for the full L0→L3 optimization ladder plus
# the dataset-scale correctness test, use `bash run_optim_ladder_gen.sh`.
#
# Hardware: works on CUDA (default) or CPU (auto-detected).
#
# Usage:
#   1. uv sync                               # first time only
#   2. uv run python scripts/download.py     # pre-fetch 0.6B models + dataset
#   3. bash run_all.sh                       # quick headline
#   OR bash run_optim_ladder_gen.sh          # full ladder + correctness
#
# Outputs: one JSON per run under results/.

set -euo pipefail
source "$(dirname "$0")/lib.sh"
qg_setup_env

LENGTHS="${LENGTHS:-32 64 128 256 512 1024 2048 4096 8192 16384}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3Guard-Gen-0.6B}"
STREAM_MODEL="${STREAM_MODEL:-Qwen/Qwen3Guard-Stream-0.6B}"

DEVICE="${DEVICE:-$(qg_detect_device)}"
echo "[run_all] device=$DEVICE"

# --- Gen × PyTorch (L1 prefix-cache, the default) ---
uv run python src/bench_gen_pytorch.py    --model-id "$MODEL_ID" --device "$DEVICE"
uv run python src/bench_gen_pytorch.py    --model-id "$MODEL_ID" --device "$DEVICE" \
                                          --lengths $LENGTHS
# Uncached baseline sweep for prefix-cache delta:
uv run python src/bench_gen_pytorch.py    --model-id "$MODEL_ID" --device "$DEVICE" \
                                          --lengths $LENGTHS --no-prefix-cache

# --- Stream × PyTorch (prefill + per-token regime) ---
uv run python src/bench_stream_pytorch.py --model-id "$STREAM_MODEL" --device "$DEVICE"
uv run python src/bench_stream_pytorch.py --model-id "$STREAM_MODEL" --device "$DEVICE" \
                                          --lengths $LENGTHS
echo "[run_all] done. Results in $(pwd)/results/"
