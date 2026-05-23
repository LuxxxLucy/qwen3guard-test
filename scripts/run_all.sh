#!/usr/bin/env bash
# Qwen3Guard-Stream quick-headline runner — one representative + one sweep
# for Stream-0.6B (prefill + per-token classifier). The Gen variant is
# benchmarked by `bash run_gen_cpu.sh`, which carries its own ladder and
# multi-runtime matrix; this script covers Stream only.
#
# Hardware: works on CUDA (default) or CPU (auto-detected).
#
# Usage:
#   1. uv sync                               # first time only
#   2. uv run python scripts/download.py     # pre-fetch 0.6B models + dataset
#   3. bash run_all.sh                       # Stream headline
#
# Outputs: one JSON per run under results/.

set -euo pipefail
source "$(dirname "$0")/lib.sh"
qg_setup_env

LENGTHS="${LENGTHS:-32 64 128 256 512 1024 2048 4096 8192 16384}"
STREAM_MODEL="${STREAM_MODEL:-Qwen/Qwen3Guard-Stream-0.6B}"

DEVICE="${DEVICE:-$(qg_detect_device)}"
echo "[run_all] device=$DEVICE"

# --- Stream × PyTorch (prefill + per-token regime) ---
uv run python src/bench_stream_pytorch.py --model-id "$STREAM_MODEL" --device "$DEVICE"
uv run python src/bench_stream_pytorch.py --model-id "$STREAM_MODEL" --device "$DEVICE" \
                                          --lengths $LENGTHS
echo "[run_all] done. Results in $(pwd)/results/"
