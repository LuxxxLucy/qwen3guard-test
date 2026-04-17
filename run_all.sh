#!/usr/bin/env bash
# Qwen3Guard quick-headline runner — one representative + one sweep per
# variant, at the "realistic deployment" L1 (prefix-cache only) setting.
# This is the fast smoke run; for the full L0→L3 optimization ladder plus
# the dataset-scale correctness test, use `bash run_optim_ladder.sh`.
#
# Hardware: works on CUDA (default) or CPU (auto-detected).
#
# Usage:
#   1. uv sync                               # first time only
#   2. uv run python scripts/download.py     # pre-fetch 0.6B models + dataset
#   3. bash run_all.sh                       # quick headline
#   OR bash run_optim_ladder.sh              # full ladder + correctness
#
# Outputs: one JSON per run under results/.

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

LENGTHS="${LENGTHS:-32 64 128 256 512 1024 2048 4096 8192 16384}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3Guard-Gen-0.6B}"
STREAM_MODEL="${STREAM_MODEL:-Qwen/Qwen3Guard-Stream-0.6B}"

DEVICE="${DEVICE:-auto}"
if [[ "$DEVICE" == "auto" ]]; then
    if uv run python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi
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

# --- 4B / 8B / ONNX combos: disabled by default. Uncomment to enable. ---
# uv run python src/bench_gen_pytorch.py    --model-id Qwen/Qwen3Guard-Gen-4B    --lengths $LENGTHS
# uv run python src/bench_gen_pytorch.py    --model-id Qwen/Qwen3Guard-Gen-8B    --lengths $LENGTHS
# uv run python src/bench_stream_pytorch.py --model-id Qwen/Qwen3Guard-Stream-4B --lengths $LENGTHS
# uv run python src/bench_stream_pytorch.py --model-id Qwen/Qwen3Guard-Stream-8B --lengths $LENGTHS
#
# ONNX (Gen + Stream): DISABLED. Gen no-KV export is unusably slow on CPU;
# KV-cache export trips optimum on Qwen3 GQA head_dim != hidden/heads.
# Stream uses a stateful custom forward with two classification heads that
# don't cleanly export. Code left in-tree under src/bench_gen_onnx.py +
# scripts/export_gen_onnx.py for future revival.

echo "[run_all] done. Results in $(pwd)/results/"
