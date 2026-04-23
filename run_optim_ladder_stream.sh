#!/usr/bin/env bash
# Optimization ladder runner for Qwen3Guard-Stream-0.6B (streaming classifier).
#
# Stream-0.6B uses a custom stateful forward (`stream_moderate_from_ids`)
# whose opaque state object is not trivially cloneable, so it does not
# plug into the Gen ladder's prefix-cache / forced-prefix levers. Per-token
# latency is already the right regime for A8 T2 (< 8 ms/token); the only
# lever that applies here is kernel fusion (torch.compile) and — eventually
# — a static-cache rewrite.
#
# Runs:
#   - representative bench (one dialogue from Qwen3GuardTest)
#   - length sweep (synthetic prompts at fixed user-content lengths)
#
# Both write per-run JSONs into results/ with variant=stream so they group
# cleanly alongside the Gen ladder output.
#
# Usage:
#   bash run_optim_ladder_stream.sh                        # default: 0.6B
#   MODEL_SIZE=4B  bash run_optim_ladder_stream.sh         # 4B variant
#   MODEL_SIZE=8B  bash run_optim_ladder_stream.sh         # 8B variant
#   LENGTHS="32 128 1024" bash run_optim_ladder_stream.sh  # restrict sweep
#   STREAM_MODEL=Qwen/Qwen3Guard-Stream-4B bash ...        # full HF id override

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

LENGTHS="${LENGTHS:-32 64 128 256 512 1024 2048 4096}"
# Model selection: either set STREAM_MODEL directly to a full HF id, or set
# MODEL_SIZE=0.6B|4B|8B to pick the Qwen3Guard-Stream variant at that size.
MODEL_SIZE="${MODEL_SIZE:-0.6B}"
STREAM_MODEL="${STREAM_MODEL:-Qwen/Qwen3Guard-Stream-${MODEL_SIZE}}"

DEVICE="${DEVICE:-auto}"
if [[ "$DEVICE" == "auto" ]]; then
    if uv run python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi
echo "[run_optim_ladder_stream] device=$DEVICE model=$STREAM_MODEL"

echo "===================================================================="
echo "Stream variant: representative + length sweep"
echo "===================================================================="
uv run python src/bench_stream_pytorch.py \
    --model-id "$STREAM_MODEL" --device "$DEVICE"
uv run python src/bench_stream_pytorch.py \
    --model-id "$STREAM_MODEL" --device "$DEVICE" \
    --lengths $LENGTHS

echo ""
echo "[run_optim_ladder_stream] done. Results in $(pwd)/results/"
