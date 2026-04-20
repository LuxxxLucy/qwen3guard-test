#!/usr/bin/env bash
# Investigation runner for the per-token Stream latency gap.
#
# Two probes, both on 0.6B (smallest model shows the biggest relative slack
# between the memory-bandwidth floor and measured per-token latency):
#
#   (1) torch.profiler breakdown of one per-token forward — tells us what
#       fraction of wall time is GPU kernel execution vs host-side overhead
#       (Python + CUDA dispatch + kernel launch), and which kernels dominate.
#
#   (2) Chunked-streaming sweep — tests whether advancing the stream in
#       fixed chunks of k ∈ {1, 2, 4, 8, 16, 32, 64} assistant tokens at a
#       time reduces effective per-token latency. Hypothesis: yes, roughly
#       linearly in k until compute or KV bandwidth binds, because a single
#       forward pays weight-read cost once regardless of input length at
#       short context.
#
# Both probe the output shape of stream_moderate_from_ids (API discovery).
#
# Usage:
#   bash debug_stream.sh                        # default 0.6B
#   MODEL_SIZE=4B bash debug_stream.sh
#   STREAM_MODEL=Qwen/Qwen3Guard-Stream-0.6B bash debug_stream.sh
#
# Everything writes to stdout; the profiler also drops a chrome-trace
# JSON into results/ (loadable at chrome://tracing for a visual timeline).

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

MODEL_SIZE="${MODEL_SIZE:-0.6B}"
STREAM_MODEL="${STREAM_MODEL:-Qwen/Qwen3Guard-Stream-${MODEL_SIZE}}"

# Chunk sizes for the chunked-stream sweep. Powers-of-two from 1 to 64 cover
# the realistic "LLM delivers tokens in bursts of 1-8" range plus an upper
# bound that starts seeing compute bind.
CHUNKS="${CHUNKS:-1 2 4 8 16 32 64}"

echo "===================================================================="
echo "[debug_stream] model=$STREAM_MODEL"
echo "[debug_stream] chunk sizes=$CHUNKS"
echo "===================================================================="

echo ""
echo "### (1) torch.profiler — single per-token forward breakdown"
echo ""
uv run python src/profile_stream.py --model-id "$STREAM_MODEL" --n-tokens 32

echo ""
echo "===================================================================="
echo "### (2) chunked-streaming sweep — effective per-token vs chunk size"
echo "===================================================================="
echo ""
uv run python src/bench_stream_chunked.py \
    --model-id "$STREAM_MODEL" \
    --chunks $CHUNKS \
    --n-assistant-tokens 128 \
    --n-prompts 20

echo ""
echo "[debug_stream] done. Results/traces in $(pwd)/results/"
