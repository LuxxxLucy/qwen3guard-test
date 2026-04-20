#!/usr/bin/env bash
# Direct-path Qwen3Guard-Stream length × chunk-size sweep across all sizes.
#
# Feeds into the REPORT_STREAM rewrite: once the shipped stream_generate
# loop is replaced with a correct incremental forward, where does per-token
# latency cross the 8 ms T2 budget as accumulated user context grows?
#
# Per size, runs src/bench_stream_direct_length_sweep.py over:
#   user_len ∈ {81, 256, 1024, 2048, 4096}
#   k       ∈ {1, 4, 8, 16}
#
# Each cell: 2 warmup + 5 timed streaming runs of 64 asst tokens.
# Sample counts per cell: 320 at k=1, 80 at k=4, 40 at k=8, 20 at k=16 —
# enough for P50/P95, marginal for P99 at k=16.
#
# Usage:
#   bash debug_stream_direct_lengths.sh                   # 0.6B, 4B, 8B
#   SIZES="0.6B" bash debug_stream_direct_lengths.sh      # single size
#   LENGTHS="81 1024" bash debug_stream_direct_lengths.sh # restrict lengths
#
# Expected 3090 runtime: ~5 min (0.6B) + ~10 min (4B) + ~20 min (8B) ≈ 35 min.

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

SIZES="${SIZES:-0.6B 4B 8B}"
LENGTHS="${LENGTHS:-81 256 1024 2048 4096}"
CHUNKS="${CHUNKS:-1 4 8 16}"

for size in $SIZES; do
    MODEL="Qwen/Qwen3Guard-Stream-${size}"
    echo ""
    echo "########################################################################"
    echo "### Qwen3Guard-Stream-${size}"
    echo "########################################################################"
    uv run python src/bench_stream_direct_length_sweep.py \
        --model-id "$MODEL" \
        --lengths $LENGTHS \
        --chunks $CHUNKS \
        --n-assistant-tokens 64 \
        --n-prompts 5 \
        --n-warmup 2
done

echo ""
echo "[debug_stream_direct_lengths] done."
