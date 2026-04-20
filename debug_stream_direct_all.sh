#!/usr/bin/env bash
# Direct-path Qwen3Guard-Stream bench across all three sizes, in one go.
#
# Per size, this runs src/bench_stream_direct_heads.py which:
#   1. Verifies per-position verdict equivalence between
#        - direct path at k=1 (clean causal streaming)
#        - direct path at k=16 (chunked, amortised)
#        - stock API stream_moderate_from_ids at k=1
#      All three should agree at every asst position if the paper's
#      "per-token moderation without reprocessing" semantic holds.
#   2. Sweeps chunk size k ∈ {1, 2, 4, 8, 16, 32, 64} with the
#      classification heads INSIDE the timed region, so the per-chunk
#      numbers include the full production inference cost (backbone +
#      risk_level_head + category_head).
#
# Usage:
#   bash debug_stream_direct_all.sh                   # 0.6B, 4B, 8B
#   SIZES="0.6B" bash debug_stream_direct_all.sh      # single size
#
# Expected runtime on a 3090: ~3 min for 0.6B, ~5 min for 4B, ~8 min for
# 8B (dominated by the 3200-forward latency sweep at each k).

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

SIZES="${SIZES:-0.6B 4B 8B}"
CHUNKS="${CHUNKS:-1 2 4 8 16 32 64}"

for size in $SIZES; do
    MODEL="Qwen/Qwen3Guard-Stream-${size}"
    echo ""
    echo "===================================================================="
    echo "### Qwen3Guard-Stream-${size}"
    echo "===================================================================="
    uv run python src/bench_stream_direct_heads.py \
        --model-id "$MODEL" \
        --chunks $CHUNKS \
        --n-assistant-tokens 128 \
        --n-prompts 20 \
        --equivalence-n-tokens 64 \
        --equivalence-chunk 16
done

echo ""
echo "[debug_stream_direct_all] done."
