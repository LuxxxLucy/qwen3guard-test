#!/usr/bin/env bash
# Optimization ladder runner for Qwen3Guard-Gen-0.6B (full-text classifier).
#
# Step 0: Dataset-scale correctness test — verify every optimization agrees
#         with the L0 baseline verdict on N real Qwen3GuardTest samples.
# Step 1: Benchmark each level (representative + length sweep), one at a time.
#
# Each bench run writes an independent JSON into results/ with opt_level
# tagged in the extra field so the report script can group cleanly.
#
# Ladder:
#   LP — prefill-only:        one forward + restricted 3-way verdict readout.
#                             Isolates pure prefill cost at each length.
#   L0 — naive:               no prefix cache, no forced prefix, no compile
#   L1 — + prefix cache       (template head KV reused across calls)
#   L2 — + forced-prefix      (teacher-force "Safety: ", single forward)
#   L3 — + torch.compile      (CUDA-only, mode=reduce-overhead)
#
# With LP and L0 measured at each sweep length, the per-token decode cost
# is derivable as (L0 - LP) / N_decode_tokens — the breakdown
# REPORT_GEN.md uses.
#
# The Stream variant is orthogonal to this ladder; see
# `run_optim_ladder_stream.sh`.
#
# Usage:
#   bash run_optim_ladder_gen.sh                           # default: 0.6B
#   MODEL_SIZE=4B  bash run_optim_ladder_gen.sh            # 4B variant
#   MODEL_SIZE=8B  bash run_optim_ladder_gen.sh            # 8B variant
#   CORRECTNESS_N=200 bash run_optim_ladder_gen.sh         # more correctness samples
#   MODEL_ID=Qwen/Qwen3Guard-Gen-4B bash ...               # full HF id override

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

LENGTHS="${LENGTHS:-32 64 128 256 512 1024 2048 4096}"
CORRECTNESS_N="${CORRECTNESS_N:-20}"
# Model selection: either set MODEL_ID directly to a full HF id, or set
# MODEL_SIZE=0.6B|4B|8B to pick the Qwen3Guard-Gen variant at that size.
MODEL_SIZE="${MODEL_SIZE:-0.6B}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3Guard-Gen-${MODEL_SIZE}}"

# Auto-detect device so the ladder does the right thing on either CUDA or CPU.
# L3 (torch.compile) is skipped on CPU: first-call inductor compilation is
# multi-minute on CPU and the steady-state speedup is marginal. On CUDA the
# same flag turns on kernel fusion + CUDA graphs.
DEVICE="${DEVICE:-auto}"
if [[ "$DEVICE" == "auto" ]]; then
    if uv run python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi
echo "[run_optim_ladder_gen] device=$DEVICE model=$MODEL_ID"

echo "===================================================================="
echo "Step 0: dataset correctness check (N=${CORRECTNESS_N})"
echo "===================================================================="
uv run python scripts/correctness_test.py \
    --model-id "$MODEL_ID" \
    --n "$CORRECTNESS_N" \
    --device "$DEVICE"

echo ""
echo "===================================================================="
echo "Step 1: benchmark the ladder (Gen variant)"
echo "===================================================================="

run_level () {
    local tag="$1"
    local flags="$2"
    echo ""
    echo "---- $tag : $flags ----"
    echo "[representative]"
    uv run python src/bench_gen_pytorch.py \
        --model-id "$MODEL_ID" --opt-level "$tag" --device "$DEVICE" $flags
    echo "[length sweep]"
    uv run python src/bench_gen_pytorch.py \
        --model-id "$MODEL_ID" --opt-level "$tag" --device "$DEVICE" \
        --lengths $LENGTHS $flags
}

run_level "LP" "--prefill-only    --no-prefix-cache  --no-forced-prefix --no-compile"
run_level "L0" "--no-prefix-cache --no-forced-prefix --no-compile"
run_level "L1" "--prefix-cache    --no-forced-prefix --no-compile"
run_level "L2" "--prefix-cache    --forced-prefix    --no-compile"

if [[ "$DEVICE" == "cuda" ]]; then
    run_level "L3" "--prefix-cache    --forced-prefix    --compile"
else
    echo ""
    echo "[run_optim_ladder_gen] skipping L3 on $DEVICE: torch.compile is CUDA-primary"
    echo "  (multi-minute first-call inductor compile on CPU, marginal steady-state gain)"
fi

echo ""
echo "[run_optim_ladder_gen] done. Results in $(pwd)/results/"
