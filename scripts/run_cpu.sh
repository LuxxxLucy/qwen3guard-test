#!/usr/bin/env bash
# Qwen3Guard CPU benchmark — one script, all output to stdout.
#
# Syncs deps, downloads weights, exports the model to every CPU runtime, then
# benchmarks Qwen3Guard-Gen (L2 forced-prefix path) across PyTorch / ONNX
# Runtime / OpenVINO / llama.cpp and Qwen3Guard-Stream on PyTorch-CPU. A
# comparison table is printed at the end.
#
# Usage:
#   bash scripts/run_cpu.sh
#
# Knobs (environment variables):
#   DRY_RUN    set to 1 for a smoke test — tiny warmup/iters/length, every
#              runtime still exported and exercised, latency numbers not real
#   THREADS    CPU threads each runtime is pinned to   (default: physical cores)
#   NSAMPLES   Gen timed iterations per cell            (default: 100)
#   LENGTHS    if set, also run a Gen length sweep, e.g. LENGTHS="32 128 512 2048"
#   TEMPLATE   Gen input template: original (built-in ~296-tok prompt) or
#              test-200 (compressed ~130-tok prompt)    (default: original)
#   VERIFY     "--no-verify" to skip the cross-runtime verdict check
#   MODEL_ID / STREAM_MODEL   override the benchmarked checkpoints
#
# Each runtime/precision step is independent: a failure is recorded and the run
# continues. A PASS/FAIL ledger is printed before the final summary table.

# No `set -u`: expanding an empty array under nounset errors on the bash 3.2
# that ships with macOS. Every variable below already has a `:-` default.
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3Guard-Gen-0.6B}"
STREAM_MODEL="${STREAM_MODEL:-Qwen/Qwen3Guard-Stream-0.6B}"
NSAMPLES="${NSAMPLES:-100}"
TEMPLATE="${TEMPLATE:-original}"
VERIFY="${VERIFY:---verify}"
BASENAME="$(basename "$MODEL_ID")"

# Dry run: tiny benchmark params (every runtime is still exported and run, only
# the warmup / iteration / length counts shrink). bench_gen_cpu.py --dry-run
# fixes the Gen params; the Stream params shrink via the STREAM_* defaults.
DRY_GEN=()
if [[ -n "${DRY_RUN:-}" ]]; then
    DRY_GEN=(--dry-run)
    STREAM_SAMPLES="${STREAM_SAMPLES:-2}"; STREAM_TOKENS="${STREAM_TOKENS:-8}"
    STREAM_PROMPTS="${STREAM_PROMPTS:-2}"; STREAM_ASST="${STREAM_ASST:-16}"
fi

detect_threads() {
    if [[ "$(uname)" == "Darwin" ]]; then
        sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu
    else
        local n
        n=$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2);c=$2}
                                          /^Socket\(s\)/{gsub(/ /,"",$2);s=$2}
                                          END{if(c&&s)print c*s}')
        [[ -n "$n" ]] && echo "$n" || nproc
    fi
}
THREADS="${THREADS:-$(detect_threads)}"

LEN_ARGS=()
[[ -n "${LENGTHS:-}" ]] && LEN_ARGS=(--lengths $LENGTHS)

section() { echo; echo "######## $* ########"; echo; }

# Per-step pass/fail ledger, printed before the summary so a failed runtime is
# named explicitly rather than inferred from a missing row in the table.
LEDGER=()
step() {  # step "label" cmd args...
    local label="$1"; shift
    if "$@"; then LEDGER+=("PASS  $label"); else LEDGER+=("FAIL  $label"); fi
}

echo "[run_cpu] host=$(uname -srm)  threads=$THREADS  n_samples=$NSAMPLES"
echo "[run_cpu] gen=$MODEL_ID  stream=$STREAM_MODEL  template=$TEMPLATE"
[[ -n "${DRY_RUN:-}" ]] && echo "[run_cpu] DRY RUN — smoke test only, latency numbers are not meaningful."

section "1/6  uv sync"
uv sync || { echo "[fatal] uv sync failed — fix the environment and retry."; exit 1; }
LEDGER+=("PASS  uv sync")

section "2/6  download weights + Qwen3GuardTest dataset"
step "download" uv run python scripts/download.py --variants gen stream --sizes 0.6B

section "3/6  export ONNX (fp32, int8, int4)"
step "export onnx" uv run python scripts/export_gen_onnx.py \
    --model-id "$MODEL_ID" --precisions fp32 int8 int4

section "4/6  export OpenVINO (fp32, int8, int4)"
step "export openvino" uv run python scripts/export_gen_openvino.py \
    --model-id "$MODEL_ID" --precisions fp32 int8 int4

section "5/6  export GGUF (q8_0, q4_k_m)"
step "export gguf" uv run python scripts/export_gen_gguf.py \
    --model-id "$MODEL_ID" --quants q8_0 q4_k_m

section "6/6  benchmarks"
gen() {
    echo "--- Gen: $* ---"
    if uv run python src/bench_gen_cpu.py --model-id "$MODEL_ID" \
        --template "$TEMPLATE" --n-samples "$NSAMPLES" --threads "$THREADS" $VERIFY \
        "${LEN_ARGS[@]}" "${DRY_GEN[@]}" "$@"; then
        LEDGER+=("PASS  gen $*")
    else
        LEDGER+=("FAIL  gen $*")
    fi
    echo
}
gen --runtime pytorch
gen --runtime onnx     --precision fp32 --artifact "onnx_models/$BASENAME/fp32"
gen --runtime onnx     --precision int8 --artifact "onnx_models/$BASENAME/int8"
gen --runtime onnx     --precision int4 --artifact "onnx_models/$BASENAME/int4"
gen --runtime openvino --precision fp32 --artifact "ov_models/$BASENAME/fp32"
gen --runtime openvino --precision int8 --artifact "ov_models/$BASENAME/int8"
gen --runtime openvino --precision int4 --artifact "ov_models/$BASENAME/int4"
gen --runtime llamacpp --precision q8_0   --artifact "gguf_models/$BASENAME.q8_0.gguf"
gen --runtime llamacpp --precision q4_k_m --artifact "gguf_models/$BASENAME.q4_k_m.gguf"
# llama.cpp again with the shared system-prompt prefix KV-cached (suffix-only
# forward per request) — the head-to-head for the prefix-cache speedup.
gen --runtime llamacpp --precision q8_0   --artifact "gguf_models/$BASENAME.q8_0.gguf" --kv-cache
gen --runtime llamacpp --precision q4_k_m --artifact "gguf_models/$BASENAME.q4_k_m.gguf" --kv-cache

echo "--- Stream: PyTorch-CPU shipped API ---"
step "stream shipped-api" uv run python src/bench_stream_pytorch.py \
    --model-id "$STREAM_MODEL" --device cpu \
    --n-samples "${STREAM_SAMPLES:-20}" --stream-tokens "${STREAM_TOKENS:-32}"
echo
echo "--- Stream: PyTorch-CPU direct path (chunk sweep) ---"
step "stream direct-path" uv run python src/bench_stream_direct.py \
    --model-id "$STREAM_MODEL" --device cpu \
    --n-prompts "${STREAM_PROMPTS:-5}" --n-assistant-tokens "${STREAM_ASST:-64}" \
    --chunks 1 4 8 16

section "step ledger"
for line in "${LEDGER[@]}"; do echo "  $line"; done

section "summary"
uv run python scripts/summarize_cpu.py
echo "[run_cpu] done. Raw JSON under results/."
