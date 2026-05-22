#!/usr/bin/env bash
# Qwen3Guard-Gen CPU benchmark — one script, all output to stdout.
#
# Syncs deps, downloads weights, exports the model to every CPU runtime
# (ONNX / OpenVINO / GGUF / Rust candle), then benchmarks Qwen3Guard-Gen
# across PyTorch / ONNX Runtime / OpenVINO / llama.cpp / Rust candle. Each
# benchmark cell runs BOTH input templates (original, test-200) in one call.
# A method x template comparison table is printed at the end.
#
# Usage:
#   bash scripts/run_gen_cpu.sh [--dry-run]
#
#   --dry-run   smoke test — tiny warmup/iteration counts, every runtime still
#               exported and exercised, latency numbers are not meaningful.
#
# Each runtime/precision step is independent: a failure is recorded and the run
# continues. A PASS/FAIL ledger is printed before the final summary table.

# No `set -u`: expanding an empty array under nounset errors on the bash 3.2
# that ships with macOS.
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

MODEL_ID="Qwen/Qwen3Guard-Gen-0.6B"
N_SAMPLES=100
BASENAME="$(basename "$MODEL_ID")"

DRY_RUN=
DRY_GEN=()
DRY_RUST=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1; DRY_GEN=(--dry-run); DRY_RUST=(--dry-run) ;;
        *) echo "[run_gen_cpu] unknown argument: $arg" >&2; exit 2 ;;
    esac
done

detect_threads() {  # physical cores — CPU detection, not a user knob
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
THREADS="$(detect_threads)"

section() { echo; echo "######## $* ########"; echo; }

# Per-step pass/fail ledger, printed before the summary so a failed runtime is
# named explicitly rather than inferred from a missing row in the table.
LEDGER=()
step() {  # step "label" cmd args...
    local label="$1"; shift
    if "$@"; then LEDGER+=("PASS  $label"); else LEDGER+=("FAIL  $label"); fi
}

echo "[run_gen_cpu] host=$(uname -srm)  threads=$THREADS  n_samples=$N_SAMPLES  model=$MODEL_ID"
[[ -n "$DRY_RUN" ]] && echo "[run_gen_cpu] DRY RUN — smoke test only, latency numbers are not meaningful."

section "1/8  uv sync (llama-cpp-python built from source)"
# Rebuild llama-cpp-python from source every run: the prebuilt wheel is generic
# (no AVX2 / ARM dot-product) and benchmarks 5-7x slow. no-binary-package in
# pyproject makes the rebuild native; --reinstall-package forces it even when a
# wheel is already installed, which a plain `uv sync` would keep.
uv sync --reinstall-package llama-cpp-python \
    || { echo "[fatal] uv sync failed — fix the environment and retry."; exit 1; }
LEDGER+=("PASS  uv sync")

section "2/8  download weights + Qwen3GuardTest dataset"
step "download" uv run python scripts/download.py --variants gen --sizes 0.6B

section "3/8  export ONNX (fp32, int8, with-past)"
step "export onnx" uv run python scripts/export_gen_onnx.py \
    --model-id "$MODEL_ID" --precisions fp32 int8 --with-past

section "4/8  export OpenVINO (fp32, int8)"
step "export openvino" uv run python scripts/export_gen_openvino.py \
    --model-id "$MODEL_ID" --precisions fp32 int8

section "5/8  export GGUF (q8_0)"
step "export gguf" uv run python scripts/export_gen_gguf.py \
    --model-id "$MODEL_ID" --quants q8_0

section "6/8  build Rust candle backend"
step "cargo build" bash -c "cd rust && cargo build --release"

section "7/8  dump Rust benchmark inputs"
step "dump rust inputs" uv run python scripts/dump_rust_inputs.py

section "8/8  benchmarks"
gen() {
    echo "--- Gen: $* ---"
    step "gen $*" uv run python src/bench_gen_cpu.py --model-id "$MODEL_ID" \
        --n-samples "$N_SAMPLES" --threads "$THREADS" --verify \
        "${DRY_GEN[@]}" "$@"
    echo
}
# Each cell runs both templates (original, test-200) internally.
gen --runtime pytorch  --precision fp32
gen --runtime pytorch  --precision fp32  --unoptimized
gen --runtime onnx     --precision fp32  --artifact "onnx_models/$BASENAME/fp32"
gen --runtime onnx     --precision int8  --artifact "onnx_models/$BASENAME/int8"
# ONNX again with the shared system-prompt prefix KV-cached (with-past graph).
gen --runtime onnx     --precision fp32  --artifact "onnx_models/$BASENAME/withpast" --kv-cache
gen --runtime openvino --precision fp32  --artifact "ov_models/$BASENAME/fp32"
gen --runtime openvino --precision int8  --artifact "ov_models/$BASENAME/int8"
gen --runtime openvino --precision fp32  --artifact "ov_models/$BASENAME/fp32" --unoptimized
gen --runtime llamacpp --precision q8_0   --artifact "gguf_models/$BASENAME.q8_0.gguf"
# llama.cpp again with the shared system-prompt prefix KV-cached.
gen --runtime llamacpp --precision q8_0   --artifact "gguf_models/$BASENAME.q8_0.gguf"   --kv-cache
gen --runtime llamacpp --precision q8_0   --artifact "gguf_models/$BASENAME.q8_0.gguf"   --unoptimized

echo "--- Gen: rust-candle ---"
step "gen rust-candle" rust/target/release/qwen3guard-bench \
    --input rust/bench_inputs.json --out-dir results "${DRY_RUST[@]}"
echo

section "step ledger"
for line in "${LEDGER[@]}"; do echo "  $line"; done

section "summary"
uv run python scripts/summarize_cpu.py
echo "[run_gen_cpu] done. Raw JSON under results/."
