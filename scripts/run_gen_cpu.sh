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
source "$(dirname "$0")/lib.sh"
qg_setup_env

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

# Cap at 16: this batch=1 prefill-bound workload tops out around there;
# pinning to a higher physical-core count just adds scheduling noise.
THREADS="$(qg_detect_threads)"
[[ "$THREADS" -gt 16 ]] && THREADS=16

echo "[run_gen_cpu] host=$(uname -srm)  threads=$THREADS  n_samples=$N_SAMPLES  model=$MODEL_ID"
[[ -n "$DRY_RUN" ]] && echo "[run_gen_cpu] DRY RUN — smoke test only, latency numbers are not meaningful."

qg_section "1/8  uv sync (llama-cpp-python built from source)"
# Rebuild llama-cpp-python from source every run: the prebuilt wheel is generic
# (no AVX2 / ARM dot-product) and benchmarks 5-7x slow. no-binary-package in
# pyproject makes the build native. uv keys its build cache by source-dist
# hash, not by CMAKE_ARGS, so a cached wheel is reused even with different
# flags — `uv cache clean` evicts it so the rebuild actually honors CMAKE_ARGS.
# GGML_NATIVE builds for the host CPU; CMAKE_ARGS adds a BLAS backend for the
# fp16/fp32 prefill GEMM — Accelerate on macOS, OpenBLAS on Linux.
if [[ "$(uname)" == "Darwin" ]]; then
    export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple -DGGML_NATIVE=ON"
else
    export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_NATIVE=ON"
fi
uv cache clean llama-cpp-python
uv sync --reinstall-package llama-cpp-python \
    || { echo "[fatal] uv sync failed — fix the environment and retry."; exit 1; }
qg_step "uv sync" true

qg_section "2/8  download weights + Qwen3GuardTest dataset"
qg_step "download" uv run python scripts/download.py --variants gen --sizes 0.6B

qg_section "3/8  export ONNX (fp32, int8, with-past)"
qg_step "export onnx" uv run python scripts/export_gen_onnx.py \
    --model-id "$MODEL_ID" --precisions fp32 int8 --with-past

qg_section "4/8  export OpenVINO (fp16, int8)"
qg_step "export openvino" uv run python scripts/export_gen_openvino.py \
    --model-id "$MODEL_ID" --precisions fp16 int8

qg_section "5/8  export GGUF (f16, q8_0)"
qg_step "export gguf" uv run python scripts/export_gen_gguf.py \
    --model-id "$MODEL_ID" --quants f16 q8_0

qg_section "6/8  build Rust candle backend"
qg_step "cargo build" bash -c "cd rust && cargo build --release"

qg_section "7/8  dump Rust benchmark inputs"
qg_step "dump rust inputs" uv run python scripts/dump_rust_inputs.py

qg_section "8/8  benchmarks"
gen() {
    qg_step "gen $*" uv run python src/bench_gen_cpu.py --model-id "$MODEL_ID" \
        --n-samples "$N_SAMPLES" --threads "$THREADS" --verify \
        "${DRY_GEN[@]}" "$@"
    echo
}
# Each cell runs both templates (original, test-200) internally.
# pytorch runs the L2 forced-prefix forward both ways: the full-sequence
# output projection, then --last-pos-logits restricting it to the last
# position. onnx/openvino bake that restriction into the export.
gen --runtime pytorch  --precision fp32
gen --runtime pytorch  --precision fp32  --last-pos-logits
gen --runtime pytorch  --precision fp32  --unoptimized
gen --runtime onnx     --precision fp32  --artifact "onnx_models/$BASENAME/fp32"
gen --runtime onnx     --precision int8  --artifact "onnx_models/$BASENAME/int8"
# ONNX again with the shared system-prompt prefix KV-cached (with-past graph).
gen --runtime onnx     --precision fp32  --artifact "onnx_models/$BASENAME/withpast" --kv-cache
gen --runtime openvino --precision fp16  --artifact "ov_models/$BASENAME/fp16"
gen --runtime openvino --precision int8  --artifact "ov_models/$BASENAME/int8"
gen --runtime openvino --precision fp16  --artifact "ov_models/$BASENAME/fp16" --unoptimized
gen --runtime llamacpp --precision q8_0   --artifact "gguf_models/$BASENAME.q8_0.gguf"
# llama.cpp again with the shared system-prompt prefix KV-cached.
gen --runtime llamacpp --precision q8_0   --artifact "gguf_models/$BASENAME.q8_0.gguf"   --kv-cache
gen --runtime llamacpp --precision q8_0   --artifact "gguf_models/$BASENAME.q8_0.gguf"   --unoptimized
# f16 GGUF — with a BLAS backend its prefill GEMM is the fastest CPU path.
gen --runtime llamacpp --precision f16    --artifact "gguf_models/$BASENAME.f16.gguf"
gen --runtime llamacpp --precision f16    --artifact "gguf_models/$BASENAME.f16.gguf"    --kv-cache

echo "--- Gen: rust-candle ---"
qg_step "gen rust-candle" rust/target/release/qwen3guard-bench \
    --input rust/bench_inputs.json --out-dir results "${DRY_RUST[@]}"
echo

qg_section "step ledger"
qg_ledger_print

qg_section "summary"
uv run python scripts/summarize_cpu.py
echo "[run_gen_cpu] done. Raw JSON under results/."
