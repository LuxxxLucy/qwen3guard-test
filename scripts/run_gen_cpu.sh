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

# --model-id accepts a HF repo id (default) or a local directory. A local path
# is useful when the HF hub is unreachable (e.g. Huawei Cloud regions where the
# upstream LFS CDN is throttled): pre-fetch via ModelScope or rsync, then
# `bash scripts/run_gen_cpu.sh --model-id /local/path`. The download step is
# skipped automatically when --model-id resolves to an existing directory.
MODEL_ID="Qwen/Qwen3Guard-Gen-0.6B"
N_SAMPLES=100

DRY_RUN=
DRY_GEN=()
DRY_RUST=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; DRY_GEN=(--dry-run); DRY_RUST=(--dry-run); shift ;;
        --model-id) MODEL_ID="$2"; shift 2 ;;
        --model-id=*) MODEL_ID="${1#--model-id=}"; shift ;;
        *) echo "[run_gen_cpu] unknown argument: $1" >&2; exit 2 ;;
    esac
done
BASENAME="$(basename "$MODEL_ID")"

# Cap at 16: this batch=1 prefill-bound workload tops out around there;
# pinning to a higher physical-core count just adds scheduling noise.
THREADS="$(qg_detect_threads)"
[[ "$THREADS" -gt 16 ]] && THREADS=16
qg_export_thread_caps "$THREADS"

echo "[run_gen_cpu] host=$(uname -srm)  threads=$THREADS  n_samples=$N_SAMPLES  model=$MODEL_ID"
[[ -n "$DRY_RUN" ]] && echo "[run_gen_cpu] DRY RUN — smoke test only, latency numbers are not meaningful."

qg_section "1/10 uv sync (llama-cpp-python built from source with native ISA)"
# The prebuilt llama-cpp-python wheel is generic (no AVX2 / ARM dot-product) and
# benchmarks 5-7x slow. Build from source with native ISA via CMAKE_ARGS. uv
# keys its build cache by source-dist hash, not by CMAKE_ARGS, so a CMAKE_ARGS
# change requires cache clean + reinstall to honor the new flags.
#
# Skip the rebuild if the installed wheel's fingerprint already matches current
# CMAKE_ARGS — saves bandwidth on flaky networks and ~5 min compile per run.
if [[ "$(uname)" == "Darwin" ]]; then
    export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple -DGGML_NATIVE=ON"
else
    export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_NATIVE=ON -DGGML_CPU_KLEIDIAI=ON"
fi
FINGERPRINT_FILE=".venv/.llama_cpp_cmake_args"
if command -v sha256sum >/dev/null; then HASHER="sha256sum"; else HASHER="shasum -a 256"; fi
WANT_FP="$(printf '%s' "$CMAKE_ARGS" | $HASHER | cut -c1-16)"
HAVE_FP="$(cat "$FINGERPRINT_FILE" 2>/dev/null || true)"
if [[ "$WANT_FP" == "$HAVE_FP" ]] && uv run python -c "import llama_cpp" 2>/dev/null; then
    qg_step "uv sync (llama-cpp-python wheel cached, CMAKE_ARGS unchanged)" true
else
    uv cache clean llama-cpp-python
    for i in 1 2 3 4 5; do
        if uv sync --reinstall-package llama-cpp-python; then
            echo "$WANT_FP" > "$FINGERPRINT_FILE"
            qg_step "uv sync" true
            break
        fi
        [[ $i -lt 5 ]] || { echo "[fatal] uv sync failed after 5 attempts."; exit 1; }
        echo "[uv sync] attempt $i failed; retrying in 15s..."
        sleep 15
    done
fi

qg_section "2/10 download weights + Qwen3GuardTest dataset"
if [[ -d "$MODEL_ID" ]]; then
    qg_step "download (model local, dataset only)" uv run python scripts/download.py \
        --variants gen --sizes 0.6B --skip-model
else
    qg_step "download" uv run python scripts/download.py --variants gen --sizes 0.6B
fi

qg_section "3/10 export ONNX (fp32, int8, with-past)"
qg_step "export onnx" uv run python -m exporters.onnx \
    --model-id "$MODEL_ID" --precisions fp32 int8 --with-past

qg_section "4/10 export GGUF (f32, f16, q8_0)"
qg_step "export gguf" uv run python -m exporters.gguf \
    --model-id "$MODEL_ID" --quants f32 f16 q8_0

qg_section "5/10 export CTranslate2 (fp32)"
qg_step "export ctranslate2" uv run python -m exporters.ctranslate2 \
    --model-id "$MODEL_ID" --precisions fp32

qg_section "6/10 export MNN-LLM (fp16)"
# MNN-LLM's converter needs the alibaba/MNN source tree (transformers/llm/export
# /llmexport.py). $MNN_HOME points at a local clone. MNN tops out at fp16
# weights for the language path (1/2/4/8-bit otherwise) — there is no fp32
# weight export. precision="high" at runtime keeps accumulators in fp32.
# Skipped if pymnn isn't installed or $MNN_HOME isn't set.
if uv run python -c "import MNN.llm" 2>/dev/null; then
    if [[ -n "${MNN_HOME:-}" && -d "$MNN_HOME" ]]; then
        qg_step "export mnn" uv run python -m exporters.mnn \
            --model-id "$MODEL_ID" --precisions fp16 --mnn-home "$MNN_HOME"
    else
        echo "[skip] MNN_HOME not set or not a directory; skipping MNN export."
    fi
else
    echo "[skip] MNN.llm not importable on this host (pymnn not installed)."
fi

qg_section "7/10 build Rust candle backend"
qg_step "cargo build" bash -c "cd src/backends/rust && cargo build --release"

qg_section "8/10 dump Rust benchmark inputs"
qg_step "dump rust inputs" uv run python scripts/dump_rust_inputs.py \
    --model-id "$MODEL_ID"

qg_section "9/10 cross-impl trick-correctness gate"
# Per-row verify against PyTorch fp32 L0 reference. Standalone runnable via
# `bash scripts/verify_correctness.sh`. Drift on any row records FAIL; the
# bench section continues regardless so latency cells still populate.
qg_step "verify (all rows)" bash scripts/verify_correctness.sh --model-id "$MODEL_ID"

qg_section "10/10 benchmarks"
gen() {
    qg_step "gen $*" uv run python src/bench_gen_cpu.py --model-id "$MODEL_ID" \
        --n-samples "$N_SAMPLES" --threads "$THREADS" \
        "${DRY_GEN[@]}" "$@"
    echo
}
# gen_llama: per-cell env override for llama.cpp. ARMV8 (NEON-only) sgemm in
# OpenBLAS 0.3.26 is ~13% faster than the default armv8sve dispatch for
# llama.cpp's ggml-blas matmul path on Kunpeng 920. For pytorch the same swap
# regresses L2 lastpos (the M=1 lm_head GEMV, 540 -> 826 ms) because SVE's
# wider lanes win on small-M GEMV. Scoping the env to llama.cpp gen() calls
# keeps both winning paths.
gen_llama() {
    qg_step "gen $*" env OPENBLAS_CORETYPE=ARMV8 \
        uv run python src/bench_gen_cpu.py --model-id "$MODEL_ID" \
        --n-samples "$N_SAMPLES" --threads "$THREADS" \
        "${DRY_GEN[@]}" "$@"
    echo
}
# gen_llama_kopt: f32 +kernel-opt cell. Two layered swaps over gen_llama:
#  1. LD_PRELOAD the OpenMP variant of system OpenBLAS (libopenblas0-openmp).
#     The default Ubuntu package ships a pthread-built OpenBLAS; llama.cpp's
#     libggml-cpu.so already links libgomp.so for ggml's own intra-op
#     parallelism. Two thread pools (pthread + OpenMP) each claim N threads
#     and oversubscribe the cores — perf shows ~30% of wall time inside
#     do_sched_yield / arch_local_irq_enable from the pthread spin-wait.
#     The OpenMP variant routes OpenBLAS through the same libgomp pool ggml
#     already runs, collapsing the two pools into one and erasing the
#     yield storm.
#  2. OPENBLAS_CORETYPE=NEOVERSEN2 selects OpenBLAS's Neoverse-N2 sgemm
#     kernel (SVE + I8MM tuned) over the default armv8sve dispatch. Kunpeng
#     920 part 0xd02 (TaiShan v110, A76-class with SVE/I8MM) closely matches
#     the N2 microarch; the N2 path is ~9% faster than ARMV8 alone on this
#     prefill.
# Scoped per-cell so the existing `llamacpp f32` rows still use the default
# OpenBLAS-pthread + ARMV8 path (regression baseline preserved). On macOS /
# any host without the OpenMP OpenBLAS, LD_PRELOAD silently skips and the
# cell falls back to the same path as gen_llama.
gen_llama_kopt() {
    local preload=
    for cand in /usr/lib/aarch64-linux-gnu/openblas-openmp/libopenblas.so.0 \
                /usr/lib/aarch64-linux-gnu/openblas-openmp/libopenblas.so; do
        [[ -e "$cand" ]] && { preload="$cand"; break; }
    done
    qg_step "gen $*" env \
        ${preload:+LD_PRELOAD="$preload"} \
        OPENBLAS_CORETYPE=NEOVERSEN2 \
        uv run python src/bench_gen_cpu.py --model-id "$MODEL_ID" \
        --n-samples "$N_SAMPLES" --threads "$THREADS" \
        "${DRY_GEN[@]}" "$@"
    echo
}
# Each cell runs both templates (original, test-200) internally.
# pytorch runs the forced-prefix forward three ways: L1 (full-sequence output
# projection), L2 (--last-pos-logits restricts to last position), L3 (prefix
# KV cached). L0 is the model-card decode loop. onnx bakes lastpos into export.
gen --runtime pytorch  --precision fp32  --unoptimized
gen --runtime pytorch  --precision fp32
gen --runtime pytorch  --precision fp32  --last-pos-logits
gen --runtime pytorch  --precision fp32  --kv-cache
# ONNX fp32 ladder: L0 (decode loop, full lm_head) -> +L1 (forced prefix, full
# lm_head) -> +L2 (forced prefix, lastpos sliced) -> +L3 (forced prefix, lastpos
# sliced + prefix-KV reused). L0/L1 use fp32/ (no slice); L2/L3 use fp32-lastpos/.
# --last-pos-logits on the L2 cell is a manifest (the slice is graph-baked).
gen --runtime onnx     --precision fp32 --artifact "var/models/onnx/$BASENAME/fp32"         --unoptimized
gen --runtime onnx     --precision fp32 --artifact "var/models/onnx/$BASENAME/fp32"
gen --runtime onnx     --precision fp32 --artifact "var/models/onnx/$BASENAME/fp32-lastpos" --last-pos-logits
gen --runtime onnx     --precision fp32 --artifact "var/models/onnx/$BASENAME/fp32-lastpos" --kv-cache
# ONNX int8 ladder: same four rows on the dynamic-INT8 quantized graphs.
gen --runtime onnx     --precision int8 --artifact "var/models/onnx/$BASENAME/int8"         --unoptimized
gen --runtime onnx     --precision int8 --artifact "var/models/onnx/$BASENAME/int8"
gen --runtime onnx     --precision int8 --artifact "var/models/onnx/$BASENAME/int8-lastpos" --last-pos-logits
gen --runtime onnx     --precision int8 --artifact "var/models/onnx/$BASENAME/int8-lastpos" --kv-cache
# llama.cpp: f32 first (precision ladder anchor), then +kernel-opt (env tweaks
# that close the gap to ONNX), then f16 (BLAS prefill), then q8_0 (KleidiAI
# int8). Each precision runs the full L0 / L1 / L3 ladder; L2 is baked.
# f32 GGUF — uncompressed baseline.
gen_llama --runtime llamacpp --precision f32      --artifact "var/models/gguf/$BASENAME.f32.gguf"   --unoptimized
gen_llama --runtime llamacpp --precision f32      --artifact "var/models/gguf/$BASENAME.f32.gguf"
gen_llama --runtime llamacpp --precision f32      --artifact "var/models/gguf/$BASENAME.f32.gguf"   --kv-cache
# f32 +kernel-opt — same f32 GGUF, env-only kernel swap (see gen_llama_kopt).
# Closes the gap vs ONNX Runtime fp32 by removing the OpenBLAS-pthread /
# libgomp oversubscription and routing sgemm through the Neoverse-N2 kernel.
gen_llama_kopt --runtime llamacpp --precision f32-kopt --artifact "var/models/gguf/$BASENAME.f32.gguf" --unoptimized
gen_llama_kopt --runtime llamacpp --precision f32-kopt --artifact "var/models/gguf/$BASENAME.f32.gguf"
gen_llama_kopt --runtime llamacpp --precision f32-kopt --artifact "var/models/gguf/$BASENAME.f32.gguf" --kv-cache
# f16 GGUF — with a BLAS backend its prefill GEMM is the fastest CPU fp path.
gen_llama --runtime llamacpp --precision f16      --artifact "var/models/gguf/$BASENAME.f16.gguf"   --unoptimized
gen_llama --runtime llamacpp --precision f16      --artifact "var/models/gguf/$BASENAME.f16.gguf"
gen_llama --runtime llamacpp --precision f16      --artifact "var/models/gguf/$BASENAME.f16.gguf"   --kv-cache
# q8_0 GGUF — KleidiAI int8 dispatch; the int-quant floor on Kunpeng 920.
gen_llama --runtime llamacpp --precision q8_0     --artifact "var/models/gguf/$BASENAME.q8_0.gguf"  --unoptimized
gen_llama --runtime llamacpp --precision q8_0     --artifact "var/models/gguf/$BASENAME.q8_0.gguf"
gen_llama --runtime llamacpp --precision q8_0     --artifact "var/models/gguf/$BASENAME.q8_0.gguf"  --kv-cache

echo "--- Gen: rust-candle ---"
qg_step "gen rust-candle" src/backends/rust/target/release/qwen3guard-bench \
    --input src/backends/rust/bench_inputs.json --out-dir var/results "${DRY_RUST[@]}"
echo

# CTranslate2 CPU: single fp32 row, all tricks baked in (forward_batch's
# last-position slice gives L2 baked + L1 forced-prefix).
gen --runtime ctranslate2 --precision fp32 --artifact "var/models/ctranslate2/$BASENAME-fp32"

# MNN-LLM CPU: Alibaba's Arm-tuned LLM runtime. `forward(input_ids)` returns
# last-position logits by default (L2 lastpos baked); L1 is one forced-prefix
# forward. fp16 weights + precision="high" (fp32 accumulators) is the
# highest-precision MNN-LLM path. Skipped when pymnn isn't installed or the
# converted model dir is missing.
if uv run python -c "import MNN.llm" 2>/dev/null \
   && [[ -d "var/models/mnn/$BASENAME-fp16" ]]; then
    gen --runtime mnn --precision fp16 --artifact "var/models/mnn/$BASENAME-fp16"
    gen --runtime mnn --precision fp16 --artifact "var/models/mnn/$BASENAME-fp16" --unoptimized
else
    echo "[skip] mnn (MNN.llm not importable or model dir missing)."
fi

# vLLM CPU: single-row baseline, no trick ladder. vLLM bakes its own paged
# attention, KV management, and last-position sampling. The Qwen3-supporting
# vLLM releases require torch>=2.7; the main .venv pins torch<2.7 for the
# GPU box's CUDA-driver compat. vLLM therefore lives in a sibling .venv-vllm/
# (bootstrap with scripts/setup_vllm_venv.sh on Kunpeng aarch64). The bench
# script is invoked directly via that interpreter, bypassing `uv run`.
VLLM_PY=".venv-vllm/bin/python"
if [[ -x "$VLLM_PY" ]] && "$VLLM_PY" -c "import vllm" 2>/dev/null; then
    # vLLM's CPU dispatcher honours VLLM_CPU_OMP_THREADS_BIND; pin to the same
    # physical-core budget the other backends use. tcmalloc reduces allocator
    # pressure for vLLM's eager-mode dispatch; LD_PRELOAD it when present.
    VLLM_TCMALLOC="$(find /usr -name 'libtcmalloc_minimal.so.4' 2>/dev/null | head -1)"
    qg_step "gen vllm-cpu fp32" env \
        VLLM_CPU_OMP_THREADS_BIND="0-$((THREADS-1))" \
        ${VLLM_TCMALLOC:+LD_PRELOAD="$VLLM_TCMALLOC"} \
        PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 \
        "$VLLM_PY" src/bench_gen_cpu.py --model-id "$MODEL_ID" \
        --n-samples "$N_SAMPLES" --threads "$THREADS" \
        "${DRY_GEN[@]}" --runtime vllm-cpu --precision fp32
    echo
else
    echo "[skip] vllm not installed in .venv-vllm (Qwen3 needs vllm>=0.10 / torch>=2.7; bootstrap with scripts/setup_vllm_venv.sh on Kunpeng aarch64)."
fi

qg_section "step ledger"
qg_ledger_print

qg_section "summary"
uv run python scripts/summarize_cpu.py
echo "[run_gen_cpu] done. Raw JSON under var/results/."
