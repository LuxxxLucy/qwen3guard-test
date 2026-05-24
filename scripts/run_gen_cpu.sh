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

qg_section "1/10 uv sync (llama-cpp-python built from source, +kopt3 patches on aarch64)"
# Rebuild llama-cpp-python from source every run: the prebuilt wheel is generic
# (no AVX2 / ARM dot-product) and benchmarks 5-7x slow. uv keys its build cache
# by source-dist hash, not by CMAKE_ARGS, so `uv cache clean` evicts the cached
# wheel and the rebuild honors current CMAKE_ARGS.
if [[ "$(uname)" == "Darwin" ]]; then
    export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple -DGGML_NATIVE=ON"
else
    export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_NATIVE=ON -DGGML_CPU_KLEIDIAI=ON"
fi
uv cache clean llama-cpp-python
uv sync --reinstall-package llama-cpp-python \
    || { echo "[fatal] uv sync failed — fix the environment and retry."; exit 1; }
# Note: scripts/patch_llamacpp_aarch64.py contains source-level SVE/NEON
# patches for ggml-cpu (RMSNorm fused, fp32<->fp16 SIMD). Validated in R7.kernel
# but showed no measurable speedup — the dominant cost is OpenBLAS sgemm, not
# the patched ops. Auto-apply during build is disabled because the wheel-cache
# eviction step it requires occasionally produces a malformed wheel on Kunpeng.
# To apply manually: uv run python scripts/patch_llamacpp_aarch64.py then
# uv cache clean llama-cpp-python && uv sync --reinstall-package llama-cpp-python.
qg_step "uv sync" true

qg_section "2/10 download weights + Qwen3GuardTest dataset"
if [[ -d "$MODEL_ID" ]]; then
    qg_step "download (model local, dataset only)" uv run python scripts/download.py \
        --variants gen --sizes 0.6B --skip-model
else
    qg_step "download" uv run python scripts/download.py --variants gen --sizes 0.6B
fi

qg_section "3/10 export ONNX (fp32, int8, with-past)"
qg_step "export onnx" uv run python scripts/export_gen_onnx.py \
    --model-id "$MODEL_ID" --precisions fp32 int8 --with-past

qg_section "4/10 export ONNX Runtime GenAI (fp32, prune_lm_head)"
# onnxruntime-genai has no Linux aarch64 wheel (Mac arm64, Linux x86_64, and
# Windows only). The dep is platform-gated in pyproject.toml; both the export
# and bench cells gate on `import onnxruntime_genai` and skip cleanly.
if uv run python -c "import onnxruntime_genai" 2>/dev/null; then
    qg_step "export onnx-genai" uv run python scripts/export_gen_onnx_genai.py \
        --model-id "$MODEL_ID" --precisions fp32
else
    echo "[skip] onnxruntime-genai not importable on this host (no Linux aarch64 wheel)."
fi

qg_section "5/10 export OpenVINO (fp16, int8)"
qg_step "export openvino" uv run python scripts/export_gen_openvino.py \
    --model-id "$MODEL_ID" --precisions fp16 int8

qg_section "6/10 export GGUF (f32, f16, q8_0, q4_K_M)"
qg_step "export gguf" uv run python scripts/export_gen_gguf.py \
    --model-id "$MODEL_ID" --quants f32 f16 q8_0 q4_K_M

qg_section "6b/10 export CTranslate2 (fp32)"
qg_step "export ctranslate2" uv run python scripts/export_gen_ctranslate2.py \
    --model-id "$MODEL_ID" --precisions fp32

qg_section "6c/10 export MNN-LLM (fp16)"
# MNN-LLM's converter needs the alibaba/MNN source tree (transformers/llm/export
# /llmexport.py). $MNN_HOME points at a local clone. MNN tops out at fp16
# weights for the language path (1/2/4/8-bit otherwise) — there is no fp32
# weight export. precision="high" at runtime keeps accumulators in fp32.
# Skipped if pymnn isn't installed or $MNN_HOME isn't set.
if uv run python -c "import MNN.llm" 2>/dev/null; then
    if [[ -n "${MNN_HOME:-}" && -d "$MNN_HOME" ]]; then
        qg_step "export mnn" uv run python scripts/export_gen_mnn.py \
            --model-id "$MODEL_ID" --precisions fp16 --mnn-home "$MNN_HOME"
    else
        echo "[skip] MNN_HOME not set or not a directory; skipping MNN export."
    fi
else
    echo "[skip] MNN.llm not importable on this host (pymnn not installed)."
fi

qg_section "7/10 build Rust candle backend"
qg_step "cargo build" bash -c "cd rust && cargo build --release"

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
# gen_llama_kopt2: f32 +kernel-opt round 2. Same OpenBLAS-OMP + NEOVERSEN2
# base as gen_llama_kopt, with four extra knobs layered on:
#  - OPENBLAS_BLOCK_FACTOR=2: halves the GEBP block size; smaller A-panels
#    fit cleaner into the 64 KB L1d on each Kunpeng core.
#  - OMP_PROC_BIND=spread + OMP_PLACES=cores: pins the 16 OpenMP threads to
#    16 distinct cores in spread order so libgomp does not migrate them and
#    so two threads never share an L1d set.
#  - OPENBLAS_THREAD_TIMEOUT=0: keeps the OpenBLAS worker threads parked
#    across consecutive sgemm calls instead of re-spawning per call.
# Net effect on Kunpeng 920 (24 physical cores, 16 used) is within
# measurement noise of kopt — see plan.md "R7 wall" for the cost stack.
# The cell exists so the table publishes the post-R6 wall measurement
# independently of the kopt row.
gen_llama_kopt2() {
    local preload=
    for cand in /usr/lib/aarch64-linux-gnu/openblas-openmp/libopenblas.so.0 \
                /usr/lib/aarch64-linux-gnu/openblas-openmp/libopenblas.so; do
        [[ -e "$cand" ]] && { preload="$cand"; break; }
    done
    qg_step "gen $*" env \
        ${preload:+LD_PRELOAD="$preload"} \
        OPENBLAS_CORETYPE=NEOVERSEN2 \
        OPENBLAS_BLOCK_FACTOR=2 \
        OPENBLAS_THREAD_TIMEOUT=0 \
        OMP_PROC_BIND=spread \
        OMP_PLACES=cores \
        uv run python src/bench_gen_cpu.py --model-id "$MODEL_ID" \
        --n-samples "$N_SAMPLES" --threads "$THREADS" \
        "${DRY_GEN[@]}" "$@"
    echo
}
# gen_llama_kopt3: f32 +kernel-opt round 3. Same OpenBLAS-OMP + NEOVERSEN2 env
# as kopt; the win comes from C/C++-source patches to ggml-cpu applied by
# scripts/patch_llamacpp_aarch64.py at build time. The patches add SVE/NEON
# fast paths to three previously-scalar (or x86-only) hot functions:
#   - ggml_compute_forward_rms_norm_f32 (ops.cpp): sum-of-squares now via
#     ggml_vec_dot_f32 (SVE 8x-unroll), fused y = x*scale*w loop now SVE/NEON.
#   - ggml_cpu_fp32_to_fp16 (ggml-cpu.c): SVE svcvt_f16_f32 fast path.
#   - ggml_cpu_fp16_to_fp32 (ggml-cpu.c): SVE svcvt_f32_f16 fast path.
# Targets the ~15% non-sgemm tail in the post-R6 profile (60% sgemm wall is
# untouchable without writing a custom Neoverse-N2 sgemm). Realistic gain
# bounded by Amdahl: cutting half off the 15% tail = 7.5% overall.
gen_llama_kopt3() {
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
# ONNX L0 baseline: hand-rolled decode loop over the with-past graph — same
# per-call KV behaviour as generate(use_cache=True) on the other backends.
gen --runtime onnx     --precision fp32  --artifact "onnx_models/$BASENAME/withpast" --unoptimized
# onnx-genai (microsoft/onnxruntime-genai). Built with prune_lm_head=true, so
# L2 (lastpos) is baked into both the L0 decode loop and the L1 forced-prefix
# row. Cross-call prefix-KV reuse is not in the GenAI Generator API; no L3.
if uv run python -c "import onnxruntime_genai" 2>/dev/null; then
    gen --runtime onnx-genai --precision fp32 --artifact "ortgenai_models/$BASENAME/fp32"
    gen --runtime onnx-genai --precision fp32 --artifact "ortgenai_models/$BASENAME/fp32" --unoptimized
else
    echo "[skip] onnxruntime-genai not importable; dashed row in summary."
fi
gen --runtime openvino --precision fp16  --artifact "ov_models/$BASENAME/fp16"
gen --runtime openvino --precision int8  --artifact "ov_models/$BASENAME/int8"
gen --runtime openvino --precision fp16  --artifact "ov_models/$BASENAME/fp16" --unoptimized
gen_llama --runtime llamacpp --precision q8_0   --artifact "gguf_models/$BASENAME.q8_0.gguf"
# llama.cpp again with the shared system-prompt prefix KV-cached.
gen_llama --runtime llamacpp --precision q8_0   --artifact "gguf_models/$BASENAME.q8_0.gguf"   --kv-cache
gen_llama --runtime llamacpp --precision q8_0   --artifact "gguf_models/$BASENAME.q8_0.gguf"   --unoptimized
# f16 GGUF — with a BLAS backend its prefill GEMM is the fastest CPU path.
gen_llama --runtime llamacpp --precision f16    --artifact "gguf_models/$BASENAME.f16.gguf"
gen_llama --runtime llamacpp --precision f16    --artifact "gguf_models/$BASENAME.f16.gguf"    --kv-cache
# f32 GGUF — uncompressed baseline. Slowest llama.cpp row but anchors the
# precision ladder against pytorch/onnx fp32.
gen_llama --runtime llamacpp --precision f32    --artifact "gguf_models/$BASENAME.f32.gguf"
gen_llama --runtime llamacpp --precision f32    --artifact "gguf_models/$BASENAME.f32.gguf"    --kv-cache
# f32 +kernel-opt — same f32 GGUF, kernel-level swap (see gen_llama_kopt).
# Closes the gap vs ONNX Runtime fp32 by removing the OpenBLAS-pthread /
# libgomp oversubscription and routing sgemm through the Neoverse-N2 kernel.
gen_llama_kopt --runtime llamacpp --precision f32-kopt --artifact "gguf_models/$BASENAME.f32.gguf"
gen_llama_kopt --runtime llamacpp --precision f32-kopt --artifact "gguf_models/$BASENAME.f32.gguf" --kv-cache
# f32 +kernel-opt round 2 — adds GEBP block tuning, OMP thread pinning, and
# OpenBLAS thread-keep-alive on top of the kopt env. Within-noise of kopt on
# Kunpeng 920: the post-kopt cost stack is 60% sgemm_kernel_NEOVERSEN2 +
# 10% ggml_vec_dot_f16 + 6% sgemm_incopy, all in already-tuned aarch64
# kernels with no further f32 headroom available. See gen_llama_kopt2.
gen_llama_kopt2 --runtime llamacpp --precision f32-kopt2 --artifact "gguf_models/$BASENAME.f32.gguf"
gen_llama_kopt2 --runtime llamacpp --precision f32-kopt2 --artifact "gguf_models/$BASENAME.f32.gguf" --kv-cache
# f32 +kernel-opt round 3 — same env as kopt, gain comes from the ggml-cpu
# C/C++-source patches applied at build time (see gen_llama_kopt3 above and
# scripts/patch_llamacpp_aarch64.py).
gen_llama_kopt3 --runtime llamacpp --precision f32-kopt3 --artifact "gguf_models/$BASENAME.f32.gguf"
gen_llama_kopt3 --runtime llamacpp --precision f32-kopt3 --artifact "gguf_models/$BASENAME.f32.gguf" --kv-cache
# q4_K_M GGUF — K-quant 4-bit, the recommended "small but accurate" path.
# Uses ggml's K-quant matmul kernels (KleidiAI int4/int8 dispatch on aarch64).
gen_llama --runtime llamacpp --precision q4_K_M --artifact "gguf_models/$BASENAME.q4_K_M.gguf"
gen_llama --runtime llamacpp --precision q4_K_M --artifact "gguf_models/$BASENAME.q4_K_M.gguf" --kv-cache

echo "--- Gen: rust-candle ---"
qg_step "gen rust-candle" rust/target/release/qwen3guard-bench \
    --input rust/bench_inputs.json --out-dir results "${DRY_RUST[@]}"
echo

# CTranslate2 CPU: single fp32 row, all tricks baked in (forward_batch's
# last-position slice gives L2 baked + L1 forced-prefix).
gen --runtime ctranslate2 --precision fp32 --artifact "ct2_models/$BASENAME-fp32"

# MNN-LLM CPU: Alibaba's Arm-tuned LLM runtime. `forward(input_ids)` returns
# last-position logits by default (L2 lastpos baked); L1 is one forced-prefix
# forward. fp16 weights + precision="high" (fp32 accumulators) is the
# highest-precision MNN-LLM path. Skipped when pymnn isn't installed or the
# converted model dir is missing.
if uv run python -c "import MNN.llm" 2>/dev/null \
   && [[ -d "mnn_models/$BASENAME-fp16" ]]; then
    gen --runtime mnn --precision fp16 --artifact "mnn_models/$BASENAME-fp16"
    gen --runtime mnn --precision fp16 --artifact "mnn_models/$BASENAME-fp16" --unoptimized
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
echo "[run_gen_cpu] done. Raw JSON under results/."
