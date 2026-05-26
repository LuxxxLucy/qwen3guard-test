#!/usr/bin/env bash
# Cross-impl trick-correctness gate. One verify call per row of the
# bench matrix. Reference = PyTorch fp32 L0 verdict (argmax) and
# L1 logits (max abs diff). fp32 rows: strict; fp16/quant rows: argmax-only
# tripwire (>=80%). Exits 0 if every row passes its gate; non-zero otherwise.
#
# Usage:
#   bash scripts/verify_correctness.sh
#
# Pre-reqs:
#   - uv sync has run (artifact + reference models live in HF cache)
#   - python -m exporters.<runtime> has produced the exported artifacts
#   - src/backends/rust/target/release/qwen3guard-bench is built
#   - scripts/dump_rust_inputs.py has written src/backends/rust/bench_inputs.json

set -o pipefail
source "$(dirname "$0")/lib.sh"
qg_setup_env

# --model-id accepts a HF repo id (default) or a local directory; BASENAME
# stays as the model's short name and drives the export-artifact paths.
MODEL_ID="Qwen/Qwen3Guard-Gen-0.6B"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id) MODEL_ID="$2"; shift 2 ;;
        --model-id=*) MODEL_ID="${1#--model-id=}"; shift ;;
        *) echo "[verify_correctness] unknown argument: $1" >&2; exit 2 ;;
    esac
done
BASENAME="$(basename "$MODEL_ID")"

verify() {
    qg_step "verify $*" uv run python src/verify_lm_head.py --model-id "$MODEL_ID" "$@"
    echo
}

# verify_llama_kopt: scope the +kernel-opt env (OpenMP OpenBLAS via LD_PRELOAD,
# Neoverse-N2 sgemm kernel) to the kernel-opt verify rows only. See
# scripts/run_gen_cpu.sh gen_llama_kopt for the rationale. LD_PRELOAD silently
# skips on hosts without libopenblas0-openmp installed.
verify_llama_kopt() {
    local preload=
    for cand in /usr/lib/aarch64-linux-gnu/openblas-openmp/libopenblas.so.0 \
                /usr/lib/aarch64-linux-gnu/openblas-openmp/libopenblas.so; do
        [[ -e "$cand" ]] && { preload="$cand"; break; }
    done
    qg_step "verify $*" env \
        ${preload:+LD_PRELOAD="$preload"} \
        OPENBLAS_CORETYPE=NEOVERSEN2 \
        uv run python src/verify_lm_head.py --model-id "$MODEL_ID" "$@"
    echo
}

# vLLM lives in a sibling .venv-vllm (see scripts/setup_vllm_venv.sh). Its
# verify call must run under that interpreter, not the main uv venv.
verify_vllm() {
    if [[ ! -x .venv-vllm/bin/python ]]; then
        echo "[skip] verify vllm-cpu: .venv-vllm not built (run scripts/setup_vllm_venv.sh)."
        return 0
    fi
    if ! .venv-vllm/bin/python -c "import vllm" 2>/dev/null; then
        echo "[skip] verify vllm-cpu: import vllm failed in .venv-vllm."
        return 0
    fi
    local threads
    threads="$(qg_detect_threads)"
    [[ "$threads" -gt 16 ]] && threads=16
    local tcmalloc
    tcmalloc="$(find /usr -name 'libtcmalloc_minimal.so.4' 2>/dev/null | head -1)"
    qg_step "verify $*" env \
        VLLM_CPU_OMP_THREADS_BIND="0-$((threads-1))" \
        ${tcmalloc:+LD_PRELOAD="$tcmalloc"} \
        PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 \
        .venv-vllm/bin/python src/verify_lm_head.py --model-id "$MODEL_ID" "$@"
    echo
}

qg_section "cross-impl correctness gate (every row vs PyTorch fp32 L0)"

# pytorch fp32 -- full ladder L0 / L1 / L2 / L3
verify --runtime pytorch --precision fp32 --opt-level L0
verify --runtime pytorch --precision fp32 --opt-level L1
verify --runtime pytorch --precision fp32 --opt-level L2
verify --runtime pytorch --precision fp32 --opt-level L3

# onnx fp32 -- full L0/+L1/+L2/+L3 ladder. L0/+L1 on fp32/ (no slice), +L2/+L3 on
# fp32-lastpos/ (lm_head sliced). Strict gate (atol=1e-2 + argmax).
verify --runtime onnx --precision fp32 --opt-level L0 \
    --artifact "var/models/onnx/$BASENAME/fp32"
verify --runtime onnx --precision fp32 --opt-level L1 \
    --artifact "var/models/onnx/$BASENAME/fp32"
verify --runtime onnx --precision fp32 --opt-level L2 \
    --artifact "var/models/onnx/$BASENAME/fp32-lastpos"
verify --runtime onnx --precision fp32 --opt-level L3 \
    --artifact "var/models/onnx/$BASENAME/fp32-lastpos"

# onnx int8 -- same ladder against the dynamic-INT8 graphs. Tripwire gate.
verify --runtime onnx --precision int8 --opt-level L0 \
    --artifact "var/models/onnx/$BASENAME/int8"
verify --runtime onnx --precision int8 --opt-level L1 \
    --artifact "var/models/onnx/$BASENAME/int8"
verify --runtime onnx --precision int8 --opt-level L2 \
    --artifact "var/models/onnx/$BASENAME/int8-lastpos"
verify --runtime onnx --precision int8 --opt-level L3 \
    --artifact "var/models/onnx/$BASENAME/int8-lastpos"

# llamacpp f32 -- full ladder L0 / L1 / L3 (L2 baked)
verify --runtime llamacpp --precision f32  --opt-level L0 \
    --artifact "var/models/gguf/$BASENAME.f32.gguf"
verify --runtime llamacpp --precision f32  --opt-level L1 \
    --artifact "var/models/gguf/$BASENAME.f32.gguf"
verify --runtime llamacpp --precision f32  --opt-level L3 \
    --artifact "var/models/gguf/$BASENAME.f32.gguf"
# llamacpp f32 +kernel-opt -- same f32 GGUF, scoped LD_PRELOAD + NEOVERSEN2.
verify_llama_kopt --runtime llamacpp --precision f32-kopt --opt-level L0 \
    --artifact "var/models/gguf/$BASENAME.f32.gguf"
verify_llama_kopt --runtime llamacpp --precision f32-kopt --opt-level L1 \
    --artifact "var/models/gguf/$BASENAME.f32.gguf"
verify_llama_kopt --runtime llamacpp --precision f32-kopt --opt-level L3 \
    --artifact "var/models/gguf/$BASENAME.f32.gguf"
# llamacpp f16 -- full ladder L0 / L1 / L3 (L2 baked)
verify --runtime llamacpp --precision f16  --opt-level L0 \
    --artifact "var/models/gguf/$BASENAME.f16.gguf"
verify --runtime llamacpp --precision f16  --opt-level L1 \
    --artifact "var/models/gguf/$BASENAME.f16.gguf"
verify --runtime llamacpp --precision f16  --opt-level L3 \
    --artifact "var/models/gguf/$BASENAME.f16.gguf"
# llamacpp q8_0 -- full ladder L0 / L1 / L3 (L2 baked)
verify --runtime llamacpp --precision q8_0 --opt-level L0 \
    --artifact "var/models/gguf/$BASENAME.q8_0.gguf"
verify --runtime llamacpp --precision q8_0 --opt-level L1 \
    --artifact "var/models/gguf/$BASENAME.q8_0.gguf"
verify --runtime llamacpp --precision q8_0 --opt-level L3 \
    --artifact "var/models/gguf/$BASENAME.q8_0.gguf"

# mnn-llm -- fp16 weights (MNN-LLM has no fp32 weight path), runtime
# precision="high" so accumulators stay fp32. Tripwire gate (fp16 weights are
# expected to drift logits by O(1) on borderline samples).
if uv run python -c "import MNN.llm" 2>/dev/null \
   && [[ -d "var/models/mnn/$BASENAME-fp16" ]]; then
    verify --runtime mnn --precision fp16 --opt-level L0 \
        --artifact "var/models/mnn/$BASENAME-fp16"
    verify --runtime mnn --precision fp16 --opt-level L1 \
        --artifact "var/models/mnn/$BASENAME-fp16"
else
    echo "[skip] mnn verify (MNN.llm not importable or model dir missing)."
fi

# vllm-cpu -- L1 (forced-prefix). vLLM exposes top-K logprobs (monotone in the
# raw logits) rather than the raw lm_head row, so the gate is argmax-only.
verify_vllm --runtime vllm-cpu --precision fp32 --opt-level L1

# rust-candle -- dump L0/L1/L3 once, verify each from the same JSON.
qg_step "rust verify dump" src/backends/rust/target/release/qwen3guard-bench \
    --input src/backends/rust/bench_inputs.json --verify-out src/backends/rust/verify_logits.json
echo
verify --runtime rust-candle --opt-level L0 \
    --logits-json src/backends/rust/verify_logits.json \
    --rust-inputs src/backends/rust/bench_inputs.json
verify --runtime rust-candle --opt-level L1 \
    --logits-json src/backends/rust/verify_logits.json \
    --rust-inputs src/backends/rust/bench_inputs.json
verify --runtime rust-candle --opt-level L3 \
    --logits-json src/backends/rust/verify_logits.json \
    --rust-inputs src/backends/rust/bench_inputs.json

qg_section "correctness ledger"
qg_ledger_print

# Exit non-zero if any verify failed.
for entry in "${LEDGER[@]}"; do
    [[ "$entry" == FAIL* ]] && exit 1
done
echo
echo "[verify_correctness] all rows passed their gate."
exit 0
