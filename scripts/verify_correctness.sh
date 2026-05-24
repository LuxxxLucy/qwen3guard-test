#!/usr/bin/env bash
# Cross-impl trick-correctness gate. One verify call per row of the
# CPU_GEN_REPORT table. Reference = PyTorch fp32 L0 verdict (argmax) and
# L1 logits (max abs diff). fp32 rows: strict; fp16/quant rows: argmax-only
# tripwire (>=80%). Exits 0 if every row passes its gate; non-zero otherwise.
#
# Usage:
#   bash scripts/verify_correctness.sh
#
# Pre-reqs:
#   - uv sync has run (artifact + reference models live in HF cache)
#   - scripts/export_gen_*.py have produced the exported artifacts
#   - rust/target/release/qwen3guard-bench is built
#   - scripts/dump_rust_inputs.py has written rust/bench_inputs.json

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

qg_section "cross-impl correctness gate (every row vs PyTorch fp32 L0)"

# pytorch fp32 -- L0 (model-card), L1 (forced-prefix), L2 (lastpos lm_head)
verify --runtime pytorch --precision fp32 --opt-level L0
verify --runtime pytorch --precision fp32 --opt-level L1
verify --runtime pytorch --precision fp32 --opt-level L2

# onnx fp32 -- L1 (no-past artifact); L0 + L3 (with-past artifact)
verify --runtime onnx --precision fp32 --opt-level L1 \
    --artifact "onnx_models/$BASENAME/fp32"
verify --runtime onnx --precision fp32 --opt-level L0 \
    --artifact "onnx_models/$BASENAME/withpast"
verify --runtime onnx --precision fp32 --opt-level L3 \
    --artifact "onnx_models/$BASENAME/withpast"

# onnx int8 -- L1 only (tripwire gate)
verify --runtime onnx --precision int8 --opt-level L1 \
    --artifact "onnx_models/$BASENAME/int8"

# onnx-genai -- only if importable (no Linux aarch64 wheel)
if uv run python -c "import onnxruntime_genai" 2>/dev/null; then
    verify --runtime onnx-genai --precision fp32 --opt-level L0 \
        --artifact "ortgenai_models/$BASENAME/fp32"
    verify --runtime onnx-genai --precision fp32 --opt-level L1 \
        --artifact "ortgenai_models/$BASENAME/fp32"
else
    echo "[skip] onnx-genai not importable; rows omitted from gate."
fi

# openvino -- fp16 (L0 + L1) and int8 (L1). Bench-time compile may fail
# under colima/Apple-Silicon oneDNN; that surfaces as a verify FAIL here.
verify --runtime openvino --precision fp16 --opt-level L0 \
    --artifact "ov_models/$BASENAME/fp16"
verify --runtime openvino --precision fp16 --opt-level L1 \
    --artifact "ov_models/$BASENAME/fp16"
verify --runtime openvino --precision int8 --opt-level L1 \
    --artifact "ov_models/$BASENAME/int8"

# llamacpp -- q8_0 (L0 + L1 + L3) and f16 (L1 + L3)
verify --runtime llamacpp --precision q8_0 --opt-level L0 \
    --artifact "gguf_models/$BASENAME.q8_0.gguf"
verify --runtime llamacpp --precision q8_0 --opt-level L1 \
    --artifact "gguf_models/$BASENAME.q8_0.gguf"
verify --runtime llamacpp --precision q8_0 --opt-level L3 \
    --artifact "gguf_models/$BASENAME.q8_0.gguf"
verify --runtime llamacpp --precision f16  --opt-level L1 \
    --artifact "gguf_models/$BASENAME.f16.gguf"
verify --runtime llamacpp --precision f16  --opt-level L3 \
    --artifact "gguf_models/$BASENAME.f16.gguf"
verify --runtime llamacpp --precision f32  --opt-level L1 \
    --artifact "gguf_models/$BASENAME.f32.gguf"
verify --runtime llamacpp --precision f32  --opt-level L3 \
    --artifact "gguf_models/$BASENAME.f32.gguf"
verify --runtime llamacpp --precision q4_K_M --opt-level L1 \
    --artifact "gguf_models/$BASENAME.q4_K_M.gguf"
verify --runtime llamacpp --precision q4_K_M --opt-level L3 \
    --artifact "gguf_models/$BASENAME.q4_K_M.gguf"

# rust-candle -- dump L0/L1/L3 once, verify each from the same JSON.
qg_step "rust verify dump" rust/target/release/qwen3guard-bench \
    --input rust/bench_inputs.json --verify-out rust/verify_logits.json
echo
verify --runtime rust-candle --opt-level L0 \
    --logits-json rust/verify_logits.json \
    --rust-inputs rust/bench_inputs.json
verify --runtime rust-candle --opt-level L1 \
    --logits-json rust/verify_logits.json \
    --rust-inputs rust/bench_inputs.json
verify --runtime rust-candle --opt-level L3 \
    --logits-json rust/verify_logits.json \
    --rust-inputs rust/bench_inputs.json

qg_section "correctness ledger"
qg_ledger_print

# Exit non-zero if any verify failed.
for entry in "${LEDGER[@]}"; do
    [[ "$entry" == FAIL* ]] && exit 1
done
echo
echo "[verify_correctness] all rows passed their gate."
exit 0
