#!/usr/bin/env bash
# Qwen3Guard-Gen CPU kernel comparison — BLAS-backend sweep.
#
# Rebuilds llama-cpp-python once per BLAS backend and benchmarks each
# llama.cpp cell (q8_0 / f16, L2 / +kv) under it. Comparing backends
# needs a rebuild between each — they are different compiled extensions, so
# this is a separate script from scripts/run_gen_cpu.sh (one build, every
# runtime). Run on each Linux box; capture all stdout.
#
# Usage:  bash kernels/run_kernel_cpu.sh [--dry-run]
#   --dry-run   smoke test — the tinyblas config only, tiny bench counts.
#
# Prep — install the BLAS dev libs first, or that config's build is recorded
# FAIL in the ledger and the run continues with the rest:
#   Debian/Ubuntu:  apt install libopenblas-dev libblis-dev
#   x86 MKL:        apt install intel-mkl

# No `set -u`: expanding an empty bash array under nounset errors on old bash.
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

MODEL_ID="Qwen/Qwen3Guard-Gen-0.6B"
BASENAME="$(basename "$MODEL_ID")"
THREADS=16          # user-set. On the 5800X (8 cores / 16 SMT) this is SMT-on.
N_SAMPLES=100
ARCH="$(uname -m)"

# A GGML_BLAS=ON build runs the GEMM on the BLAS library's own thread pool, not
# ggml's --threads. Pin every library's pool so all configs use $THREADS.
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       MKL_NUM_THREADS="$THREADS" BLIS_NUM_THREADS="$THREADS"

DRY_RUN=
DRY=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1; DRY=(--dry-run) ;;
        *) echo "[run_kernel_cpu] unknown argument: $arg" >&2; exit 2 ;;
    esac
done

section() { echo; echo "######## $* ########"; echo; }
LEDGER=()
step() {  # step "label" cmd args...
    local label="$1"; shift
    if "$@"; then LEDGER+=("PASS  $label"); else LEDGER+=("FAIL  $label"); fi
}

echo "[run_kernel_cpu] host=$(uname -srm)  arch=$ARCH  threads=$THREADS"
[[ -n "$DRY_RUN" ]] && echo "[run_kernel_cpu] DRY RUN — tinyblas config only, latency numbers are not meaningful."

# -DGGML_NATIVE=ON: build for the host CPU (AVX2/FMA on x86, dotprod on Arm).
# Without it a from-source build can fall back to a portable, slow SIMD baseline.
BASE_CMAKE="-DGGML_NATIVE=ON"

# config: name | arch filter | CMAKE_ARGS
CONFIGS=(
    "tinyblas|all|-DGGML_BLAS=OFF"
    "openblas|all|-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
    "blis|all|-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME"
    "mkl|x86_64|-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp"
    "kleidiai|aarch64|-DGGML_BLAS=OFF -DGGML_CPU_KLEIDIAI=ON"
)
[[ -n "$DRY_RUN" ]] && CONFIGS=("${CONFIGS[0]}")

section "download model + Qwen3GuardTest dataset"
step "download" uv run python scripts/download.py --variants gen --sizes 0.6B

section "export GGUF (f16, q8_0)"
step "export gguf" uv run python scripts/export_gen_gguf.py \
    --model-id "$MODEL_ID" --quants f16 q8_0

F16="gguf_models/$BASENAME.f16.gguf"
Q8="gguf_models/$BASENAME.q8_0.gguf"

# One llama.cpp build per BLAS backend; benchmark every llama.cpp cell under it.
# Results go to kernels/profile/<backend>/ — summarize_kernel.py pivots them.
bench() {  # bench <backend> <label> <bench_gen_cpu.py args...>
    local backend="$1" label="$2"; shift 2
    echo "--- $backend / $label ---"
    step "$backend  $label" uv run python src/bench_gen_cpu.py \
        --model-id "$MODEL_ID" --n-samples "$N_SAMPLES" --threads "$THREADS" \
        --verify --out-dir "kernels/profile/$backend" "${DRY[@]}" "$@"
    echo
}

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name afilter cmake <<< "$cfg"
    if [[ "$afilter" != "all" && "$afilter" != "$ARCH" ]]; then
        echo "[skip] $name — $afilter only (host is $ARCH)"
        continue
    fi
    section "config $name   CMAKE_ARGS=$BASE_CMAKE $cmake"
    export CMAKE_ARGS="$BASE_CMAKE $cmake"
    # uv keys its build cache by sdist hash, not CMAKE_ARGS — without this
    # eviction every config reuses the first wheel and the sweep is void.
    uv cache clean llama-cpp-python
    if ! uv sync --reinstall-package llama-cpp-python; then
        echo "[run_kernel_cpu] $name build failed — the BLAS dev lib for this"
        echo "                 vendor is likely not installed (see prep notes)."
        LEDGER+=("FAIL  $name (build)")
        continue
    fi
    # Gate: the build must link the BLAS backend it claims, else it measured
    # the wrong kernel. This is the check the cached-wheel bug slipped past.
    if ! uv run python kernels/build_probe.py "$name"; then
        echo "[run_kernel_cpu] $name — build identity mismatch; skipping."
        LEDGER+=("FAIL  $name (build identity)")
        continue
    fi
    mkdir -p "kernels/profile/$name"
    bench "$name" "q8_0 L2"  --runtime llamacpp --precision q8_0 --artifact "$Q8"
    bench "$name" "q8_0 +kv" --runtime llamacpp --precision q8_0 --artifact "$Q8" --kv-cache
    bench "$name" "f16 L2"   --runtime llamacpp --precision f16  --artifact "$F16"
    bench "$name" "f16 +kv"  --runtime llamacpp --precision f16  --artifact "$F16" --kv-cache
done

section "step ledger"
for line in "${LEDGER[@]}"; do echo "  $line"; done

section "summary — llama.cpp cell x BLAS backend"
uv run python kernels/summarize_kernel.py

echo
echo "[run_kernel_cpu] done. Save this whole stdout into linux_box_log/."
