#!/usr/bin/env bash
# Shared preamble for the qwen3guard-test run scripts.
# Source with: source "$(dirname "$0")/lib.sh"

[[ -n "${QG_LIB_LOADED:-}" ]] && return
QG_LIB_LOADED=1

qg_setup_env() {
    cd "$(dirname "$0")/.."
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
}

qg_export_thread_caps() {
    # Cap the underlying BLAS / OpenMP thread pools to the same N the bench
    # passes via --threads. Without these, PyTorch's set_num_threads(N) only
    # caps the intraop pool; MKL / OpenBLAS / OpenMP keep their defaults
    # (often = logical CPU count), so the actual core load drifts above N.
    local n="$1"
    export OMP_NUM_THREADS="$n"
    export MKL_NUM_THREADS="$n"
    export OPENBLAS_NUM_THREADS="$n"
    export VECLIB_MAXIMUM_THREADS="$n"
    export NUMEXPR_NUM_THREADS="$n"
    # Rust candle's CPU backend uses rayon for parallel ops; without this,
    # rayon defaults to num_cpus (= 24 on Kunpeng 920 / 24 vCPU), oversubscribing.
    export RAYON_NUM_THREADS="$n"
}

qg_detect_threads() {
    # Physical-core count, perf-cores only when the OS exposes the split.
    # Logical / SMT counts make benchmarks noisier without raising throughput
    # for this batch=1 prefill-bound workload.
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

qg_detect_device() {
    uv run python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo cpu
}

qg_section() {
    echo
    echo "######## $* ########"
    echo
}

LEDGER=()

qg_step() {
    local label="$1"
    shift
    if "$@"; then
        LEDGER+=("PASS  $label")
    else
        LEDGER+=("FAIL  $label")
    fi
}

qg_ledger_print() {
    for l in "${LEDGER[@]}"; do
        echo "  $l"
    done
}
