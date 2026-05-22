#!/usr/bin/env bash
# Qwen3Guard-Gen CPU benchmark — full run. Two scripts, two tables:
#
#   scripts/run_gen_cpu.sh     every CPU runtime under one llama.cpp build
#                              (PyTorch / ONNX / OpenVINO / llama.cpp / Rust)
#   kernels/run_kernel_cpu.sh  llama.cpp alone, rebuilt under each BLAS backend
#                              (tinyblas / openblas / blis / mkl / kleidiai)
#
# Comparing BLAS backends needs a llama-cpp-python rebuild between each, so the
# BLAS sweep is a separate script from the single-build cross-runtime table.
#
# Usage:  bash run.sh [--dry-run]
#   --dry-run   smoke test — tiny bench counts; passed through to both scripts.

set -o pipefail
cd "$(dirname "$0")"

echo "############################################################"
echo "## 1/2  cross-runtime table — scripts/run_gen_cpu.sh"
echo "############################################################"
bash scripts/run_gen_cpu.sh "$@"

echo
echo "############################################################"
echo "## 2/2  BLAS-backend sweep — kernels/run_kernel_cpu.sh"
echo "############################################################"
bash kernels/run_kernel_cpu.sh "$@"

echo
echo "[run] both done in $((SECONDS / 60)) min. Save this whole stdout into linux_box_log/."
