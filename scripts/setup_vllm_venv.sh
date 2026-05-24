#!/usr/bin/env bash
# One-shot bootstrap for the .venv-vllm/ sibling environment.
#
# vLLM requires torch>=2.7 (current Linux aarch64 CPU wheels target torch
# 2.9.1+cpu and pull in setuptools>=77). The main .venv pins torch<2.7 to keep
# the GPU box's CUDA driver compatible, so vLLM lives in its own venv.
#
# Usage (on Kunpeng aarch64):
#   bash scripts/setup_vllm_venv.sh
#
# Idempotent: re-running rebuilds .venv-vllm/ from scratch.

set -o pipefail
cd "$(dirname "$0")/.."

# vLLM CPU aarch64 wheels are published at github releases under
# `vllm-project/vllm`. v0.13.0 is the oldest tag with an aarch64 cpu wheel
# (it targets torch==2.9.1+cpu); newer tags follow the same pattern.
VLLM_VERSION="${VLLM_VERSION:-0.13.0}"
WHEEL_NAME="vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_aarch64.whl"
WHEEL_URL_GH="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/${WHEEL_NAME}"
# China-friendly github mirror; falls back to direct github if it 404s.
WHEEL_URL_MIRROR="https://ghfast.top/${WHEEL_URL_GH}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
WHEEL_PATH="/tmp/${WHEEL_NAME}"

PY_VER="${PY_VER:-3.12}"
# Aliyun mirror gives ~5 MB/s from Huawei Cloud while pypi.org direct sits at
# ~30 KB/s; the install would take an hour otherwise. Override via env var if
# Aliyun is unreachable from your network.
PIP_MIRROR="${PIP_MIRROR:-https://mirrors.aliyun.com/pypi/simple}"
PIP_TRUSTED="${PIP_TRUSTED:-mirrors.aliyun.com}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

# tcmalloc reduces vLLM's eager-mode allocator pressure; install if missing.
if ! find /usr -name 'libtcmalloc_minimal.so.4' 2>/dev/null | grep -q .; then
    echo "[setup-vllm] installing libtcmalloc-minimal4 (apt)…"
    apt-get install -y libtcmalloc-minimal4 || \
        echo "[setup-vllm] tcmalloc install failed — vLLM will still run, just slower allocator path."
fi

# 1. fetch wheel (idempotent)
if [[ ! -s "$WHEEL_PATH" ]]; then
    echo "[setup-vllm] downloading $WHEEL_NAME via mirror…"
    if ! curl -L --max-time 300 -o "$WHEEL_PATH" "$WHEEL_URL_MIRROR" -w "\n  mirror: http=%{http_code} size=%{size_download}\n"; then
        echo "[setup-vllm] mirror failed, retry direct github…"
        curl -L --max-time 600 -o "$WHEEL_PATH" "$WHEEL_URL_GH" -w "\n  github: http=%{http_code} size=%{size_download}\n"
    fi
fi
[[ -s "$WHEEL_PATH" ]] || { echo "[fatal] wheel download failed: $WHEEL_PATH"; exit 1; }

# 2. fresh venv (uv venv is fast and consistent with the main .venv)
echo "[setup-vllm] creating .venv-vllm/ (Python $PY_VER)…"
rm -rf .venv-vllm
uv venv --python "$PY_VER" .venv-vllm

# 3. torch first to pin the cpu wheel. Use plain pip (not uv pip) because uv
# was observed to deadlock on the heavy multi-wheel unpack phase under the
# default io_uring concurrency on this aarch64 box.
echo "[setup-vllm] installing pip + torch==${TORCH_VERSION}+cpu…"
uv pip install --python .venv-vllm/bin/python pip
.venv-vllm/bin/python -m pip install --no-cache-dir \
    --index-url "$PIP_MIRROR" --trusted-host "$PIP_TRUSTED" \
    --extra-index-url "$TORCH_INDEX" \
    "torch==${TORCH_VERSION}"

# 4. vllm wheel + remaining deps from the Aliyun mirror (the PyPI default
# resolves to Fastly = ~30 KB/s from Huawei Cloud Kunpeng).
echo "[setup-vllm] installing vllm + deps via Aliyun mirror…"
.venv-vllm/bin/python -m pip install --no-cache-dir \
    --index-url "$PIP_MIRROR" --trusted-host "$PIP_TRUSTED" \
    "$WHEEL_PATH"

# 5. sanity check
echo "[setup-vllm] sanity check…"
.venv-vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
echo "[setup-vllm] done."
