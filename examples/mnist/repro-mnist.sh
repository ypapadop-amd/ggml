#!/usr/bin/env bash
#
# Reproduce the MNIST FC (784->500->10) eval numbers on this box.
#
# Runs the mnist-eval binary over the 10000 MNIST test images and prints the
# key summary lines (accuracy, loss, timing) with the noisy per-image ASCII
# image dump stripped out.
#
# Usage:
#   ./repro-mnist.sh [npu|cpu|both]     # default: both
#
# Environment overrides:
#   BUILD_DIR   build dir with bin/mnist-eval (default: build-release)
#   MODEL       gguf model file             (default: mnist-fc-f32.gguf)
#   REGEN=1     regenerate model + download data before running
#
# Notes:
#   - HIP_VISIBLE_DEVICES=-1 hides the gfx1150 iGPU from the ROCm backend,
#     which otherwise SIGSEGVs during scheduler alloc (see repo memory).
#   - The repo-root .venv is activated so the HSA AIE-kernel JIT (embedded
#     pybind11 interpreter) can find the IRON toolchain. First NPU run
#     JIT-compiles kernels into $XDG_CACHE_HOME/ggml/aie2p (slow); later
#     runs reuse the cache.
#   - Expected: test_acc ~98.17%, NPU ~35 us/image, CPU ~2.4 us/image.

set -euo pipefail

# Resolve paths relative to this script so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-build-release}"
MODEL="${MODEL:-mnist-fc-f32.gguf}"
TARGET="${1:-both}"

EVAL_BIN="${REPO_ROOT}/${BUILD_DIR}/bin/mnist-eval"
IMAGES="${SCRIPT_DIR}/data/MNIST/raw/t10k-images-idx3-ubyte"
LABELS="${SCRIPT_DIR}/data/MNIST/raw/t10k-labels-idx1-ubyte"

cd "${SCRIPT_DIR}"

# Optionally regenerate the model + download the MNIST dataset.
if [[ "${REGEN:-0}" == "1" ]]; then
    echo ">>> Regenerating model and data (PyTorch CPU env)..."
    "${SCRIPT_DIR}/venv/bin/python" mnist-train-fc.py "${MODEL}"
fi

# Sanity checks.
[[ -x "${EVAL_BIN}" ]] || { echo "ERROR: ${EVAL_BIN} not found. Build it first (BUILD_DIR=${BUILD_DIR})." >&2; exit 1; }
[[ -f "${MODEL}"    ]] || { echo "ERROR: model ${MODEL} not found. Run with REGEN=1 to create it." >&2; exit 1; }
[[ -f "${IMAGES}"   ]] || { echo "ERROR: test images not found. Run with REGEN=1 to download the dataset." >&2; exit 1; }
[[ -f "${LABELS}"   ]] || { echo "ERROR: test labels not found. Run with REGEN=1 to download the dataset." >&2; exit 1; }

# Activate the IRON/HSA JIT env for the NPU path.
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

run_eval() {
    local label="$1" dev="$2"
    echo
    echo "======================================================================"
    echo ">>> ${label} (primary backend: ${dev})"
    echo "======================================================================"
    # Strip the ANSI-colored per-image ASCII dump; keep the summary lines.
    HIP_VISIBLE_DEVICES=-1 "${EVAL_BIN}" "${MODEL}" "${IMAGES}" "${LABELS}" "${dev}" 2>&1 \
        | grep -avE $'\e\\[48;2' \
        | grep -aE "loaded model|predicted digit|test_loss|test_acc|us/image|as primary|fallback"
}

case "${TARGET}" in
    npu)  run_eval "NPU"          "HSA0" ;;
    cpu)  run_eval "CPU"          "CPU"  ;;
    both) run_eval "NPU"          "HSA0"
          run_eval "CPU"          "CPU"  ;;
    *)    echo "Usage: $0 [npu|cpu|both]" >&2; exit 1 ;;
esac

echo
echo ">>> Done."
