#!/usr/bin/env bash
#
# Reproduce the MNIST FC (784->500->10) eval numbers on this box.
#
# Runs the mnist-eval binary over the 10000 MNIST test images and prints the
# key summary lines (accuracy, loss, timing) with the noisy per-image ASCII
# image dump stripped out.
#
# Usage:
#   ./repro-mnist.sh [npu|cpu|gpu|both]     # default: both (npu+cpu)
#
# Environment overrides:
#   BUILD_DIR   build dir with bin/mnist-eval (default: build-release)
#   MODEL       gguf model file             (default: mnist-fc-f32.gguf)
#   RUNS        number of eval invocations per target (default: 1). When >1,
#               per-run us/image is printed and a min/mean/max summary is shown.
#   WARMUP=1    discard the first run from the timing summary (useful on the
#               NPU where the first run may JIT-compile kernels)
#   REGEN=1     regenerate model + download data before running
#
# Targets / primary backend device:
#   npu -> HSA0    cpu -> CPU    gpu -> ROCm0 (HIP)
#
# Notes:
#   - For the cpu/npu targets, HIP_VISIBLE_DEVICES=-1 hides the gfx1150 iGPU
#     from the ROCm backend, which otherwise SIGSEGVs when synchronized as a
#     *fallback* during scheduler alloc (see repo memory). The gpu target
#     leaves HIP visible (ROCm0 as primary works fine).
#   - The repo-root .venv is activated so the HSA AIE-kernel JIT (embedded
#     pybind11 interpreter) can find the IRON toolchain. First NPU run
#     JIT-compiles kernels into $XDG_CACHE_HOME/ggml/aie2p (slow); later
#     runs reuse the cache.
#   - Expected: test_acc ~98.17%, NPU ~36 us/image, CPU ~2.6 us/image,
#     GPU ~5.8 us/image. (Device availability depends on the build.)

set -euo pipefail

# Resolve paths relative to this script so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-build-release}"
MODEL="${MODEL:-mnist-fc-f32.gguf}"
RUNS="${RUNS:-1}"
WARMUP="${WARMUP:-0}"
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

# Run one eval invocation; echo the filtered summary lines so the caller can
# aggregate. For non-GPU targets the iGPU is hidden from the ROCm backend to
# avoid the fallback-synchronize crash; the gpu target needs it visible.
run_once() {
    local dev="$1" hipvis="$2"
    HIP_VISIBLE_DEVICES="${hipvis}" "${EVAL_BIN}" "${MODEL}" "${IMAGES}" "${LABELS}" "${dev}" 2>&1 \
        | grep -avE $'\e\\[48;2' \
        | grep -aE "test_loss|test_acc|us/image"
}

run_eval() {
    local label="$1" dev="$2" hipvis="$3"
    echo
    echo "======================================================================"
    echo ">>> ${label} (primary backend: ${dev}) — ${RUNS} run(s)"
    echo "======================================================================"

    local times=() acc="" loss="" out us
    for ((i = 1; i <= RUNS; i++)); do
        out="$(run_once "${dev}" "${hipvis}")"
        us="$(sed -n 's/.*, \([0-9.]*\) us\/image/\1/p' <<<"${out}")"
        acc="$(sed -n 's/.*test_acc=\([0-9.]*\).*/\1/p' <<<"${out}")"
        loss="$(sed -n 's/.*test_loss=\([0-9.]*\).*/\1/p' <<<"${out}")"
        if [[ ${RUNS} -gt 1 ]]; then
            printf '  run %2d/%d: %8s us/image  (test_acc=%s%%)\n' "${i}" "${RUNS}" "${us}" "${acc}"
        else
            printf '  %s us/image, test_acc=%s%%, test_loss=%s\n' "${us}" "${acc}" "${loss}"
        fi
        # Optionally drop the first (warmup) run from the timing summary.
        if [[ ${WARMUP} -eq 1 && ${i} -eq 1 && ${RUNS} -gt 1 ]]; then
            continue
        fi
        times+=("${us}")
    done

    if [[ ${RUNS} -gt 1 ]]; then
        printf '%s\n' "${times[@]}" | awk -v w="${WARMUP}" '
            NR==1 { min=max=sum=$1; next }
            { sum+=$1; if ($1<min) min=$1; if ($1>max) max=$1 }
            END {
                printf "  ------------------------------------------------------------\n"
                printf "  us/image over %d run(s)%s: min=%.2f  mean=%.2f  max=%.2f\n",
                       NR, (w==1 ? " (warmup dropped)" : ""), min, sum/NR, max
            }'
        printf '  final test_acc=%s%%, test_loss=%s\n' "${acc}" "${loss}"
    fi
}

# run_eval <label> <primary-device> <HIP_VISIBLE_DEVICES value>
#   -1  = hide the iGPU from ROCm (cpu/npu); 0 = expose ROCm0 (gpu)
case "${TARGET}" in
    npu)  run_eval "NPU"          "HSA0"  "-1" ;;
    cpu)  run_eval "CPU"          "CPU"   "-1" ;;
    gpu)  run_eval "GPU (HIP)"    "ROCm0" "0"  ;;
    both) run_eval "NPU"          "HSA0"  "-1"
          run_eval "CPU"          "CPU"   "-1" ;;
    *)    echo "Usage: $0 [npu|cpu|gpu|both]" >&2; exit 1 ;;
esac

echo
echo ">>> Done."
