#!/usr/bin/env bash
#
# Reproduce the MNIST FC (784->500->10) eval numbers on this box.
#
# Runs the mnist-eval binary over the 10000 MNIST test images and prints the
# key summary lines (accuracy, loss, timing) with the noisy per-image ASCII
# image dump stripped out. Also writes a JSON + markdown result file per
# sub-run, named after the hardware architecture actually running it (e.g.
# results-cpu-znver4.json, results-gpu-gfx1150.json, results-npu-aie2p.json),
# so results from different machines/architectures don't overwrite each other.
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
#   OUTDIR      where result JSON + markdown files are written (default: script dir)
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
OUTDIR="${OUTDIR:-${SCRIPT_DIR}}"

EVAL_BIN="${REPO_ROOT}/${BUILD_DIR}/bin/mnist-eval"
IMAGES="${SCRIPT_DIR}/data/MNIST/raw/t10k-images-idx3-ubyte"
LABELS="${SCRIPT_DIR}/data/MNIST/raw/t10k-labels-idx1-ubyte"

# Detect the hardware architecture actually running each target, so result
# files are named after it instead of just the generic target.
arch_for() {
    case "$1" in
        cpu) gcc -march=native -Q --help=target 2>/dev/null | awk '$1=="-march="{print $2}' ;;
        gpu) rocm_agent_enumerator 2>/dev/null | awk 'NR==1' ;;
        npu) rocminfo 2>/dev/null | grep -oE 'Name:\s+aie2p?' | awk 'NR==1{print $2}' ;;
    esac || true
}

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
    local target="$1" label="$2" dev="$3" hipvis="$4"
    local arch; arch="$(arch_for "${target}")"; arch="${arch:-unknown}"
    echo
    echo "======================================================================"
    echo ">>> ${label} (primary backend: ${dev}, arch: ${arch}) — ${RUNS} run(s)"
    echo "======================================================================"

    local times=() runs_json=() acc="" loss="" out us
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
        runs_json+=("{\"run\":${i},\"us_per_image\":${us:-null},\"test_acc\":${acc:-null},\"test_loss\":${loss:-null}}")
        # Optionally drop the first (warmup) run from the timing summary.
        if [[ ${WARMUP} -eq 1 && ${i} -eq 1 && ${RUNS} -gt 1 ]]; then
            continue
        fi
        times+=("${us}")
    done

    local min max mean
    if [[ ${RUNS} -gt 1 ]]; then
        read -r min mean max < <(printf '%s\n' "${times[@]}" | awk '
            NR==1 { min=max=sum=$1; next }
            { sum+=$1; if ($1<min) min=$1; if ($1>max) max=$1 }
            END { printf "%.6f %.6f %.6f\n", min, sum/NR, max }')
        printf '  ------------------------------------------------------------\n'
        printf '  us/image over %d run(s)%s: min=%.2f  mean=%.2f  max=%.2f\n' \
            "${#times[@]}" "$([[ ${WARMUP} -eq 1 ]] && echo ' (warmup dropped)')" "${min}" "${mean}" "${max}"
        printf '  final test_acc=%s%%, test_loss=%s\n' "${acc}" "${loss}"
    else
        min="${times[0]}"; mean="${times[0]}"; max="${times[0]}"
    fi

    # Write a JSON + markdown result file per sub-run, named after the
    # detected architecture (not just the generic cpu/gpu/npu target), so
    # results from different machines/architectures don't overwrite each other.
    local stem="${OUTDIR}/results-${target}-${arch}"
    local runs_arr; runs_arr="[$(IFS=,; echo "${runs_json[*]}")]"
    jq -n \
        --arg target "${target}" --arg arch "${arch}" --arg label "${label}" --arg device "${dev}" \
        --argjson runs "${runs_arr}" --argjson warmup "$([[ ${WARMUP} -eq 1 ]] && echo true || echo false)" \
        --arg min "${min}" --arg mean "${mean}" --arg max "${max}" \
        --arg final_acc "${acc}" --arg final_loss "${loss}" \
        '{target: $target, arch: $arch, label: $label, device: $device, warmup_dropped: $warmup,
          runs: $runs,
          summary: {
            min_us_per_image: ($min | tonumber),
            mean_us_per_image: ($mean | tonumber),
            max_us_per_image: ($max | tonumber),
            final_test_acc: (if $final_acc == "" then null else ($final_acc | tonumber) end),
            final_test_loss: (if $final_loss == "" then null else ($final_loss | tonumber) end)
          }}' \
        > "${stem}.json"

    {
        echo "# mnist-eval — ${label} (${arch})"
        echo
        echo "\`mnist-eval\` on the 10000-image MNIST test set (FC 784->500->10), primary backend \`${dev}\`."
        echo
        echo "| | |"
        echo "|---|---|"
        echo "| Architecture | ${arch} |"
        echo "| Runs | ${RUNS}$([[ ${WARMUP} -eq 1 && ${RUNS} -gt 1 ]] && echo ' (first run/warmup dropped from timing)') |"
        printf '| us/image (min / mean / max) | %.2f / %.2f / %.2f |\n' "${min}" "${mean}" "${max}"
        echo "| test_acc | ${acc}% |"
        echo "| test_loss | ${loss} |"
    } > "${stem}.md"

    echo "  wrote ${stem}.json"
    echo "  wrote ${stem}.md"
}

# run_eval <target> <label> <primary-device> <HIP_VISIBLE_DEVICES value>
#   -1  = hide the iGPU from ROCm (cpu/npu); 0 = expose ROCm0 (gpu)
case "${TARGET}" in
    npu)  run_eval "npu" "NPU"       "HSA0"  "-1" ;;
    cpu)  run_eval "cpu" "CPU"       "CPU"   "-1" ;;
    gpu)  run_eval "gpu" "GPU (HIP)" "ROCm0" "0"  ;;
    both) run_eval "npu" "NPU"       "HSA0"  "-1"
          run_eval "cpu" "CPU"       "CPU"   "-1" ;;
    *)    echo "Usage: $0 [npu|cpu|gpu|both]" >&2; exit 1 ;;
esac

echo
echo ">>> Done."
