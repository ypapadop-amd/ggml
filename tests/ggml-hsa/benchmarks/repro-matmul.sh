#!/usr/bin/env bash
# Reproduce the bench-mul-mat-hsa matmul benchmark for one backend and emit both
# a Google Benchmark JSON file and a markdown report.
#
# The benchmark binary contains CPU, GPU (HIP via the CUDA path) and HSA (NPU)
# registrations; we isolate one backend per invocation with --benchmark_filter.
#
# Output files are named after the hardware architecture actually running the
# benchmark (detected at run time), not just the generic target: e.g.
# results-cpu-znver4.json, results-gpu-gfx1150.json, results-npu-aie2p.json.
#
# Usage:
#   ./repro-matmul.sh cpu|gpu|npu
#
# Env vars:
#   BUILD_DIR    build directory containing the benchmark  (default: build-bench-<target>)
#   REPS         --benchmark_repetitions                    (default: 10)
#   MIN_TIME     per-benchmark min wall time, e.g. 0.5s     (default: 0.5s)
#   OUTDIR       where JSON + reports are written           (default: script dir)
set -euo pipefail

TARGET="${1:-}"
case "${TARGET}" in
    cpu|gpu|npu) ;;
    *) echo "Usage: $0 [cpu|gpu|npu]" >&2; exit 1 ;;
esac

# resolve paths relative to this script so it works from any cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# HIP and HSA must never be compiled into the same binary, so each backend uses
# its own isolated build directory by default.
BUILD_DIR="${BUILD_DIR:-build-bench-${TARGET}}"
REPS="${REPS:-10}"
MIN_TIME="${MIN_TIME:-0.5s}"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}}"

# locate the benchmark binary inside the build tree (bin/ is where ggml puts it,
# but fall back to the source-mirrored path just in case)
BENCH_BIN=""
for cand in \
    "${REPO_ROOT}/${BUILD_DIR}/bin/bench-mul-mat-hsa" \
    "${REPO_ROOT}/${BUILD_DIR}/tests/ggml-hsa/benchmarks/bench-mul-mat-hsa"; do
    if [[ -x "${cand}" ]]; then BENCH_BIN="${cand}"; break; fi
done
if [[ -z "${BENCH_BIN}" ]]; then
    echo "error: bench-mul-mat-hsa binary not found under ${REPO_ROOT}/${BUILD_DIR}" >&2
    echo "       build it with: cmake --build ${BUILD_DIR} --target bench-mul-mat-hsa" >&2
    exit 1
fi

# detect the hardware architecture actually running the benchmark, so the
# output files are named after it instead of just the generic target
case "${TARGET}" in
    cpu) ARCH="$(gcc -march=native -Q --help=target 2>/dev/null | awk '$1=="-march="{print $2}')" || true ;;
    # Detect the *physical* GPU arch: strip HSA_OVERRIDE_GFX_VERSION so the file
    # is named after the real hardware (e.g. gfx1103), not an override target
    # (e.g. gfx1150) used to borrow another arch's rocBLAS TensileLibrary.
    gpu) ARCH="$(env -u HSA_OVERRIDE_GFX_VERSION rocm_agent_enumerator 2>/dev/null | awk 'NR==1')" || true ;;
    npu) ARCH="$(rocminfo 2>/dev/null | grep -oE 'Name:\s+aie2p?' | awk 'NR==1{print $2}')" || true ;;
esac
ARCH="${ARCH:-unknown}"

# map target -> (google-benchmark filter regex, HIP visibility, output stem)
case "${TARGET}" in
    cpu) FILTER="BackendType::CPU"; HIPVIS="-1"; STEM="results-cpu-${ARCH}" ;;
    gpu) FILTER="BackendType::GPU"; HIPVIS="0";  STEM="results-gpu-${ARCH}" ;;
    npu) FILTER="BackendType::HSA"; HIPVIS="-1"; STEM="results-npu-${ARCH}" ;;
esac

# GGML_HSA_MUL_MAT_PREFER_TRITON flips the NPU kernel to the Triton path; tag the
# output so it doesn't overwrite the default (IRON) NPU results.
if [[ "${TARGET}" == "npu" && "${GGML_HSA_MUL_MAT_PREFER_TRITON:-0}" == "1" ]]; then
    STEM="${STEM}-triton"
fi

JSON="${OUTDIR}/${STEM}.json"
REPORT="${OUTDIR}/${STEM}.md"

# the NPU JIT toolchain lives in the repo-root .venv (IRON / mlir_aie)
if [[ "${TARGET}" == "npu" && -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

echo "==> ${TARGET}: ${BENCH_BIN}"
echo "    filter=${FILTER}  reps=${REPS}  min_time=${MIN_TIME}  HIP_VISIBLE_DEVICES=${HIPVIS}"
echo "    json=${JSON}"

# The HSA backend currently segfaults during process teardown *after* all
# results are written, so don't let a nonzero exit abort the report step: check
# the JSON was produced instead.
rc=0
HIP_VISIBLE_DEVICES="${HIPVIS}" "${BENCH_BIN}" \
    --benchmark_filter="${FILTER}" \
    --benchmark_repetitions="${REPS}" \
    --benchmark_min_time="${MIN_TIME}" \
    --benchmark_report_aggregates_only=false \
    --benchmark_format=json \
    --benchmark_out="${JSON}" || rc=$?

if [[ ! -s "${JSON}" ]]; then
    echo "error: benchmark produced no JSON output (exit ${rc})" >&2
    exit "${rc:-1}"
fi
if [[ "${rc}" -ne 0 ]]; then
    echo "warning: benchmark exited with code ${rc} (likely HSA teardown crash);" \
         "JSON was written, continuing to report generation" >&2
fi

# generate the markdown report from the JSON
case "${TARGET}" in
    gpu) LABEL="GPU (HIP, ${ARCH})" ;;
    npu) LABEL="NPU (HSA, ${ARCH})" ;;
    cpu) LABEL="CPU (${ARCH})" ;;
esac
python3 "${SCRIPT_DIR}/report_benchmarks.py" "${JSON}" -o "${REPORT}" --title "${LABEL}"

echo "==> done: ${JSON}"
echo "         ${REPORT}"
