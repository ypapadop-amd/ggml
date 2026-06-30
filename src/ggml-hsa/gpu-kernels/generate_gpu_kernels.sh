#!/usr/bin/env bash
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Generates shape-specialized HIP elementwise kernels and compiles them to
# loadable AMDGPU code objects (.hsaco) for the GGML HSA GPU backend.
#
# The GGML HSA backend searches for code objects in (in order):
#     1. ${GGML_HSA_KERNEL_DIR}/<device_name>/<kernel_symbol>.hsaco   (pregenerated)
#     2. <cache_dir>/<device_name>/<kernel_symbol>.hsaco             (cache)
# and resolves the kernel descriptor symbol "<kernel_symbol>.kd". The kernel
# symbol is the (sanitized) ggml kernel name, e.g. add_8f32_8f32_8f32.
#
# By default this script writes into the same cache directory the backend uses
# at runtime (mirroring the AIE JIT cache), so generated kernels are found
# automatically without setting GGML_HSA_KERNEL_DIR and without depending on the
# CMake build folder. The cache directory is resolved exactly as the backend
# does: GGML_HSA_KERNEL_CACHE_DIR, else $XDG_CACHE_HOME/ggml, else
# $HOME/.cache/ggml, else /tmp/ggml/ggml-hsa.
#
# Usage:
#     generate_gpu_kernels.sh <arch> [names_file] [out_dir]
#
#   arch       AMDGPU target, e.g. gfx1151 (default: gfx1151)
#   names_file file with one kernel name per line, or "-" for stdin (default: -)
#   out_dir    output kernel directory (the <arch> subdir is created inside);
#              defaults to the runtime cache directory described above
#
# Only elementwise add_* and scale_* kernel names are supported; others are
# skipped with a warning.
#
# Environment:
#   ROCM_PATH  ROCm installation (default: /opt/rocm)

set -euo pipefail

# Resolves the runtime kernel cache directory, matching ggml_hsa_cached_kernel_dir().
default_cache_dir() {
    if [ -n "${GGML_HSA_KERNEL_CACHE_DIR:-}" ]; then
        echo "${GGML_HSA_KERNEL_CACHE_DIR}"
    elif [ -n "${XDG_CACHE_HOME:-}" ]; then
        echo "${XDG_CACHE_HOME}/ggml"
    elif [ -n "${HOME:-}" ]; then
        echo "${HOME}/.cache/ggml"
    else
        echo "/tmp/ggml/ggml-hsa"
    fi
}

ARCH="${1:-gfx1151}"
NAMES_ARG="${2:--}"
OUT_DIR="${3:-$(default_cache_dir)}"
NAMES_FILE="${NAMES_ARG}"
if [ "${NAMES_FILE}" = "-" ]; then
    NAMES_FILE="/dev/stdin"
fi

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${ROCM_PATH}/bin/hipcc"
DEVICE_CLANG="${ROCM_PATH}/llvm/bin/clang"

ARCH_DIR="${OUT_DIR}/${ARCH}"
mkdir -p "${ARCH_DIR}"
echo "generating ${ARCH} kernels into ${ARCH_DIR}" >&2

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

# Maps a ggml dtype tag to a C scalar type.
dtype_to_ctype() {
    case "$1" in
        f32) echo "float" ;;
        f16) echo "_Float16" ;;
        *)   echo "" ;;
    esac
}

while IFS= read -r name; do
    name="$(echo "${name}" | tr -d '[:space:]')"
    [ -z "${name}" ] && continue

    body=""

    # add: elementwise binary, C[i] = A[i] + B[i].
    # Name: add_<N><dtype>_<N><dtype>_<N><dtype> with all three components equal.
    if [[ "${name}" =~ ^add_([0-9]+)(f32|f16)_([0-9]+)(f32|f16)_([0-9]+)(f32|f16)$ ]] && \
       [ "${BASH_REMATCH[1]}" = "${BASH_REMATCH[3]}" ] && \
       [ "${BASH_REMATCH[1]}" = "${BASH_REMATCH[5]}" ] && \
       [ "${BASH_REMATCH[2]}" = "${BASH_REMATCH[4]}" ] && \
       [ "${BASH_REMATCH[2]}" = "${BASH_REMATCH[6]}" ]; then
        ctype="$(dtype_to_ctype "${BASH_REMATCH[2]}")"
        body="extern \"C\" __global__
void ${name}(const ${ctype} * A, const ${ctype} * B, ${ctype} * C, unsigned long long N) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}"

    # scale: elementwise unary with two scalars, C[i] = A[i] * scale + bias.
    # Name: scale_<N><dtype>_<N><dtype>_<op_params_hash> (dst and src shapes equal).
    elif [[ "${name}" =~ ^scale_([0-9]+)(f32|f16)_([0-9]+)(f32|f16)_[0-9a-f]+$ ]] && \
         [ "${BASH_REMATCH[1]}" = "${BASH_REMATCH[3]}" ] && \
         [ "${BASH_REMATCH[2]}" = "${BASH_REMATCH[4]}" ]; then
        ctype="$(dtype_to_ctype "${BASH_REMATCH[2]}")"
        body="extern \"C\" __global__
void ${name}(const ${ctype} * A, ${ctype} * C, unsigned long long N, float scale, float bias) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = (${ctype})((float)A[i] * scale + bias);
    }
}"
    fi

    if [ -z "${body}" ]; then
        echo "skip (unsupported name): ${name}" >&2
        continue
    fi
    if [ -z "${ctype}" ]; then
        echo "skip (unknown dtype): ${name}" >&2
        continue
    fi

    out="${ARCH_DIR}/${name}.hsaco"
    if [ -f "${out}" ]; then
        continue
    fi

    cu="${WORK_DIR}/${name}.cu"
    asm="${WORK_DIR}/${name}.s"
    printf '#include <hip/hip_runtime.h>\n\n%s\n' "${body}" > "${cu}"

    "${HIPCC}" -S --cuda-device-only -Wno-unused-command-line-argument \
        --offload-arch="${ARCH}" "${cu}" -o "${asm}"
    "${DEVICE_CLANG}" -target amdgcn-amd-amdhsa -mcpu="${ARCH}" "${asm}" -o "${out}"
    echo "built ${out}"
done < "${NAMES_FILE}"
