#!/usr/bin/env bash
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Generates shape-specialized HIP elementwise kernels and compiles them to
# loadable AMDGPU code objects (.hsaco) for the GGML HSA GPU backend.
#
# The GGML HSA backend looks up code objects at:
#     ${GGML_HSA_KERNEL_DIR}/<device_name>/<kernel_symbol>.hsaco
# and resolves the kernel descriptor symbol "<kernel_symbol>.kd". The kernel
# symbol is the (sanitized) ggml kernel name, e.g. add_8f32_8f32_8f32.
#
# Usage:
#     generate_gpu_kernels.sh <arch> <out_dir> [names_file]
#
#   arch       AMDGPU target, e.g. gfx1151 (default: gfx1151)
#   out_dir    output kernel directory (the <arch> subdir is created inside)
#   names_file file with one kernel name per line; if omitted, names are read
#              from stdin. Only pure elementwise add_<N><dtype>_<N><dtype>_<N><dtype>
#              names are supported; others are skipped with a warning.
#
# Environment:
#   ROCM_PATH  ROCm installation (default: /opt/rocm)

set -euo pipefail

ARCH="${1:-gfx1151}"
OUT_DIR="${2:?usage: generate_gpu_kernels.sh <arch> <out_dir> [names_file]}"
NAMES_FILE="${3:-/dev/stdin}"

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${ROCM_PATH}/bin/hipcc"
DEVICE_CLANG="${ROCM_PATH}/llvm/bin/clang"

ARCH_DIR="${OUT_DIR}/${ARCH}"
mkdir -p "${ARCH_DIR}"

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

    # Expect: add_<N><dtype>_<N><dtype>_<N><dtype> with all three components equal.
    if [[ ! "${name}" =~ ^add_([0-9]+)(f32|f16)_([0-9]+)(f32|f16)_([0-9]+)(f32|f16)$ ]]; then
        echo "skip (unsupported name): ${name}" >&2
        continue
    fi
    if [ "${BASH_REMATCH[1]}" != "${BASH_REMATCH[3]}" ] || \
       [ "${BASH_REMATCH[1]}" != "${BASH_REMATCH[5]}" ] || \
       [ "${BASH_REMATCH[2]}" != "${BASH_REMATCH[4]}" ] || \
       [ "${BASH_REMATCH[2]}" != "${BASH_REMATCH[6]}" ]; then
        echo "skip (non-elementwise): ${name}" >&2
        continue
    fi

    ctype="$(dtype_to_ctype "${BASH_REMATCH[2]}")"
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
    cat > "${cu}" <<EOF
#include <hip/hip_runtime.h>

extern "C" __global__
void ${name}(const ${ctype} * A, const ${ctype} * B, ${ctype} * C, unsigned long long N) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
EOF

    "${HIPCC}" -S --cuda-device-only -Wno-unused-command-line-argument \
        --offload-arch="${ARCH}" "${cu}" -o "${asm}"
    "${DEVICE_CLANG}" -target amdgcn-amd-amdhsa -mcpu="${ARCH}" "${asm}" -o "${out}"
    echo "built ${out}"
done < "${NAMES_FILE}"
