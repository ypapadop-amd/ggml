#!/usr/bin/env bash
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Builds the standalone GPU .hsaco dispatch reference (test-gpu-hsaco-dispatch.cpp).
#
# This is a minimal, ggml-independent HSA program that loads a HIP-compiled
# elementwise add code object and dispatches it on a GPU agent, deriving the
# kernarg layout from the code object metadata via comgr (the same technique the
# ggml-hsa GPU backend uses). It is a developer reference / sanity check, not part
# of the automated test build.
#
# Usage:
#   ./compile-gpu-hsaco-dispatch.sh                 # build the binary
#   ./test-gpu-hsaco-dispatch <N>                   # run for vector size N
#
# At run time it expects a matching code object "add_<N>f32_<N>f32_<N>f32.hsaco"
# in the current directory (or ./<agent_name>/). Generate one with, e.g.:
#   echo add_8f32_8f32_8f32 | \
#     ROCM_PATH=/opt/rocm ../../src/ggml-hsa/gpu-kernels/generate_gpu_kernels.sh gfx1151 - "$PWD"
#   ln -sf gfx1151/add_8f32_8f32_8f32.hsaco .   # or run from the gfx1151/ dir
#
# Environment:
#   ROCM_PATH  ROCm installation (default: /opt/rocm)

set -euo pipefail

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${ROCM_PATH}/llvm/bin/clang++" \
    "${SCRIPT_DIR}/test-gpu-hsaco-dispatch.cpp" \
    -o "${SCRIPT_DIR}/test-gpu-hsaco-dispatch" \
    -I"${ROCM_PATH}/include" -L"${ROCM_PATH}/lib" \
    -lhsa-runtime64 -lamd_comgr \
    -O2 -std=c++17

echo "built ${SCRIPT_DIR}/test-gpu-hsaco-dispatch"
