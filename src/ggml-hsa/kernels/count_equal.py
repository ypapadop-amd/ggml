#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML count equal operation (GGML_OP_COUNT_EQUAL)."""

from .kernel import Backend, KernelSpec


def _make_triton_gpu_count_equal_kernel_spec(
    arch: str, input_tensors: list, output_tensor
) -> KernelSpec:
    """Create a TRITON-backend (GPU AOT) KernelSpec for COUNT_EQUAL.

    Counts the elementwise-equal entries between two same-shaped, contiguous I32 tensors with a
    single Triton program that loops over the flattened elements in BLOCK-sized tiles (element and
    tile counts baked as constexprs), storing the total into an I64 scalar. Only the contiguous I32
    case is supported; other cases raise ValueError so the op is reported unsupported.
    """
    import numpy as np

    from .triton_kernels.gpu_ops import count_equal_kernel, next_pow2

    src0, src1 = input_tensors[0], input_tensors[1]
    i32 = np.dtype(np.int32)
    if src0.dtype != i32 or src1.dtype != i32:
        msg = "Triton GPU COUNT_EQUAL supports I32 inputs only."
        raise ValueError(msg)
    if not src0.contiguous or not src1.contiguous:
        msg = "Triton GPU COUNT_EQUAL requires contiguous inputs."
        raise ValueError(msg)
    if src0.numel() != src1.numel():
        msg = "Triton GPU COUNT_EQUAL requires inputs with matching element counts."
        raise ValueError(msg)

    n = int(src0.numel())
    block = min(next_pow2(n), 1024)
    num_tiles = (n + block - 1) // block

    signature = {
        "S0": "*i32",
        "S1": "*i32",
        "DST": "*i64",
        "N": "constexpr",
        "NUM_TILES": "constexpr",
        "BLOCK": "constexpr",
    }
    config = {
        "triton_fn": count_equal_kernel,
        "signature": signature,
        "constexprs": {"N": n, "NUM_TILES": int(num_tiles), "BLOCK": int(block)},
        "num_warps": 4,
        "num_programs": 1,
    }
    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_COUNT_EQUAL",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=count_equal_kernel,
        config=config,
    )


def ggml_op_count_equal(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_COUNT_EQUAL.

    Counts elementwise-equal entries between two same-shaped I32 tensors.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two contiguous I32 tensors.
        output_tensor: I64 scalar [1, 1, 1, 1] holding the count.
        op_params: Unused.

    Returns:
        KernelSpec for the COUNT_EQUAL operation.

    """
    from functools import partial

    from .triton_kernels.utils import is_gpu_arch

    if is_gpu_arch(arch):
        return _make_triton_gpu_count_equal_kernel_spec(
            arch, input_tensors, output_tensor
        )

    from .iron_kernels.count_equal import count_equal_op

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_COUNT_EQUAL",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=partial(
            count_equal_op,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
        ),
    )
