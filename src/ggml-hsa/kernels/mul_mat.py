#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

from os import path

from aie.iron import dtype_to_str
from aie.extras.context import mlir_mod_ctx

from core_function import core_function, CoreFunctionInfo


def mul_mat_core_function_info(arch: str, input_tensors: list, output_tensor):
    """Returns a compilation specification for matrix multiplication."""

    m = 8
    n = 8
    k = 8
    use_scalar = False

    scalar_suffix = "_scalar" if use_scalar else ""
    current_dir = path.dirname(path.realpath(__file__))
    return CoreFunctionInfo(
        source_file=path.join(current_dir, arch, "mm.cc"),
        exported_function={
            "matmul": f"matmul{scalar_suffix}_{dtype_to_str(input_tensors[0].dtype)}_{dtype_to_str(output_tensor.dtype)}",
            "zero": f"zero{scalar_suffix}_{dtype_to_str(output_tensor.dtype)}",
        },
        compile_args=[
            f"-DDIM_M={m}",
            f"-DDIM_N={n}",
            f"-DDIM_K={k}",
            f"-D{dtype_to_str(input_tensors[0].dtype)}_{dtype_to_str(output_tensor.dtype)}_ONLY",
            "-DB_COL_MAJ",
            "-DC_COL_MAJ",
        ],
    )


def create_mul_mat_external_functions(
    arch: str,
    input_tensors: list,
    output_tensor,
):
    """
    Returns the parameters and names of the external functions for matrix multiplication.

    Args:
        arch (str): Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.

    Returns:
        A tuple containing:
            - m: The block size in the M dimension.
            - n: The block size in the N dimension.
            - k: The block size in the K dimension.
            - use_scalar: Boolean indicating if scalar multiplication is used.
            - mm_fn: The name of the matrix multiplication function.
            - zero_fn: The name of the zeroing function.
    """
    m = 8
    n = 8
    k = 8
    use_scalar = False
    scalar_suffix = "_scalar" if use_scalar else ""

    return (
        m,
        n,
        k,
        use_scalar,
        f"matmul{scalar_suffix}_{dtype_to_str(input_tensors[0].dtype)}_{dtype_to_str(output_tensor.dtype)}",
        f"zero{scalar_suffix}_{dtype_to_str(output_tensor.dtype)}",
    )


@core_function(mul_mat_core_function_info)
def ggml_op_mul_mat(
    arch: str,
    input_tensors: list,
    output_tensor,
    core_function_info: CoreFunctionInfo,
):
    mat_mul_fn = None
    dev = None
    if arch == "aie2":
        from aie2 import mul_mat

        mat_mul_fn = mul_mat.my_matmul
        dev = "npu"
    elif arch == "aie2p":
        dev = "npu2"

        raise ValueError(f"Not implemented for {arch}")
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    if len(input_tensors) != 2:
        raise ValueError("Requires two input tensors")

    A = input_tensors[0]  # MxK = A.shape(1) x A.shape(0)
    B = input_tensors[1]  # KxN = B.shape(0) x B.shape(1)
    C = output_tensor  # MxN = C.shape(0) x C.shape(1)

    if not A.contiguous or not B.contiguous or not C.contiguous:
        raise ValueError("Tensors must be contiguous")

    if A.shape[1] != C.shape[0]:
        raise ValueError(f"Incompatible M for A and C ({A.shape[1]} != {C.shape[0]})")

    if B.shape[1] != C.shape[1]:
        raise ValueError(f"Incompatible N for B and C ({B.shape[1]} != {C.shape[1]})")

    if A.shape[0] != B.shape[0]:
        raise ValueError(f"Incompatible K for A and B ({A.shape[0]} != {B.shape[0]})")

    m, n, k, use_scalar, mm_fn, zero_fn = create_mul_mat_external_functions(
        arch=arch, input_tensors=input_tensors, output_tensor=output_tensor
    )
    object_file = core_function_info.object_file

    with mlir_mod_ctx() as ctx:
        mat_mul_fn(
            dev=dev,
            M=A.shape[1],
            N=B.shape[1],
            K=A.shape[0],
            m=m,
            n=n,
            k=k,
            n_aie_cols=4,
            dtype_in_str=dtype_to_str(A.dtype),
            dtype_out_str=dtype_to_str(C.dtype),
            b_col_maj=True,
            c_col_maj=True,
            use_scalar=use_scalar,
            emulate_bf16_mmul_with_bfp16=False,
            trace_size=0,
            mm_fn=mm_fn,
            zero_fn=zero_fn,
            object_file=object_file,
        )
        return ctx.module
