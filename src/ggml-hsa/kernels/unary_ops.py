#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry points for GGML unary operations."""

from pathlib import Path

from .kernel import Backend, KernelSpec, order_kernel_specs


def _make_iron_unary_kernel_spec(
    arch: str,
    input_tensors: list,
    output_tensor,
    op_name: str,
) -> KernelSpec:
    """Create an IRON-backend KernelSpec for a unary operation.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_name: Name of the GGML operation.

    Returns:
        KernelSpec configured for the IRON backend.

    """
    from functools import partial

    from .iron_kernels.unary_ops import unary_op

    return KernelSpec(
        backend=Backend.IRON,
        op_name=op_name,
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=partial(
            unary_op,
            op_name=op_name,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
        ),
    )


def _make_triton_relu_kernel_spec(
    arch: str,
    input_tensors: list,
    output_tensor,
) -> KernelSpec:
    """Create a TRITON-backend KernelSpec for RELU.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.

    Returns:
        KernelSpec configured for the TRITON backend.

    Raises:
        ValueError: If the tensors are non-contiguous (raised lazily when the
            returned compile function is invoked).
    """
    n_elements = output_tensor.numel()

    def _compile(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        n_elements=n_elements,
    ):
        # All imports, grid specialisation, and tensor creation are deferred into
        # _compile so that any failure is caught by the try/except in build.py.
        # IRON is the primary backend (tried first); this Triton spec is the
        # fallback, reached only if IRON compilation fails.
        import torch
        import triton

        from .triton_kernels.relu import relu
        from .triton_kernels.utils import numpy_dtype_to_torch, triton_device

        if any(not t.contiguous for t in (*input_tensors, output_tensor)):
            msg = "Non-contiguous tensors detected."
            raise ValueError(msg)

        block_size = 1 << (min(1024, n_elements) - 1).bit_length()
        block_size = min(block_size, 1024)
        grid = (triton.cdiv(n_elements, block_size),)
        device = triton_device(arch)
        x = torch.randn(
            n_elements,
            device=device,
            dtype=numpy_dtype_to_torch(input_tensors[0].dtype),
        )
        y = torch.empty(
            n_elements,
            device=device,
            dtype=numpy_dtype_to_torch(output_tensor.dtype),
        )
        return relu[grid](X=x, Y=y, n_elements=n_elements, BLOCK_SIZE_N=block_size)

    # The bf16 transform script (relu_{arch}.mlir) pads with a bf16 zero, which
    # aircc rejects for f32 tensors. Select an f32-padding variant for f32 inputs.
    import numpy as np

    script_stem = f"relu_{arch}"
    if np.dtype(output_tensor.dtype) == np.float32:
        script_stem += "_f32"

    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_UNARY_OP_RELU",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=_compile,
        config={
            "transform_script": str(
                Path(__file__).parent / "triton_kernels" / f"{script_stem}.mlir"
            ),
        },
    )


def ggml_op_sqr(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_SQR.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the SQR operation.

    """
    return _make_iron_unary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_OP_SQR"
    )


def ggml_op_sqrt(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_SQRT.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Raises:
        NotImplementedError: SQRT is not yet implemented.

    """
    raise NotImplementedError


def ggml_op_log(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_LOG.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the LOG operation.

    """
    return _make_iron_unary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_OP_LOG"
    )


def ggml_op_sin(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_SIN.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Raises:
        NotImplementedError: SIN is not yet implemented.

    """
    raise NotImplementedError


def ggml_op_cos(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_COS.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Raises:
        NotImplementedError: COS is not yet implemented.

    """
    raise NotImplementedError


def ggml_unary_op_abs(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_ABS.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the ABS operation.

    """
    return _make_iron_unary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_UNARY_OP_ABS"
    )


def ggml_unary_op_sgn(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_SGN.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the SGN operation.

    """
    return _make_iron_unary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_UNARY_OP_SGN"
    )


def ggml_unary_op_neg(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_NEG.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the NEG operation.

    """
    return _make_iron_unary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_UNARY_OP_NEG"
    )


def ggml_unary_op_step(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_STEP.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the STEP operation.

    """
    return _make_iron_unary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_UNARY_OP_STEP"
    )


def ggml_unary_op_tanh(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_TANH.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Raises:
        NotImplementedError: TANH is not yet implemented.

    """
    raise NotImplementedError


def ggml_unary_op_elu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_ELU.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Raises:
        NotImplementedError: ELU is not yet implemented.

    """
    raise NotImplementedError


def ggml_unary_op_relu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> list[KernelSpec]:
    """Return KernelSpecs for GGML_UNARY_OP_RELU (IRON primary, Triton fallback).

    IRON is tried first; the Triton spec is the fallback, reached only if IRON
    compilation fails. Set ``GGML_HSA_JIT_COMPILER_ORDER=triton,iron`` to flip the
    order so Triton is tried first (used to benchmark or exercise the Triton path).

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        List of KernelSpecs for the RELU operation: IRON first, then Triton as a
        fallback (reordered by CompilerConfig.compilers /
        ``GGML_HSA_JIT_COMPILER_ORDER``).

    """
    return [
        _make_iron_unary_kernel_spec(
            arch, input_tensors, output_tensor, "GGML_UNARY_OP_RELU"
        ),
        _make_triton_relu_kernel_spec(arch, input_tensors, output_tensor),
    ]


def ggml_unary_op_sigmoid(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_SIGMOID.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Raises:
        NotImplementedError: SIGMOID is not yet implemented.

    """
    raise NotImplementedError


def ggml_unary_op_gelu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_GELU.

    Uses the tanh approximation of GELU, matching GGML's GGML_UNARY_OP_GELU.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the GELU operation.

    """
    return _make_iron_unary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_UNARY_OP_GELU"
    )


def ggml_unary_op_gelu_quick(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_GELU_QUICK.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Raises:
        NotImplementedError: GELU_QUICK is not yet implemented.

    """
    raise NotImplementedError


def ggml_unary_op_silu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_SILU.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Raises:
        NotImplementedError: SILU is not yet implemented.

    """
    raise NotImplementedError


def ggml_unary_op_hardswish(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_HARDSWISH.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the HARDSWISH operation.

    """
    return _make_iron_unary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_UNARY_OP_HARDSWISH"
    )


def ggml_unary_op_hardsigmoid(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_HARDSIGMOID.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the HARDSIGMOID operation.

    """
    return _make_iron_unary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_UNARY_OP_HARDSIGMOID"
    )


def ggml_unary_op_exp(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_EXP.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Raises:
        NotImplementedError: EXP is not yet implemented.

    """
    raise NotImplementedError


def ggml_unary_op_gelu_erf(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_GELU_ERF.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Raises:
        NotImplementedError: GELU_ERF is not yet implemented.

    """
    raise NotImplementedError


def ggml_unary_op_xielu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_XIELU.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Raises:
        NotImplementedError: XIELU is not yet implemented.

    """
    raise NotImplementedError


def ggml_unary_op_floor(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_FLOOR.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the FLOOR operation.

    """
    return _make_iron_unary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_UNARY_OP_FLOOR"
    )


def ggml_unary_op_ceil(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_CEIL.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the CEIL operation.

    """
    return _make_iron_unary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_UNARY_OP_CEIL"
    )


def ggml_unary_op_round(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_ROUND.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the ROUND operation.

    """
    return _make_iron_unary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_UNARY_OP_ROUND"
    )


def ggml_unary_op_trunc(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_UNARY_OP_TRUNC.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the TRUNC operation.

    """
    return _make_iron_unary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_UNARY_OP_TRUNC"
    )
