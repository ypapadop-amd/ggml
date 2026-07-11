#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry points for GGML binary operations."""

from pathlib import Path

from .kernel import Backend, KernelSpec


def _validate_binary_inputs(input_tensors: list) -> None:
    """Validate that a binary operation has exactly two input tensors.

    Parameters:
        input_tensors: List of input tensors to validate.

    Raises:
        ValueError: If the number of input tensors is not exactly two.

    """
    if len(input_tensors) != 2:
        msg = f"Operation requires exactly two input tensors, got {len(input_tensors)}."
        raise ValueError(msg)


def _numpy_to_triton_elt(dtype) -> str:
    """Map a numpy dtype to a Triton pointer element type string (e.g. 'fp32').

    Parameters:
        dtype: The numpy dtype to map.

    Raises:
        ValueError: If the dtype has no Triton element-type equivalent.

    """
    import numpy as np

    name = np.dtype(dtype).name
    mapping = {
        "float32": "fp32",
        "float16": "fp16",
        "bfloat16": "bf16",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
    }
    try:
        return mapping[name]
    except KeyError:
        msg = f"Unsupported dtype for Triton GPU kernel: {name}"
        raise ValueError(msg) from None


def _make_iron_binary_kernel_spec(
    arch: str,
    input_tensors: list,
    output_tensor,
    op_name: str,
) -> KernelSpec:
    """Create an IRON-backend KernelSpec for a binary operation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_name: Name of the GGML operation.

    Returns:
        KernelSpec configured for the IRON backend.

    """
    from functools import partial

    from .iron_kernels.binary_ops import binary_op

    return KernelSpec(
        backend=Backend.IRON,
        op_name=op_name,
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=partial(
            binary_op,
            op_name=op_name,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
        ),
    )


def _make_triton_add_kernel_spec(
    arch: str,
    input_tensors: list,
    output_tensor,
) -> KernelSpec:
    """Create a TRITON-backend KernelSpec for ADD (NPU path).

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.

    Returns:
        KernelSpec configured for the TRITON backend.

    Raises:
        ValueError: If the tensors require broadcasting or are non-contiguous
            (raised lazily when the returned compile function is invoked).

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

        from .triton_kernels.utils import numpy_dtype_to_torch, triton_device
        from .triton_kernels.vecadd import vecadd

        broadcast = input_tensors[0].shape != input_tensors[1].shape
        if broadcast or any(not t.contiguous for t in (*input_tensors, output_tensor)):
            msg = "Broadcasting or non-contiguous tensors detected."
            raise ValueError(msg)

        block_size = 1 << (min(1024, n_elements) - 1).bit_length()
        block_size = min(block_size, 1024)
        grid = (triton.cdiv(n_elements, block_size),)
        device = triton_device(arch)
        a = torch.randn(
            n_elements,
            device=device,
            dtype=numpy_dtype_to_torch(input_tensors[0].dtype),
        )
        b = torch.randn(
            n_elements,
            device=device,
            dtype=numpy_dtype_to_torch(input_tensors[1].dtype),
        )
        c = torch.empty(
            n_elements,
            device=device,
            dtype=numpy_dtype_to_torch(output_tensor.dtype),
        )
        return vecadd[grid](
            A=a, B=b, C=c, n_elements=n_elements, BLOCK_SIZE_N=block_size
        )

    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_ADD",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=_compile,
        config={
            "transform_script": str(
                Path(__file__).parent / "triton_kernels" / f"vecadd_{arch}.mlir"
            ),
        },
    )


def _make_triton_gpu_add_kernel_spec(
    arch: str,
    input_tensors: list,
    output_tensor,
) -> KernelSpec:
    """Create a TRITON-backend KernelSpec for ADD targeting an AMD GPU (gfx).

    Unlike the NPU path (which JIT-launches the kernel), the GPU path is ahead-of-time
    compiled: the returned spec's ``config`` carries a backend-neutral description of the
    kernel's argument contract and launch geometry (``triton_fn``, ``signature``,
    ``constexprs``, ``num_warps``, ``num_programs``). ``build_triton.py`` consumes those to
    emit a ``.hsaco`` plus a ``<name>.hsaco.json`` launch sidecar required by the C++ dispatch.

    Parameters:
        arch: Target GPU architecture (e.g. "gfx1151").
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.

    Returns:
        KernelSpec configured for the TRITON backend (GPU AOT path).

    Raises:
        ValueError: If the tensors require broadcasting (unmasked kernel).

    """
    import triton

    from .triton_kernels.vecadd import vecadd

    n_elements = output_tensor.numel()

    # vecadd is an unmasked elementwise kernel; it is only valid when both sources and the
    # destination have the same number of elements (no broadcasting).
    if input_tensors[0].numel() != n_elements or input_tensors[1].numel() != n_elements:
        msg = "Triton vecadd requires matching element counts (no broadcast)."
        raise ValueError(msg)

    # Triton's tl.arange requires a power-of-two range, so BLOCK_SIZE_N must be a power of two
    # that divides n_elements (the kernel is unmasked). Pick the largest such block.
    cap = min(1024, n_elements)
    block_size = 1
    power = 1
    while power <= cap:
        if n_elements % power == 0:
            block_size = power
        power *= 2
    num_programs = triton.cdiv(n_elements, block_size)

    # ONE canonical, backend-neutral description of the kernel's argument contract and launch
    # geometry. The GPU compiler consumes signature/constexprs directly via ASTSource.
    elt = _numpy_to_triton_elt(output_tensor.dtype)
    signature = {
        "A": f"*{elt}",
        "B": f"*{elt}",
        "C": f"*{elt}",
        "n_elements": "constexpr",
        "BLOCK_SIZE_N": "constexpr",
    }
    config = {
        "triton_fn": vecadd,
        "signature": signature,
        "constexprs": {"n_elements": n_elements, "BLOCK_SIZE_N": block_size},
        "num_warps": 4,
        "num_programs": num_programs,
    }

    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_ADD",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=vecadd,
        config=config,
    )


def ggml_op_add(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> list[KernelSpec]:
    """Return KernelSpecs for GGML_OP_ADD.

    On NPU targets IRON is the primary backend with a Triton fallback. On GPU (gfx) targets
    the Triton -> hsaco AOT path is used; unsupported cases (e.g. broadcasting) yield no spec,
    so the op is reported unsupported and skipped.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        List of KernelSpecs for the ADD operation.

    """
    from .triton_kernels.utils import is_gpu_arch

    _validate_binary_inputs(input_tensors)

    if is_gpu_arch(arch):
        try:
            return [
                _make_triton_gpu_add_kernel_spec(arch, input_tensors, output_tensor)
            ]
        except ValueError:
            return []

    return [
        _make_iron_binary_kernel_spec(
            arch, input_tensors, output_tensor, "GGML_OP_ADD"
        ),
        _make_triton_add_kernel_spec(arch, input_tensors, output_tensor),
    ]


def ggml_op_sub(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_SUB.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the SUB operation.

    """
    _validate_binary_inputs(input_tensors)

    return _make_iron_binary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_OP_SUB"
    )


def ggml_op_mul(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_MUL.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the MUL operation.

    """
    _validate_binary_inputs(input_tensors)

    return _make_iron_binary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_OP_MUL"
    )


def ggml_op_div(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_DIV.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the DIV operation.

    """
    _validate_binary_inputs(input_tensors)

    return _make_iron_binary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_OP_DIV"
    )
