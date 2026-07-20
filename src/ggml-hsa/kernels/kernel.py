# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Kernel dispatch and backend-selection types for the GGML HSA backend.

Two layers separate concerns:

1. Kernel (static): maps a GGML operation name to its dispatch module.
2. KernelSpec (runtime): returned by a dispatch function to specify which
   backend and function to compile.

Example:
    # In op_to_kernel_map (static)
    "ADD": Kernel("ggml_op_add", "binary_ops.py")

    # At runtime, ggml_op_add() returns:
    KernelSpec(backend=Backend.IRON, function=iron_add_fn)

"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

# Common env var gating backend preference for ops that ship both an IRON and a Triton
# kernel (MUL_MAT, ADD, RELU). When set to "1", Triton specs are tried first and IRON
# becomes the fallback; otherwise IRON is primary. See order_kernel_specs().
PREFER_TRITON_ENV = "GGML_HSA_PREFER_TRITON"


class Backend(Enum):
    """Supported kernel compilation backends.

    - IRON: MLIR-AIE/IRON framework for optimized AIE kernels.
    - TRITON: Triton-XDNA for compiler-driven generation via MLIR-AIR/AIE.
    """

    IRON = auto()
    TRITON = auto()


@dataclass(frozen=True)
class Kernel:
    """Static op_to_kernel_map entry identifying a dispatch function and its module.

    Attributes:
        name: Name of the dispatch function to call (e.g., "ggml_op_add").
        source_file: Python module containing the dispatch function.

    """

    name: str
    source_file: str | Path


@dataclass(frozen=True)
class KernelSpec:
    """Specification returned by a kernel dispatch function.

    Tells the build system which backend to use and which function generates the
    IR, enabling per-invocation backend selection based on tensor shapes, dtypes,
    and other runtime parameters.

    Attributes:
        backend: The compilation backend to use.
        op_name: Name of the operation.
        arch: Target architecture for the kernel.
        input_tensors: List of input tensors for the operation.
        output_tensor: Output tensor for the operation.
        op_params: Operation parameters.
        function: Callable that generates the backend-specific IR.
        config: Dictionary for additional configuration parameters.

    """

    backend: Backend
    op_name: str
    arch: str
    input_tensors: list
    output_tensor: Any
    function: Callable[..., Any]
    op_params: bytearray | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate backend."""
        if not isinstance(self.backend, Backend):
            backend_type = type(self.backend).__name__
            msg = f"backend must be a Backend enum, got {backend_type}"
            raise TypeError(msg)


def order_kernel_specs(specs: list[KernelSpec]) -> list[KernelSpec]:
    """Order candidate KernelSpecs so the preferred backend is tried first.

    The build system compiles the specs in order and uses the first that succeeds,
    falling back to the next. For ops that ship both an IRON and a Triton kernel, the
    default is IRON-first (Triton is a fallback reached only if IRON compilation fails).
    Setting ``GGML_HSA_PREFER_TRITON=1`` promotes every Triton-backed spec ahead of the
    rest, so Triton is tried first and IRON becomes the fallback.

    Ordering is stable: within each group (Triton-backed vs. the rest) the caller's
    original order is preserved, so a caller that already lists IRON before Triton gets
    IRON-first by default and Triton-first under the flag with no other change.

    Args:
        specs: Candidate KernelSpecs for one op, in the caller's default (IRON-first)
            order.

    Returns:
        The specs reordered with Triton-backed specs first when the flag is set;
        otherwise the input order unchanged.
    """
    if os.environ.get(PREFER_TRITON_ENV, "0") != "1":
        return specs

    triton_specs = [s for s in specs if s.backend == Backend.TRITON]
    other_specs = [s for s in specs if s.backend != Backend.TRITON]
    return triton_specs + other_specs
