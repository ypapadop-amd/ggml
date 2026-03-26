# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Kernel specification types for the GGML HSA backend.

This module defines the core data structures used for kernel dispatch and
compilation backend selection. The two-layer architecture separates:

1. Static mapping (Kernel): Maps GGML operation names to dispatch modules
2. Runtime dispatch (KernelSpec): Returned by dispatch functions to specify
   which backend and function to use for compilation

Example:
    # In op_to_kernel_map (static)
    "ADD": Kernel("ggml_op_add", "binary_ops.py")

    # At runtime, ggml_op_add() returns:
    KernelSpec(backend=Backend.IRON, function=iron_add_fn)

"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any


class Backend(Enum):
    """Supported kernel compilation backends.

    Each backend has its own compilation pipeline:
    - IRON: Uses MLIR-AIE/IRON framework for optimized AIE kernels
    """

    IRON = auto()


@dataclass(frozen=True)
class Kernel:
    """Static mapping entry from GGML operation to dispatch module.

    This dataclass represents an entry in op_to_kernel_map. It identifies
    which Python module contains the dispatch function for a given operation.

    Attributes:
        name: Name of the dispatch function to call (e.g., "ggml_op_add").
        source_file: Path to the Python module containing the
            dispatch function.

    """

    name: str
    source_file: str | Path


@dataclass(frozen=True)
class KernelSpec:
    """Specification returned by kernel dispatch functions.

    When a kernel dispatch function (e.g., ggml_op_add) is called, it examines
    the input parameters and returns a KernelSpec that tells the build system:
    1. Which backend to use for compilation
    2. Which function to call to generate the IR

    This enables per-invocation backend selection based on tensor shapes,
    dtypes, and other runtime parameters.

    Attributes:
        backend: The compilation backend to use.
        op_name: Name of the operation.
        arch: Target architecture for the kernel.
        input_tensors: List of input tensors for the operation.
        output_tensor: Output tensor for the operation.
        op_params: Operation parameters.
        function: Callable that generates the backend-specific IR.

    """

    backend: Backend
    op_name: str
    arch: str
    input_tensors: list
    output_tensor: Any
    op_params: bytearray
    function: Callable[..., Any]
    config: dict | None = None  # Optional field for additional configuration parameters

    def __post_init__(self):
        """Validate that backend is a Backend enum instance."""
        if not isinstance(self.backend, Backend):
            backend_type = type(self.backend).__name__
            raise TypeError(f"backend must be a Backend enum, got {backend_type}")
