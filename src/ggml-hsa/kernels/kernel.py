# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Kernel specification types for the GGML HSA backend.

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
    """
    Supported kernel compilation backends.

    Each backend has its own compilation pipeline:
    - IRON: Uses MLIR-AIE/IRON framework for optimized AIE kernels
    """

    IRON = auto()


@dataclass(frozen=True)
class Kernel:
    """
    Static mapping entry from GGML operation to dispatch module.

    This dataclass represents an entry in op_to_kernel_map. It identifies
    which Python module contains the dispatch function for a given operation.

    Attributes:
        name: Name of the dispatch function to call (e.g., "ggml_op_add").
        source_file: Path to the Python module containing the dispatch function.
    """

    name: str
    source_file: str | Path


@dataclass(frozen=True)
class KernelSpec:
    """
    Specification returned by kernel dispatch functions.

    When a kernel dispatch function (e.g., ggml_op_add) is called, it examines
    the input parameters and returns a KernelSpec that tells the build system:
    1. Which backend to use for compilation
    2. Which function to call to generate the IR

    This enables per-invocation backend selection based on tensor shapes,
    dtypes, and other runtime parameters.

    Attributes:
        backend: The compilation backend to use (see Backend enum).
        function: Callable that generates the backend-specific IR.
        op_name: Name of the operation (set by ggml_compile_op from Kernel.name).
    """

    backend: Backend
    function: Callable[..., Any]
    op_name: str = ""

    def __post_init__(self):
        """Validate that backend is a Backend enum instance."""
        if not isinstance(self.backend, Backend):
            raise TypeError(
                f"backend must be a Backend enum, got {type(self.backend).__name__}"
            )
