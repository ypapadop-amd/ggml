# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Triton-XDNA backend compiler for GGML HSA kernels.
"""

import logging
from pathlib import Path

from kernel import KernelSpec


def _validate_triton_available() -> None:
    """
    Validate that triton-xdna is installed and available.

    Raises:
        ImportError: If triton-xdna package is not installed.
    """
    try:
        import triton  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Triton-XDNA backend requires triton-xdna package.\n"
            "Install with: pip install -r requirements-triton.txt"
        ) from e


def compile_triton_kernel(
    kernel_spec: KernelSpec,
    work_dir: Path,
    exported_name: str,
    output_directory: Path,
    logger: logging.Logger,
    verbose: bool,
) -> None:
    """
    Compile a Triton kernel to PDI and instructions files.

    This function executes the Triton-XDNA compilation pipeline:
    1. Validates that triton-xdna is available
    2. Translates the kernel specification into a Triton kernel
    3. Compiles via the MLIR-AIR/AIE stack to produce PDI and instructions

    Parameters:
        kernel_spec: The KernelSpec containing the Triton kernel function.
        work_dir: Working directory for intermediate files.
        exported_name: Name for the exported kernel files.
        output_directory: Directory for output PDI and instruction files.
        logger: Logger for status messages.
        verbose: If True, enables verbose compilation output.
    """
    _validate_triton_available()

    raise NotImplementedError(
        f"Triton-XDNA compilation for kernel '{exported_name}' is not yet implemented.\n"
        f"This is a skeleton implementation for future Triton backend support."
    )
