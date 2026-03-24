"""
Triton-XDNA kernel compilation backend.

This module provides kernel compilation support for the Triton-XDNA backend.
Triton-XDNA uses a compiler-driven approach to automatically generate optimized
AIE kernels through the MLIR-AIR/AIE stack.

Dependencies:
    - triton-xdna: Triton compiler backend for AMD XDNA devices

Example:
    >>> from kernel import Kernel, KernelConfig, Backend
    >>> kernel = Kernel(
    ...     name="matmul",
    ...     backend=Backend.TRITON,
    ...     config=KernelConfig(M=64, N=64, K=64),
    ...     spec={
    ...         "operation": "matmul",
    ...         "block_size": {"M": 64, "N": 64, "K": 64}
    ...     }
    ... )
    >>> compile_triton_kernel(kernel)  # doctest: +SKIP
"""

from pathlib import Path
from typing import Any, Dict

from kernel import Kernel


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


def _validate_kernel_spec(kernel: Kernel) -> None:
    """
    Validate that kernel specification contains required Triton fields.

    Args:
        kernel: Kernel instance to validate.

    Raises:
        ValueError: If kernel.spec is missing required fields for Triton compilation.
    """
    if not isinstance(kernel.spec, dict):
        raise ValueError(
            f"Kernel {kernel.name} spec must be a dictionary for Triton backend, "
            f"got {type(kernel.spec).__name__}"
        )

    required_fields = ["operation"]
    missing = [field for field in required_fields if field not in kernel.spec]
    if missing:
        raise ValueError(
            f"Kernel {kernel.name} spec missing required fields for Triton: "
            f"{', '.join(missing)}"
        )


def compile_triton_kernel(kernel: Kernel, output_dir: Path) -> Dict[str, Any]:
    """
    Compile a kernel using the Triton-XDNA backend.

    This function translates the kernel specification into a Triton kernel,
    compiles it via the MLIR-AIR/AIE stack, and generates the binary artifacts.

    Args:
        kernel: Kernel instance containing:
            - name: Kernel identifier
            - backend: Must be Backend.TRITON
            - config: KernelConfig with dimension parameters
            - spec: Dictionary with Triton-specific fields:
                * operation: Operation type (e.g., "matmul", "conv2d")
                * block_size: Dictionary of blocking parameters (optional)
                * Additional operation-specific parameters
        output_dir: Directory where compiled artifacts will be written:
            - {kernel.name}.xclbin: Compiled binary
            - {kernel.name}_insts.txt: Instruction sequence
            - {kernel.name}_metadata.json: Compilation metadata

    Returns:
        Dictionary containing compilation metadata:
            - xclbin_path: Path to compiled binary
            - insts_path: Path to instruction sequence
            - metadata_path: Path to metadata file
            - triton_version: Version of triton-xdna used
            - compilation_flags: Flags used during compilation

    Raises:
        ImportError: If triton-xdna package is not installed.
        ValueError: If kernel.spec is missing required fields.
        NotImplementedError: Currently raised as skeleton implementation.

    Example:
        >>> kernel = Kernel(
        ...     name="matmul_64x64x64",
        ...     backend=Backend.TRITON,
        ...     config=KernelConfig(M=64, N=64, K=64),
        ...     spec={
        ...         "operation": "matmul",
        ...         "block_size": {"M": 64, "N": 64, "K": 64}
        ...     }
        ... )
        >>> result = compile_triton_kernel(kernel, Path("/tmp/kernels"))
    """
    _validate_triton_available()
    _validate_kernel_spec(kernel)

    raise NotImplementedError(
        f"Triton-XDNA compilation for kernel '{kernel.name}' is not yet implemented.\n"
        f"This is a skeleton implementation for future Triton backend support."
    )
