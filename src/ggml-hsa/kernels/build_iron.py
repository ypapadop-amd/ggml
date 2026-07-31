# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON backend compiler for GGML HSA kernels."""

import logging
from pathlib import Path

from aie.iron import ExternalFunction
from aie.utils.compile import compile_external_kernel, compile_mlir_module
from kernel import KernelSpec


def compile_iron_kernel(
    kernel_spec: KernelSpec,
    exported_name: str,
    output_directory: Path,
    logger: logging.Logger,
    verbose: bool,
) -> None:
    """Run the IRON compilation pipeline for a kernel.

    Runs the kernel's Python function to generate an MLIR module, compiles any
    external C++ core functions to object files, then compiles the module into
    PDI and instruction binaries.

    Args:
        kernel_spec: The KernelSpec containing the IRON kernel function.
        exported_name: Name for the exported kernel files.
        output_directory: Directory for output PDI and instruction files.
        logger: Logger for status messages.
        verbose: If True, enables verbose compilation output.
    """
    work_dir = output_directory / f"{exported_name}-iron-artifacts"
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Working directory: %s", str(work_dir))

    # Clear any existing external functions from previous compilations
    ExternalFunction._instances.clear()

    # Generate MLIR module by calling the kernel function
    # (this populates ExternalFunction._instances)
    mlir_module = kernel_spec.function()

    # Compile any external C++ core functions. The objects land in work_dir,
    # which is also compile_mlir_module's work_dir, so the relative link_with
    # paths in the MLIR resolve.
    for func in ExternalFunction._instances:
        compile_external_kernel(func, str(work_dir), kernel_spec.arch)

    # Clear external functions after compilation
    ExternalFunction._instances.clear()

    # Write MLIR module to file for debugging/inspection
    mlir_path = work_dir / f"{exported_name}.mlir"
    logger.info(
        "Writing MLIR module for operation %s in %s",
        kernel_spec.op_name,
        mlir_path,
    )
    with mlir_path.open("w", encoding="utf-8") as file:
        file.write(str(mlir_module))

    # Generate PDI and instructions files from MLIR
    pdi_path = output_directory / f"{exported_name}.pdi"
    insts_path = output_directory / f"{exported_name}_insts.bin"
    compile_mlir_module(
        mlir_module=mlir_module,
        options=["--alloc-scheme=basic-sequential"],
        insts_path=str(insts_path),
        pdi_path=str(pdi_path),
        verbose=verbose,
        work_dir=str(work_dir),
    )

    logger.info(
        "IRON compilation successful\n  PDI Path: %s\n  Instructions Path: %s",
        pdi_path,
        insts_path,
    )
