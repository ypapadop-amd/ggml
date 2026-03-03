# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""
IRON backend compiler for GGML HSA kernels.
"""

import logging
from collections.abc import Iterable
from pathlib import Path

from kernel import KernelSpec
from tensor_desc import TensorDesc
from iron.utils import suppress_import_pyxrt_msg

suppress_import_pyxrt_msg()

from aie.iron import ExternalFunction
from aie.utils.compile import compile_cxx_core_function
from aie.utils.compile import compile_mlir_module


def _compile_aie_core_kernels(
    arch: str,
    functions: Iterable[ExternalFunction],
    work_dir: Path,
    verbose: bool,
) -> None:
    """
    Compile AIE core functions to object files.

    This function compiles the C++ source files for external functions
    (core compute kernels) into object files that will be linked into
    the final PDI.

    Parameters:
        arch: Target architecture (e.g., "aie2", "aie2p").
        functions: Iterable of ExternalFunction objects to compile.
        work_dir: Working directory for intermediate files.
        verbose: If True, enables verbose compilation output.
    """
    for func in functions:
        compile_cxx_core_function(
            source_path=func._source_file,
            target_arch=arch,
            output_path=func.bin_name,
            include_dirs=func._include_dirs,
            compile_args=func._compile_flags,
            cwd=str(work_dir),
            verbose=verbose,
        )


def compile_iron_kernel(
    kernel_spec: KernelSpec,
    arch: str,
    input_tensors: list[TensorDesc],
    output_tensor: TensorDesc,
    op_params: bytearray,
    work_dir: Path,
    exported_name: str,
    output_directory: Path,
    logger: logging.Logger,
    verbose: bool,
) -> None:
    """
    Compile an IRON kernel to PDI and instructions files.

    This function executes the IRON compilation pipeline:
    1. Executes the kernel's Python function to generate an MLIR module
    2. Compiles any external C++ core functions to object files
    3. Compiles the MLIR module to produce PDI and instructions binaries

    Parameters:
        kernel_spec: The KernelSpec containing the IRON kernel function.
        arch: Target architecture (e.g., "aie2", "aie2p").
        input_tensors: List of input tensor descriptions.
        output_tensor: Output tensor description.
        op_params: Operation-specific parameters.
        work_dir: Working directory for intermediate files.
        exported_name: Name for the exported kernel files.
        output_directory: Directory for output PDI and instruction files.
        logger: Logger for status messages.
        verbose: If True, enables verbose compilation output.
    """
    # Clear any existing external functions from previous compilations
    ExternalFunction._instances.clear()

    # Generate MLIR module by calling the kernel function
    # (this also populates ExternalFunction._instances)
    mlir_module = kernel_spec.function(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
    )

    # Compile any external C++ core functions
    _compile_aie_core_kernels(
        arch=arch,
        functions=ExternalFunction._instances,
        work_dir=work_dir,
        verbose=verbose,
    )

    # Clear external functions after compilation
    ExternalFunction._instances.clear()

    # Write MLIR module to file for debugging/inspection
    mlir_path = work_dir / f"{exported_name}.mlir"
    logger.info(
        "Writing MLIR module for operation %s in %s", kernel_spec.op_name, mlir_path
    )
    with open(mlir_path, "wt", encoding="utf-8") as file:
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
