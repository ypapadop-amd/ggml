# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON backend compiler for GGML HSA kernels."""

import logging
from pathlib import Path

from aie import ir
from aie.dialects.aie import AIEDevice, get_target_model
from aie.iron import ExternalFunction
from aie.utils.compile import compile_external_kernel, compile_mlir_module
from aie_hsaco import pack_aie_hsaco
from kernel import KernelSpec


def _partition_columns(mlir_module: ir.Module) -> int:
    """Return the column count of the device the module's ``aie.device`` targets.

    This is the width of the partition the kernel was compiled for (e.g. 1 for
    ``npu1_1col``, 4 for ``npu1``), not the number of columns the design uses.
    """
    for op in mlir_module.body.operations:
        if op.operation.name == "aie.device":
            device = AIEDevice(ir.IntegerAttr(op.attributes["device"]).value)
            return get_target_model(device).columns()
    msg = "MLIR module has no aie.device operation"
    raise ValueError(msg)


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
    PDI and instruction binaries, and packs those into an hsaco.

    Args:
        kernel_spec: The KernelSpec containing the IRON kernel function.
        exported_name: Name for the exported kernel files.
        output_directory: Directory for the output hsaco.
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
    #
    # compile_external_kernel skips compilation when the object already exists, and it
    # decides that on existence alone -- it never compares the object against the source.
    # That cache is wrong across builds: this function only runs when the enclosing PDI is
    # missing and the kernel is being rebuilt, so an object left behind by an earlier run
    # (most easily by one where aiecc failed *after* the object was written) would be
    # linked into the new PDI in place of the edited source, and the edit would appear to
    # have no effect. Drop the stale objects so every build recompiles them.
    #
    # Within a single build that same existence check is load-bearing, so drop each
    # distinct path exactly once, before the loop: several ExternalFunctions may share one
    # object file (gemm.py registers zero_fn and matmul_fn against one matmul_core_functions_*.o,
    # two symbols in one translation unit). Unlinking inside the loop would delete the
    # object the previous iteration just produced and compile the same source again.
    for object_file_name in {
        func.object_file_name for func in ExternalFunction._instances
    }:
        (work_dir / object_file_name).unlink(missing_ok=True)

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
    pdi_path = work_dir / f"{exported_name}.pdi"
    insts_path = work_dir / f"{exported_name}_insts.bin"
    compile_mlir_module(
        mlir_module=mlir_module,
        insts_path=str(insts_path),
        pdi_path=str(pdi_path),
        verbose=verbose,
        work_dir=str(work_dir),
    )

    # Pack PDI and instructions into an hsaco
    hsaco_path = output_directory / f"{exported_name}.hsaco"
    pack_aie_hsaco(
        hsaco_path,
        arch=kernel_spec.arch,
        kernel_name=exported_name,
        insts=insts_path.read_bytes(),
        pdi=pdi_path.read_bytes(),
        num_kernargs=len(kernel_spec.input_tensors) + 1,
        num_cols=_partition_columns(mlir_module),
    )

    logger.info("IRON compilation successful\n  HSACO Path: %s", hsaco_path)
