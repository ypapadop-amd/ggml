# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Packs a compiled AIE kernel into an hsaco (HSA code object) for the ROCr loader."""

import contextlib
import io
import tempfile
from pathlib import Path

from aie.compiler.hsaco import pack
from kernel import KernelSpec

# Each kernel argument occupies two uint64 kernarg entries: its device address and its size.
_KERNARG_BYTES_PER_ARG = 16


def pack_aie_hsaco(
    kernel_spec: KernelSpec,
    exported_name: str,
    output_directory: Path,
    xclbin_path: Path,
    insts_path: Path,
) -> Path:
    """Write ``<exported_name>.hsaco`` from an xclbin and its instruction sequence.

    Runs MLIR-AIE's ``aie-hsaco`` packer, which takes the PDI and the partition's
    column count from the xclbin's ``AIE_PARTITION`` section. The hsaco has one
    kernel, named ``exported_name``, in the section for ``kernel_spec.arch``; its
    arguments are the kernel's input tensors followed by its output tensor. The
    hsaco is assembled in a scratch directory and renamed into place only once
    complete, so a failed pack never leaves a partial hsaco that the runtime would
    find and fail to load.

    Args:
        kernel_spec: The compiled kernel's KernelSpec.
        exported_name: Kernel name, used for the HSA symbol and the hsaco file name.
        output_directory: Directory for the output hsaco.
        xclbin_path: xclbin holding the kernel's PDI.
        insts_path: Instruction sequence (``insts.bin``).

    Returns:
        The path of the written hsaco.

    Raises:
        RuntimeError: If the packer rejects the inputs.
    """
    num_kernargs = len(kernel_spec.input_tensors) + 1  # inputs, then the output
    hsaco_path = output_directory / f"{exported_name}.hsaco"
    with tempfile.TemporaryDirectory(dir=output_directory) as scratch_dir:
        scratch = Path(scratch_dir) / hsaco_path.name
        argv = [
            "--hsaco",
            str(scratch),
            "--arch",
            kernel_spec.arch,
            "--kernel-name",
            exported_name,
            "--kernel-xclbin",
            str(xclbin_path),
            "--kernel-insts",
            str(insts_path),
            "--kernel-kernarg",
            str(_KERNARG_BYTES_PER_ARG * num_kernargs),
        ]
        # The packer is a command-line tool: it reports bad input on stderr and exits.
        # Keep that report so the exception says why the pack failed.
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                pack.main(argv)
        except SystemExit as e:
            # The report follows argparse's usage text and may span several lines, e.g. when
            # it quotes xclbinutil's or llvm-objcopy's own output.
            report = stderr.getvalue()
            start = report.find("aie-hsaco: error:")
            reason = (report[start:] if start >= 0 else report).strip()
            msg = f"aie-hsaco could not pack kernel {exported_name}: " + (
                reason or f"exit status {e.code}"
            )
            raise RuntimeError(msg) from e
        scratch.replace(hsaco_path)
    return hsaco_path
