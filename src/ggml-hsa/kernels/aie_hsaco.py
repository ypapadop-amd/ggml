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
    xclbin_path: Path | None = None,
    insts_path: Path | None = None,
    full_elf_path: Path | None = None,
) -> Path:
    """Write ``<exported_name>.hsaco`` from an xclbin and its instruction sequence, or a full ELF.

    Runs MLIR-AIE's ``aie-hsaco`` packer. Give it either ``xclbin_path`` and
    ``insts_path``, or ``full_elf_path`` (aie2p only):

    - From an xclbin, the packer takes the PDI and the partition's column count from
      the xclbin's ``AIE_PARTITION`` section, and the kernel is named ``exported_name``.
    - From a full ELF, the PDI, control code and column count all come from the ELF,
      and so does the kernel name (``<device>:<sequence>``, e.g. ``main:sequence``).
      Unlike a PDI-plus-instructions kernel, it takes no PDI slot on the queue.

    The hsaco has one kernel, in the section for ``kernel_spec.arch``; its arguments
    are the kernel's input tensors followed by its output tensor. The hsaco is
    assembled in a scratch directory and renamed into place only once complete, so a
    failed pack never leaves a partial hsaco that the runtime would find and fail to
    load.

    Args:
        kernel_spec: The compiled kernel's KernelSpec.
        exported_name: Name of the hsaco file, and of the kernel when packing an xclbin.
        output_directory: Directory for the output hsaco.
        xclbin_path: xclbin holding the kernel's PDI.
        insts_path: Instruction sequence (``insts.bin``).
        full_elf_path: Full ELF (``aiecc --get-full-elf``).

    Returns:
        The path of the written hsaco.

    Raises:
        ValueError: If not given exactly one of the two input forms.
        RuntimeError: If the packer rejects the inputs.
    """
    if (full_elf_path is None) == (xclbin_path is None or insts_path is None):
        msg = "pack_aie_hsaco needs either xclbin_path and insts_path, or full_elf_path"
        raise ValueError(msg)
    if full_elf_path is not None:
        kernel_args = ["--kernel-elf", str(full_elf_path)]
    else:
        kernel_args = [
            "--kernel-name",
            exported_name,
            "--kernel-xclbin",
            str(xclbin_path),
            "--kernel-insts",
            str(insts_path),
        ]
    num_kernargs = len(kernel_spec.input_tensors) + 1  # inputs, then the output
    hsaco_path = output_directory / f"{exported_name}.hsaco"
    with tempfile.TemporaryDirectory(dir=output_directory) as scratch_dir:
        scratch = Path(scratch_dir) / hsaco_path.name
        argv = [
            "--hsaco",
            str(scratch),
            "--arch",
            kernel_spec.arch,
            *kernel_args,
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
