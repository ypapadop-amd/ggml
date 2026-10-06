# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Packs a compiled AIE kernel into an hsaco (HSA code object) for the ROCr loader."""

import tempfile
from pathlib import Path

from aie.compiler.hsaco import pack

# Each kernel argument occupies two uint64 kernarg entries: its device address and its size.
_KERNARG_BYTES_PER_ARG = 16


def pack_aie_hsaco(
    hsaco_path: Path,
    arch: str,
    kernel_name: str,
    xclbin_path: Path,
    insts_path: Path,
    num_kernargs: int,
) -> None:
    """Write a single-kernel hsaco from an xclbin and its instruction sequence.

    Runs MLIR-AIE's ``aie-hsaco`` packer, which takes the PDI and the partition's
    column count from the xclbin's ``AIE_PARTITION`` section. The hsaco is assembled
    in a scratch directory next to ``hsaco_path`` and renamed over it only once
    complete, so a failed pack never leaves a partial hsaco that the runtime would
    find and fail to load.

    Args:
        hsaco_path: Output hsaco path.
        arch: AIE architecture, which is also the hsaco section name ("aie2" or "aie2p").
        kernel_name: HSA symbol name of the kernel.
        xclbin_path: xclbin holding the kernel's PDI.
        insts_path: Instruction sequence (``insts.bin``).
        num_kernargs: Number of kernel arguments (buffers).

    Raises:
        RuntimeError: If the packer rejects the inputs.
    """
    with tempfile.TemporaryDirectory(dir=hsaco_path.parent) as scratch_dir:
        scratch = Path(scratch_dir) / hsaco_path.name
        argv = [
            "--hsaco",
            str(scratch),
            "--arch",
            arch,
            "--kernel-name",
            kernel_name,
            "--kernel-xclbin",
            str(xclbin_path),
            "--kernel-insts",
            str(insts_path),
            "--kernel-kernarg",
            str(_KERNARG_BYTES_PER_ARG * num_kernargs),
        ]
        # The packer is a command-line tool and reports bad input by exiting; it
        # prints the reason to stderr.
        try:
            pack.main(argv)
        except SystemExit as e:
            msg = (
                f"aie-hsaco could not pack kernel {kernel_name} (exit status {e.code})"
            )
            raise RuntimeError(msg) from e
        scratch.replace(hsaco_path)
