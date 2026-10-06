# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Packs a compiled AIE kernel into an hsaco (HSA code object) for the ROCr loader."""

import os
import tempfile
from pathlib import Path

from aie.compiler.hsaco.elf import make_empty_elf64
from aie.compiler.hsaco.pack import build_section, inject

# Each kernel argument occupies two uint64 kernarg entries: its device address and its size.
_KERNARG_BYTES_PER_ARG = 16


def pack_aie_hsaco(
    hsaco_path: Path,
    arch: str,
    kernel_name: str,
    insts: bytes,
    pdi: bytes,
    num_kernargs: int,
    num_cols: int,
) -> None:
    """Write a single-kernel hsaco holding a PDI and instruction sequence.

    The hsaco is assembled in a scratch file next to ``hsaco_path`` and renamed over it
    only once complete, so a failed or interrupted pack never leaves a partial hsaco
    that the runtime would find and fail to load.

    Args:
        hsaco_path: Output hsaco path.
        arch: AIE architecture, which is also the hsaco section name ("aie2" or "aie2p").
        kernel_name: HSA symbol name of the kernel.
        insts: Instruction sequence (``insts.bin``).
        pdi: PDI.
        num_kernargs: Number of kernel arguments (buffers).
        num_cols: Column count of the partition the kernel was compiled for.
    """
    section = build_section(
        arch,
        [
            {
                "name": kernel_name,
                "insts": insts,
                "pdi": pdi,
                "kernarg_size": _KERNARG_BYTES_PER_ARG * num_kernargs,
                "num_cols": num_cols,
            }
        ],
    )
    fd, scratch = tempfile.mkstemp(
        dir=hsaco_path.parent, prefix=hsaco_path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(make_empty_elf64())
        inject(scratch, arch, section)
        Path(scratch).replace(hsaco_path)
    finally:
        Path(scratch).unlink(missing_ok=True)
