# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Utility functions for IRON kernel implementations."""

from dataclasses import dataclass

import numpy as np
from aie.iron import ExternalFunction
from aie.iron.device import NPU1, NPU2

# Per-architecture on-tile resources. Add a new NPU generation by adding one entry.
_ARCH_PARAMS = {
    "aie2": {
        "core_data_mem_bytes": 64 * 1024,
        "vector_reg_bits": 512,
    },  # NPU1/Phoenix (AIE-ML)
    "aie2p": {
        "core_data_mem_bytes": 64 * 1024,
        "vector_reg_bits": 512,
    },  # NPU2/Strix (XDNA2)
}


def _arch_params(arch: str) -> dict:
    """Return the on-tile resource parameters for an architecture.

    Args:
        arch: Target architecture.

    Returns:
        The parameter dict for the architecture.

    Raises:
        ValueError: If the architecture is unknown.
    """
    params = _ARCH_PARAMS.get(arch)
    if params is None:
        msg = f"Unsupported architecture: {arch}"
        raise ValueError(msg)
    return params


def align_to_arch(
    arch: str, size: int, dtype: np.dtype, alignment_bytes: int = 4
) -> int:
    """Align an element count so its byte size is a multiple of alignment_bytes.

    Args:
        arch: Target architecture.
        size: Element count to align.
        dtype: Element data type.
        alignment_bytes: Byte boundary to align to.

    Returns:
        The aligned element count.
    """
    if arch in ["aie2", "aie2p"]:
        dtype_size = dtype.itemsize
        data_size = size * dtype_size
        if data_size % alignment_bytes != 0:
            return (
                alignment_bytes
                * ((data_size + (alignment_bytes - 1)) // alignment_bytes)
                // dtype_size
            )
        return size
    msg = f"Unsupported architecture: {arch}"
    raise ValueError(msg)


def row_dimensions(tensor) -> tuple[int, int]:
    """Return (row_length, num_rows) for a GGML-ordered tensor.

    Row-structured ops treat dim 0 (ne00) as the row: row_length = ne00 and
    num_rows = product of the remaining dimensions (ne01 * ne02 * ne03). Rows are
    laid out slice-major (ne01 consecutive rows per ne02/ne03 slice), matching the
    contiguous element order the runtime streams.

    Args:
        tensor: GGML-ordered tensor to inspect.

    Returns:
        The (row_length, num_rows) pair.

    Raises:
        ValueError: If the tensor rank is unsupported.
    """
    shape = tensor.shape
    if not 1 <= len(shape) <= 4:
        msg = f"Unsupported tensor rank: {len(shape)}"
        raise ValueError(msg)
    row_length = shape[0]
    num_rows = 1
    for dim in shape[1:]:
        num_rows *= dim
    return row_length, num_rows


def arch_aligned_num_elements(arch: str, tensor) -> int:
    """Tensor element count aligned to the architecture for its dtype.

    Args:
        arch: Target architecture.
        tensor: Tensor whose element count is aligned.

    Returns:
        The arch-aligned element count.
    """
    return align_to_arch(arch, tensor.numel(), tensor.dtype)


def max_tile_size(arch: str, dtype: np.dtype, num_elements: int) -> int:
    """Largest power-of-two tile within a 512-bit vector dividing num_elements.

    Args:
        arch: Target architecture.
        dtype: Element data type.
        num_elements: Total number of elements to tile.

    Returns:
        The chosen tile size.
    """
    vector_register_width = _arch_params(arch)["vector_reg_bits"]
    tile_size = int(vector_register_width / dtype.itemsize)

    while num_elements % tile_size != 0 and tile_size > 1:
        tile_size //= 2

    assert num_elements % tile_size == 0, (
        f"Number of elements ({num_elements}) must be a multiple of "
        f"tile size ({tile_size})."
    )

    return tile_size


def tiled_tile_size(arch: str, dtype: np.dtype, num_elements: int) -> int:
    """Largest tile that divides num_elements and fits half the core data memory.

    The tile must divide num_elements exactly: an ObjectFifo's DMA transfer size is
    fixed at construction, so every acquire moves a full tile regardless of any
    runtime element count. A tile that did not divide num_elements would make the
    total streamed (num_tiles * tile) exceed the tensor and deadlock the DMA. Prefer
    the largest multiple-of-V divisor within the L1 budget; if none exists (V does not
    divide num_elements), fall back to max_tile_size, which halves V down to a divisor.

    Args:
        arch: Target architecture.
        dtype: Element data type.
        num_elements: Total number of elements to tile.

    Returns:
        The chosen tile size in elements; always divides num_elements.
    """
    params = _arch_params(arch)
    v = params["vector_reg_bits"] // (8 * dtype.itemsize)
    budget = (
        params["core_data_mem_bytes"] // 2
    )  # half DM: leave room for stack + locals
    # in + out fifos, each double-buffered (depth 2) => 4 buffers of tile*itemsize bytes.
    max_by_mem = (budget // (4 * dtype.itemsize) // v) * v
    cap = min(max_by_mem, num_elements)

    # Largest multiple of V that is <= cap and divides num_elements exactly.
    best = 0
    tile = v
    while tile <= cap:
        if num_elements % tile == 0:
            best = tile
        tile += v
    if best:
        return best

    # V does not divide num_elements: fall back to a power-of-two divisor.
    return max_tile_size(arch, dtype, num_elements)


# Minimum valid elements a worker must process for its share of the copy/convert to outweigh the
# fixed per-worker cost (an extra shim-DMA fill+drain pair and its completion fence). Below this,
# fanning out costs more than the single-core compute it saves; small ops (e.g. the MNIST FC
# transform operands, <= ~400k elements) stay single-worker and match the original fast path.
_MIN_ELEMS_PER_WORKER = 256 * 1024


def fan_out_worker_count(
    arch: str, total_elems: int, n_units: int, max_workers: int | None = None
) -> int:
    """Number of data-parallel workers to fan a transform across, scaled to the work size.

    Capped by (a) the caller's max_workers (defaults to the array column count), (b) the number of
    independent units to split (n_units), and (c) the work: at least _MIN_ELEMS_PER_WORKER valid
    elements per worker, so tiny ops stay single-worker and avoid per-worker DMA/fence overhead.

    Args:
        arch: Target architecture.
        total_elems: Total valid elements the transform processes.
        n_units: Number of independent units available to split across workers.
        max_workers: Hard cap on workers; defaults to the array's column count.

    Returns:
        The worker count (>= 1).
    """
    if max_workers is None:
        max_workers = arch_num_columns(arch)
    by_work = max(1, total_elems // _MIN_ELEMS_PER_WORKER)
    return max(1, min(max_workers, n_units, by_work))


def partition_units(num_workers: int, n_units: int) -> tuple[list[int], list[int]]:
    """Split n_units contiguous units as evenly as possible across num_workers.

    The first (n_units % num_workers) workers get one extra unit so the slices
    cover all n_units exactly.

    Args:
        num_workers: Number of workers to split across.
        n_units: Total number of independent units to distribute.

    Returns:
        The (counts, starts) pair: counts[w] is the number of units assigned to
        worker w, and starts[w] is the index of its first unit.
    """
    base, rem = divmod(n_units, num_workers)
    counts = [base + (1 if w < rem else 0) for w in range(num_workers)]
    starts = []
    acc = 0
    for count in counts:
        starts.append(acc)
        acc += count
    return counts, starts


@dataclass(frozen=True)
class CoreFunctionSpec:
    """Core function plus total element count for an element-wise op.

    Attributes:
        external_function: External function implementing the operation.
        num_elements: Total number of elements to process.

    """

    external_function: ExternalFunction
    num_elements: int

    @property
    def tile_size(self) -> int:
        """Tile size used by the external function."""
        return self.external_function.tile_size(0)


def arch_num_columns(arch: str) -> int:
    """Number of array columns for an architecture (aie2p -> 8, aie2 -> 4).

    Each column has its own shim tile / external-DMA path, so the column count is the natural
    upper bound on independent data-parallel workers streaming to/from external memory: one worker
    per column saturates the available shim-DMA bandwidth without oversubscribing a column.

    Args:
        arch: Target architecture.

    Returns:
        The number of columns in the array.
    """
    return arch_to_device(arch).cols


def arch_to_device(device):
    """Map "aie2" -> NPU1, "aie2p" -> NPU2; pass through existing device objects.

    Args:
        device: Architecture string or an existing device object.

    Returns:
        The corresponding device object.
    """
    if isinstance(device, str):
        if device == "aie2":
            return NPU1()
        if device == "aie2p":
            return NPU2()
        msg = f"Unsupported device: {device}"
        raise ValueError(msg)
    return device
