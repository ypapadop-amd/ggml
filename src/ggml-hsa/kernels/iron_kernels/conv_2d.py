#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for GGML_OP_CONV_2D (direct 2D convolution)."""

import struct
from pathlib import Path

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
    ExternalFunction,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    dtype_to_str,
)
from aie.iron.controlflow import range_
from ml_dtypes import bfloat16

from .utils import arch_to_device, batch_slice_tap, partition_units, shape_defines

# Number of int32 op_params: {s0, s1, p0, p1, d0, d1}
_CONV2D_PARAMS_SIZE = 6 * 4

# Cap on data-parallel workers (compute tiles). Each worker needs one shim DMA
# for its image slice and one for its output slice; the weights are broadcast
# through a single shared shim channel. The shim budget is ~16 endpoints on NPU1
# (aie2), so 2*W + 1 (weights) <= 16 gives W <= 7.
_MAX_WORKERS = 7

# The core's frame exceeds the AIE core's 1024-byte default stack once the tap loops are unrolled.
# Without an explicit size the core silently writes past the end of its stack into neighbouring core
# data memory. Same pattern as cross_entropy_loss.py, softmax.py and gemm.py.
#
# The frame is shape-dependent, but bounded: conv_2d.cc only unrolls the tap loops when
# KW * KH <= 9 (see GGML_CONV2D_UNROLL_TAPS), so the worst case is a 3x3 window, for which aiecc
# measures 2752 bytes. Larger windows compile without the unroll and need far less. Verified at
# 3x3, 5x5, 7x7 and 11x11. If a future change raises the unroll bound, mlir-aie fails the build
# with "stack_size = N is insufficient: this core needs M bytes", which is the signal to raise this
# constant rather than a silent overrun. On aie2 the kernel's vector paths (conv_2d.cc) need more:
# the split-image path's frame measured 3776 B (MNIST conv1), and the ring path adds its ring of
# split rows to the frame (see _ring_bytes).
_STACK_SIZE_BYTES = {"aie2": 6144}
_DEFAULT_STACK_SIZE_BYTES = 4096

# Stack the aie2 ring path's frame needs beside its ring: it measured at most 1344 B (MNIST conv1)
# and the core's main adds ~128 B.
_RING_FRAME_BYTES = 2048

# Largest ring the aie2 ring path may keep on the stack; a larger one takes the per-row path.
_MAX_RING_BYTES = 8192

# Lanes of the aie2 vector paths (f32 in a 512-bit register).
_V = 16

# A compute tile's data memory (aie2), which holds every buffer of a worker and its stack.
_TILE_MEMORY_BYTES = 64 * 1024


# Lanes of the aie2p flat path's bf16 limb products.
_FLAT_V = 64

# Stack of the aie2p flat path's core, at least.
_FLAT_STACK_BYTES = 8192


def _flat_geometry(shape: dict) -> tuple[int, int, int] | None:
    """Return conv_2d.cc's aie2p flat layout (namespace flat): row pitch, output vectors, plane length.

    Args:
        shape: The -DGGML_CONV2D_* extents and op_params (S0 = S1 = 1).

    Returns:
        (pw, n_vec, L): the row pitch, the 64-lane output vectors of one plane (in pairs), and the bf16
        elements of one limb plane of one input channel; None if a row's taps span 32 columns or
        more, which the flat path does not take.

    """
    span = (shape["KW"] - 1) * shape["D0"]
    if span >= 32:
        return None
    pw = shape["OW"] + span
    while any((ikh * shape["D1"] * pw) % 32 + span > 31 for ikh in range(shape["KH"])):
        pw += 1
    n_vec = -(-shape["OH"] * pw // (2 * _FLAT_V)) * 2
    rows = (shape["KH"] - 1) * shape["D1"] * pw
    return pw, n_vec, -(-(n_vec * _FLAT_V + rows + 32) // 32) * 32


def _split_image_elems(shape: dict) -> int:
    """Return the bf16 elements of conv_2d.cc's split image (limb::image_rows split rows per channel).

    Args:
        shape: The -DGGML_CONV2D_* extents and op_params.

    Returns:
        IC * image_rows * 3 * pw, as conv_2d.cc's namespace limb lays it out.

    """
    pw = ((shape["OW"] + _V - 1) // _V + 1) * _V
    image_rows = (shape["OH"] - 1) * shape["S1"] + (shape["KH"] - 1) * shape["D1"] + 1
    return shape["IC"] * image_rows * 3 * pw


def _ring_bytes(shape: dict) -> int:
    """Return the bytes of conv_2d.cc's aie2 ring: R split rows per input channel, and the weights.

    Args:
        shape: The -DGGML_CONV2D_* extents and op_params.

    Returns:
        (R * IC * 3 * pw + 3 * n_wts_pad) bf16, as conv_2d.cc's ring path declares them.

    """
    pw = ((shape["OW"] + _V - 1) // _V + 1) * _V
    rows = (shape["KH"] - 1) * shape["D1"] + 1
    n_wts_pad = -(-shape["IC"] * shape["KH"] * shape["KW"] // _V) * _V
    return (rows * shape["IC"] * 3 * pw + 3 * n_wts_pad) * 2


def conv_2d(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """Build the CONV_2D IRON program.

    Args:
        arch: Target architecture.
        input_tensors: [kernel src0 (KW, KH, IC, OC), image src1 (IW, IH, IC, N)].
        output_tensor: Output tensor, shape (OW, OH, OC, N).
        op_params: {s0, s1, p0, p1, d0, d1} as 6 x int32.

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: On invalid tensor count, dtype, contiguity, or op_params.

    """
    if len(input_tensors) != 2:
        msg = "CONV_2D requires exactly two input tensors (kernel and image)."
        raise ValueError(msg)

    kernel_tensor = input_tensors[0]
    image_tensor = input_tensors[1]

    if kernel_tensor.dtype != np.float32:
        msg = f"CONV_2D only supports float32 kernel input; got {kernel_tensor.dtype}."
        raise ValueError(msg)

    if image_tensor.dtype != np.float32:
        msg = f"CONV_2D only supports float32 image input; got {image_tensor.dtype}."
        raise ValueError(msg)

    if output_tensor.dtype != np.float32:
        msg = f"CONV_2D only supports float32 output; got {output_tensor.dtype}."
        raise ValueError(msg)

    if (
        not kernel_tensor.contiguous
        or not image_tensor.contiguous
        or not output_tensor.contiguous
    ):
        msg = "Kernel, image and output tensors must be contiguous in memory."
        raise ValueError(msg)

    if len(op_params) < _CONV2D_PARAMS_SIZE:
        msg = (
            f"op_params too short: expected at least {_CONV2D_PARAMS_SIZE} bytes, "
            f"got {len(op_params)}."
        )
        raise ValueError(msg)

    s0, s1, p0, p1, d0, d1 = struct.unpack_from("6i", op_params, 0)

    if s0 <= 0 or s1 <= 0:
        msg = f"Strides must be positive; got s0={s0}, s1={s1}."
        raise ValueError(msg)

    if p0 < 0 or p1 < 0:
        msg = f"Padding must be non-negative; got p0={p0}, p1={p1}."
        raise ValueError(msg)

    if d0 <= 0 or d1 <= 0:
        msg = f"Dilation must be positive; got d0={d0}, d1={d1}."
        raise ValueError(msg)

    # Kernel (src0): [KW, KH, IC, OC]
    kw, kh, ic, oc = (
        kernel_tensor.shape[0],
        kernel_tensor.shape[1],
        kernel_tensor.shape[2],
        kernel_tensor.shape[3],
    )
    # Image (src1): [IW, IH, IC, N]
    iw, ih, _ic, n = image_tensor.shape
    # Output (dst): [OW, OH, OC, N]
    ow, oh, _oc, out_n = output_tensor.shape

    if _ic != ic:
        msg = f"Channel mismatch: kernel IC={ic}, image IC={_ic}."
        raise ValueError(msg)

    if _oc != oc:
        msg = f"Output channel mismatch: kernel OC={oc}, output OC={_oc}."
        raise ValueError(msg)

    if out_n != n:
        msg = f"Batch mismatch: image N={n}, output N={out_n}."
        raise ValueError(msg)

    if n <= 0:
        msg = f"Batch size must be positive; got N={n}."
        raise ValueError(msg)

    if ic <= 0 or oc <= 0:
        msg = f"Channel counts must be positive; got IC={ic}, OC={oc}."
        raise ValueError(msg)

    if ow <= 0 or oh <= 0:
        msg = f"Output extents must be positive; got OW={ow}, OH={oh}."
        raise ValueError(msg)

    # Element counts per tile
    image_size = ic * ih * iw  # one full image (all channels)
    wts_size = kw * kh * ic * oc  # full weight tensor (same for all images)
    plane_size = ow * oh  # one output plane for a single output channel

    shape = {
        "IW": iw,
        "IH": ih,
        "IC": ic,
        "KW": kw,
        "KH": kh,
        "OW": ow,
        "OH": oh,
        "S0": s0,
        "S1": s1,
        "P0": p0,
        "P1": p1,
        "D0": d0,
        "D1": d1,
    }

    # On aie2, shapes the vector paths take split each image into bf16 limbs once, into a buffer
    # of the worker's, which every output channel's call then reads (conv_2d.cc, namespace limb),
    # if that buffer fits in the tile next to the double-buffered image and planes, the weights
    # and the stack. Otherwise each call splits the rows it reads itself.
    vector_path = arch == "aie2" and s0 == 1 and (kw - 1) * d0 <= _V
    stack_size = _STACK_SIZE_BYTES.get(arch, _DEFAULT_STACK_SIZE_BYTES)
    split_elems = _split_image_elems(shape)
    split_tile_bytes = (
        2 * image_size * 4
        + wts_size * 4
        + 2 * plane_size * 4
        + split_elems * 2
        + stack_size
    )
    use_split_image = vector_path and split_tile_bytes <= _TILE_MEMORY_BYTES - 4096
    # On aie2p, at unit strides, each image is split into flat bf16 limb planes once, into a buffer
    # of the worker's, and each output channel's call computes its plane at flat positions, whose
    # DMA streams only the OW columns of each row (conv_2d.cc, namespace flat).
    flat_elems = 0
    flat_plane = plane_size
    to_stream = None
    geometry = _flat_geometry(shape) if arch != "aie2" and s0 == 1 and s1 == 1 else None
    if geometry is not None:
        pw, n_vec, flat_len = geometry
        # The flat path's frames measured up to 4352 B at 3x3 (MNIST conv2) and 7360 B for a 9-tap
        # row; ggml_op_conv_2d_flat_image also stages one padded channel in f32 on the stack.
        flat_stack = max(_FLAT_STACK_BYTES, -(-(flat_len * 4 + 1024) // 256) * 256)
        # conv_2d.cc's static table of broadcast weights: 384 B a tap of one pass.
        pass_rows = kh if kw * kh <= 9 else 1
        flat_tile_bytes = (
            2 * image_size * 4
            + wts_size * 4
            + 2 * n_vec * _FLAT_V * 4
            + ic * 3 * flat_len * 2
            + pass_rows * kw * 384
            + flat_stack
        )
        if flat_tile_bytes <= _TILE_MEMORY_BYTES - 4096:
            flat_elems = ic * 3 * flat_len
            flat_plane = n_vec * _FLAT_V
            stack_size = flat_stack
            to_stream = TensorAccessPattern((flat_plane,), 0, [oh, ow], [pw, 1])
    # Otherwise the ring path, if its ring fits the stack it is given; the cap passed to
    # conv_2d.cc (GGML_CONV2D_RING_BYTES) is derived from that stack, so the two cannot disagree.
    # A shape that still overflows fails aiecc's stack check rather than running.
    ring_bytes = _ring_bytes(shape)
    if vector_path and not use_split_image and ring_bytes <= _MAX_RING_BYTES:
        stack_size = max(stack_size, -(-(ring_bytes + _RING_FRAME_BYTES) // 256) * 256)

    functions = _create_external_functions(
        image_tensor=image_tensor,
        output_tensor=output_tensor,
        image_size=image_size,
        wts_size=wts_size,
        plane_size=plane_size,
        shape=shape,
        split_elems=split_elems if use_split_image else 0,
        ring_cap=stack_size - _RING_FRAME_BYTES,
        flat_elems=flat_elems,
        flat_plane=flat_plane,
    )

    # Distribute the N images across compute tiles. Each worker owns independent
    # weight/image/output fifos and processes a contiguous slice of the batch.
    num_workers = min(_MAX_WORKERS, n)
    images_per_worker, image_starts = partition_units(num_workers, n)

    # One output plane per (image, oc); planes per image = oc.
    out_per_image = plane_size * oc

    image_tile_ty = np.ndarray[(image_size,), np.dtype[image_tensor.dtype]]
    wts_tile_ty = np.ndarray[(wts_size,), np.dtype[kernel_tensor.dtype]]
    plane_tile_ty = np.ndarray[(flat_plane,), np.dtype[output_tensor.dtype]]

    # depth=1: the full weight tensor is broadcast once to all workers through a
    # single shim channel (one producer, one consumer handle per worker), then
    # held for the whole run. Per-worker weight fifos would each need their own
    # shim DMA and exhaust the array's DMA budget.
    of_wts = ObjectFifo(wts_tile_ty, depth=1, name="wts")
    wts_conss = [of_wts.cons() for _ in range(num_workers)]
    of_ins = [ObjectFifo(image_tile_ty, name=f"in{w}") for w in range(num_workers)]
    of_outs = [
        ObjectFifo(plane_tile_ty, name=f"out{w}", to_stream=to_stream)
        for w in range(num_workers)
    ]

    # Outer loop: one image per iteration (AIE loop).
    # Inner loop: plain Python range over OC channels — unrolled at build time
    # to avoid nested range_ miscompilation (mlir-aie issue #1547). Emitting
    # planes in (batch, oc) order matches the contiguous dst layout.
    def make_core_fn(image_count):
        def ext_core_fn(of_wts, of_in, of_out, function):
            wts = of_wts.acquire(1)
            for _ in range_(image_count):
                img = of_in.acquire(1)
                for oc_idx in range(oc):
                    plane = of_out.acquire(1)
                    # The shape and op_params reach the core function as
                    # -DGGML_CONV2D_* literals, so only oc_idx is passed.
                    function(img, wts, plane, oc_idx)
                    of_out.release(1)
                of_in.release(1)
            of_wts.release(1)

        return ext_core_fn

    # The split image is complete before the first plane, so the image goes back to its fifo
    # (and the DMA can fill the next one) while this one's planes are computed.
    def make_split_core_fn(image_count):
        def ext_core_fn(of_wts, of_in, of_out, split, split_function, function):
            wts = of_wts.acquire(1)
            for _ in range_(image_count):
                img = of_in.acquire(1)
                split_function(img, split)
                of_in.release(1)
                for oc_idx in range(oc):
                    plane = of_out.acquire(1)
                    function(split, wts, plane, oc_idx)
                    of_out.release(1)
            of_wts.release(1)

        return ext_core_fn

    if use_split_image or flat_elems:
        split_ty = np.ndarray[(flat_elems or split_elems,), np.dtype[bfloat16]]
        workers = [
            Worker(
                make_split_core_fn(images_per_worker[w]),
                fn_args=[
                    wts_conss[w],
                    of_ins[w].cons(),
                    of_outs[w].prod(),
                    Buffer(split_ty, name=f"split{w}"),
                    *functions,
                ],
                stack_size=stack_size,
            )
            for w in range(num_workers)
        ]
    else:
        workers = [
            Worker(
                make_core_fn(images_per_worker[w]),
                fn_args=[wts_conss[w], of_ins[w].cons(), of_outs[w].prod(), *functions],
                stack_size=stack_size,
            )
            for w in range(num_workers)
        ]

    # Buffer-mapping contract: declare buffers in src0, src1, dst order.
    # src0 is the kernel weights; src1 is the input image.
    kernel_numel = kernel_tensor.numel()
    kernel_tensor_ty = np.ndarray[(kernel_numel,), np.dtype[kernel_tensor.dtype]]
    image_tensor_ty = np.ndarray[(image_size * n,), np.dtype[image_tensor.dtype]]
    output_tensor_ty = np.ndarray[(out_per_image * n,), np.dtype[output_tensor.dtype]]

    def sequence(a_kernel, a_in, b_out, wts_prod, in_prods, out_conses):
        # Broadcast the full weight tensor to every worker through one channel.
        wts_prod.fill(a_kernel)
        # Each worker reads its contiguous slice of the batch.
        for w in range(num_workers):
            count = images_per_worker[w]
            in_tap = batch_slice_tap(n, image_size, image_starts[w], count)
            in_prods[w].fill(a_in, in_tap)
        for w in range(num_workers):
            count = images_per_worker[w]
            out_tap = batch_slice_tap(n, out_per_image, image_starts[w], count)
            out_conses[w].drain(b_out, out_tap, wait=True)

    rt = Runtime(
        sequence,
        [
            kernel_tensor_ty,
            image_tensor_ty,
            output_tensor_ty,
            of_wts.prod(),
            [of_in.prod() for of_in in of_ins],
            [of_out.cons() for of_out in of_outs],
        ],
    )

    return Program(arch_to_device(arch), rt, workers=workers).resolve_program()


def _create_external_functions(
    image_tensor,
    output_tensor,
    image_size: int,
    wts_size: int,
    plane_size: int,
    shape: dict,
    split_elems: int,
    ring_cap: int,
    flat_elems: int = 0,
    flat_plane: int = 0,
) -> list[ExternalFunction]:
    """Create the ExternalFunctions for the conv_2d core functions.

    Args:
        image_tensor: Image input tensor (src1); also supplies the dtype for
            the weight buffer (kernel and image always share a dtype here).
        output_tensor: Output tensor.
        image_size: Elements in one input image (IC * IH * IW).
        wts_size: Elements in the full weight tensor (KW * KH * IC * OC).
        plane_size: Elements in one output plane (OW * OH).
        shape: Extents and op_params to bake in as -DGGML_CONV2D_* literals, so
            the core function compiles with constant loop bounds and strides
            instead of runtime arguments.
        split_elems: Elements of the split image (aie2), or 0 to compute each plane straight
            from the image.
        ring_cap: Bytes the aie2 ring path may keep on the stack (GGML_CONV2D_RING_BYTES).
        flat_elems: Elements of the aie2p flat split image, or 0 for the other paths.
        flat_plane: Elements of the aie2p flat output plane (with flat_elems).

    Returns:
        [ggml_op_conv_2d] or, with a split image, [ggml_op_conv_2d_split_image,
        ggml_op_conv_2d_split] (aie2) or [ggml_op_conv_2d_flat_image, ggml_op_conv_2d_flat]
        (aie2p): the core functions in the order the worker takes them.

    """
    op_name = "GGML_OP_CONV_2D"
    current_dir = Path(__file__).resolve().parent
    object_file_name = f"{op_name.lower()}_core_function.o"
    source_file = str(current_dir / "conv_2d.cc")
    compile_flags = [
        f"-DINPUT_DTYPE={dtype_to_str(image_tensor.dtype)}",
        f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        *shape_defines("GGML_CONV2D", shape),
        f"-DGGML_CONV2D_RING_BYTES={ring_cap}",
    ]
    image_ty = np.ndarray[(image_size,), np.dtype[image_tensor.dtype]]
    wts_ty = np.ndarray[(wts_size,), np.dtype[image_tensor.dtype]]
    plane_ty = np.ndarray[(plane_size,), np.dtype[output_tensor.dtype]]
    if flat_elems:
        split_ty = np.ndarray[(flat_elems,), np.dtype[bfloat16]]
        flat_plane_ty = np.ndarray[(flat_plane,), np.dtype[output_tensor.dtype]]
        compile_flags.append(f"-DGGML_CONV2D_FLAT={flat_elems}")
        return [
            ExternalFunction(
                name=f"{op_name.lower()}_flat_image",
                object_file_name=object_file_name,
                source_file=source_file,
                arg_types=[image_ty, split_ty],
                compile_flags=compile_flags,
            ),
            ExternalFunction(
                name=f"{op_name.lower()}_flat",
                object_file_name=object_file_name,
                source_file=source_file,
                arg_types=[split_ty, wts_ty, flat_plane_ty, np.int32],
                compile_flags=compile_flags,
            ),
        ]
    if split_elems == 0:
        return [
            ExternalFunction(
                name=op_name.lower(),
                object_file_name=object_file_name,
                source_file=source_file,
                arg_types=[image_ty, wts_ty, plane_ty, np.int32],
                compile_flags=compile_flags,
            )
        ]
    split_ty = np.ndarray[(split_elems,), np.dtype[bfloat16]]
    compile_flags.append(f"-DGGML_CONV2D_SPLIT_IMAGE={split_elems}")
    return [
        ExternalFunction(
            name=f"{op_name.lower()}_split_image",
            object_file_name=object_file_name,
            source_file=source_file,
            arg_types=[image_ty, split_ty],
            compile_flags=compile_flags,
        ),
        ExternalFunction(
            name=f"{op_name.lower()}_split",
            object_file_name=object_file_name,
            source_file=source_file,
            arg_types=[split_ty, wts_ty, plane_ty, np.int32],
            compile_flags=compile_flags,
        ),
    ]
