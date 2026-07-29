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
    ExternalFunction,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    dtype_to_str,
)
from aie.iron.controlflow import range_

from .utils import arch_to_device, partition_units

# Number of int32 op_params: {s0, s1, p0, p1, d0, d1}
_CONV2D_PARAMS_SIZE = 6 * 4

# Cap on data-parallel workers (compute tiles). Each worker needs one shim DMA
# for its image slice and one for its output slice; the weights are broadcast
# through a single shared shim channel. The shim budget is ~16 endpoints on NPU1
# (aie2), so 2*W + 1 (weights) <= 16 gives W <= 7.
_MAX_WORKERS = 7


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

    # Element counts per tile
    image_size = ic * ih * iw  # one full image (all channels)
    wts_size = kw * kh * ic * oc  # full weight tensor (same for all images)
    plane_size = ow * oh  # one output plane for a single output channel

    function = _create_external_function(
        image_tensor=image_tensor,
        output_tensor=output_tensor,
        image_size=image_size,
        wts_size=wts_size,
        plane_size=plane_size,
    )

    # Distribute the N images across compute tiles. Each worker owns independent
    # weight/image/output fifos and processes a contiguous slice of the batch.
    num_workers = min(_MAX_WORKERS, n)
    images_per_worker, image_starts = partition_units(num_workers, n)

    # One output plane per (image, oc); planes per image = oc.
    out_per_image = plane_size * oc

    image_tile_ty = np.ndarray[(image_size,), np.dtype[image_tensor.dtype]]
    wts_tile_ty = np.ndarray[(wts_size,), np.dtype[kernel_tensor.dtype]]
    plane_tile_ty = np.ndarray[(plane_size,), np.dtype[output_tensor.dtype]]

    # depth=1: the full weight tensor is broadcast once to all workers through a
    # single shim channel (one producer, one consumer handle per worker), then
    # held for the whole run. Per-worker weight fifos would each need their own
    # shim DMA and exhaust the array's DMA budget.
    of_wts = ObjectFifo(wts_tile_ty, depth=1, name="wts")
    wts_conss = [of_wts.cons() for _ in range(num_workers)]
    of_ins = [ObjectFifo(image_tile_ty, name=f"in{w}") for w in range(num_workers)]
    of_outs = [ObjectFifo(plane_tile_ty, name=f"out{w}") for w in range(num_workers)]

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
                    function(
                        img,
                        wts,
                        plane,
                        oc_idx,
                        iw,
                        ih,
                        ic,
                        kw,
                        kh,
                        ow,
                        oh,
                        s0,
                        s1,
                        p0,
                        p1,
                        d0,
                        d1,
                    )
                    of_out.release(1)
                of_in.release(1)
            of_wts.release(1)

        return ext_core_fn

    workers = [
        Worker(
            make_core_fn(images_per_worker[w]),
            fn_args=[wts_conss[w], of_ins[w].cons(), of_outs[w].prod(), function],
        )
        for w in range(num_workers)
    ]

    # Buffer-mapping contract: declare buffers in src0, src1, dst order.
    # src0 is the kernel weights; src1 is the input image.
    rt = Runtime()
    kernel_numel = kernel_tensor.numel()
    kernel_tensor_ty = np.ndarray[(kernel_numel,), np.dtype[kernel_tensor.dtype]]
    image_tensor_ty = np.ndarray[(image_size * n,), np.dtype[image_tensor.dtype]]
    output_tensor_ty = np.ndarray[(out_per_image * n,), np.dtype[output_tensor.dtype]]

    with rt.sequence(kernel_tensor_ty, image_tensor_ty, output_tensor_ty) as (
        a_kernel,
        a_in,
        b_out,
    ):
        rt.start(*workers)
        # Broadcast the full weight tensor to every worker through one channel.
        rt.fill(of_wts.prod(), a_kernel)
        # Each worker reads its contiguous slice of the batch.
        for w in range(num_workers):
            count = images_per_worker[w]
            in_tap = TensorAccessPattern(
                (n, image_size),
                image_starts[w] * image_size,
                [1, count, 1, image_size],
                [0, image_size, 0, 1],
            )
            rt.fill(of_ins[w].prod(), a_in, in_tap)
        for w in range(num_workers):
            count = images_per_worker[w]
            out_tap = TensorAccessPattern(
                (n, out_per_image),
                image_starts[w] * out_per_image,
                [1, count, 1, out_per_image],
                [0, out_per_image, 0, 1],
            )
            rt.drain(of_outs[w].cons(), b_out, out_tap, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program()


def _create_external_function(
    image_tensor,
    output_tensor,
    image_size: int,
    wts_size: int,
    plane_size: int,
) -> ExternalFunction:
    """Create the ExternalFunction for the conv_2d core function.

    Args:
        image_tensor: Image input tensor (src1); also supplies the dtype for
            the weight buffer (kernel and image always share a dtype here).
        output_tensor: Output tensor.
        image_size: Elements in one input image (IC * IH * IW).
        wts_size: Elements in the full weight tensor (KW * KH * IC * OC).
        plane_size: Elements in one output plane (OW * OH).

    Returns:
        The configured ExternalFunction.

    """
    op_name = "GGML_OP_CONV_2D"
    current_dir = Path(__file__).resolve().parent
    return ExternalFunction(
        name=op_name.lower(),
        object_file_name=f"{op_name.lower()}_core_function.o",
        source_file=str(current_dir / "conv_2d.cc"),
        arg_types=[
            np.ndarray[(image_size,), np.dtype[image_tensor.dtype]],  # in
            np.ndarray[(wts_size,), np.dtype[image_tensor.dtype]],  # wts
            np.ndarray[(plane_size,), np.dtype[output_tensor.dtype]],  # out
            np.int32,  # oc_idx
            np.int32,  # iw
            np.int32,  # ih
            np.int32,  # ic
            np.int32,  # kw
            np.int32,  # kh
            np.int32,  # ow
            np.int32,  # oh
            np.int32,  # s0
            np.int32,  # s1
            np.int32,  # p0
            np.int32,  # p1
            np.int32,  # d0
            np.int32,  # d1
        ],
        compile_flags=[
            f"-DINPUT_DTYPE={dtype_to_str(image_tensor.dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
