# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Run a GPU KernelSpec's Triton kernel in the Triton interpreter (CPU); used by the tests.

The interpreter has to be enabled (TRITON_INTERPRET=1) before triton is imported, because the
@triton.jit helpers a kernel calls (e.g. tl.zeros) are bound at import time, so it runs in its
own process. Usage:

    TRITON_INTERPRET=1 python triton_interpret_run.py ARGS_JSON

ARGS_JSON holds ``module``, ``fn`` (a dispatch function in kernels/<module>.py), ``arch``,
``inputs`` (GGML shapes), ``output`` (GGML shape), ``input_files`` (.npy, f32), ``output_file``
(.npy written with the flat result) and, optionally, ``op_params`` (a list of bytes, default
empty) and ``out_dtype`` (the output's GGML dtype, ``f32`` or ``i32``, default ``f32``).
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import _PKG, _ensure_package


def main() -> None:
    """Build the spec, run its kernel with the baked launch geometry and save the output."""
    args = json.loads(sys.argv[1])
    _ensure_package()
    module = importlib.import_module(f"{_PKG}.{args['module']}")
    TensorDesc = importlib.import_module(f"{_PKG}.tensor_desc").TensorDesc

    def td(shape, dtype="f32"):
        return TensorDesc(dtype=dtype, shape=(*shape, *(1,) * (4 - len(shape))))

    out_dtype = args.get("out_dtype", "f32")
    (spec,) = getattr(module, args["fn"])(
        args["arch"],
        [td(s) for s in args["inputs"]],
        td(args["output"], out_dtype),
        bytearray(args.get("op_params", [])),
    )
    cfg = spec.config
    inputs = [torch.from_numpy(np.load(f)) for f in args["input_files"]]
    n_out = int(np.prod(args["output"]))
    output = (
        torch.full((n_out,), -1, dtype=torch.int32)
        if out_dtype == "i32"
        else torch.full((n_out,), float("nan"))
    )
    pointers = [name for name, kind in cfg["signature"].items() if kind != "constexpr"]
    cfg["triton_fn"][(cfg["num_programs"],)](
        **dict(zip(pointers, [*inputs, output], strict=True)), **cfg["constexprs"]
    )
    np.save(args["output_file"], output.numpy())


if __name__ == "__main__":
    main()
