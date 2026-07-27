# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""GGML HSA kernels for AMD AIE devices: compilation entry point and tensor descriptors."""

from .build import CompilerConfig, ggml_compile_op
from .kernel import Kernel
from .tensor_desc import TensorDesc, ggml_tensor_to_tensordesc

__all__ = [
    "CompilerConfig",
    "Kernel",
    "TensorDesc",
    "ggml_compile_op",
    "ggml_tensor_to_tensordesc",
]
