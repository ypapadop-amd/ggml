# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""GGML HSA Kernels package.

This package provides IRON-based kernel implementations for GGML operations
targeting AMD AIE (AI Engine) devices. It exposes the main compilation function
and tensor descriptor utilities needed for JIT kernel compilation.
"""

from .build import ggml_compile_op
from .kernel import Kernel
from .tensor_desc import TensorDesc, ggml_tensor_to_tensordesc

__all__ = [
    "Kernel",
    "TensorDesc",
    "ggml_compile_op",
    "ggml_tensor_to_tensordesc",
]
