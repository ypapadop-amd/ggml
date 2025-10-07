#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

from dataclasses import dataclass
from typing import Callable


@dataclass
class CoreFunctionInfo:
    """Core function information.

    This class provides information necessary to compile a core function via Peano and use it in a kernel.

    Attributes:
        source_file (str): Path to the source file containing the core function.
        exported_function (str): Name of the function to be exported.
        compile_args (None | list[str]): List of arguments to be passed to the compiler.
        object_file (None | str): Path to the object file generated after compilation.
        additional_args (None | dict[str, str]): Additional arguments that may be required for the function.
    """

    source_file: str
    exported_function: str
    compile_args: None | list[str] = None
    additional_args: None | dict[str, str] = None
    object_file: None | str = None


def core_function(function_info=None) -> Callable:
    """Associates a core function with a kernel."""

    def wrapper(func):
        if function_info:
            func.core_function_info = function_info
        return func

    return wrapper
