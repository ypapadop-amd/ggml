# core_function.py -*- Python -*-
#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

from typing import Callable


class CoreFunctionInfo:
    """Core function information.

    This class provides information necessary to compile a core function via Peano and use it in a kernel.

    Attributes:
        source_file (str): Path to the source file containing the core function.
        exported_function (str): Name of the function to be exported.
        compile_args (list): List of arguments to be passed to the compiler.
        object_file (str): Path to the object file generated after compilation.
        additional_args (dict): Additional arguments that may be required for the function.
    """

    def __init__(
        self,
        source_file: str,
        exported_function: str | dict[str, str],
        compile_args: list[str],
        additional_args=None,
    ):
        self.source_file = source_file
        if compile_args is None:
            self.compile_args = []
        else:
            self.compile_args = compile_args
        self.exported_function = exported_function
        self.object_file = None
        if additional_args is None:
            self.additional_args = {}
        else:
            self.additional_args = additional_args

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(source_file={self.source_file} "
            f"compile_args={self.compile_args} "
            f"exported_function={self.exported_function} "
            f"object_file={self.object_file} "
            f"additional_args={self.additional_args})"
        )


def core_function(function_info=None) -> Callable:
    """Associates a core function with a kernel."""

    def wrapper(func):
        if function_info:
            func.core_function_info = function_info
        return func

    return wrapper
