# core_function.py -*- Python -*-
#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

from typing import Callable


class CoreFunctionInfo:
    """
    Core function information.

    This class provides information necessary to compile a core function via Peano and use it in a kernel.
    """

    def __init__(
        self,
        source_file: str,
        exported_function,
        compile_args,
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

    def __str__(self):
        return (
            f'Source file: "{self.source_file}", '
            + f"Compile args: {self.compile_args}, "
            + f'Exported function: "{self.exported_function}", '
            + f'Object file: "{self.object_file}", '
            + f"Additional args: {self.additional_args}"
        )


def core_function(function_info=None) -> Callable:
    """Associates a core function with a kernel."""

    def wrapper(func):
        if function_info:
            func.core_function_info = function_info
        return func

    return wrapper
