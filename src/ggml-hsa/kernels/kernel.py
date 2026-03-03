# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Kernel:
    """Dataclass representing a kernel."""

    name: str
    source_file: str
    function: Any = None
