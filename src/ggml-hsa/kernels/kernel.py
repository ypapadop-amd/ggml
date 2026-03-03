# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

from dataclasses import dataclass
from typing import Any
from pathlib import Path


@dataclass(frozen=True)
class Kernel:
    """Dataclass representing a kernel."""

    name: str
    source_file: str | Path
    function: Any = None
