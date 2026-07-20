# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Shared pytest fixtures for the ggml-hsa kernel tests.

Modules under ``src/ggml-hsa/kernels`` (e.g. ``mul_mat``) use package-relative
imports (``from .kernel import ...``), so they cannot be imported flat. This
maps a synthetic package onto the kernels directory once and exposes a loader,
avoiding the heavy ``kernels/__init__.py`` (which imports ``build.py``).
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

KERNELS_DIR = Path(__file__).resolve().parents[2] / "src" / "ggml-hsa" / "kernels"
_PKG = "_ggml_hsa_kernels"


def _ensure_package() -> None:
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(KERNELS_DIR)]
        pkg.__package__ = _PKG
        sys.modules[_PKG] = pkg


@pytest.fixture(scope="session")
def import_kernel_module():
    """Return a loader that imports a kernels submodule under package context.

    Returns:
        A callable ``load(name)`` that imports ``kernels/<name>.py`` (resolving
        its package-relative imports) and returns the module object.
    """
    _ensure_package()

    def load(name: str):
        return importlib.import_module(f"{_PKG}.{name}")

    return load
