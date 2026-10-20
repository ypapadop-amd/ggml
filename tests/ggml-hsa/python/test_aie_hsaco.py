# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for the hsaco packer wrapper (no NPU or packer binaries required).

aie_hsaco.py uses flat imports (from kernel import ...), so the kernels dir must be on
sys.path, like build_triton.py's own tests. The packer itself (aie.compiler.hsaco.pack)
is monkeypatched: these tests check pack_aie_hsaco's own input validation and error
translation, not the packer.
"""

import sys
from pathlib import Path

import pytest

from conftest import KERNELS_DIR

sys.path.insert(0, str(KERNELS_DIR))

pytest.importorskip("aie.compiler.hsaco")

import aie_hsaco  # noqa: E402
from kernel import Backend, KernelSpec  # noqa: E402


def _spec(num_inputs=2):
    return KernelSpec(
        backend=Backend.IRON,
        op_name="ADD",
        arch="aie2p",
        input_tensors=[object()] * num_inputs,
        output_tensor=object(),
        function=lambda: None,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "needs either"),
        ({"xclbin_path": "a.xclbin"}, "must be given together"),
        ({"insts_path": "a.bin"}, "must be given together"),
        (
            {"full_elf_path": "a.elf", "xclbin_path": "a.xclbin"},
            "mutually exclusive",
        ),
        (
            {"full_elf_path": "a.elf", "insts_path": "a.bin"},
            "mutually exclusive",
        ),
        (
            {"full_elf_path": "a.elf", "xclbin_path": "a.xclbin", "insts_path": "a.bin"},
            "mutually exclusive",
        ),
    ],
)
def test_invalid_input_combinations_raise(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        aie_hsaco.pack_aie_hsaco(_spec(), "k", tmp_path, **kwargs)


def test_full_elf_without_kernel_name_requires_exactly_one_kernel(tmp_path, monkeypatch):
    monkeypatch.setattr(
        aie_hsaco.pack,
        "kernels_from_full_elf",
        lambda *a, **k: [{"name": "main:sequence"}, {"name": "main:extra"}],
    )
    with pytest.raises(RuntimeError, match="holds 2 kernels"):
        aie_hsaco.pack_aie_hsaco(_spec(), "k", tmp_path, full_elf_path=tmp_path / "a.elf")


def test_full_elf_kernel_name_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(
        aie_hsaco.pack,
        "kernels_from_full_elf",
        lambda *a, **k: [{"name": "main:sequence"}],
    )
    with pytest.raises(RuntimeError, match="no kernel named"):
        aie_hsaco.pack_aie_hsaco(
            _spec(),
            "k",
            tmp_path,
            full_elf_path=tmp_path / "a.elf",
            elf_kernel_name="does-not-exist",
        )


def test_full_elf_single_kernel_packs_without_name(tmp_path, monkeypatch):
    monkeypatch.setattr(
        aie_hsaco.pack, "kernels_from_full_elf", lambda *a, **k: [{"name": "main:sequence"}]
    )
    monkeypatch.setattr(aie_hsaco.pack, "build_section", lambda *a, **k: b"")
    # Real ensure_hsaco/inject write the scratch file on disk; pack_aie_hsaco renames it into
    # place afterward, so the mock has to leave something there for that rename to find.
    monkeypatch.setattr(aie_hsaco.pack, "ensure_hsaco", lambda p, *a, **k: Path(p).touch())
    monkeypatch.setattr(aie_hsaco.pack, "inject", lambda *a, **k: None)
    out = aie_hsaco.pack_aie_hsaco(_spec(), "k", tmp_path, full_elf_path=tmp_path / "a.elf")
    assert out == tmp_path / "k.hsaco"


def test_packer_cli_failure_surfaces_stderr(tmp_path, monkeypatch):
    def fake_main(argv):
        sys.stderr.write("usage: aie-hsaco ...\naie-hsaco: error: bad xclbin\n")
        raise SystemExit(2)

    monkeypatch.setattr(aie_hsaco.pack, "main", fake_main)
    with pytest.raises(RuntimeError, match="bad xclbin"):
        aie_hsaco.pack_aie_hsaco(
            _spec(), "k", tmp_path, xclbin_path=tmp_path / "a.xclbin", insts_path=tmp_path / "a.bin"
        )


def test_packer_cli_failure_without_parseable_reason_uses_exit_code(tmp_path, monkeypatch):
    def fake_main(argv):
        raise SystemExit(1)

    monkeypatch.setattr(aie_hsaco.pack, "main", fake_main)
    with pytest.raises(RuntimeError, match="exit status 1"):
        aie_hsaco.pack_aie_hsaco(
            _spec(), "k", tmp_path, xclbin_path=tmp_path / "a.xclbin", insts_path=tmp_path / "a.bin"
        )


def test_packer_cli_failure_with_no_exit_code_normalizes_to_one(tmp_path, monkeypatch):
    def fake_main(argv):
        raise SystemExit  # e.code is None, not 0 or a string

    monkeypatch.setattr(aie_hsaco.pack, "main", fake_main)
    with pytest.raises(RuntimeError, match="exit status 1"):
        aie_hsaco.pack_aie_hsaco(
            _spec(), "k", tmp_path, xclbin_path=tmp_path / "a.xclbin", insts_path=tmp_path / "a.bin"
        )


def test_output_directory_is_created_if_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        aie_hsaco.pack, "kernels_from_full_elf", lambda *a, **k: [{"name": "main:sequence"}]
    )
    monkeypatch.setattr(aie_hsaco.pack, "build_section", lambda *a, **k: b"")
    monkeypatch.setattr(aie_hsaco.pack, "ensure_hsaco", lambda p, *a, **k: Path(p).touch())
    monkeypatch.setattr(aie_hsaco.pack, "inject", lambda *a, **k: None)
    missing_dir = tmp_path / "does" / "not" / "exist"
    out = aie_hsaco.pack_aie_hsaco(_spec(), "k", missing_dir, full_elf_path=tmp_path / "a.elf")
    assert out == missing_dir / "k.hsaco"
