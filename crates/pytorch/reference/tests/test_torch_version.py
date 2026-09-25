"""Reject incompatible PyTorch builds before loading native extensions."""

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from luminal_reference import _torch_version


@pytest.fixture
def guard_path():
    return _torch_version.__file__


@pytest.mark.parametrize(
    "version", ["2.11.0", "2.13.1+cu130", "2.14.0rc1", "2.14.0.dev20260901"]
)
def test_rejects_older_torch(monkeypatch, version, guard_path):
    monkeypatch.setattr(torch, "__version__", version)
    with pytest.raises(ImportError, match=r"requires PyTorch >= 2\.14\.0; found"):
        runpy.run_path(guard_path)


@pytest.mark.parametrize("version", ["2.14.0", "2.14.0+cu130", "2.14.1", "2.15.0"])
def test_accepts_supported_torch(monkeypatch, version, guard_path):
    monkeypatch.setattr(torch, "__version__", version)
    runpy.run_path(guard_path)


def test_package_imports_enforce_minimum_before_native_loading():
    root = Path(__file__).resolve().parents[4]
    paths = [
        root / "crates/pytorch/reference/python",
        root / "crates/pytorch/cuda_lite/python",
    ]
    script = """
import importlib
import torch
torch.__version__ = "2.13.0"
for package in ("luminal_reference", "luminal_cuda_lite"):
    try:
        importlib.import_module(package)
    except ImportError as error:
        assert "requires PyTorch >= 2.14.0; found 2.13.0" in str(error), str(error)
    else:
        raise AssertionError(f"{package} accepted unsupported PyTorch")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths))},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_no_generic_luminal_package():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; assert importlib.util.find_spec('luminal') is None",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
