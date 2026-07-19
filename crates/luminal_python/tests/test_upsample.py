from typing import Callable

import pytest
import torch
import torch._dynamo
from test_models import (
    UpsampleNearestScaleModel,
    UpsampleNearestScaleHWModel,
    UpsampleNearestSizeModel,
)

from luminal import luminal_backend

# aten.upsample_nearest2d.vec (SD UNet Upsample2D blocks)


def test_upsample_nearest_scale_2x(device):
    """scale_factor=2.0 -> output_size=None, scale_factors=[2,2] (the SD path)."""
    model = UpsampleNearestScaleModel(scale_factor=2.0).to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((2, 3, 5, 7), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_upsample_nearest_scale_non_square(device):
    """Per-axis scale (2x height, 3x width)."""
    model = UpsampleNearestScaleHWModel(scale_h=2.0, scale_w=3.0).to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((2, 3, 4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_upsample_nearest_size(device):
    """size=(H, W) -> output_size=[H, W], scale_factors=None branch."""
    model = UpsampleNearestSizeModel(size=(16, 16)).to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((1, 4, 8, 8), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


# Per-axis skip branch (scale == 1 on one axis)


def test_upsample_nearest_height_only(device):
    """(2, 1): hits the `scale_width == 1` skip."""
    model = UpsampleNearestScaleHWModel(scale_h=2.0, scale_w=1.0).to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((2, 3, 4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_upsample_nearest_width_only(device):
    """(1, 2): hits the `scale_height == 1` skip."""
    model = UpsampleNearestScaleHWModel(scale_h=1.0, scale_w=2.0).to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((2, 3, 4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


# General path: fractional scales, downsampling, divergent explicit scales


def test_upsample_nearest_fractional_scale(device):
    """1.5x -> gather path; nearest is pure selection, must match eager exactly."""
    model = UpsampleNearestScaleModel(scale_factor=1.5).to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((1, 3, 4, 6), device=device)
    assert torch.equal(model_compiled(x), model(x))


def test_upsample_nearest_downsample(device):
    """0.5x nearest downsample."""
    model = UpsampleNearestScaleModel(scale_factor=0.5).to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((2, 3, 8, 6), device=device)
    assert torch.equal(model_compiled(x), model(x))


def test_upsample_nearest_size_fractional(device):
    """size=(5, 7) from (3, 4): non-divisible output_size overload."""
    model = UpsampleNearestSizeModel(size=(5, 7)).to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((1, 2, 3, 4), device=device)
    assert torch.equal(model_compiled(x), model(x))


def test_upsample_nearest_mixed_paths(device):
    """(2.0, 1.5): H takes the integer view path, W the gather path."""
    model = UpsampleNearestScaleHWModel(scale_h=2.0, scale_w=1.5).to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((2, 3, 4, 6), device=device)
    assert torch.equal(model_compiled(x), model(x))


def test_upsample_nearest_scale_nonintegral_product(device):
    """scale=1.7 on in=5: in*s non-integral, so floor(j/s) != floor(j*in/out)
    — indices must honor the provided scale (ATen's general branch)."""
    model = UpsampleNearestScaleModel(scale_factor=1.7).to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((1, 2, 5, 5), device=device)
    assert torch.equal(model_compiled(x), model(x))


def test_upsample_nearest_downsample_nonintegral(device):
    """scale=0.7 on in=6 (out=4): fractional downsample with provided scale."""
    model = UpsampleNearestScaleModel(scale_factor=0.7).to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((1, 2, 6, 6), device=device)
    assert torch.equal(model_compiled(x), model(x))


def test_upsample_nearest_noninteger_scale_same_size(device):
    """scale=1.05, in=10 -> out=10: ATen's out==in kernel fast path ignores
    the scale — identity, NOT floor(j/1.05)."""
    model = UpsampleNearestScaleModel(scale_factor=1.05).to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((1, 2, 10, 10), device=device)
    ref = model(x)
    assert torch.equal(ref, x)
    assert torch.equal(model_compiled(x), ref)


# dtype: fp16


def test_upsample_nearest_fp16(device):
    """Pure movement: fp16 must round-trip bit-exactly."""
    if device.type != "cuda":
        pytest.skip("fp16 upsample exercised on the CUDA backend")
    model = UpsampleNearestScaleModel(scale_factor=2.0).to(device).half()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((2, 3, 5, 7), device=device, dtype=torch.float16)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-3)
