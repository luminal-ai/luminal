"""YOLO v11n end-to-end tests using the luminal_cuda_lite backend.

This module exercises the YOLO v11n building blocks (Conv + BN, C3k2, the
SPPF/C2PSA backbone, the Detect head) and finally the full model through
``torch.compile(..., backend=luminal_backend)``.

The smaller per-block tests are useful when triaging which part of the
architecture starts diverging: incrementally building a model up is much
easier than debugging a 100-layer mismatch in one go.

Marked ``slow`` because the first run downloads ~6 MB of weights and the
luminal e-graph compile of the full model is non-trivial. Run with::

    uv run pytest tests/test_yolo_v11.py -v -s
"""

from typing import Callable

import pytest
import torch
import torch._dynamo

from luminal import luminal_backend


def _require_cuda(device: torch.device):
    if device.type != "cuda":
        pytest.skip("YOLO v11 examples require the CUDA backend.")


def _require_ultralytics():
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"ultralytics not installed: {exc}")


def _yolo_model(device: torch.device, decode_only: bool = True):
    """Load yolo11n with BN folded into Conv. Returns the eager torch model."""
    from ultralytics import YOLO

    yolo = YOLO("yolo11n.pt")
    pt_model = yolo.model.eval()
    pt_model.fuse()
    if decode_only:
        pt_model.model[-1].export = True
    pt_model.to(device)
    return pt_model


@pytest.mark.slow
def test_yolo_v11n_first_three_layers(device: torch.device):
    """Compile only the first three layers (Conv, Conv, C3k2) — exercises the
    chunk + bottleneck residual + concat pattern that's the trickiest piece
    of the model graph."""
    _require_cuda(device)
    _require_ultralytics()

    pt_model = _yolo_model(device, decode_only=True)

    class FirstThree(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.layers = torch.nn.ModuleList([backbone[i] for i in range(3)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    sub = FirstThree(pt_model.model).to(device).eval()
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640, dtype=torch.float32, device=device)

    with torch.no_grad():
        ref = sub(x)
    torch._dynamo.reset()
    compiled: Callable = torch.compile(sub, backend=luminal_backend)
    with torch.no_grad():
        out = compiled(x)

    max_diff = torch.max(torch.abs(out - ref)).item()
    print(f"yolo11n[:3] max_diff vs PyTorch eager: {max_diff:.4e}")
    assert torch.allclose(out, ref, atol=1e-3), (
        f"yolo11n[:3] outputs differ — max_diff={max_diff:.4e}"
    )


@pytest.mark.slow
def test_yolo_v11n_end_to_end(device: torch.device):
    """Full yolo11n forward via torch.compile. The compile may be slow on
    machines without strong egglog parallelism — see the example README for
    the standalone Rust binary alternative."""
    _require_cuda(device)
    _require_ultralytics()

    pt_model = _yolo_model(device)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640, dtype=torch.float32, device=device)

    with torch.no_grad():
        ref = pt_model(x)
    if isinstance(ref, (list, tuple)):
        ref = ref[0]

    torch._dynamo.reset()
    compiled: Callable = torch.compile(pt_model, backend=luminal_backend)
    with torch.no_grad():
        out = compiled(x)
    if isinstance(out, (list, tuple)):
        out = out[0]

    max_diff = torch.max(torch.abs(out - ref)).item()
    print(f"YOLO v11n max_diff vs PyTorch eager: {max_diff:.4e}")
    assert torch.allclose(out, ref, atol=1e-3), (
        f"YOLO v11n outputs differ from PyTorch eager — max_diff={max_diff:.4e}"
    )
