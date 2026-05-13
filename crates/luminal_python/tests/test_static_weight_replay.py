from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from luminal import process_pt2
from luminal.dtype_util import torch_dtype_code
from luminal.luminal import _native_factory_capsule
from luminal.main import luminal_backend
from luminal.pt2 import _decomp_table
from luminal.pt2 import compile as luminal_compile

try:
    from luminal.luminal import _cuda_lite_factory_capsule
except ImportError:
    _cuda_lite_factory_capsule = None


def _first_output(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _export_pt2(
    model: torch.nn.Module, example_input: torch.Tensor
) -> tuple[tempfile.TemporaryDirectory, str]:
    tmpdir = tempfile.TemporaryDirectory()
    pt2_path = Path(tmpdir.name) / "model.pt2"
    ep = torch.export.export(model, (example_input,), strict=False)
    ep = ep.run_decompositions(_decomp_table())
    torch.export.save(ep, str(pt2_path))
    return tmpdir, str(pt2_path)


def _run_host_graph(graph, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu().contiguous()
    graph.set_input_from_ptr(
        graph.input_names[0],
        x.data_ptr(),
        x.numel() * x.element_size(),
        torch_dtype_code(x.dtype),
    )
    graph.run()
    out = torch.tensor(graph.get_output(graph.output_names[0]), dtype=torch.float32)
    return out.reshape(tuple(graph.output_shapes[0]))


def _run_device_graph(graph, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().contiguous()
    graph.set_input_device_ptr(
        graph.input_names[0],
        x.data_ptr(),
        x.numel() * x.element_size(),
    )
    graph.run()
    out = torch.tensor(graph.get_output(graph.output_names[0]), dtype=torch.float32)
    return out.reshape(tuple(graph.output_shapes[0]))


def test_compile_cpu_static_weights_replayed_across_runs():
    class Mdl(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("bias", torch.tensor([1.0, -2.0], dtype=torch.float32))

        def forward(self, x):
            return x + self.bias + torch.tensor([0.25, 0.5], dtype=torch.float32)

    model = Mdl().eval()
    compiled = luminal_compile(model, torch.randn(2), search_iterations=3)

    for x in (torch.tensor([3.0, 4.0]), torch.tensor([-5.0, 6.0])):
        ref = model(x)
        out = _first_output(compiled(x))
        assert torch.allclose(out, ref, atol=1e-5), (
            f"max_diff={torch.max(torch.abs(out - ref)).item():.2e}"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-only — exercises lifted original-weight device-pointer replay",
)
def test_torch_compile_cuda_weight_device_ptrs_replayed(device: torch.device):
    class Mdl(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 4)

        def forward(self, x):
            return torch.relu(self.lin(x))

    model = Mdl().eval().to(device)
    compiled = torch.compile(model, backend=luminal_backend)

    with torch.no_grad():
        for _ in range(3):
            x = torch.randn(4, 8, device=device)
            ref = model(x)
            out = compiled(x)
            assert torch.allclose(out, ref, atol=1e-5), (
                f"max_diff={torch.max(torch.abs(out - ref)).item():.2e}"
            )


def test_process_pt2_set_weight_from_ptr_persists_across_runs():
    class Mdl(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

        def forward(self, x):
            return x * self.w

    model = Mdl().eval()
    tmpdir, pt2_path = _export_pt2(model, torch.randn(2))
    try:
        graph = process_pt2(pt2_path, "", 1, _native_factory_capsule(), None)

        weight = torch.tensor([2.0, 3.0], dtype=torch.float32)
        graph.set_weight_from_ptr(
            "w",
            weight.data_ptr(),
            weight.numel() * weight.element_size(),
            torch_dtype_code(weight.dtype),
        )

        for x in (torch.tensor([5.0, 7.0]), torch.tensor([-11.0, 13.0])):
            out = _run_host_graph(graph, x)
            ref = x * weight
            assert torch.allclose(out, ref, atol=1e-5), (
                f"max_diff={torch.max(torch.abs(out - ref)).item():.2e}"
            )
    finally:
        tmpdir.cleanup()


@pytest.mark.skipif(
    not torch.cuda.is_available() or _cuda_lite_factory_capsule is None,
    reason="CUDA-only — exercises persistent zero-copy device weight replay",
)
def test_process_pt2_set_weight_device_ptr_persists_across_runs():
    class Mdl(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

        def forward(self, x):
            return x * self.w

    model = Mdl().eval()
    tmpdir, pt2_path = _export_pt2(model, torch.randn(2))
    try:
        graph = process_pt2(pt2_path, "", 1, _cuda_lite_factory_capsule(), None)

        weight = torch.tensor([2.0, 3.0], dtype=torch.float32, device="cuda")
        graph.set_weight_device_ptr(
            "w",
            weight.data_ptr(),
            weight.numel() * weight.element_size(),
        )

        for x in (
            torch.tensor([5.0, 7.0], dtype=torch.float32, device="cuda"),
            torch.tensor([-11.0, 13.0], dtype=torch.float32, device="cuda"),
        ):
            out = _run_device_graph(graph, x)
            ref = x.cpu() * weight.cpu()
            assert torch.allclose(out, ref, atol=1e-5), (
                f"max_diff={torch.max(torch.abs(out - ref)).item():.2e}"
            )
    finally:
        tmpdir.cleanup()
