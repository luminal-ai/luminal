"""Outputs the caller keeps: every call's output is its own live storage.

A caller may hold every call's result at once and read it later, or drop
each before the next call. Either way each output holds its own call's
numbers, at the shape and strides eager gives it. These hold for every
backend.
"""

import importlib

import pytest
import torch
from backend_test_utils import selected_backend

ATOL = 1e-3
RTOL = 1e-3


def _compiler():
    return importlib.import_module(
        f"luminal_torch_{selected_backend()}_compiler"
    ).Compiler()


def _linear_stack():
    """Three matmuls, each with a bias."""
    return torch.nn.Sequential(
        torch.nn.Linear(64, 64),
        torch.nn.Linear(64, 64),
        torch.nn.Linear(64, 64),
    ).eval()


def _elementwise():
    """A small pointwise chain."""

    class Chain(torch.nn.Module):
        def forward(self, x):
            return (x * 2 + 1) * x - 0.5

    return Chain().eval()


MODELS = {"linear_stack": _linear_stack, "elementwise": _elementwise}


@pytest.mark.parametrize("build", list(MODELS.values()), ids=list(MODELS))
def test_outputs_held_across_calls_stay_correct(device, build):
    """Twelve calls, every output kept: each holds its own call's numbers,
    so no call wrote into storage an earlier one still owns."""
    torch.manual_seed(0)
    model = build().to(device)
    compiled = torch.compile(model, backend=_compiler())
    calls = []
    with torch.no_grad():
        for _ in range(12):
            x = torch.randn(8, 64, device=device)
            calls.append((x, compiled(x)))
        pointers = [got.data_ptr() for _, got in calls]
        assert len(set(pointers)) == len(pointers), "two held outputs share an address"
        for x, got in calls:
            torch.testing.assert_close(got, model(x), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("build", list(MODELS.values()), ids=list(MODELS))
def test_rebinding_loop_matches_eager(device, build):
    """The other half: each call's output is released before the next, so
    the allocator may hand the same block straight back. Reuse must be as
    correct as holding."""
    torch.manual_seed(0)
    model = build().to(device)
    compiled = torch.compile(model, backend=_compiler())
    with torch.no_grad():
        for _ in range(12):
            x = torch.randn(8, 64, device=device)
            y = compiled(x)
            torch.testing.assert_close(y, model(x), atol=ATOL, rtol=RTOL)


def test_a_fresh_output_has_eagers_layout(device):
    """One matmul, one call: the returned tensor has eager's shape, strides
    and numbers."""
    torch.manual_seed(0)
    model = torch.nn.Linear(64, 64).to(device).eval()
    compiled = torch.compile(model, backend=_compiler())
    x = torch.randn(8, 64, device=device)
    with torch.no_grad():
        expected = model(x)
        got = compiled(x)
    assert got.shape == expected.shape
    assert got.stride() == expected.stride()
    torch.testing.assert_close(got, expected, atol=ATOL, rtol=RTOL)
