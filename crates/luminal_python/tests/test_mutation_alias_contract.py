"""Regression coverage for torch.compile mutation and alias contracts.

PyTorch backends are expected to preserve the semantics of the traced graph.
After torch.export functionalization, input mutations are represented as
leading mutation outputs before user outputs. Luminal currently treats every
compiled graph output as a user output and also materializes inputs at the
boundary, so caller-visible mutation and aliasing semantics are not preserved.
"""

import pytest
import torch

from luminal import luminal_backend


class MutateInputThenCompute(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.add_(1.0)
        return x * 2.0


class MutateInputReturnAlias(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.add_(1.0)
        return x


class MutateOverlappingInputAlias(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x.add_(10.0)
        return y * 2.0


class ReturnInputView(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.t()


def _assert_same_storage(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "inductor"])
def test_stock_torch_compile_preserves_input_mutation_writeback(backend: str) -> None:
    model = MutateInputThenCompute()
    expected_input = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    actual_input = expected_input.clone()

    expected = model(expected_input)
    compiled = torch.compile(model, backend=backend)
    actual = compiled(actual_input)

    assert torch.equal(actual, expected)
    assert torch.equal(actual_input, expected_input)


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "inductor"])
def test_stock_torch_compile_preserves_mutated_return_alias(backend: str) -> None:
    model = MutateInputReturnAlias()
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    compiled = torch.compile(model, backend=backend)
    out = compiled(x)

    assert torch.equal(x, torch.arange(1, 7, dtype=torch.float32).reshape(2, 3))
    _assert_same_storage(out, x)


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "inductor"])
def test_stock_torch_compile_preserves_returned_view_alias(backend: str) -> None:
    model = ReturnInputView()
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    compiled = torch.compile(model, backend=backend)
    out = compiled(x)

    assert torch.equal(out, x.t())
    assert out.stride() == (1, 3)
    _assert_same_storage(out, x)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Luminal currently treats functionalized input-mutation outputs as user "
        "outputs and does not copy mutation outputs back to caller inputs."
    ),
)
def test_luminal_input_mutation_writeback_contract(device: torch.device) -> None:
    model = MutateInputThenCompute().to(device)
    x = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)

    compiled = torch.compile(model, backend=luminal_backend)
    out = compiled(x)

    expected_x = torch.arange(1, 7, dtype=torch.float32, device=device).reshape(2, 3)
    expected_out = expected_x * 2.0
    assert torch.equal(out, expected_out)
    assert torch.equal(x, expected_x)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Luminal does not preserve caller-visible overlapping input aliasing "
        "when one aliased input is mutated."
    ),
)
def test_luminal_overlapping_input_alias_mutation_contract(
    device: torch.device,
) -> None:
    model = MutateOverlappingInputAlias().to(device)

    eager_base = torch.arange(6, dtype=torch.float32, device=device)
    expected = model(eager_base[:4], eager_base[1:5])

    base = torch.arange(6, dtype=torch.float32, device=device)
    compiled = torch.compile(model, backend=luminal_backend)
    actual = compiled(base[:4], base[1:5])

    assert torch.equal(actual, expected)
    assert torch.equal(base, eager_base)


@pytest.mark.xfail(
    strict=True,
    reason="Luminal materializes returned input views instead of preserving aliasing.",
)
def test_luminal_returned_view_alias_contract(device: torch.device) -> None:
    model = ReturnInputView().to(device)
    x = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)

    compiled = torch.compile(model, backend=luminal_backend)
    out = compiled(x)

    assert torch.equal(out, x.t())
    assert out.stride() == (1, 3)
    _assert_same_storage(out, x)
