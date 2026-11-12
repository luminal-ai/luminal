"""Mutation / aliasing contract tests.

PyTorch semantics: an in-place op on an input is visible in the caller's
tensor, and a model that returns the mutated input returns the SAME storage
(``data_ptr`` unchanged).
"""

from typing import Any, Tuple

import luminal_reference
import torch
import torch.nn as nn


class InPlaceAdd(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.add_(2.0)
        return x


class InPlaceThenMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.add_(1.0)
        return x * 3.0


class InPlaceRelu(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.relu_()
        return x


class CopyFrom(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x.copy_(y)
        return x


class ViewMutation(nn.Module):
    """A write through a view of the input: functionalisation spells it as a
    scatter into the input, and the caller's tensor must show it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[0].add_(1.0)
        x[:, 1].mul_(2.0)
        return x


class BufferAdd(nn.Module):
    """A module buffer mutated in place.

    Under ``torch.compile``, Dynamo lifts ``self.cache`` into a placeholder and
    the functionalised export tags the write ``user_input_mutation`` on it.
    Exporting the module directly gives the form whose spec is
    ``buffer_mutation``; both are covered below.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("cache", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.cache.add_(x)
        return self.cache * 1.0


class TwoMutations(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x.add_(1.0)
        y.mul_(2.0)
        return x + y


def _run(cls: type, inputs: Tuple[torch.Tensor, ...]) -> Tuple[Any, ...]:
    torch.manual_seed(0)
    model = cls()
    base = [t.clone() for t in inputs]
    eager_inputs = [t.clone() for t in inputs]
    eager = model(*eager_inputs)

    compiled = torch.compile(model, backend=luminal_reference.Compiler())
    compiled_inputs = [t.clone() for t in base]
    out = compiled(*compiled_inputs)

    assert torch.allclose(out, eager, atol=1e-5)
    for compiled_in, eager_in in zip(compiled_inputs, eager_inputs):
        assert torch.allclose(compiled_in, eager_in, atol=1e-5), (
            f"caller tensor not mutated: {compiled_in} != {eager_in}"
        )
    return out, compiled_inputs, eager


def test_in_place_add_returns_caller_storage() -> None:
    x = torch.randn(3, 4)
    out, compiled_inputs, _ = _run(InPlaceAdd, (x,))
    assert out.data_ptr() == compiled_inputs[0].data_ptr(), (
        "a returned mutation must alias the caller's tensor"
    )


def test_in_place_then_mul_returns_fresh_value() -> None:
    out, compiled_inputs, _ = _run(InPlaceThenMul, (torch.randn(3, 4),))
    assert out.data_ptr() != compiled_inputs[0].data_ptr()
    assert torch.allclose(out, compiled_inputs[0] * 3.0, atol=1e-5)


def test_in_place_relu() -> None:
    _run(InPlaceRelu, (torch.randn(3, 4),))


def test_mutation_through_a_view_reaches_the_caller() -> None:
    x = torch.randn(3, 4)
    out, compiled_inputs, _ = _run(ViewMutation, (x,))
    assert out.data_ptr() == compiled_inputs[0].data_ptr()


def test_copy_from() -> None:
    _run(CopyFrom, (torch.randn(3, 4), torch.randn(3, 4)))


def test_two_mutation_targets() -> None:
    _run(TwoMutations, (torch.randn(3, 4), torch.randn(3, 4)))


def test_buffer_mutation() -> None:
    """A mutated module buffer must end up holding the eager value."""
    torch.manual_seed(0)
    x = torch.randn(4)

    eager_model = BufferAdd()
    eager = eager_model(x.clone())

    model = BufferAdd()
    compiled = torch.compile(model, backend=luminal_reference.Compiler())
    out = compiled(x.clone())

    assert torch.allclose(out, eager, atol=1e-5)
    assert torch.allclose(model.cache, eager_model.cache, atol=1e-5), (
        f"buffer not mutated: {model.cache} != {eager_model.cache}"
    )
