"""Regression coverage for typed real and compound-complex constructors."""

import torch
from luminal.pt2 import compile as luminal_compile


def _compile(module: torch.nn.Module, *inputs: torch.Tensor, dynamic_shapes=None):
    return luminal_compile(
        module,
        inputs,
        search_iterations=1,
        dynamic_shapes={} if dynamic_shapes is None else dynamic_shapes,
    )


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


class Full(torch.nn.Module):
    def __init__(self, value, dtype: torch.dtype):
        super().__init__()
        self.value = value
        self.dtype = dtype

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.full.default(
            [anchor.shape[0], 2],
            self.value,
            dtype=self.dtype,
            device=anchor.device,
        )


class FullLike(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.full_like.default(value, self.value)


class ScalarTensor(torch.nn.Module):
    def __init__(self, value, dtype: torch.dtype):
        super().__init__()
        self.value = value
        self.dtype = dtype

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.scalar_tensor.default(
            self.value, dtype=self.dtype, device=anchor.device
        )


class ConstantPad(torch.nn.Module):
    def __init__(self, padding: list[int], value):
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.constant_pad_nd.default(tensor, self.padding, self.value)


class DefaultConstantPad(torch.nn.Module):
    def __init__(self, padding: list[int]):
        super().__init__()
        self.padding = padding

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.constant_pad_nd.default(tensor, self.padding)


class RuntimeScalarConstructors(torch.nn.Module):
    def forward(
        self, template: torch.Tensor, scalar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = scalar.item()
        return (
            torch.ops.aten.full_like.default(template, value),
            torch.ops.aten.scalar_tensor.default(
                value, dtype=template.dtype, device=template.device
            ),
        )


def test_full_preserves_dtype_and_literal_value() -> None:
    anchor = torch.ones(3)
    cases = (
        (1.2345678901234567, torch.float64),
        (2**40 + 3, torch.int64),
        (-101, torch.int8),
        (513, torch.int16),
        (231, torch.uint8),
        (1.25 - 2.5j, torch.complex128),
        (True, torch.bool),
    )
    for value, dtype in cases:
        module = Full(value, dtype)
        expected = module(anchor)
        (actual,) = _compile(module, anchor)(anchor)
        _assert_exact(actual, expected)


def test_full_supports_symbolic_shapes() -> None:
    module = Full(3.25 - 1.5j, torch.complex64)
    anchor = torch.ones(3)
    batch = torch.export.Dim("batch", min=1, max=8)
    compiled = _compile(module, anchor, dynamic_shapes=({0: batch},))
    for length in (2, 5):
        value = torch.ones(length)
        (actual,) = compiled(value)
        _assert_exact(actual, module(value))


def test_full_like_supports_nonzero_complex_fill() -> None:
    for dtype in (torch.complex64, torch.complex128):
        value = torch.tensor([[1 + 2j, -3 + 4j], [5 - 6j, -7 - 8j]], dtype=dtype)
        module = FullLike(-2.25 + 3.5j)
        (actual,) = _compile(module, value)(value)
        _assert_exact(actual, module(value))


def test_scalar_tensor_is_typed_zero_dimensional() -> None:
    anchor = torch.ones(1)
    for value, dtype in (
        (1.2345678901234567, torch.float64),
        (2**40 + 3, torch.int64),
        (-2.25 + 3.5j, torch.complex128),
    ):
        module = ScalarTensor(value, dtype)
        (actual,) = _compile(module, anchor)(anchor)
        _assert_exact(actual, module(anchor))
        assert actual.shape == torch.Size([])


def test_constructors_accept_tensor_backed_runtime_scalars() -> None:
    module = RuntimeScalarConstructors()
    for template, scalar in (
        (torch.zeros(2, 3, dtype=torch.float64), torch.tensor(1.2345678901234567)),
        (torch.zeros(2, 3, dtype=torch.int64), torch.tensor(2**40 + 3)),
    ):
        expected = module(template, scalar)
        actual = _compile(module, template, scalar)(template, scalar)
        for result, reference in zip(actual, expected):
            _assert_exact(result, reference)


def test_constant_pad_nd_preserves_typed_fill_and_ieee_values() -> None:
    cases = (
        (
            torch.tensor(
                [float("nan"), float("inf"), -float("inf")], dtype=torch.float64
            ),
            [2, 1],
            1.2345678901234567,
        ),
        (
            torch.arange(12, dtype=torch.int64).reshape(3, 4),
            [1, 2, -1, 1],
            2**40 + 3,
        ),
        (
            torch.tensor(
                [[1 + 2j, complex(float("nan"), 3)], [4 - 5j, 6 + 7j]],
                dtype=torch.complex128,
            ),
            [1, 2, -1, 1],
            -2.25 + 3.5j,
        ),
    )
    for value, padding, fill in cases:
        module = ConstantPad(padding, fill)
        (actual,) = _compile(module, value)(value)
        _assert_exact(actual, module(value))

    value = torch.arange(6, dtype=torch.int16).reshape(2, 3)
    module = DefaultConstantPad([1, 2, 3, 0])
    (actual,) = _compile(module, value)(value)
    _assert_exact(actual, module(value))
