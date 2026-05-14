import torch
import pytest

from luminal import luminal_backend


class StrideSensitiveInputModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "coeff",
            torch.tensor([1.0, 10.0, 100.0], dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.coeff


class TwoInputReadModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * 2.0 + y * 3.0


class ReturnInputModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ReturnInputAndComputedModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x + 1.0


class CloneThenMutateModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        y.add_(1.0)
        return y, x * 2.0


def _base_view(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    return base, base.t()


def _assert_non_contiguous_storage_alias(base: torch.Tensor, view: torch.Tensor) -> None:
    assert not view.is_contiguous()
    assert view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()


def _assert_same(actual, expected) -> None:
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_same(actual_item, expected_item)
        return

    assert torch.allclose(actual, expected)


def _single_non_contiguous_view(device: torch.device):
    base, view = _base_view(device)
    _assert_non_contiguous_storage_alias(base, view)
    return StrideSensitiveInputModel().to(device), (view,), base


def _same_view_twice(device: torch.device):
    base, view = _base_view(device)
    _assert_non_contiguous_storage_alias(base, view)
    return TwoInputReadModel().to(device), (view, view), base


def _overlapping_views(device: torch.device):
    base = torch.arange(20, dtype=torch.float32, device=device).reshape(4, 5)
    x = base[:3, :4]
    y = base[1:, 1:]
    assert not x.is_contiguous()
    assert not y.is_contiguous()
    assert x.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert y.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    return TwoInputReadModel().to(device), (x, y), base


def _return_input(device: torch.device):
    base, view = _base_view(device)
    _assert_non_contiguous_storage_alias(base, view)
    return ReturnInputModel().to(device), (view,), base


def _return_input_and_computed(device: torch.device):
    base, view = _base_view(device)
    _assert_non_contiguous_storage_alias(base, view)
    return ReturnInputAndComputedModel().to(device), (view,), base


def _internal_clone_inplace(device: torch.device):
    base, view = _base_view(device)
    _assert_non_contiguous_storage_alias(base, view)
    return CloneThenMutateModel().to(device), (view,), base


@pytest.mark.parametrize(
    "make_case",
    [
        pytest.param(
            _single_non_contiguous_view,
            id="single_non_contiguous_view_stride_sensitive_read",
        ),
        pytest.param(_same_view_twice, id="same_view_passed_as_two_read_inputs"),
        pytest.param(_overlapping_views, id="overlapping_views_as_two_read_inputs"),
        pytest.param(_return_input, id="return_input_boundary_value"),
        pytest.param(
            _return_input_and_computed,
            id="return_input_boundary_value_and_computed_value",
        ),
        pytest.param(_internal_clone_inplace, id="inplace_mutation_on_internal_clone"),
    ],
)
def test_input_boundary_contiguous_materialization_cases(
    device: torch.device, make_case
) -> None:
    model, inputs, base = make_case(device)
    compiled = torch.compile(model, backend=luminal_backend)

    base_before = base.clone()
    expected = model(*inputs)
    actual = compiled(*inputs)

    _assert_same(actual, expected)
    assert torch.allclose(base, base_before)


def test_non_contiguous_view_input_fails_if_raw_storage_order_is_used(
    device: torch.device,
) -> None:
    model, (view,), base = _single_non_contiguous_view(device)

    wrong_if_storage_order_used = model(base.reshape(view.shape))
    expected = model(view)

    assert not torch.allclose(wrong_if_storage_order_used, expected)
