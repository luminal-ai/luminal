"""Export may omit cat's default dimension from its serialized arguments."""

import pytest
import torch
from backend_test_utils import compile_for_test


@pytest.mark.parametrize("sizes", [(3, 2), (0, 2), (3, 0), (0, 0)])
def test_cat_omitted_default_dimension(sizes):
    class Model(torch.nn.Module):
        def forward(self, left, right):
            return torch.cat([left, right])

    inputs = tuple(torch.arange(float(size * 2)).reshape(size, 2) for size in sizes)
    model = Model()
    compiled = compile_for_test(model, inputs, search_iterations=1)
    (actual,) = compiled(*inputs)
    torch.testing.assert_close(actual, model(*inputs))
