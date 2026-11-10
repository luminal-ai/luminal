import pytest
import torch

import luminal_cuda_lite


class ComplexLogcumsumexpEdges(torch.nn.Module):
    def forward(self, value):
        return torch.ops.aten.logcumsumexp.default(value, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_complex_logcumsumexp_preserves_ieee_edges():
    value = torch.tensor(
        [
            0.0 + 0.0j,
            80.0 + 1.0j,
            complex(float("inf"), 0.0),
            complex(float("-inf"), 2.0),
            complex(float("nan"), 0.0),
            1.0 - 2.0j,
        ],
        dtype=torch.complex64,
        device="cuda",
    )
    module = ComplexLogcumsumexpEdges()
    compiled = torch.compile(
        module,
        backend=luminal_cuda_lite.Compiler(search_iterations=1),
        fullgraph=True,
    )
    torch.testing.assert_close(
        compiled(value),
        module(value),
        rtol=5e-4,
        atol=5e-5,
        equal_nan=True,
    )
