"""Triangular translation preserves symbolic extents through the public compiler."""

import pytest
import torch
from backend_test_utils import _backend


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("diagonal", [-2, 0, 2])
def test_dynamic_triangular(upper, diagonal, device):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.triu(x, diagonal) if upper else torch.tril(x, diagonal)

    model = Model().eval()
    example = torch.randn(2, 4, 7, device=device)
    torch._dynamo.mark_dynamic(example, 1)
    torch._dynamo.mark_dynamic(example, 2)
    compiler = _backend((example,)).Compiler()
    compiled = torch.compile(model, backend=compiler, fullgraph=True)
    with torch.no_grad():
        torch.testing.assert_close(compiled(example), model(example))
        for rows, cols in ((6, 3), (5, 8)):
            x = torch.randn(2, rows, cols, device=device)
            x[:, 0, :] = float("nan")
            x[:, 1, :] = float("inf")
            x[:, 2, :] = -0.0
            actual, expected = compiled(x), model(x)
            torch.testing.assert_close(actual, expected, equal_nan=True)
            torch.testing.assert_close(torch.signbit(actual), torch.signbit(expected))
    if hasattr(compiler, "graphs"):
        assert len(compiler.graphs) == 1


def test_whisper_dynamic_causal_mask(device):
    class Model(torch.nn.Module):
        def forward(self, tokens):
            seq = tokens.shape[0]
            return torch.triu(
                torch.full((seq, seq), -1e10, device=tokens.device), diagonal=1
            )

    model = Model().eval()
    example = torch.zeros(2, dtype=torch.int64, device=device)
    torch._dynamo.mark_dynamic(example, 0, min=2, max=448)
    compiler = _backend((example,)).Compiler()
    compiled = torch.compile(model, backend=compiler, fullgraph=True, dynamic=True)
    with torch.no_grad():
        for size in (2, 3, 7):
            tokens = (
                example
                if size == 2
                else torch.zeros(size, dtype=torch.int64, device=device)
            )
            torch.testing.assert_close(compiled(tokens), model(tokens))
    if hasattr(compiler, "graphs"):
        assert len(compiler.graphs) == 1
