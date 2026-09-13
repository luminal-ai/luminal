"""End-to-end MLP via `torch.compile(backend=luminal_reference)`."""

import torch
import torch.nn as nn

import luminal_reference


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_mlp_matches_eager_module_backend() -> None:
    torch.manual_seed(0)
    model = MLP()
    x = torch.randn(3, 4)
    eager = model(x)

    compiled = torch.compile(model, backend=luminal_reference)
    out = compiled(x)

    assert out.shape == eager.shape
    assert torch.allclose(out, eager, atol=1e-5)


def test_mlp_matches_eager_function_backend() -> None:
    torch.manual_seed(0)
    model = MLP()
    x = torch.randn(3, 4)
    eager = model(x)

    compiled = torch.compile(model, backend=luminal_reference.luminal_reference)
    out = compiled(x)

    assert torch.allclose(out, eager, atol=1e-5)


def test_mlp_matches_eager_backend_string() -> None:
    torch.manual_seed(0)
    model = MLP()
    x = torch.randn(3, 4)
    eager = model(x)

    compiled = torch.compile(model, backend="luminal_reference")
    out = compiled(x)

    assert torch.allclose(out, eager, atol=1e-5)


def test_mlp_rebinds_new_inputs_without_recompile() -> None:
    torch.manual_seed(0)
    model = MLP()
    compiled = torch.compile(model, backend=luminal_reference)

    for _ in range(3):
        x = torch.randn(3, 4)
        assert torch.allclose(compiled(x), model(x), atol=1e-5)
