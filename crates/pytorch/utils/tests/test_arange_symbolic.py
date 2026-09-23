"""Symbolic arange bounds must remain dynamic through PT2 translation."""

import pytest
import torch
from backend_test_utils import compile_for_test


@pytest.mark.parametrize("variant", ["end", "start", "step", "descending"])
def test_arange_symbolic_length(variant):
    class Model(torch.nn.Module):
        def forward(self, tokens):
            length = tokens.shape[0]
            if variant == "end":
                return torch.arange(length, dtype=torch.int64, device=tokens.device)
            if variant == "start":
                return torch.arange(1, length, dtype=torch.int64, device=tokens.device)
            if variant == "step":
                return torch.arange(
                    1, length, 2, dtype=torch.int64, device=tokens.device
                )
            return torch.arange(length, 0, -2, dtype=torch.int64, device=tokens.device)

    model = Model()
    compiled = compile_for_test(
        model,
        torch.zeros(8, dtype=torch.int64),
        search_iterations=1,
        dynamic_shapes=({0: torch.export.Dim("length", min=6, max=24)},),
    )
    for length in (8, 9, 14):
        tokens = torch.zeros(length, dtype=torch.int64)
        (actual,) = compiled(tokens)
        torch.testing.assert_close(actual, model(tokens))


def test_whisper_position_embedding_changes_with_sequence_length():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.positions = torch.nn.Embedding(448, 8)

        def forward(self, tokens):
            positions = torch.arange(
                tokens.shape[0], dtype=torch.long, device=tokens.device
            )
            return self.positions(positions)

    torch.manual_seed(19)
    model = Model().eval()
    compiled = compile_for_test(
        model, torch.zeros(2, dtype=torch.int64), search_iterations=1, dynamic_dim=0
    )
    for length in (2, 3, 7):
        tokens = torch.zeros(length, dtype=torch.int64)
        (actual,) = compiled(tokens)
        torch.testing.assert_close(actual, model(tokens))


def test_whisper_causal_mask_changes_with_sequence_length():
    class Model(torch.nn.Module):
        def forward(self, tokens):
            seq = tokens.shape[0]
            return torch.triu(
                torch.full((seq, seq), -1e10, device=tokens.device), diagonal=1
            )

    model = Model()
    compiled = compile_for_test(
        model, torch.zeros(2, dtype=torch.int64), search_iterations=1, dynamic_dim=0
    )
    for length in (2, 3, 7):
        tokens = torch.zeros(length, dtype=torch.int64)
        (actual,) = compiled(tokens)
        torch.testing.assert_close(actual, model(tokens))
