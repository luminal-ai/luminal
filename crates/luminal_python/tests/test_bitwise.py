"""Integer bitwise ATen translation tests.

The GPTQ reference path unpacks int4 values from int32 storage with a
broadcasted right shift, an internal int8 cast, and a scalar mask. Keep a
regression test for that exact shape/layout sequence so the right shift and
the mask's canonical normalized-Mod lowering remain visible to Egglog.
"""

from typing import Callable

import pytest
import torch

from luminal import luminal_backend


class GptqInt4DecodeFragment(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "wf",
            torch.tensor([[0, 4, 8, 12, 16, 20, 24, 28]], dtype=torch.int32),
        )

    def forward(self, qweight: torch.Tensor) -> torch.Tensor:
        shifted = torch.bitwise_right_shift(
            qweight.unsqueeze(1).expand(-1, 8, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int8)
        unpacked = torch.bitwise_and(shifted, 15)
        return unpacked.reshape(qweight.shape[0] * 8, qweight.shape[1]).to(
            torch.int32
        )


def test_gptq_int4_decode_fragment_reaches_luminal(device: torch.device) -> None:
    if device.type != "cpu":
        pytest.skip("integer bitwise HLIR execution is currently reference-backend only")
    model = GptqInt4DecodeFragment().to(device)
    compiled: Callable = torch.compile(model, backend=luminal_backend)
    qweight = torch.tensor(
        [
            [0x76543210, -19088744, -1],
            [0x01234567, -2147483648, 0x13579BDF],
        ],
        dtype=torch.int32,
        device=device,
    )

    expected = model(qweight)
    actual = compiled(qweight)

    assert actual.dtype == torch.int32
    assert torch.equal(actual.cpu(), expected.cpu())
