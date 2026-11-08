"""Translator batch 6: attention (SDPA) and upsample / resize."""

import luminal_reference
import torch
import torch.nn as nn
import torch.nn.functional as F
from luminal_reference.backend import _compile_local_graph


def _check(model: nn.Module, *inputs: torch.Tensor, atol: float = 1e-4) -> None:
    torch.manual_seed(0)
    eager = model(*inputs)
    compiled = torch.compile(model, backend=luminal_reference.Compiler())
    out = compiled(*inputs)
    assert out.shape == eager.shape, f"{out.shape} != {eager.shape}"
    assert torch.allclose(out.to(torch.float32), eager.to(torch.float32), atol=atol), (
        f"{out} != {eager}"
    )


class SdpaCausal(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class SdpaMasked(nn.Module):
    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask)


class SdpaGqa(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)


class UpNearest(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")


class UpBilinear(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class UpBilinearAA(nn.Module):
    def forward(self, x):
        return F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False, antialias=True
        )


def test_sdpa_causal() -> None:
    shape = (2, 5, 8)
    _check(
        SdpaCausal(),
        torch.randn(*shape),
        torch.randn(*shape),
        torch.randn(*shape),
    )


def test_sdpa_additive_mask() -> None:
    mask = torch.zeros(5, 5)
    _check(
        SdpaMasked(mask),
        torch.randn(2, 5, 8),
        torch.randn(2, 5, 8),
        torch.randn(2, 5, 8),
    )


def test_sdpa_gqa() -> None:
    _check(
        SdpaGqa(),
        torch.randn(2, 2, 5, 8),
        torch.randn(2, 2, 5, 8),
        torch.randn(2, 2, 5, 8),
    )


def test_sdpa_tensor_scale_stays_runtime_bound() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 5, 8)
    k = torch.randn(2, 5, 8)
    v = torch.randn(2, 5, 8)
    graph = torch.fx.Graph()
    q_node = graph.placeholder("q")
    k_node = graph.placeholder("k")
    v_node = graph.placeholder("v")
    scale_node = graph.placeholder("scale")
    scale_item = graph.call_function(torch.ops.aten.item.default, (scale_node,))
    attention = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q_node, k_node, v_node),
        {"scale": scale_item},
    )
    graph.output((attention,))
    module = torch.fx.GraphModule({}, graph)
    compiled = _compile_local_graph(module, (q, k, v, torch.tensor(0.125)))

    outputs = []
    for value in (0.125, 0.75):
        scale = torch.tensor(value)
        (actual,) = compiled(q, k, v, scale)
        expected = F.scaled_dot_product_attention(q, k, v, scale=scale.item())
        torch.testing.assert_close(actual, expected)
        outputs.append(actual)

    assert not torch.equal(outputs[0], outputs[1])


def test_upsample_nearest() -> None:
    _check(UpNearest(), torch.randn(1, 2, 4, 4))


def test_upsample_bilinear() -> None:
    _check(UpBilinear(), torch.randn(1, 2, 4, 4))


def test_upsample_bilinear_antialias() -> None:
    _check(UpBilinearAA(), torch.randn(1, 2, 4, 4), atol=1e-4)
