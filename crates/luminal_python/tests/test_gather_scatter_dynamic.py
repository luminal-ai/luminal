"""Dynamic-shape regression coverage for the movement ops Qwen3-MoE /
Gemma4-MoE exercise via `torch.compile`.

Three failure modes surfaced while debugging the Qwen3-30B-A3B path:

1. `gather_elements: index dim must be concrete` — `gather_elements`
   / `scatter_elements` collected index dims as `Vec<usize>` via
   `.to_usize().expect(...)`. First forward worked; the second forward
   at a different seq_len made Dynamo emit a SymInt dim and tripped
   the assertion.
2. `Dims must match to add tensors. left: [(a*8), 2048] right: [(8*a), 2048]`
   — different translator paths produced semantically-equal but
   syntactically-different `Expression` dims.
3. `scatter_nd: data dim must be concrete` — same family as (1),
   reached via `translate_index_put` (HF's MoE accumulator).
"""

from __future__ import annotations

import torch

from luminal.main import luminal_backend


def _compile(model):
    return torch.compile(model, backend=luminal_backend)


def test_gather_elements_dynamic_index_shape(device: torch.device) -> None:
    """`torch.gather` with a dynamic batch dim on the index tensor."""

    class GatherModel(torch.nn.Module):
        def forward(self, table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            expanded = table.unsqueeze(0).expand(indices.shape[0], -1, -1)
            idx = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 32)
            return torch.gather(expanded, 1, idx).squeeze(1)

    model = GatherModel().to(device)
    compiled = _compile(model)
    table = torch.randn(8, 32, device=device)

    for batch in [4, 7, 11, 4]:
        idx = torch.randint(0, 8, (batch,), device=device, dtype=torch.int64)
        assert torch.allclose(compiled(table, idx), model(table, idx), atol=1e-4)


def test_scatter_elements_dynamic_index_shape(device: torch.device) -> None:
    """`torch.scatter` with a dynamic batch dim on the index tensor."""

    class ScatterModel(torch.nn.Module):
        def forward(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            dest = torch.zeros(
                values.shape[0], 16, device=values.device, dtype=values.dtype
            )
            return dest.scatter(1, indices, values)

    model = ScatterModel().to(device)
    compiled = _compile(model)

    for batch in [4, 7, 11, 4]:
        # Distinct indices per row → no-overlap scatter for allclose.
        idx = torch.stack(
            [torch.randperm(16, device=device)[:4] for _ in range(batch)]
        ).to(torch.int64)
        vals = torch.randn(batch, 4, device=device)
        assert torch.allclose(compiled(vals, idx), model(vals, idx), atol=1e-4)


def test_scatter_nd_dynamic_data_shape(device: torch.device) -> None:
    """`tensor[idx] = value` → `translate_index_put` → `scatter_nd`."""

    class ScatterNDModel(torch.nn.Module):
        def forward(
            self, base: torch.Tensor, idx: torch.Tensor, vals: torch.Tensor
        ) -> torch.Tensor:
            out = base.clone()
            out[idx] = vals
            return out

    model = ScatterNDModel().to(device)
    compiled = _compile(model)

    for batch in [4, 7, 11, 4]:
        base = torch.randn(16, 4, device=device)
        idx = torch.randperm(16, device=device)[:batch].to(torch.int64)
        vals = torch.randn(batch, 4, device=device)
        assert torch.allclose(
            compiled(base, idx, vals), model(base, idx, vals), atol=1e-4
        )


def test_where_dynamic_shape_no_dim_mismatch_panic(device: torch.device) -> None:
    """`torch.where` over inputs whose shape derives from a SymInt:
    two translator paths can produce `a*8` vs `8*a` for the same dim,
    which trips the dim-equality assert in luminal-core's `Sub` /
    `Add` without canonical ordering in `dim_arith`.
    """

    class WhereModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.where(x > 0, x, y)

    model = WhereModel().to(device)
    compiled = _compile(model)

    for batch in [4, 7, 11, 4]:
        x = torch.randn(batch, 16, device=device)
        y = torch.randn(batch, 16, device=device)
        assert torch.allclose(compiled(x, y), model(x, y), atol=1e-4)
