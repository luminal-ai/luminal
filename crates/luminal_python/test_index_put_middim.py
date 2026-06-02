"""Isolated correctness test for the translate_index_put middle-dim scatter.

Exercises `x[:, :, idx] = v` (index_put with indices [None, None, idx]) — the
StaticCache cache-write pattern — and compares the luminal-compiled result
against eager. This is the regression test for the movement.rs change.
"""
import torch
import torch._dynamo
from luminal import luminal_backend


class WriteDim2(torch.nn.Module):
    def forward(self, x, v, idx):
        y = x.clone()
        y[:, :, idx] = v          # -> aten.index_put [None, None, idx]
        return y


def main():
    assert torch.cuda.is_available()
    torch._dynamo.reset()
    m = WriteDim2().eval().cuda()
    c = torch.compile(m, backend=luminal_backend, fullgraph=True)

    # Non-1 leading dim so torch.compile doesn't squeeze it away (which would
    # misalign the index entries vs the tensor rank). Mirrors the StaticCache
    # write a[:, :, pos] = v on a [batch, heads, cache_len, head_dim] buffer.
    x = torch.randn(2, 2, 8, 4, device="cuda")
    v = torch.randn(2, 2, 3, 4, device="cuda")
    idx = torch.tensor([1, 3, 5], device="cuda", dtype=torch.int64)

    with torch.no_grad():
        ref = m(x, v, idx)
        got = c(x, v, idx)
    diff = (ref - got).abs().max().item()
    ok = torch.allclose(ref, got, atol=1e-5)
    print(f"[index_put dim2] max_diff={diff:.3e} allclose={ok}")
    # also check a dim-0 sanity (existing all-tensors path unaffected) skipped here
    assert ok, "index_put middle-dim scatter mismatch vs eager"
    print("PASS")


if __name__ == "__main__":
    main()
