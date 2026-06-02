"""Regression tests for in-place buffer-mutation scatter (index_copy_ on a
registered buffer), with literal vs computed (arange-derived) indices.

Captures the bug: a COMPUTED index into an in-place (NoCopy) buffer scatter
writes a spurious value at cache position == seq. Tensor-target and
literal-index cases are correct; buffer + computed index is the failing case.

Run: CUDARC_CUDA_VERSION=12080 uv run --group dev python test_buffer_scatter.py
"""
import torch
import torch._dynamo
from luminal import luminal_backend

MAXC, W = 8, 4  # W must be <= MAXC (writing W positions into an MAXC-slot cache)


def _run(make_model, computed):
    me, mc = make_model().eval().cuda(), make_model().eval().cuda()
    k = torch.full((1, 2, W, 4), 5.0, device="cuda")
    cp = torch.arange(W, device="cuda")
    torch._dynamo.reset()
    c = torch.compile(mc, backend=luminal_backend, fullgraph=True)
    with torch.no_grad():
        me(k, cp)
        c(k, cp)
    return me.keys, mc.keys


def _buffer_model():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("keys", torch.zeros(1, 2, MAXC, 4))
            self.computed = True

        def forward(self, k, cp):
            idx = (torch.arange(k.shape[-2], device=k.device) + cp[0:1]
                   if self.computed else cp)
            self.keys.index_copy_(2, idx, k)
            return k.sum()
    return M


def _case(computed):
    Mk = _buffer_model()
    def make():
        m = Mk(); m.computed = computed; return m
    e, g = _run(make, computed)
    diff = (e - g).abs().max().item()
    return diff


def main():
    assert torch.cuda.is_available()
    results = {}
    for computed in (False, True):
        label = "computed(arange+cp)" if computed else "literal(cp)"
        diff = _case(computed)
        ok = diff < 1e-4
        results[label] = (diff, ok)
        print(f"[buffer index_copy_, {label}] max_diff={diff:.3e} {'PASS' if ok else 'FAIL'}")
    all_ok = all(ok for _, ok in results.values())
    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    assert all_ok, f"buffer-mutation scatter regression: {results}"


if __name__ == "__main__":
    main()
