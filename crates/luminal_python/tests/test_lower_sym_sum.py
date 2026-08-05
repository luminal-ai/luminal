"""_lower_sym_sum: sym_sum nodes must not survive to torch.export.save.

Three distinct-sized dynamic inputs concatenated force the output length
s0+s1+s2 — an n-ary sympy Add that torch FX-ifies as torch.sym_sum on
affected versions. Whether or not this torch build emits sym_sum, the
contract is the same: after the pass, no sym_sum nodes remain and save()
succeeds. CPU-only.
"""

import os
import tempfile

import torch
from torch.export import Dim, export

from luminal.pt2 import _lower_sym_sum


class Cat3(torch.nn.Module):
    def forward(self, a, b, c):
        out = torch.cat([a, b, c])
        return out + out.shape[0]


def test_lower_sym_sum_roundtrip():
    m = Cat3()
    ex = (torch.randn(3), torch.randn(5), torch.randn(7))
    dyn = {"a": {0: Dim("s0")}, "b": {0: Dim("s1")}, "c": {0: Dim("s2")}}
    ep = export(m, ex, dynamic_shapes=dyn)

    _lower_sym_sum(ep)

    assert not any(
        n.op == "call_function" and n.target is getattr(torch, "sym_sum", None)
        for n in ep.graph_module.graph.nodes
    )
    with tempfile.TemporaryDirectory() as td:
        torch.export.save(ep, os.path.join(td, "m.pt2"))  # must not raise

    # Semantics preserved.
    got = ep.module()(*ex)
    want = m(*ex)
    torch.testing.assert_close(got, want)
