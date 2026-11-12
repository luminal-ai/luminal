"""A size that sums three or more dimensions reaches the translator as a
chain of binary additions.

Three inputs of independent dynamic length, concatenated, give an output
length of ``s0 + s1 + s2``. sympy's ``Add`` is n-ary, so torch can write
that sum in the graph as one ``torch.sym_sum`` node, which torch's own node
serializer has no entry for. Whether or not this torch build emits the
node, the contract is the same: the graph handed to the translator has no
``sym_sum`` in it, and the program compiles and matches eager. CPU-only.
"""

import torch
from backend_test_utils import record_lowered_graphs
from luminal_reference import Compiler
from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec


class Cat3(torch.nn.Module):
    def forward(self, a, b, c):
        out = torch.cat([a, b, c])
        return out + out.shape[0]


def test_a_three_term_size_sum_reaches_the_translator_as_binary_adds(monkeypatch):
    lowered = record_lowered_graphs(monkeypatch)
    model = Cat3()
    spec = ShapesSpec(
        params={
            name: TensorSpec([ShapeVar(f"s{i}", min=2, max=16, optimization_hint=4)])
            for i, name in enumerate("abc")
        }
    )
    compiled = torch.compile(model, backend=Compiler(), dynamic_shapes=spec)

    for lengths in ((3, 5, 7), (2, 9, 4)):
        inputs = tuple(torch.randn(length) for length in lengths)
        torch.testing.assert_close(compiled(*inputs), model(*inputs))

    assert len(lowered) == 1, "every size ran on the one compiled graph"
    (program,) = lowered
    assert not any(
        node.op == "call_function" and node.target is getattr(torch, "sym_sum", None)
        for node in program.graph_module.graph.nodes
    )
