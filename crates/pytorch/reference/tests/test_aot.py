"""AOT contract tests: gradients, saved tensors, and runtime parameter binding."""

import pytest
import torch
from luminal_reference import Compiler


def test_forward_backward_and_optimizer_update():
    torch.manual_seed(71)
    backend = Compiler()

    def model(x, w):
        return (x @ w).square().sum()

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    weight = torch.nn.Parameter(torch.randn(4, 3))
    expected_weight = torch.nn.Parameter(weight.detach().clone())
    optimizer = torch.optim.SGD([weight], lr=0.01)
    expected_optimizer = torch.optim.SGD([expected_weight], lr=0.01)
    for _ in range(2):
        inputs = [torch.randn(2, 4, requires_grad=True) for _ in range(2)]
        expected_inputs = [x.detach().clone().requires_grad_() for x in inputs]
        # Two outstanding forwards must retain different saved activations.
        outputs = [compiled(x, weight) for x in inputs]
        expected = [model(x, expected_weight) for x in expected_inputs]
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference)
        sum(outputs).backward()
        sum(expected).backward()
        torch.testing.assert_close(weight.grad, expected_weight.grad)
        for actual, reference in zip(inputs, expected_inputs):
            torch.testing.assert_close(actual.grad, reference.grad)
        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(weight, expected_weight)
        optimizer.zero_grad()
        expected_optimizer.zero_grad()

    assert {r.phase for r in backend.graphs} == {"forward", "backward"}
    assert all(r.executions >= 4 for r in backend.graphs)


def test_output_structure_and_missing_gradient():
    backend = Compiler()

    def model(x, unused):
        return x.square() + unused.detach(), None, x.sum(), 7

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    unused = torch.ones(2, requires_grad=True)
    square, missing, total, constant = compiled(x, unused)
    assert missing is None and constant == 7
    (square.sum() + total).backward()
    torch.testing.assert_close(x.grad, 2 * x.detach() + 1)
    assert unused.grad is None


def test_inference_preserves_forwarded_input_identity():
    backend = Compiler()

    def model(x):
        return {"input": x, "computed": (x + 1, x * 2), "empty": None}

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    x = torch.arange(4.0)
    actual = compiled(x)
    expected = model(x)
    assert actual["input"] is x
    assert actual["empty"] is None
    torch.testing.assert_close(actual["computed"], expected["computed"])
    assert {r.phase for r in backend.graphs} == {"inference"}


def test_dynamic_forward_backward_reuses_graphs():
    backend = Compiler()

    def model(x):
        # Backward must retain symbolic dimensions even though this sum's
        # saved state contains no full-sized activation.
        return x.sum()

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)
    for size in (3, 7, 5):
        x = torch.randn(size, 4, requires_grad=True)
        actual = compiled(x)
        torch.testing.assert_close(actual, x.sum())
        actual.backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
    assert len(backend.graphs) == 2
    assert {g.phase for g in backend.graphs} == {"forward", "backward"}


def test_compiler_trains_with_dynamic_shapes():
    compiled = torch.compile(
        lambda x: x.square().sum(), backend=Compiler(), fullgraph=True, dynamic=True
    )
    for size in (3, 8):
        x = torch.randn(size, 4, requires_grad=True)
        output = compiled(x)
        torch.testing.assert_close(output, x.square().sum())
        output.backward()
        torch.testing.assert_close(x.grad, 2 * x.detach())


def test_default_backend_dynamic_mutation_and_alias():
    import luminal_reference

    def model(x):
        x.add_(2)
        return x.view(-1), x * 3

    compiled = torch.compile(
        model, backend=luminal_reference.Compiler(), fullgraph=True, dynamic=True
    )
    for size in (3, 7):
        x = torch.randn(size, 4)
        expected_input = x.clone()
        expected = model(expected_input)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x, expected_input)
        assert actual[0].data_ptr() == x.data_ptr()


def test_dynamic_scalar_shape_output():
    import luminal_reference

    compiled = torch.compile(
        lambda x: (x + 1, x.shape[0] // 2),
        backend=luminal_reference.Compiler(),
        fullgraph=True,
        dynamic=True,
    )
    for rows in (6, 9, 12):
        x = torch.randn(rows, 4)
        tensor, count = compiled(x)
        torch.testing.assert_close(tensor, x + 1)
        assert count == rows // 2


def test_literal_constant_in_graph():
    backend = Compiler()

    def model(x):
        # A literal created in the traced frame reaches the backend as a graph
        # constant behind ``lift_fresh_copy``; the compiled program must carry
        # it rather than re-export the surrounding graph.
        return x + torch.tensor([1.0, 2.0, 3.0, 4.0])

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    x = torch.randn(3, 4)
    torch.testing.assert_close(compiled(x), model(x))
    assert len(backend.graphs) == 1


def test_collective_graphs_are_refused(tmp_path):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "rendezvous").as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        group = dist.group.WORLD.group_name
        ops = torch.ops._c10d_functional

        def model(x):
            return ops.wait_tensor(ops.all_reduce(x + 1, "sum", group))

        compiled = torch.compile(
            model, backend=Compiler(), fullgraph=True, dynamic=False
        )
        with pytest.raises(RuntimeError, match="collective operations.*all_reduce"):
            compiled(torch.ones(2))
    finally:
        dist.destroy_process_group()


def test_in_graph_input_mutations_compile_through_the_shared_adapter():
    """An input mutation stays in the graph as AOTAutograd's trailing
    ``copy_``; the shared adapter turns it into a mutation output the runtime
    writes in place, and a graph whose only work is such a writeback still
    runs the native program."""

    def returns_input_twice(x):
        x.add_(1)
        return x, x

    def mutates_then_clones(x):
        x.add_(1)
        return x.clone()

    for fn in (returns_input_twice, mutates_then_clones):
        torch._dynamo.reset()
        backend = Compiler()
        eager_x = torch.arange(6.0).reshape(2, 3)
        x = torch.arange(6.0).reshape(2, 3)
        want = fn(eager_x)
        got = torch.compile(fn, backend=backend)(x)
        torch.testing.assert_close(got, want)
        torch.testing.assert_close(x, eager_x, msg="the mutation landed in the input")
        assert backend.graphs[0].program is not None, "the writeback ran natively"
        assert "aten.copy_.default" in backend.graphs[0].targets
        if fn is returns_input_twice:
            assert got[0] is x and got[1] is x


def test_symbolic_scalar_outputs_keep_their_python_type():
    """AOTAutograd returns a symbolic int, float or bool that the backward
    graph needs as an output leaf of the forward graph, and torch's contract
    for such a leaf is the Python ``int``, ``float`` or ``bool`` at call
    time. The shared adapter evaluates those leaves from the call's sizes.

    The tensor leaves here come from running the graph in eager torch, so
    the test covers the adapter alone, whatever the translator lowers."""
    from functorch.compile import make_boxed_func
    from luminal_reference.aot_graph import prepare_local_graph
    from torch._dynamo.backends.common import aot_autograd
    from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec

    def through_adapter(seen):
        def forward(gm, inputs):
            local = prepare_local_graph(gm, inputs, device_type="cpu")

            def run(*args):
                # The rows the program would write, taken from eager.
                eager = gm.forward(*local.bind(args))
                rows = [None] * len(local.written)
                for slot, value in zip(local.output_slots, eager):
                    if slot.source == "program":
                        rows[slot.index] = value
                leaves = local.assemble(args, rows)
                seen.append((local, leaves))
                return leaves

            return make_boxed_func(run)

        # The backward graph reads the saved scalar as an input, which the
        # adapter does not compile yet; torch runs that graph itself.
        return aot_autograd(
            fw_compiler=forward,
            bw_compiler=lambda gm, inputs: make_boxed_func(gm.forward),
        )

    cases = [
        (
            "float",
            float,
            lambda t: (t ** (t.shape[0] / 2.0)).sum(),
            lambda rows: rows / 2.0,
        ),
        (
            "bool",
            bool,
            lambda t: (
                torch.where(torch.tensor(t.shape[0] > 3), t, -t).sum()
                * (t.shape[0] > 3)
            ),
            lambda rows: rows > 3,
        ),
    ]
    for kind, python, fn, expected in cases:
        torch._dynamo.reset()
        seen = []
        size = ShapeVar("n", min=2, max=8, optimization_hint=4)
        compiled = torch.compile(
            fn,
            backend=through_adapter(seen),
            dynamic_shapes=ShapesSpec(params={"t": TensorSpec([size, 3])}),
        )
        for rows in (4, 5, 2):
            torch.manual_seed(rows)
            t = torch.rand(rows, 3, requires_grad=True)
            out, want = compiled(t), fn(t)
            torch.testing.assert_close(out, want)
            (grad,) = torch.autograd.grad(out, t)
            (reference,) = torch.autograd.grad(want, t)
            torch.testing.assert_close(grad, reference)

            local, leaves = seen[-1]
            positions = [
                position
                for position, slot in enumerate(local.output_slots)
                if slot.source == "expression" and slot.kind == kind
            ]
            assert positions, f"the forward graph returns no {kind} leaf"
            for position in positions:
                assert type(leaves[position]) is python
                assert leaves[position] == expected(rows)


def test_every_placeholder_and_output_leaf_has_a_slot():
    """torch calls a compiled graph with one value per placeholder and
    expects one value per output leaf. The boundary states each of those
    positions as a slot whose kind is the type of the value: a tensor, an
    ``int``, a ``float`` or a ``bool``.

    The backward graph of each program takes the symbolic float or bool the
    forward graph saved. Such an argument is the expression torch recorded
    for it, so its slot is stated and the call's value is never read.
    """
    from functorch.compile import make_boxed_func
    from luminal_reference.aot_graph import prepare_local_graph
    from torch._dynamo.backends.common import aot_autograd
    from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec

    def recording(seen):
        def compiler(phase):
            def compile_graph(gm, inputs):
                seen[phase] = prepare_local_graph(gm, inputs, device_type="cpu")
                return make_boxed_func(gm.forward)

            return compile_graph

        return aot_autograd(
            fw_compiler=compiler("forward"), bw_compiler=compiler("backward")
        )

    cases = [
        ("float", lambda t: (t ** (t.shape[0] / 2.0)).sum()),
        (
            "bool",
            lambda t: (
                torch.where(torch.tensor(t.shape[0] > 3), t, -t).sum()
                * (t.shape[0] > 3)
            ),
        ),
    ]
    for scalar, fn in cases:
        torch._dynamo.reset()
        seen = {}
        size = ShapeVar("n", min=2, max=8, optimization_hint=4)
        compiled = torch.compile(
            fn,
            backend=recording(seen),
            dynamic_shapes=ShapesSpec(params={"t": TensorSpec([size, 3])}),
        )
        compiled(torch.rand(4, 3, requires_grad=True)).backward()

        forward = seen["forward"]
        # The dynamic size and the tensor, in torch's order. The size is an
        # integer argument that defines its symbol.
        assert [(s.kind, s.read) for s in forward.input_slots] == [
            ("int", True),
            ("tensor", True),
        ]
        assert forward.input_slots[0].symbol is not None
        sources = {(slot.kind, slot.source) for slot in forward.output_slots}
        assert ("tensor", "program") in sources, "the loss is computed"
        assert ("int", "input") in sources, "the saved size is the argument itself"
        assert (scalar, "expression") in sources, f"the saved {scalar} is a slot"

        backward = seen["backward"]
        (saved,) = [slot for slot in backward.input_slots if slot.kind == scalar]
        assert saved.read, f"the backward graph reads the saved {scalar}"
        assert saved.symbol is None, "its value is an expression, not a binding"
        # A call passes one value per slot.
        with pytest.raises(RuntimeError, match="one per placeholder"):
            backward.bind([0] * (len(backward.input_slots) - 1))


def test_a_saved_symbolic_bool_is_read_as_its_expression():
    """A bool computed from a size and saved for the backward graph reaches
    that graph as an argument. The program reads the expression torch
    recorded for it, over the size the integer argument defines, so the
    forward and the backward graph both compile and match eager."""
    from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec

    def fn(t):
        keep = t.shape[0] > 3
        return torch.where(torch.tensor(keep), t, -t).sum() * keep

    backend = Compiler()
    size = ShapeVar("n", min=2, max=8, optimization_hint=4)
    compiled = torch.compile(
        fn,
        backend=backend,
        dynamic_shapes=ShapesSpec(params={"t": TensorSpec([size, 3])}),
    )
    for rows in (4, 2, 7):
        torch.manual_seed(rows)
        t = torch.rand(rows, 3, requires_grad=True)
        out, want = compiled(t), fn(t)
        torch.testing.assert_close(out, want)
        (grad,) = torch.autograd.grad(out, t)
        (reference,) = torch.autograd.grad(want, t)
        torch.testing.assert_close(grad, reference)
    assert {record.phase for record in backend.graphs} == {"forward", "backward"}
    assert len(backend.graphs) == 2, "every size ran on the one pair of graphs"


def test_a_symbolic_float_is_refused_where_a_tensor_op_reads_it():
    """A float computed from a size has no expression the translator reads
    yet. The graph that uses one in tensor math is refused at that op, by
    the value's name and the expression torch recorded for it."""
    from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec

    size = ShapeVar("n", min=2, max=8, optimization_hint=4)
    compiled = torch.compile(
        lambda t: (t ** (t.shape[0] / 2.0)).sum(),
        backend=Compiler(),
        dynamic_shapes=ShapesSpec(params={"t": TensorSpec([size, 3])}),
    )
    with pytest.raises(Exception, match=r"pow\.Tensor_Scalar.*truediv.*FloatTrueDiv"):
        compiled(torch.rand(4, 3, requires_grad=True))
