from __future__ import annotations

import operator

import pytest
import torch
import torch.nn.functional as F
from torch import fx

from luminal.region_export import _fresh_export_inputs, export_region


def _linear_graph() -> fx.GraphModule:
    graph = fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    bias = graph.placeholder("bias")
    linear = graph.call_function(F.linear, (x, weight, bias))
    output = graph.call_function(F.silu, (linear,))
    graph.output(output)
    return fx.GraphModule(torch.nn.Module(), graph)


def _symbolic_inputs():
    from torch._dynamo.source import LocalSource
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv(tracked_fakes=[])
    symbol = shape_env.create_symintnode(
        shape_env.create_symbol(4, LocalSource("num_tokens")), hint=4
    )
    fake_mode = FakeTensorMode(shape_env=shape_env)
    with fake_mode:
        tensor = torch.empty_strided(
            (symbol, 8), (8, 1), device="cuda:0", dtype=torch.float16
        )
    return symbol, tensor


def test_export_region_normalizes_to_aten_and_matches_eager(tmp_path) -> None:
    graph = _linear_graph()
    inputs = [torch.randn(3, 8), torch.randn(4, 8), torch.randn(4)]

    result = export_region(graph, inputs)
    exported = result.program

    assert result.input_indices == (0, 1, 2)
    assert not getattr(exported, "_guards_code", [])
    targets = [
        node.target
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    ]
    assert targets
    assert all(isinstance(target, torch._ops.OpOverload) for target in targets)
    torch.testing.assert_close(exported.module()(*inputs), graph(*inputs))

    path = tmp_path / "region.pt2"
    torch.export.save(exported, path)
    loaded = torch.export.load(path)
    torch.testing.assert_close(loaded.module()(*inputs), graph(*inputs))


def test_export_region_does_not_modify_original_graph() -> None:
    graph = _linear_graph()
    before = str(graph.graph)

    export_region(
        graph,
        [torch.randn(3, 8), torch.randn(4, 8), torch.randn(4)],
    )

    assert str(graph.graph) == before


def test_export_region_treats_data_attr_as_input_alias() -> None:
    graph = fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    data = graph.call_function(torch._C._autograd._get_data_attr, (weight,))
    graph.output(graph.call_function(torch.ops.aten.mul.Tensor, (x, data)))
    graph_module = fx.GraphModule(torch.nn.Module(), graph)
    inputs = [torch.randn(3, 8), torch.randn(3, 8)]

    result = export_region(graph_module, inputs)
    exported = result.program

    assert result.input_indices == (0, 1)
    assert not exported.constants
    assert all(
        node.target is not torch._C._autograd._get_data_attr
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    )
    torch.testing.assert_close(exported.module()(*inputs), graph_module(*inputs))


def test_export_region_preserves_symbolic_fake_tensor_dimension() -> None:
    symbol, tensor = _symbolic_inputs()
    graph = fx.Graph()
    size = graph.placeholder("num_tokens")
    size.meta["example_value"] = symbol
    x = graph.placeholder("x")
    x.meta["example_value"] = tensor
    output = graph.call_method("view", (x, size, 2, 4))
    graph.output(output)
    graph_module = fx.GraphModule(torch.nn.Module(), graph)

    result = export_region(graph_module, [symbol, tensor])
    exported = result.program

    assert result.input_indices == (1,)
    placeholders = list(exported.graph_module.graph.find_nodes(op="placeholder"))
    assert len(placeholders) == 1
    assert exported.range_constraints
    assert any(
        node.target == torch.ops.aten.sym_size.int
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    )


def test_export_region_applies_bounded_dynamic_range() -> None:
    symbol, tensor = _symbolic_inputs()
    graph = fx.Graph()
    size = graph.placeholder("num_tokens")
    size.meta["example_value"] = symbol
    x = graph.placeholder("x")
    x.meta["example_value"] = tensor
    graph.output(graph.call_method("view", (x, size, 2, 4)))

    result = export_region(
        fx.GraphModule(torch.nn.Module(), graph),
        [symbol, tensor],
        dynamic_range=(1, 8),
    )
    exported = result.program

    assert exported.range_constraints
    assert result.dynamic_ranges[0].minimum == 2
    assert result.dynamic_ranges[0].maximum == 8
    assert all(int(value.lower) == 2 for value in exported.range_constraints.values())
    assert all(int(value.upper) == 8 for value in exported.range_constraints.values())


def test_export_region_preserves_shared_dynamic_dimension() -> None:
    symbol, left_value = _symbolic_inputs()
    with left_value.fake_mode:
        right_value = torch.empty_strided(
            (symbol, 8), (8, 1), device="cuda:0", dtype=torch.float16
        )
    graph = fx.Graph()
    left = graph.placeholder("left")
    left.meta["example_value"] = left_value
    right = graph.placeholder("right")
    right.meta["example_value"] = right_value
    graph.output(graph.call_function(torch.ops.aten.cat.default, ([left, right], 0)))

    result = export_region(
        fx.GraphModule(torch.nn.Module(), graph),
        [left_value, right_value],
        dynamic_range=(1, 8),
    )
    exported = result.program

    assert len(exported.range_constraints) == 1
    assert len(result.dynamic_ranges) == 1
    assert int(next(iter(exported.range_constraints.values())).upper) == 8


def test_export_region_uses_a_fresh_fake_mode() -> None:
    symbol, tensor = _symbolic_inputs()
    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["example_value"] = tensor
    graph.output(x)

    result = export_region(
        fx.GraphModule(torch.nn.Module(), graph),
        [tensor],
        dynamic_range=(1, 8),
    )

    example_inputs, _ = result.program.example_inputs
    assert example_inputs[0].fake_mode is not tensor.fake_mode
    assert example_inputs[0].fake_mode.shape_env is not None
    assert isinstance(example_inputs[0].shape[0], torch.SymInt)


def test_fresh_export_inputs_do_not_duck_shape_equal_dimensions() -> None:
    symbol, left = _symbolic_inputs()
    with left.fake_mode:
        right = torch.empty_strided(
            (symbol, 8), (8, 1), device="cuda:0", dtype=torch.float16
        )

    fresh_left, fresh_right = _fresh_export_inputs([left, right])

    assert fresh_left.shape[0].node.expr != fresh_right.shape[0].node.expr


def test_export_region_inside_outer_tracing_context() -> None:
    from torch._guards import TracingContext, tracing

    symbol, left = _symbolic_inputs()
    with left.fake_mode:
        right = torch.empty_strided(
            (symbol, 8), (8, 1), device="cuda:0", dtype=torch.float16
        )
    graph = fx.Graph()
    left_node = graph.placeholder("left")
    left_node.meta["example_value"] = left
    right_node = graph.placeholder("right")
    right_node.meta["example_value"] = right
    graph.output(
        graph.call_function(torch.ops.aten.cat.default, ([left_node, right_node], 0))
    )

    outer_context = TracingContext(left.fake_mode)
    with tracing(outer_context):
        result = export_region(
            fx.GraphModule(torch.nn.Module(), graph),
            [left, right],
            dynamic_range=(1, 8),
        )
        assert TracingContext.get() is outer_context

    assert len(result.program.range_constraints) == 1


def test_export_region_specializes_exact_range() -> None:
    symbol, tensor = _symbolic_inputs()
    graph = fx.Graph()
    size = graph.placeholder("num_tokens")
    size.meta["example_value"] = symbol
    x = graph.placeholder("x")
    x.meta["example_value"] = tensor
    graph.output(graph.call_method("view", (x, size, 2, 4)))

    with tensor.fake_mode:
        concrete_tensor = torch.empty_strided(
            (8, 8), (8, 1), device="cuda:0", dtype=torch.float16
        )
    result = export_region(
        fx.GraphModule(torch.nn.Module(), graph),
        [8, concrete_tensor],
        dynamic_range=(8, 8),
    )
    exported = result.program

    assert not result.dynamic_ranges
    assert not exported.range_constraints
    assert "Sym(" not in exported.graph_module.print_readable(print_output=False)


def test_export_region_rejects_static_graph_for_ranged_artifact() -> None:
    graph = fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_function(torch.ops.aten.relu.default, (x,)))

    with pytest.raises(RuntimeError, match="exactly one dynamic dimension"):
        export_region(
            fx.GraphModule(torch.nn.Module(), graph),
            [torch.randn(4, 8)],
            dynamic_range=(1, 8),
        )


def test_export_region_rejects_unrepresentable_symbol() -> None:
    symbol, tensor = _symbolic_inputs()
    graph = fx.Graph()
    size = graph.placeholder("unrelated_symbol")
    size.meta["example_value"] = symbol + 1
    x = graph.placeholder("x")
    x.meta["example_value"] = tensor
    output = graph.call_function(operator.mul, (x, size))
    graph.output(output)
    graph_module = fx.GraphModule(torch.nn.Module(), graph)

    with pytest.raises(RuntimeError, match="could not be derived"):
        export_region(graph_module, [symbol + 1, tensor])


def test_export_region_surfaces_export_error(monkeypatch) -> None:
    def fail_export(*args, **kwargs):
        raise ValueError("original export failure")

    monkeypatch.setattr(torch.export, "export", fail_export)

    with pytest.raises(RuntimeError, match="ValueError: original export failure"):
        export_region(_linear_graph(), [torch.randn(3, 8)] * 3)
