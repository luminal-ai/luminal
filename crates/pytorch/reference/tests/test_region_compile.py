from __future__ import annotations

import json

import luminal_reference.region_compile as region_compile_module
import pytest
import torch
from luminal_reference.artifact_cache import (
    CompiledArtifact,
    artifact_cache_stats,
    clear_artifact_cache,
    get_or_compile,
    get_or_load,
    region_artifact_key,
)
from luminal_reference.region_compile import compile_region, load_region_artifact
from luminal_reference.region_export import export_region
from torch import fx


def _add_graph() -> fx.GraphModule:
    graph = fx.Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    result = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
    graph.output((result,))
    return fx.GraphModule(torch.nn.Module(), graph)


def test_compile_region_preserves_runtime_input_indices(monkeypatch):
    inputs = [torch.randn(2, 4) for _ in range(2)]
    region = export_region(_add_graph(), inputs)
    received = {}
    sentinel = object()

    def compile_exported(program, **kwargs):
        received.update(program=program, **kwargs)
        return sentinel

    monkeypatch.setattr(region_compile_module, "compile_exported", compile_exported)
    assert compile_region(region, search_iterations=3) is sentinel
    assert received["program"] is region.program
    assert received["user_indices"] == region.input_indices
    assert received["output_spec"] == region.output_spec
    assert received["search_iterations"] == 3
    assert received["device_type"] == "cpu"


def _key_program(input_name: str, op=torch.ops.aten.relu.default):
    from types import SimpleNamespace

    graph = fx.Graph()
    value = graph.placeholder(input_name)
    value.meta["val"] = torch.empty(2, 4)
    result = graph.call_function(op, (value,))
    result.meta["val"] = torch.empty(2, 4)
    graph.output((result,))
    return SimpleNamespace(
        constants={},
        state_dict={},
        range_constraints={},
        graph_module=fx.GraphModule(torch.nn.Module(), graph),
    )


def test_artifact_key_ignores_input_names() -> None:
    options = {"device_type": "cuda", "search_iterations": 1}

    assert region_artifact_key(_key_program("layer_0"), **options) == (
        region_artifact_key(_key_program("layer_1"), **options)
    )
    assert region_artifact_key(_key_program("layer_0"), **options) != (
        region_artifact_key(
            _key_program("layer_0", torch.ops.aten.sigmoid.default), **options
        )
    )


def test_artifact_cache_compiles_once() -> None:
    clear_artifact_cache()
    artifact = object()
    calls = 0

    def compile_artifact():
        nonlocal calls
        calls += 1
        return artifact

    assert get_or_compile("region", compile_artifact) is artifact
    assert get_or_compile("region", compile_artifact) is artifact
    assert calls == 1
    stats = artifact_cache_stats()
    assert (stats.unique_artifacts, stats.reuse_hits, stats.searches) == (1, 1, 1)
    assert stats.search_seconds > 0
    clear_artifact_cache()


def test_region_artifact_loads_once_and_binds_each_model(monkeypatch) -> None:
    clear_artifact_cache()
    bindings = []

    class Artifact:
        def bind(self, **kwargs):
            bindings.append(kwargs)
            return len(bindings)

    artifact = Artifact()
    loads = 0

    def deserialize(*args, **kwargs):
        nonlocal loads
        loads += 1
        return artifact

    monkeypatch.setattr(CompiledArtifact, "deserialize", deserialize)

    common = {"device_index": 0, "external_cuda_graph": True}
    assert (
        load_region_artifact(b"{}", input_indices=(0,), output_spec="first", **common)
        == 1
    )
    assert (
        load_region_artifact(b"{}", input_indices=(1,), output_spec="second", **common)
        == 2
    )
    assert loads == 1
    assert [binding["user_indices"] for binding in bindings] == [(0,), (1,)]
    stats = artifact_cache_stats()
    assert (stats.loads, stats.load_reuse_hits) == (1, 1)
    assert stats.load_seconds > 0
    clear_artifact_cache()


def test_loaded_artifact_checks_payload_identity() -> None:
    clear_artifact_cache()
    common = {
        "luminal_artifact_key": "region",
        "schema_version": 4,
        "backend": "cuda_lite",
    }
    first = json.dumps({**common, "value": 1}).encode()
    second = json.dumps({**common, "value": 2}).encode()
    artifact = object()

    assert get_or_load(first, lambda: artifact, device_index=0) is artifact
    assert get_or_load(second, lambda: object(), device_index=0) is not artifact
    clear_artifact_cache()


def test_loaded_artifact_identity_includes_compatibility() -> None:
    clear_artifact_cache()
    first = json.dumps(
        {"luminal_artifact_key": "region", "schema_version": 4, "backend": "cuda_lite"}
    ).encode()
    second = json.dumps(
        {"luminal_artifact_key": "region", "schema_version": 4, "backend": "reference"}
    ).encode()
    artifacts = [object(), object()]

    assert get_or_load(first, lambda: artifacts[0]) is artifacts[0]
    assert get_or_load(second, lambda: artifacts[1]) is artifacts[1]
    clear_artifact_cache()


def test_compiled_artifact_serializes_structural_identity():
    clear_artifact_cache()
    inputs = [torch.randn(2, 4) for _ in range(2)]
    compiled = compile_region(export_region(_add_graph(), inputs))
    assert (
        json.loads(compiled.serialize_artifact())["luminal_artifact_key"]
        == compiled.artifact.cache_key
    )


def test_compiled_artifact_round_trip_cpu() -> None:
    from luminal_reference.frontend import compile_exported

    inputs = [torch.randn(2, 4), torch.randn(2, 4)]
    compiled = compile_exported(torch.export.export(_add_graph(), tuple(inputs)))
    loaded = CompiledArtifact.deserialize(
        compiled.serialize_artifact(),
    ).bind()

    (actual,) = loaded(*inputs)
    torch.testing.assert_close(actual, inputs[0] + inputs[1])


def test_cold_bindings_share_plan_and_keep_execution_storage_independent():
    import copy

    from luminal_reference.backend import compile_exported

    inputs = [torch.randn(2, 4), torch.randn(2, 4)]
    program = torch.export.export(_add_graph(), tuple(inputs))
    fake_inputs = [
        node.meta["val"] for node in program.graph.nodes if node.op == "placeholder"
    ]
    native = compile_exported(program, fake_inputs, search_iterations=1)
    searches = []

    class CountedGraph:
        def __init__(self, graph):
            self.graph = graph

        def __getattr__(self, name):
            return getattr(self.graph, name)

        def fork(self):
            return CountedGraph(self.graph.fork())

        def search(self, **options):
            searches.append(self.graph)
            return self.graph.search(**options)

    native._graph = CountedGraph(native._graph)
    first, second = copy.copy(native), copy.copy(native)
    assert first._pending_search is not second._pending_search
    (actual,) = first(*inputs)
    saved_bytes = first._graph.output_bytes(0)
    torch.testing.assert_close(actual, sum(inputs))
    zeros = [torch.zeros_like(value) for value in inputs]
    (other,) = second(*zeros)
    torch.testing.assert_close(other, zeros[0])
    assert first._graph.output_bytes(0) == saved_bytes
    third = copy.copy(native)
    torch.testing.assert_close(third(*inputs)[0], sum(inputs))
    warm = copy.copy(first)
    torch.testing.assert_close(warm(*zeros)[0], zeros[0])
    assert first._graph.output_bytes(0) == saved_bytes
    assert len(searches) == 1


def test_cold_bindings_isolate_mutable_exported_buffers():
    import copy

    from luminal_reference.backend import compile_exported

    class Accumulator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("total", torch.zeros(4))

        def forward(self, value):
            self.total.add_(value)
            return self.total.clone()

    value = torch.ones(4)
    program = torch.export.export(Accumulator(), (value,)).run_decompositions()
    native = compile_exported(program, (value,), search_iterations=1, defer_search=True)
    first, second = copy.copy(native), copy.copy(native)
    torch.testing.assert_close(first(value)[0], value)
    torch.testing.assert_close(first(value)[0], value * 2)
    torch.testing.assert_close(second(value)[0], value)
    warm = copy.copy(first)
    torch.testing.assert_close(warm(value)[0], value * 3)
    torch.testing.assert_close(first(value)[0], value * 3)
    assert all(torch.count_nonzero(buffer) == 0 for buffer in native._held.values())


def test_compiled_artifact_rejects_bound_weights() -> None:
    from luminal_reference.frontend import compile_exported

    class Weighted(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.randn(4))

        def forward(self, value):
            return value * self.weight

    compiled = compile_exported(torch.export.export(Weighted(), (torch.randn(4),)))

    with pytest.raises(RuntimeError, match="bound weights"):
        compiled.serialize_artifact()


def test_compiled_artifact_rejects_old_schema() -> None:
    from luminal_reference.frontend import compile_exported

    inputs = [torch.randn(2, 4), torch.randn(2, 4)]
    compiled = compile_exported(torch.export.export(_add_graph(), tuple(inputs)))
    artifact = json.loads(compiled.serialize_artifact())
    artifact["schema_version"] = 1

    with pytest.raises(RuntimeError, match="unsupported artifact schema 1"):
        CompiledArtifact.deserialize(
            json.dumps(artifact).encode(),
        )


def test_shared_artifact_rebinds_inputs_between_models():
    clear_artifact_cache()
    inputs = [torch.randn(2, 4) for _ in range(2)]
    region = export_region(_add_graph(), inputs)
    first = compile_region(region)
    second = compile_region(region)
    assert first.artifact is second.artifact
    assert first._graph is not second._graph
    first(*inputs)
    first_output = first._graph.output_bytes(0)
    second(*(torch.zeros_like(t) for t in inputs))
    assert first._graph.output_bytes(0) == first_output
    for model, values in [
        (first, inputs),
        (second, [torch.ones_like(t) for t in inputs]),
        (first, inputs),
    ]:
        (actual,) = model(*values)
        torch.testing.assert_close(actual, values[0] + values[1])


def test_compile_region_rejects_nonzero_device() -> None:
    from dataclasses import replace

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        inputs = [torch.randn(2, 4, device="cuda") for _ in range(2)]
        region = export_region(_add_graph(), inputs)

    with pytest.raises(ValueError, match="only logical CUDA device 0"):
        compile_region(replace(region, device_index=1))


def test_compiled_model_rejects_wrong_cuda_device():
    from dataclasses import replace

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        region = export_region(
            _add_graph(), [torch.empty(2, 4, device="cuda") for _ in range(2)]
        )
    with pytest.raises(ValueError, match="only logical CUDA device 0"):
        compile_region(replace(region, device_index=1), device_type="cuda")


def test_compiled_model_restores_region_output_structure():
    graph = fx.Graph()
    value = graph.placeholder("value")
    result = graph.call_function(torch.ops.aten.add.Tensor, (value, value))
    graph.output({"result": result, "optional": None})
    model = fx.GraphModule(torch.nn.Module(), graph)
    inputs = [torch.randn(2, 4)]
    compiled = compile_region(export_region(model, inputs))
    output = compiled(*inputs)
    assert output["optional"] is None
    torch.testing.assert_close(output["result"], inputs[0] * 2)


def test_equal_profiles_with_different_domains_do_not_share_artifacts():
    from luminal_reference.frontend import compile_exported

    class Model(torch.nn.Module):
        def forward(self, x):
            return x.sin()

    clear_artifact_cache()
    programs = [
        torch.export.export(
            Model(),
            (torch.ones(4),),
            dynamic_shapes={"x": {0: torch.export.Dim("n", min=2, max=hi)}},
        )
        for hi in (8, 16)
    ]
    models = [compile_exported(program) for program in programs]
    assert models[0].artifact is not models[1].artifact
    assert compile_exported(programs[0]).artifact is models[0].artifact
    assert artifact_cache_stats().searches == 2
    for model, hi in zip(models, (8, 16)):
        x = torch.randn(hi)
        (actual,) = model(x)
        torch.testing.assert_close(actual, x.sin())
    with pytest.raises(RuntimeError, match="<= 8"):
        models[0](torch.ones(9))
    clear_artifact_cache()
