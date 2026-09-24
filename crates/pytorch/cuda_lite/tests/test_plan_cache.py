import base64
import json
import threading
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("luminal_cuda_lite")

from luminal_cuda_lite.plan_cache import (  # noqa: E402
    cache_stats,
    clear_plan_cache,
    get_or_create,
    structural_fingerprint,
)
from luminal_cuda_lite.backend import luminal_cuda_lite  # noqa: E402
from luminal_cuda_lite.artifacts import load_artifact  # noqa: E402


@pytest.fixture(autouse=True)
def _empty_cache():
    clear_plan_cache()
    yield
    clear_plan_cache()


class _Add(torch.nn.Module):
    def forward(self, left, right):
        return left + right


class _Mul(torch.nn.Module):
    def forward(self, left, right):
        return left * right


def _fingerprint(module, shape=(2, 3), *, dtype=torch.float32, iterations=3):
    ep = torch.export.export(
        module, (torch.ones(shape, dtype=dtype), torch.ones(shape, dtype=dtype)), strict=False
    )
    names = [spec.arg.name for spec in ep.graph_signature.input_specs]
    output = next(
        spec.arg.name
        for spec in ep.graph_signature.output_specs
        if spec.kind.name == "USER_OUTPUT"
    )
    return structural_fingerprint(
        ep,
        [(name, "row_major", []) for name in names],
        [(output, "row_major", [])],
        [],
        search_iterations=iterations,
        dynamic_range=None,
    )


def test_structural_fingerprint_is_stable_and_sensitive_to_operations():
    assert _fingerprint(_Add()) == _fingerprint(_Add())
    assert _fingerprint(_Add()) != _fingerprint(_Mul())
    assert _fingerprint(_Add(), shape=(4, 3)) != _fingerprint(_Add(), shape=(2, 3))
    assert _fingerprint(_Add(), dtype=torch.bfloat16) != _fingerprint(_Add())


def test_structural_fingerprint_includes_layout_and_search_configuration():
    ep = torch.export.export(_Add(), (torch.ones(2, 3), torch.ones(2, 3)), strict=False)
    names = [spec.arg.name for spec in ep.graph_signature.input_specs]
    output = ep.graph_signature.output_specs[0].arg.name

    def digest(layout, iterations):
        return structural_fingerprint(
            ep,
            [(name, layout, []) for name in names],
            [(output, "row_major", [])],
            [],
            search_iterations=iterations,
            dynamic_range=None,
        )

    assert digest("row_major", 1) != digest("column_major", 1)
    assert digest("row_major", 1) != digest("row_major", 2)


def test_get_or_create_is_single_flight():
    barrier = threading.Barrier(4)
    calls = 0
    calls_lock = threading.Lock()
    results = []

    def create():
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return object()

    def worker():
        barrier.wait()
        results.append(get_or_create("same", create))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert len({id(value) for value, _ in results}) == 1
    assert sorted(hit for _, hit in results) == [False, True, True, True]
    stats = cache_stats()
    assert (stats.hits, stats.misses, stats.entries) == (3, 1, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_isomorphic_modules_share_search_but_keep_distinct_weight_bindings(tmp_path):
    torch.manual_seed(0)
    first = torch.nn.Linear(16, 8).cuda().eval()
    second = torch.nn.Linear(16, 8).cuda().eval()
    first_graph = torch.fx.symbolic_trace(first)
    second_graph = torch.fx.symbolic_trace(second)
    x = torch.randn(4, 16, device="cuda")

    first_compiled = luminal_cuda_lite(
        first_graph, [x], search_iterations=1, artifact_dir=str(tmp_path)
    )
    # Model a fresh process: the second compile can only reuse the file.
    clear_plan_cache()
    second_compiled = luminal_cuda_lite(
        second_graph, [x], search_iterations=1, artifact_dir=str(tmp_path)
    )

    assert not first_compiled.plan_cache_hit
    assert second_compiled.plan_cache_hit
    assert first_compiled.artifact_handle == second_compiled.artifact_handle
    outer = json.loads(Path(first_compiled.artifact_handle).read_text())
    native = json.loads(base64.b64decode(outer["plan"]))
    assert "kernel_sources" in native
    with pytest.raises(RuntimeError, match="wrong structural fingerprint"):
        load_artifact(Path(first_compiled.artifact_handle), "not-the-fingerprint")
    with torch.no_grad():
        first_output = first_compiled(x)[0]
        second_output = second_compiled(x)[0]
        torch.testing.assert_close(first_output, first(x), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(second_output, second(x), atol=1e-4, rtol=1e-4)
        assert not torch.equal(first_output, second_output)
