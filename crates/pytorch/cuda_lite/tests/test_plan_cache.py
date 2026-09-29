import base64
import json
import os
import subprocess
import sys
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
        module,
        (torch.ones(shape, dtype=dtype), torch.ones(shape, dtype=dtype)),
        strict=False,
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


def test_get_or_create_recovers_after_creator_failure():
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        raise ValueError("failed search")

    with pytest.raises(ValueError, match="failed search"):
        get_or_create("retry", fail)

    value = object()
    actual, hit = get_or_create("retry", lambda: value)
    assert actual is value
    assert not hit
    assert calls == 1


def test_get_or_create_evicts_the_least_recently_used_entry(monkeypatch):
    monkeypatch.setenv("LUMINAL_PLAN_CACHE_SIZE", "2")
    first = object()
    second = object()
    third = object()

    assert get_or_create("first", lambda: first) == (first, False)
    assert get_or_create("second", lambda: second) == (second, False)
    # A hit makes `first` most-recently used, so inserting `third` evicts
    # `second`, not merely the oldest key by creation time.
    assert get_or_create("first", lambda: object()) == (first, True)
    assert get_or_create("third", lambda: third) == (third, False)
    replacement = object()
    assert get_or_create("second", lambda: replacement) == (replacement, False)
    assert cache_stats().entries == 2


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

    def corrupted_artifact(name, mutate):
        document = json.loads(Path(first_compiled.artifact_handle).read_text())
        native_document = json.loads(base64.b64decode(document["plan"]))
        mutate(native_document)
        document["plan"] = base64.b64encode(
            json.dumps(native_document).encode()
        ).decode()
        path = tmp_path / name
        path.write_text(json.dumps(document))
        return path

    bad_schema = corrupted_artifact(
        "bad-schema.json", lambda document: document.__setitem__("schema", 999)
    )
    with pytest.raises(RuntimeError, match="schema 999 is not supported"):
        load_artifact(bad_schema, outer["fingerprint"])

    bad_abi = corrupted_artifact(
        "bad-abi.json",
        lambda document: document.__setitem__("op_registry_abi", 999),
    )
    with pytest.raises(RuntimeError, match="op-registry ABI 999 is not supported"):
        load_artifact(bad_abi, outer["fingerprint"])

    bad_kernels = corrupted_artifact(
        "bad-kernels.json",
        lambda document: document["kernel_sources"].append(
            {"node": 999, "label": "wrong", "sources": ["wrong"]}
        ),
    )
    with pytest.raises(RuntimeError, match="kernel sources do not match"):
        load_artifact(bad_kernels, outer["fingerprint"])

    with torch.no_grad():
        first_output = first_compiled(x)[0]
        second_output = second_compiled(x)[0]
        torch.testing.assert_close(first_output, first(x), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(second_output, second(x), atol=1e-4, rtol=1e-4)
        assert not torch.equal(first_output, second_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_persistent_artifact_loads_in_a_fresh_process(tmp_path):
    """Process B must execute process A's artifact without an in-memory hit."""
    program = r"""
import json
import sys

import torch
import torch.fx as fx

from luminal_cuda_lite.backend import luminal_cuda_lite


class Add(torch.nn.Module):
    def forward(self, left, right):
        return left + right


graph = fx.symbolic_trace(Add())
left = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
right = torch.full_like(left, 7)
compiled = luminal_cuda_lite(
    graph,
    [left, right],
    search_iterations=1,
    artifact_dir=sys.argv[1],
)
actual = compiled(left, right)[0]
torch.cuda.synchronize()
torch.testing.assert_close(actual, left + right)
print("LUMINAL_RESULT=" + json.dumps({
    "cache_hit": compiled.plan_cache_hit,
    "artifact": compiled.artifact_handle,
}))
"""

    def run_process():
        completed = subprocess.run(
            [sys.executable, "-c", program, str(tmp_path)],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=180,
        )
        line = next(
            row.removeprefix("LUMINAL_RESULT=")
            for row in completed.stdout.splitlines()
            if row.startswith("LUMINAL_RESULT=")
        )
        return json.loads(line)

    first = run_process()
    second = run_process()
    assert not first["cache_hit"]
    assert second["cache_hit"]
    assert second["artifact"] == first["artifact"]
    assert Path(first["artifact"]).is_file()
