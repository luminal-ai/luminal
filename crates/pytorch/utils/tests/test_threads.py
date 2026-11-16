"""A compiled graph across threads.

A compiled graph is an ordinary Python object: it can be handed to another
thread, called there and freed there. torch does exactly that with a backward
graph, which it calls from its autograd thread. Its calls run one at a time:
a call from another thread waits for the one in progress. Nothing about a
call depends on which thread makes it except, on CUDA, the stream it runs on,
which is the calling thread's current one.

These hold for every backend.
"""

import contextlib
import gc
import importlib
import pickle
import sys
import threading
import weakref

import pytest
import torch
from backend_test_utils import selected_backend

_TOLERANCE = {"atol": 1e-5, "rtol": 1e-5}


def _program(fn, *args):
    """The code object of ``fn`` compiled as one graph and called once."""
    compiler = importlib.import_module(
        f"luminal_torch_{selected_backend()}_compiler"
    ).Compiler()
    with torch.no_grad():
        torch.compile(fn, backend=compiler, fullgraph=True)(*args)
    return compiler.graphs[0].program


def _on_another_thread(work):
    """Run ``work`` on a new thread and return what it returned. An error
    there is raised here."""
    outcome = []

    def run():
        try:
            outcome.append((work(), None))
        except BaseException as error:  # reported on the calling thread
            outcome.append((None, error))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    ((value, error),) = outcome
    if error is not None:
        raise error
    return value


@pytest.fixture
def unraisable(monkeypatch):
    """Every error Python could not raise during the test. An object that
    cannot be freed where its last reference is dropped reports here and
    nowhere else."""
    seen = []
    monkeypatch.setattr(
        sys,
        "unraisablehook",
        lambda info: seen.append(f"{info.exc_type.__name__}: {info.exc_value}"),
    )
    return seen


def test_a_compiled_graph_is_called_on_a_thread_that_did_not_build_it(device):
    """The executable built by the first call on one thread serves a call on
    another, and the first thread's later calls are unaffected."""
    x = torch.randn(3, 4, device=device)
    code = _program(lambda x: x * 3.0, x)
    (here,) = code([x])
    (there,) = _on_another_thread(lambda: code([x]))
    torch.testing.assert_close(there, here)
    torch.testing.assert_close(code([x])[0], x * 3.0)


def test_calls_from_several_threads_each_get_their_own_result(device):
    """Four threads call one compiled graph at once, each with inputs of its
    own and, on CUDA, each on a stream of its own. The calls run one at a
    time, so every result is its own call's."""

    def affine(x, w):
        return torch.relu(x @ w) + 1.0

    threads, calls = 4, 8
    torch.manual_seed(0)
    w = torch.randn(16, 8, device=device)
    inputs = [
        [torch.randn(4, 16, device=device) for _ in range(calls)]
        for _ in range(threads)
    ]
    expected = [[affine(x, w).cpu() for x in mine] for mine in inputs]
    code = _program(affine, inputs[0][0], w)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = threading.Barrier(threads)
    results = [[] for _ in range(threads)]
    errors = []

    def work(index):
        stream = (
            torch.cuda.stream(torch.cuda.Stream(device))
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        try:
            with stream, torch.no_grad():
                start.wait()
                for x in inputs[index]:
                    (out,) = code([x, w])
                    # Read back on this thread's stream, behind the program.
                    results[index].append(out.cpu())
        except BaseException as error:  # reported on the main thread
            errors.append(error)

    workers = [threading.Thread(target=work, args=(index,)) for index in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert not errors, errors
    for index in range(threads):
        assert len(results[index]) == calls
        for got, want in zip(results[index], expected[index]):
            torch.testing.assert_close(got, want, **_TOLERANCE)


def test_a_compiled_graph_is_freed_on_whichever_thread_drops_it(device, unraisable):
    """A compiled graph used on one thread and dropped on another is freed
    there, and so is one used on another thread and dropped here. Freeing
    reports nothing."""
    x = torch.randn(3, 4, device=device)
    saved = pickle.dumps(_program(lambda x: x * 3.0, x))

    # Used here, last reference dropped on another thread.
    held = [pickle.loads(saved)]
    torch.testing.assert_close(held[0]([x])[0], x * 3.0)
    alive = weakref.ref(held[0])

    def drop():
        held.clear()
        gc.collect()

    _on_another_thread(drop)
    assert alive() is None
    assert unraisable == []

    # Used on another thread, last reference dropped here.
    held = [pickle.loads(saved)]
    (there,) = _on_another_thread(lambda: held[0]([x]))
    torch.testing.assert_close(there, x * 3.0)
    alive = weakref.ref(held[0])
    held.clear()
    gc.collect()
    assert alive() is None
    assert unraisable == []


def test_freeing_a_compiled_function_after_a_training_step_reports_nothing(
    device, unraisable
):
    """torch may run a backward graph on a thread of its own, so a backward
    graph's executable is used on one thread and freed on another. The
    gradients match eager, and freeing everything afterwards reports
    nothing."""
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 4).to(device)
    x = torch.randn(5, 8, device=device)

    model(x).square().sum().backward()
    expected = [parameter.grad.clone() for parameter in model.parameters()]
    model.zero_grad()

    compiler = importlib.import_module(
        f"luminal_torch_{selected_backend()}_compiler"
    ).Compiler()
    compiled = torch.compile(model, backend=compiler)
    loss = compiled(x).square().sum()
    loss.backward()
    for parameter, want in zip(model.parameters(), expected):
        torch.testing.assert_close(parameter.grad, want, atol=1e-4, rtol=1e-4)

    del compiled, loss, compiler
    torch._dynamo.reset()
    gc.collect()
    assert unraisable == []
