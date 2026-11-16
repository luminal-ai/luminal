"""What the reference backend's code object states about its own tables.

The facts every backend's code object has in common (its bytes, torch's side
of a call, running without the compiler) are in
``utils/tests/test_code_object.py`` and run against this backend too.
"""

import torch
from luminal_torch_reference_compiler import Compiler
from luminal_torch_reference_runtime import describe


def test_a_mutation_row_names_the_argument_it_writes_back_into():
    """The reference runtime copies a call's tensors to the host, runs, and
    copies each row back. A row that is a mutation states the argument slot
    whose tensor it is copied into; a fresh row states none. ``describe``
    leaves the plan out."""

    def fn(x, y):
        y.add_(1.0)
        return x * 2.0

    x, y = torch.randn(4, 3), torch.zeros(2)
    compiler = Compiler()
    torch.compile(fn, backend=compiler, fullgraph=True)(x, y)
    table = describe(compiler.graphs[0].program)

    written_back = [row["writes_back"] for row in table["rows"]]
    (slot,) = [slot for slot in written_back if slot is not None]
    assert written_back.count(None) == 1
    assert table["slots"][slot] == "tensor"
    # The mutated argument is also read: its slot names the input staged.
    assert "read" in table["actions"][slot]
    # Every input is staged at its own dtype and shape.
    assert all(row["dtype"] == "float32" for row in table["inputs"])
    assert table["plan"] is None, "describe leaves the plan out"


def test_other_python_threads_run_while_a_plan_computes():
    """The reference runtime releases the interpreter while a plan computes,
    as torch releases it around its own kernels. A second thread counts
    while one call runs; with the interpreter held for the whole call it
    could not count at all."""
    import threading
    import time

    def program(x, w):
        return torch.relu(x @ w) @ w.t()

    torch.manual_seed(0)
    x, w = torch.randn(128, 128), torch.randn(128, 128)
    compiler = Compiler()
    with torch.no_grad():
        torch.compile(program, backend=compiler, fullgraph=True)(x, w)
    code = compiler.graphs[0].program

    ticks = [0]
    stop = threading.Event()

    def tick():
        while not stop.is_set():
            ticks[0] += 1
            time.sleep(0.001)

    ticker = threading.Thread(target=tick)
    ticker.start()
    try:
        time.sleep(0.05)
        before = ticks[0]
        began = time.perf_counter()
        (out,) = code([x, w])
        spent = time.perf_counter() - began
        during = ticks[0] - before
    finally:
        stop.set()
        ticker.join()

    torch.testing.assert_close(out, program(x, w), atol=1e-3, rtol=1e-3)
    assert spent > 0.05, f"the call took {spent:.3f}s, too short to observe"
    # One tick a millisecond when the interpreter is free; none when held.
    assert during >= 10, f"another thread ran {during} times during a {spent:.3f}s call"
