"""What the CUDA backend's code object states about its own tables and its
program.

The facts every backend's code object has in common (its bytes, torch's side
of a call, running without the compiler) are in
``utils/tests/test_code_object.py`` and run against this backend too.
"""

import pickle
import subprocess
import sys
import textwrap

import pytest
import torch
from luminal_torch_cuda_lite_compiler import Compiler
from luminal_torch_cuda_lite_runtime import describe

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)


def _compiled(fn, *args, **options):
    compiler = Compiler()
    compiled = torch.compile(fn, backend=compiler, fullgraph=True, **options)
    with torch.no_grad():
        compiled(*args)
    return compiled, compiler


def _affine_inputs():
    return (
        torch.randn(4, 16, device="cuda"),
        torch.randn(16, 8, device="cuda"),
        torch.randn(8, device="cuda"),
    )


def test_a_code_object_states_where_each_tensor_is_bound_and_what_runs():
    """The CUDA tables of a code object: every tensor the program reads is
    bound in place, a row says whose memory it is written in, and in place
    of the program is a summary of launches, copies and library calls, with
    every kernel compiled."""
    from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec

    def fn(x, y):
        y.add_(1.0)
        return x * 2.0, x.shape[0] * 3

    size = ShapeVar("n", min=2, max=8, optimization_hint=4)
    x, y = torch.randn(4, 3, device="cuda"), torch.zeros(2, device="cuda")
    _, compiler = _compiled(
        fn, x, y, dynamic_shapes=ShapesSpec(params={"x": TensorSpec([size, 3])})
    )
    table = describe(compiler.graphs[0].program)

    # Every tensor the program reads is an argument; none is a constant.
    assert all("constant" not in row for row in table["inputs"])
    # `y` is written back into, so it is the one input marked written.
    assert sorted(row["written"] for row in table["inputs"]) == [False, True]
    # One row is the mutation of `y`, stored in its argument slot; the other
    # is a fresh tensor at eager's strides.
    storage = sorted(
        (
            next(iter(row["storage"]))
            if isinstance(row["storage"], dict)
            else row["storage"]
        )
        for row in table["rows"]
    )
    assert storage == ["fresh", "writes_back"]
    (written,) = [
        row["storage"]["writes_back"]["slot"]
        for row in table["rows"]
        if isinstance(row["storage"], dict)
    ]
    assert table["slots"][written] == "tensor"
    (fresh,) = [row for row in table["rows"] if row["storage"] == "fresh"]
    assert fresh["dtype"] == "float32"
    # The program states the range it was compiled for.
    ((_, low, high),) = table["program"]["dimensions"]
    assert (low, high) == (2, 8)
    # In place of the program is a summary: it launches kernels.
    assert table["program"]["kernel_launches"] >= 1
    assert table["program"]["kernels"] >= 1


def test_a_call_that_reenters_its_own_executable_is_refused(monkeypatch):
    """Calls run one at a time, so a call that came back in on its own
    thread could only wait for itself. It is refused by name instead, and
    the call it interrupted finishes.

    A program with no tensor argument asks torch which device is current,
    which is the one place a call runs Python that a caller can replace."""
    from torch.fx.experimental.dynamic_spec import IntVar, ShapesSpec

    def program(n):
        return torch.full((n, 3), 1.5, device="cuda") + n

    compiler = Compiler()
    compiled = torch.compile(
        program,
        backend=compiler,
        dynamic_shapes=ShapesSpec(params={"n": IntVar("n", min=1, max=6)}),
    )
    compiled(3)
    code = compiler.graphs[0].program
    current_device = torch.cuda.current_device
    refused = []

    def reentering():
        if not refused:
            try:
                code([4])
                refused.append(None)
            except RuntimeError as error:
                refused.append(str(error))
        return current_device()

    monkeypatch.setattr(torch.cuda, "current_device", reentering)
    (out,) = code([4])
    monkeypatch.undo()
    (message,) = refused
    assert message is not None, "the re-entrant call was let in"
    assert "called again from inside its own call" in message, message
    torch.testing.assert_close(out, program(4))


def test_a_saved_program_compiles_no_kernel_where_it_is_loaded(tmp_path):
    """A code object holds every kernel already compiled, so there is none
    left for the device to compile at load: its table says so. A fresh
    process that cannot import the compiler loads the program and runs it
    three times."""

    # A function of this test's own: Dynamo keeps one compilation per
    # function, and another test already compiled the shared one.
    def affine(x, w, b):
        return torch.relu(x @ w + b)

    torch.manual_seed(0)
    args = _affine_inputs()
    _, compiler = _compiled(affine, *args)
    code = compiler.graphs[0].program
    summary = describe(code)["program"]
    assert summary["kernels"] >= 1
    assert summary["compiled_kernels"] == summary["kernels"], (
        "a kernel left as source would be compiled where the program is loaded"
    )

    saved = tmp_path / "program.pkl"
    saved.write_bytes(pickle.dumps(code))
    torch.save(
        {"args": [arg.cpu() for arg in args], "expected": affine(*args).cpu()},
        tmp_path / "call.pt",
    )
    script = textwrap.dedent(
        """
        import importlib.abc, pickle, sys

        class Refuse(importlib.abc.MetaPathFinder):
            blocked = ("luminal_torch_cuda_lite_compiler", "luminal_torch_common")

            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in self.blocked:
                    raise ImportError(f"{name} is not installed in this process")

        sys.meta_path.insert(0, Refuse())

        import torch

        torch.set_float32_matmul_precision("highest")
        with open(sys.argv[1], "rb") as file:
            program = pickle.load(file)
        call = torch.load(sys.argv[2])
        args = [arg.cuda() for arg in call["args"]]
        for _ in range(3):
            (out,) = program(list(args))
            torch.testing.assert_close(out.cpu(), call["expected"], atol=1e-5, rtol=1e-5)
        print("ran", type(program).__module__)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(saved), str(tmp_path / "call.pt")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "ran luminal_torch_cuda_lite_runtime" in result.stdout, result.stdout


def test_a_tensor_a_call_cannot_bind_is_refused_by_the_runtime():
    """Dynamo guards what a tensor is, not where it starts in its storage
    or what it shares storage with. The runtime refuses both by name, with
    the one refusal class the compiler also raises."""
    from luminal_torch_cuda_lite_compiler.boundary import UnsupportedBoundary
    from luminal_torch_cuda_lite_runtime import BoundaryRefusal

    assert UnsupportedBoundary is BoundaryRefusal
    assert issubclass(BoundaryRefusal, RuntimeError)

    def bump(x, y):
        y.add_(1.0)
        return x * 2.0

    x, y = torch.randn(8, device="cuda"), torch.zeros(8, device="cuda")
    _, compiler = _compiled(bump, x, y)
    code = compiler.graphs[0].program
    table = describe(code)
    order = [
        table["inputs"][action["read"]["input"]]["name"] for action in table["actions"]
    ]
    assert len(order) == 2

    storage = torch.randn(16, device="cuda")
    with pytest.raises(BoundaryRefusal, match="storage offset 4"):
        code([storage[4:12], torch.zeros(8, device="cuda")])
    # Both arguments on one storage while one is written back into.
    with pytest.raises(BoundaryRefusal, match="share device storage"):
        code([storage[:8], storage.as_strided((8,), (1,), 0)])
