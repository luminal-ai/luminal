"""A compiled graph is a code object: plain bytes that the backend's runtime
package builds every executable from.

These are the facts every backend's code object has in common, and they run
against whichever backend is selected. What a backend states about its own
inputs, rows and program is tested beside that backend.
"""

import importlib
import pickle
import subprocess
import sys
import textwrap

import torch
from backend_test_utils import selected_backend
from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec

COMPILER = f"luminal_torch_{selected_backend()}_compiler"
RUNTIME = f"luminal_torch_{selected_backend()}_runtime"
# The frontend every compiler shares. A saved program needs neither.
FRONTEND = "luminal_torch_common"

_TOLERANCE = {"atol": 1e-5, "rtol": 1e-5}


def _runtime():
    return importlib.import_module(RUNTIME)


def _compiled(fn, *args, **options):
    compiler = importlib.import_module(COMPILER).Compiler()
    compiled = torch.compile(fn, backend=compiler, fullgraph=True, **options)
    with torch.no_grad():
        compiled(*args)
    return compiled, compiler


def _affine_inputs(device):
    torch.manual_seed(0)
    return (
        torch.randn(4, 16, device=device),
        torch.randn(16, 8, device=device),
        torch.randn(8, device=device),
    )


def test_a_compiled_graph_is_its_bytes(device):
    """Pickling a compiled graph carries its bytes and names the runtime
    package alone, so loading one imports no compiler. The copy runs."""

    def affine(x, w, b):
        return torch.relu(x @ w + b)

    args = _affine_inputs(device)
    _, compiler = _compiled(affine, *args)
    code = compiler.graphs[0].program

    payload = pickle.dumps(code)
    assert RUNTIME.encode() in payload
    assert COMPILER.encode() not in payload, "a saved program names the runtime alone"
    loaded = pickle.loads(payload)
    assert type(loaded) is _runtime().Code
    assert loaded.code == code.code
    (out,) = loaded(list(args))
    torch.testing.assert_close(out, affine(*args), **_TOLERANCE)


def test_a_code_object_states_torchs_side_of_a_call_as_plain_data(device):
    """What a call reads its arguments against and builds its outputs from
    is readable in the code object, in the same tables on every backend:
    the kind of each argument slot, what a call does with it, where each
    output leaf comes from, and the value each dimension starts at."""

    def fn(x, y):
        y.add_(1.0)
        return x * 2.0, x.shape[0] * 3

    size = ShapeVar("n", min=2, max=8, optimization_hint=4)
    x, y = torch.randn(4, 3, device=device), torch.zeros(2, device=device)
    _, compiler = _compiled(
        fn, x, y, dynamic_shapes=ShapesSpec(params={"x": TensorSpec([size, 3])})
    )
    runtime = _runtime()
    table = runtime.describe(compiler.graphs[0].program)

    assert table["format"] == runtime.FORMAT
    # torch passes the dynamic size as an argument of its own.
    assert table["slots"] == ["tensor", "int", "tensor"]
    assert len(table["slot_names"]) == 3
    # Both tensors are read by the program; the integer is the dimension.
    actions = [next(iter(action)) for action in table["actions"]]
    assert actions == ["read", "dimension", "read"]
    # Each tensor slot names an input of the backend's own table, and that
    # input is an argument, not a constant the program holds.
    read = [action["read"]["input"] for action in table["actions"] if "read" in action]
    assert len(set(read)) == 2
    assert all(not table["inputs"][input].get("constant") for input in read)
    # One row per tensor the program writes: the product, and `y`.
    assert len(table["rows"]) == 2
    # The leaves: the product, and the size expression read from the
    # integer argument that states it.
    (row,) = [leaf["row"] for leaf in table["leaves"] if "row" in leaf]
    assert row["row"] < len(table["rows"]) and not row["scalar"]
    (integer,) = [leaf["integer"] for leaf in table["leaves"] if "integer" in leaf]
    (source,) = integer["sources"]
    assert table["slots"][source["slot"]] == "int" and source["axis"] is None
    # The dimension starts at the size the program was compiled with.
    assert [value for _, value in table["dims"]] == [4]


def test_a_saved_program_runs_in_a_process_that_cannot_import_the_compiler(
    device, tmp_path
):
    """A code object is all a saved program needs. A fresh process that
    refuses to import the compiler and the shared frontend loads it and runs
    it, with nothing of ours loaded but the runtime package."""

    def affine(x, w, b):
        return torch.relu(x @ w + b)

    args = _affine_inputs(device)
    _, compiler = _compiled(affine, *args)
    code = compiler.graphs[0].program

    saved = tmp_path / "program.pkl"
    saved.write_bytes(pickle.dumps(code))
    torch.save(
        {"args": [arg.cpu() for arg in args], "expected": affine(*args).cpu()},
        tmp_path / "call.pt",
    )
    script = textwrap.dedent(
        """
        import importlib.abc, pickle, sys

        program_path, call_path, device, runtime, *blocked = sys.argv[1:]

        class Refuse(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in blocked:
                    raise ImportError(f"{name} is not installed in this process")

        sys.meta_path.insert(0, Refuse())

        import torch

        torch.set_float32_matmul_precision("highest")
        with open(program_path, "rb") as file:
            program = pickle.load(file)
        call = torch.load(call_path)
        args = [arg.to(device) for arg in call["args"]]
        for _ in range(3):
            (out,) = program(list(args))
            torch.testing.assert_close(out.cpu(), call["expected"], atol=1e-5, rtol=1e-5)
        loaded = sorted(name for name in sys.modules if name.startswith("luminal"))
        assert all(name.startswith(runtime) for name in loaded), loaded
        print("ran", type(program).__module__)
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(saved),
            str(tmp_path / "call.pt"),
            str(device),
            RUNTIME,
            COMPILER,
            FRONTEND,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert f"ran {RUNTIME}" in result.stdout, result.stdout
