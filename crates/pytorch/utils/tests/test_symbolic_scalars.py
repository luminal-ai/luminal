"""Symbolic integers and bools as operator arguments and as values.

torch hands a backend an integer that changes between calls as a ``SymInt``:
a dynamic tensor dimension, or a Python ``int`` argument declared dynamic.
Arithmetic on one stays symbolic (``n * n``, ``-n // 2``, ``(n - 7) % 4``),
and a comparison of one is a ``SymBool`` (``n > 3``). A program may use any of
these wherever it could use a Python number: as a count or an amount
(``repeat``, ``pad``, ``roll``, ``full``, ``topk``), or as a value in tensor
arithmetic.

Each case runs both ways the integer can arrive, on one compiled graph, and is
compared with eager at several values. torch's integer semantics are the
reference: ``//`` floors and ``%`` takes the divisor's sign, where the
recorder's own division and remainder truncate.

The integer reaches the compiled program as an argument of its own, an
integer slot, and its value defines its symbol. No tensor has to carry it.
"""

import importlib

import pytest
import torch
import torch.nn.functional as F
from backend_test_utils import selected_backend
from torch.fx.experimental.dynamic_spec import IntVar, ShapesSpec, ShapeVar, TensorSpec

# The integer's declared range, and the values each case is called with. The
# range is bounded so the CUDA runtime can budget its arena.
LOW, HIGH = 1, 6
VALUES = (2, 3, 5, 1, 6)

# torch computes a float from a symbolic integer, and a tensor from a symbolic
# bool, as 0-d CPU tensors, also inside a CUDA graph. A translated program
# lives on one device, so the CUDA backend refuses these until the IR records
# where a tensor lives.
_HOST_SCALAR = pytest.mark.xfail(
    selected_backend() == "cuda_lite",
    reason=(
        "LUM-1336: torch computes this scalar as a 0-d CPU tensor inside the "
        "CUDA graph, and a translated program lives on one device. "
        "https://linear.app/luminalai/issue/LUM-1336"
    ),
    strict=True,
)

CASES = {
    # ---- a symbolic integer as a count or an amount
    "repeat": lambda t, n: t.repeat(n, 1),
    "repeat_with_a_new_axis": lambda t, n: t.repeat(n, 1, 2),
    "tile": lambda t, n: t.tile((n, 1)),
    "pad": lambda t, n: F.pad(t, (0, n)),
    "pad_two_axes": lambda t, n: F.pad(t, (n, 1, 0, n), value=2.0),
    "roll": lambda t, n: t.roll(n, 0),
    "full_fill": lambda t, n: t + torch.full((2, 3), n, device=t.device)[0],
    "masked_fill": lambda t, n: t.masked_fill(t > 0, n),
    # The sorted axis has 8 entries, so every n in the range is a legal k.
    "topk_values": lambda t, n: torch.cat((t, t + 0.5)).topk(n, dim=0).values,
    "topk_indices": lambda t, n: torch.cat((t, t + 0.5)).topk(n, dim=0).indices,
    "topk_smallest": lambda t, n: (
        torch.cat((t, t + 0.5)).topk(n, dim=0, largest=False).values
    ),
    # ---- torch's integer arithmetic on a symbolic integer, used as a value
    "square": lambda t, n: t * (n * n),
    "floor_division_of_a_negative": lambda t, n: t * (-n // 2),
    "floor_division_of_a_difference": lambda t, n: t * ((n - 7) // 2),
    "modulo_of_a_negative": lambda t, n: t * ((n - 7) % 4),
    "modulo_of_a_negated_symbol": lambda t, n: t * ((4 - n) % 4),
    # ---- a symbolic bool as a value
    "greater_than": lambda t, n: t * (n > 3),
    "equal_and_not_equal": lambda t, n: t * (n == 3) + (n != 2),
    "at_most": lambda t, n: t + (n <= 3),
}

HOST_SCALAR_CASES = {
    "bool_tensor": lambda t, n: torch.where(
        torch.tensor(n > 3, device=t.device), t, -t
    ),
    "float_from_an_integer": lambda t, n: t * (n / 2.0),
}

ALL_CASES = {**CASES, **HOST_SCALAR_CASES}


def _compiler():
    return importlib.import_module(f"luminal_{selected_backend()}").Compiler()


def _case_params():
    params = [pytest.param(name, id=name) for name in sorted(CASES)]
    params += [
        pytest.param(name, id=name, marks=_HOST_SCALAR)
        for name in sorted(HOST_SCALAR_CASES)
    ]
    return params


def _assert_matches_eager(got, want, context):
    assert got.dtype == want.dtype, f"{context}: dtype {got.dtype} vs {want.dtype}"
    assert tuple(got.shape) == tuple(want.shape), (
        f"{context}: shape {tuple(got.shape)} vs {tuple(want.shape)}"
    )
    torch.testing.assert_close(got, want, msg=lambda text: f"{context}: {text}")


def _program_and_calls(fn, source, device):
    """The compiled program, its dynamic spec, and its calls, with the
    integer read off a tensor dimension or passed as a Python int."""
    torch.manual_seed(0)
    x = torch.randn(4, 3, device=device)
    if source == "tensor_dim":

        def program(t, u):
            return fn(t, u.shape[0])

        size = ShapeVar("n", min=LOW, max=HIGH, optimization_hint=2)
        spec = ShapesSpec(params={"u": TensorSpec([size])})
        calls = [(x, torch.zeros(n, device=device)) for n in VALUES]
    else:

        def program(t, n):
            return fn(t, n)

        spec = ShapesSpec(params={"n": IntVar("n", min=LOW, max=HIGH)})
        calls = [(x, n) for n in VALUES]
    return program, spec, calls


@pytest.mark.parametrize("source", ["tensor_dim", "int_arg"])
@pytest.mark.parametrize("name", _case_params())
def test_symbolic_scalar_matches_eager(device, name, source):
    program, spec, calls = _program_and_calls(ALL_CASES[name], source, device)
    compiler = _compiler()
    compiled = torch.compile(program, backend=compiler, dynamic_shapes=spec)
    for args in calls:
        value = args[1] if isinstance(args[1], int) else args[1].shape[0]
        _assert_matches_eager(compiled(*args), program(*args), f"{name} at n={value}")
    assert len(compiler.graphs) == 1, (
        f"{name}: every value should run on the one dynamic graph, "
        f"compiled {len(compiler.graphs)}"
    )


def test_a_symbolic_pad_that_may_crop_is_refused(device):
    """torch crops by a negative pad. A literal states its sign; a symbolic
    amount that the declared range lets go negative is refused by name
    rather than padded by whatever it evaluates to."""

    def program(t, u):
        return F.pad(t, (0, u.shape[0] - 3))

    size = ShapeVar("n", min=LOW, max=HIGH, optimization_hint=4)
    compiled = torch.compile(
        program,
        backend=_compiler(),
        dynamic_shapes=ShapesSpec(params={"u": TensorSpec([size])}),
    )
    with pytest.raises(Exception, match="do not prove nonnegative"):
        compiled(torch.randn(4, 3, device=device), torch.zeros(4, device=device))


def test_a_symbolic_topk_k_not_kept_inside_the_axis_is_refused(device):
    """torch refuses a ``k`` larger than the axis. Here ``k`` is half of one
    dynamic size and the axis is another, so no dimension range relates
    them; the translation is refused by name rather than slicing the sorted
    axis by whatever ``k`` evaluates to."""

    def program(t, u):
        return t.topk(u.shape[0] // 2, dim=0).values

    rows = ShapeVar("rows", min=2, max=8, optimization_hint=4)
    size = ShapeVar("n", min=2, max=12, optimization_hint=4)
    compiled = torch.compile(
        program,
        backend=_compiler(),
        dynamic_shapes=ShapesSpec(
            params={"t": TensorSpec([rows, 3]), "u": TensorSpec([size])}
        ),
    )
    with pytest.raises(Exception, match="not proved to lie within the axis"):
        compiled(torch.randn(4, 3, device=device), torch.zeros(4, device=device))


def test_an_integer_argument_is_bound_by_value(device):
    """torch hands a backend one value per graph placeholder, and a changing
    ``int`` is a placeholder of its own. The compiled program takes it as
    an integer slot whose value defines its symbol: the program has one
    tensor input, with no extra tensor standing in for the integer."""
    program, spec, calls = _program_and_calls(CASES["square"], "int_arg", device)
    compiler = _compiler()
    compiled = torch.compile(program, backend=compiler, dynamic_shapes=spec)
    for args in calls:
        _assert_matches_eager(compiled(*args), program(*args), f"n={args[1]}")

    (record,) = compiler.graphs
    # One slot per argument: `(kind, name, read, symbol)`.
    slots = record.program._graph.input_slots
    assert [(kind, read) for kind, _, read, _ in slots] == [
        ("tensor", True),
        ("int", True),
    ], slots
    assert slots[1][3] is not None, "the integer argument defines a symbol"
    assert record.input_shapes == ((4, 3),), (
        f"the program's tensor inputs are the caller's alone: {record.input_shapes}"
    )


def test_a_program_whose_only_argument_is_an_integer_matches_eager(device):
    """A graph may read no tensor argument at all. Its integer still binds,
    and its declared range still bounds the program's sizes."""

    def program(n):
        return torch.full((n, 3), 1.5, device=device) + n

    compiler = _compiler()
    compiled = torch.compile(
        program,
        backend=compiler,
        dynamic_shapes=ShapesSpec(params={"n": IntVar("n", min=LOW, max=HIGH)}),
    )
    for n in VALUES:
        _assert_matches_eager(compiled(n), program(n), f"n={n}")
    assert len(compiler.graphs) == 1, f"compiled {len(compiler.graphs)} graphs"


def test_a_negative_integer_argument_is_refused_by_name(device):
    """torch makes a changing ``int`` symbolic with no lower bound, and
    binding it adds no guard of our own: a negative value reaches the
    compiled graph. A symbol's value is unsigned in the runtimes today, so
    that call is refused by the argument's name rather than wrapped."""

    def program(t, n):
        return t * n + n

    torch.manual_seed(0)
    x = torch.randn(4, 3, device=device)
    compiler = _compiler()
    previous = torch._dynamo.config.cache_size_limit
    # The second value recompiles the frame with a symbolic integer.
    torch._dynamo.config.cache_size_limit = 4
    try:
        compiled = torch.compile(program, backend=compiler)
        for n in (3, 4, 5):
            _assert_matches_eager(compiled(x, n), program(x, n), f"n={n}")
        graphs = len(compiler.graphs)
        with pytest.raises(RuntimeError, match=r"is -4.*unsigned.*LUM-1337"):
            compiled(x, -4)
        assert len(compiler.graphs) == graphs, (
            "the negative value reached the graph compiled for a symbolic "
            "integer; no guard of ours sent it to a recompile"
        )
    finally:
        torch._dynamo.config.cache_size_limit = previous


def test_a_runtime_check_the_user_wrote_is_enforced(device):
    """``torch._check`` on a symbolic size is not a guard: torch asserts it
    inside the graph it hands the backend, and expects the backend to raise
    when a call breaks it. The declared range of the same size is guarded
    by torch before the graph is called, so only the user's check is left
    for the compiled program to evaluate."""

    def program(t):
        torch._check(t.shape[0] != 5)
        return t * 2

    size = ShapeVar("n", min=2, max=8, optimization_hint=4)
    compiler = _compiler()
    compiled = torch.compile(
        program,
        backend=compiler,
        dynamic_shapes=ShapesSpec(params={"t": TensorSpec([size, 3])}),
    )
    torch.manual_seed(0)
    for n in (4, 7):
        t = torch.randn(n, 3, device=device)
        _assert_matches_eager(compiled(t), program(t), f"n={n}")

    broken = torch.randn(5, 3, device=device)
    with pytest.raises(
        RuntimeError, match=r"Runtime assertion failed for expression Ne\(u0, 5\)"
    ):
        compiled(broken)

    (record,) = compiler.graphs
    assert record.program._graph.runtime_assertions == [
        "Runtime assertion failed for expression Ne(u0, 5) on node 'ne'"
    ], "the declared range is decided by the range itself and needs no check"
    # The checks are separate from binding: switched off, the call runs.
    record.program.runtime_assertions = False
    _assert_matches_eager(compiled(broken), broken * 2, "unchecked n=5")
