"""Sizes that depend on tensor data.

``torch.nonzero(x)`` returns one row per nonzero element, so the shape of its
result is not known until ``x`` has been read. torch gives such a size a symbol
of its own (``u0``) that no argument of the call supplies.

Whether a backend ever sees the op is torch's choice:

- ``fullgraph=False``, the default: torch splits the function at the op, runs
  the op in eager, and compiles the code on either side. The backend never
  sees a data-dependent size.
- ``fullgraph=True``, which ``aot_compile`` requires: the op reaches the
  backend.

No compiled program can state a size that is computed from tensor data
(LUM-1580), so under ``fullgraph=True`` the compile is refused by name. The
refusal is a temporary pass of the shared compiler, not a check of the
compiled program: nothing with such a size is ever built.

A value read with ``.item()`` and used as an operand is not a size, and stays
inside one compiled graph.
"""

import importlib

import pytest
import torch
from backend_test_utils import selected_backend


def _compiler():
    return importlib.import_module(
        f"luminal_torch_{selected_backend()}_compiler"
    ).Compiler()


def _a_size_read_with_item(x):
    n = (x > 0).sum().item()
    return x.flatten()[:n] * 2


def _a_count_read_with_item(x):
    n = (x > 0).sum().item()
    return torch.zeros(n, device=x.device) + x.sum()


# Each program, and the op its refusal names under ``fullgraph=True``.
PROGRAMS = {
    "boolean_mask_indexing": (lambda x: x[x > 0] * 2, "aten.index.Tensor"),
    "masked_select": (
        lambda x: torch.masked_select(x, x > 0) * 2,
        "aten.masked_select.default",
    ),
    # Cast to float before the arithmetic: integer arithmetic on indices that
    # arrive as an input with no stated range is refused for its own reason.
    "nonzero": (
        lambda x: torch.nonzero(x * 2 > 0).to(torch.float32) * 0.5,
        "aten.nonzero.default",
    ),
    "unique": (lambda x: torch.unique(x.round()) + 1, "aten._unique2.default"),
    "a_size_read_with_item": (_a_size_read_with_item, "aten.slice.Tensor"),
    "a_count_read_with_item": (
        _a_count_read_with_item,
        "aten._local_scalar_dense.default",
    ),
}

# Eager returns ``nonzero``'s result column-major, the graph after it inherits
# that layout, and no CUDA kernel writes a column-major output.
_COLUMN_MAJOR_OUTPUT = pytest.mark.xfail(
    selected_backend() == "cuda_lite",
    reason=(
        "LUM-1587: the graph after an eager nonzero has a column-major output, "
        "which the CUDA backend cannot write (LUM-830). "
        "https://linear.app/luminalai/issue/LUM-1587"
    ),
    strict=True,
)


def _programs(marks=None):
    marks = marks or {}
    return [pytest.param(name, id=name, marks=marks.get(name, ())) for name in PROGRAMS]


def _input(device):
    torch.manual_seed(0)
    return torch.randn(4, 6, device=device)


@pytest.mark.parametrize("name", _programs())
def test_a_data_dependent_size_is_refused_by_name_under_fullgraph(device, name):
    """The refusal names the op whose result has the size, says that the size
    is computed from tensor data, and points at the ticket. It is raised while
    compiling: nothing reaches the search, and no program is built."""
    program, op = PROGRAMS[name]
    compiler = _compiler()
    compiled = torch.compile(program, backend=compiler, fullgraph=True)
    with pytest.raises(torch._dynamo.exc.BackendCompilerFailed) as refusal:
        compiled(_input(device))
    message = str(refusal.value)
    assert op in message, message
    assert "from tensor data" in message, message
    assert "LUM-1580" in message, message
    assert all(record.program is None for record in compiler.graphs), (
        "a program with a data-dependent size was built"
    )


@pytest.mark.parametrize("name", _programs({"nonzero": _COLUMN_MAJOR_OUTPUT}))
def test_torch_runs_a_data_dependent_op_in_eager_without_fullgraph(device, name):
    """With ``fullgraph=False`` the op never reaches the backend, and the
    function matches eager in values and shape."""
    program, _ = PROGRAMS[name]
    x = _input(device)
    got = torch.compile(program, backend=_compiler())(x)
    expected = program(x)
    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected)


def test_a_value_read_with_item_and_used_as_an_operand_stays_in_one_graph(device):
    """``.item()`` as an operand creates no size: the whole function is one
    compiled graph under ``fullgraph=True``, and it matches eager."""

    def scaled(x):
        return x * x.sum().item()

    x = _input(device)
    compiler = _compiler()
    got = torch.compile(scaled, backend=compiler, fullgraph=True)(x)
    assert len(compiler.graphs) == 1
    torch.testing.assert_close(got, scaled(x), rtol=1e-5, atol=1e-5)
