"""PyTorch OpInfo coverage for the Luminal ``torch.compile`` backend.

This intentionally tests compiler-backend correctness, not PyTorch device-
backend conformance. By default OpInfo creates CPU tensors, so eager PyTorch
is compared with Luminal's reference backend. Setting
``LUMINAL_TEST_DEVICE=cuda`` explicitly switches both sides to CUDA. Dynamo
captures the public PyTorch operation, and Luminal compiles the resulting
graph. Failures are intentionally unmarked so unsupported operations and dtype
paths remain visible as ordinary test failures.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import nullcontext
from itertools import pairwise
from types import FunctionType
from typing import Any

import pytest
import torch
from luminal import luminal_backend
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.inductor_utils import clone_preserve_strides_offset
from torch.testing._internal.opinfo.core import OpInfo, SampleInput
from torch.testing._utils import freeze_rng_state, wrapper_set_seed
from torch.utils import _pytree as pytree

# PyTorch owns the complete operation inventory, metadata, and generated inputs
# through ``op_db``. Every OpInfo is collected; unsupported Luminal paths should
# be visible as test failures rather than filtered out by a capability allowlist.
# Every CPU-supported dtype and every generated sample are exercised in this one
# suite; there is no reduced smoke mode with a different coverage contract.
_OPINFOS = tuple(op_db)

_DTYPE_TOLERANCES = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (3e-2, 2e-2),
    torch.float32: (1e-5, 1e-5),
    torch.float64: (1e-7, 1e-7),
    torch.int32: (0.0, 0.0),
    torch.int64: (0.0, 0.0),
    torch.bool: (0.0, 0.0),
}


def _has_noncontiguous_tensor(sample: SampleInput) -> bool:
    leaves = pytree.tree_leaves((sample.input, sample.args, sample.kwargs))
    return any(
        isinstance(value, torch.Tensor)
        and value.layout is torch.strided
        and not value.is_contiguous()
        for value in leaves
    )


def _supports_noncontiguous_transform(sample: SampleInput) -> bool:
    tensors = [
        value
        for value in pytree.tree_leaves((sample.input, sample.args, sample.kwargs))
        if isinstance(value, torch.Tensor)
    ]
    return bool(tensors) and all(value.layout is torch.strided for value in tensors)


def _opinfo_dtype_cases() -> tuple:
    """Build lightweight cases for every CPU-supported OpInfo dtype."""

    cases = []
    for op in _OPINFOS:
        for dtype in sorted(op.supported_dtypes("cpu"), key=str):
            dtype_name = str(dtype).removeprefix("torch.")
            cases.append(
                pytest.param(
                    op,
                    dtype,
                    id=f"{op.formatted_name}-{dtype_name}",
                )
            )
    return tuple(cases)


def _opinfo_shard_bounds(total: int, index: int, count: int) -> tuple[int, int]:
    """Return one gap-free, near-even interval of the OpInfo parent cases."""

    if count <= 0:
        raise ValueError(f"LUMINAL_OPINFO_SHARD_COUNT must be positive, got {count}")
    if not 0 <= index < count:
        raise ValueError(
            "LUMINAL_OPINFO_SHARD_INDEX must satisfy "
            f"0 <= index < count, got index={index}, count={count}"
        )
    if count > total:
        raise ValueError(
            "LUMINAL_OPINFO_SHARD_COUNT cannot exceed the number of OpInfo "
            f"parent cases, got count={count}, total={total}"
        )
    return total * index // count, total * (index + 1) // count


def _shard_opinfo_dtype_cases(cases: tuple) -> tuple:
    """Select this process's deterministic interval of the complete suite.

    Sharding changes only how the parent cases are distributed across workers.
    Each selected parent still exercises every PyTorch-generated sample and its
    noncontiguous variant. With indices ``0..count-1``, the intervals are
    disjoint and their union is the complete OpInfo x CPU-dtype inventory.
    """

    try:
        count = int(os.environ.get("LUMINAL_OPINFO_SHARD_COUNT", "1"))
        index = int(os.environ.get("LUMINAL_OPINFO_SHARD_INDEX", "0"))
    except ValueError as error:
        raise ValueError(
            "LUMINAL_OPINFO_SHARD_INDEX and LUMINAL_OPINFO_SHARD_COUNT must be integers"
        ) from error
    start, end = _opinfo_shard_bounds(len(cases), index, count)
    return cases[start:end]


_ALL_OPINFO_DTYPE_CASES = _opinfo_dtype_cases()
_OPINFO_DTYPE_CASES = _shard_opinfo_dtype_cases(_ALL_OPINFO_DTYPE_CASES)


@pytest.mark.parametrize(("total", "count"), ((1, 1), (7, 3), (6121, 32)))
def test_opinfo_shard_bounds_cover_inventory(total: int, count: int) -> None:
    """Every shard count partitions its inventory without gaps or overlap."""

    bounds = [_opinfo_shard_bounds(total, index, count) for index in range(count)]
    assert bounds[0][0] == 0
    assert bounds[-1][1] == total
    assert all(left[1] == right[0] for left, right in pairwise(bounds))
    assert sum(end - start for start, end in bounds) == total


def _clone_sample(sample: SampleInput) -> SampleInput:
    """Clone tensors so eager and compiled calls cannot affect each other."""

    def clone(value: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        detached = value.detach()
        if detached.layout is torch.strided:
            return clone_preserve_strides_offset(detached)
        return detached.clone()

    return sample.transform(clone)


def _call(op: Callable[..., Any], sample: SampleInput) -> Any:
    return op(sample.input, *sample.args, **sample.kwargs)


def _call_without_seed_wrapper(op: Callable[..., Any], *args, **kwargs) -> Any:
    """The traceable body of PyTorch's ``wrapper_set_seed``."""

    return op(*args, **kwargs)


def _traceable_opinfo_callable(
    op: Callable[..., Any],
) -> tuple[Callable[..., Any], bool]:
    """Remove only PyTorch's RNG test wrapper from an OpInfo callable.

    Several OpInfos define their public callable as a lambda that invokes
    ``wrapper_set_seed``. That wrapper enters internal C++ context managers to
    save RNG state; Dynamo cannot trace their pybind constructors in a full
    graph. Clone the callable with only that global replaced by its operation
    body, then reproduce the wrapper's seed and state isolation outside the
    compiled graph.
    """

    if not isinstance(op, FunctionType):
        return op, False
    if "wrapper_set_seed" not in op.__code__.co_names:
        return op, False
    if op.__globals__.get("wrapper_set_seed") is not wrapper_set_seed:
        return op, False

    traceable_globals = op.__globals__.copy()
    traceable_globals["wrapper_set_seed"] = _call_without_seed_wrapper
    traceable = FunctionType(
        op.__code__,
        traceable_globals,
        name=op.__name__,
        argdefs=op.__defaults__,
        closure=op.__closure__,
    )
    traceable.__kwdefaults__ = op.__kwdefaults__
    traceable.__annotations__ = op.__annotations__
    return traceable, True


def test_seed_wrapped_opinfos_are_made_traceable() -> None:
    """Every PyTorch RNG wrapper is removed without changing other OpInfos."""

    wrapper_backed = 0
    for op in _OPINFOS:
        op_callable = op.get_op()
        traceable, changed = _traceable_opinfo_callable(op_callable)
        is_wrapper_backed = (
            isinstance(op_callable, FunctionType)
            and "wrapper_set_seed" in op_callable.__code__.co_names
            and op_callable.__globals__.get("wrapper_set_seed") is wrapper_set_seed
        )
        assert changed is is_wrapper_backed
        if changed:
            wrapper_backed += 1
            assert traceable is not op_callable

    assert wrapper_backed > 0


def _assert_close(actual: Any, expected: Any, dtype: torch.dtype) -> None:
    # Unary operations such as acos/acosh promote integral inputs to F32. Base
    # tolerances on the reference output when it has one unambiguous tensor
    # dtype; using the parametrized input dtype would incorrectly demand exact
    # integer equality from a floating-point result.
    expected_dtypes = {
        value.dtype
        for value in pytree.tree_leaves(expected)
        if isinstance(value, torch.Tensor)
    }
    if len(expected_dtypes) == 1:
        dtype = expected_dtypes.pop()

    kwargs = {
        "equal_nan": True,
        "check_device": True,
        "check_dtype": True,
        "check_layout": True,
        "check_stride": False,
    }
    if dtype in _DTYPE_TOLERANCES:
        kwargs["rtol"], kwargs["atol"] = _DTYPE_TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, **kwargs)


def _assert_sample_state_close(
    actual: SampleInput, expected: SampleInput, dtype: torch.dtype
) -> None:
    actual_leaves = pytree.tree_leaves((actual.input, actual.args, actual.kwargs))
    expected_leaves = pytree.tree_leaves(
        (expected.input, expected.args, expected.kwargs)
    )
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        if isinstance(expected_leaf, torch.Tensor):
            assert isinstance(actual_leaf, torch.Tensor)
            _assert_close(actual_leaf, expected_leaf, dtype)


def _test_opinfo_sample(
    device: torch.device,
    op: OpInfo,
    dtype: torch.dtype,
    sample_index: int,
    sample: SampleInput,
    compiled_source: SampleInput,
    input_layout: str,
) -> None:
    # Each sample/layout is an independent conformance case. Reset Dynamo so a
    # different shape or stride cannot consume another case's recompile budget.
    torch._dynamo.reset()
    eager_sample = _clone_sample(sample)
    compiled_sample = _clone_sample(compiled_source)
    if input_layout == "noncontiguous":
        assert _has_noncontiguous_tensor(compiled_sample)
    op_callable, uses_seed_wrapper = _traceable_opinfo_callable(op.get_op())
    rng_context = freeze_rng_state if uses_seed_wrapper else nullcontext
    with rng_context():
        torch.manual_seed(42 if uses_seed_wrapper else 0)
        expected = _call(op_callable, eager_sample)

    compile_count = 0

    def counting_backend(gm, example_inputs, options=None):
        nonlocal compile_count
        compile_count += 1
        return luminal_backend(
            gm,
            example_inputs,
            options={"search_iterations": 1},
        )

    def fn(*args, **kwargs):
        return op_callable(*args, **kwargs)

    compiled = torch.compile(
        fn,
        backend=counting_backend,
        fullgraph=True,
        dynamic=False,
    )
    with rng_context():
        torch.manual_seed(42 if uses_seed_wrapper else 0)
        actual = _call(compiled, compiled_sample)

    case_name = f"{op.full_name} {dtype} sample {sample_index} {input_layout}"
    assert compile_count > 0, f"Luminal backend was not invoked for {case_name}"
    _assert_close(actual, expected, dtype)
    _assert_sample_state_close(compiled_sample, eager_sample, dtype)


@pytest.mark.parametrize(("op", "dtype"), _OPINFO_DTYPE_CASES)
def test_opinfo_forward_all_samples(
    device: torch.device,
    op: OpInfo,
    dtype: torch.dtype,
    subtests,
) -> None:
    """Compare every PyTorch sample with Luminal, reporting each separately."""

    samples = tuple(op.sample_inputs("cpu", dtype, requires_grad=False))
    exercised_samples = 0
    for sample_index, sample in enumerate(samples):
        if device.type != "cpu":
            sample = sample.transform(
                lambda value: (
                    value.to(device) if isinstance(value, torch.Tensor) else value
                )
            )

        layout_samples = [("contiguous", sample)]
        if _supports_noncontiguous_transform(sample):
            noncontiguous_sample = sample.noncontiguous()
            if _has_noncontiguous_tensor(noncontiguous_sample):
                layout_samples.append(("noncontiguous", noncontiguous_sample))

        for input_layout, compiled_source in layout_samples:
            exercised_samples += 1
            with subtests.test(
                sample_index=sample_index,
                input_layout=input_layout,
            ):
                _test_opinfo_sample(
                    device,
                    op,
                    dtype,
                    sample_index,
                    sample,
                    compiled_source,
                    input_layout,
                )

    assert exercised_samples > 0


@pytest.mark.parametrize(
    "dtype", (torch.float16, torch.bfloat16), ids=("float16", "bfloat16")
)
def test_empty_low_precision_output(device: torch.device, dtype: torch.dtype) -> None:
    """Zero-sized half outputs must materialize without reading an empty buffer."""

    def backend(gm, example_inputs, options=None):
        return luminal_backend(
            gm,
            example_inputs,
            options={"search_iterations": 1},
        )

    def fn(left, right):
        return left + right

    left = torch.empty((0, 1, 3), device=device, dtype=dtype)
    right = torch.empty((0, 10, 3), device=device, dtype=dtype)
    compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)
    actual = compiled(left, right)

    assert actual.shape == (0, 10, 3)
    assert actual.dtype == dtype
    assert actual.device == device
