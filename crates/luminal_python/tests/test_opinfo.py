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

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.inductor_utils import clone_preserve_strides_offset
from torch.testing._internal.opinfo.core import OpInfo, SampleInput
from torch.utils import _pytree as pytree

from luminal import luminal_backend

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


_OPINFO_DTYPE_CASES = _opinfo_dtype_cases()


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


def _assert_close(actual: Any, expected: Any, dtype: torch.dtype) -> None:
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
    torch.manual_seed(0)
    expected = _call(op.get_op(), eager_sample)

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
        return op.get_op()(*args, **kwargs)

    compiled = torch.compile(
        fn,
        backend=counting_backend,
        fullgraph=True,
        dynamic=False,
    )
    torch.manual_seed(0)
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
