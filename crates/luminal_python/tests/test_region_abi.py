from __future__ import annotations

import pytest
import torch

from luminal.region_abi import (
    DynamicRange,
    InputKind,
    InputSpec,
    MutationSpec,
    OutputSpec,
    RegionSpec,
    SymbolSpec,
    TensorSpec,
)


def _tensor_spec(**overrides) -> TensorSpec:
    values = {
        "dtype": "torch.float16",
        "device_type": "cuda",
        "device_index": 0,
        "layout": "torch.strided",
        "shape": (4, 8),
        "stride": (8, 1),
        "storage_offset": 0,
        "requires_grad": False,
    }
    values.update(overrides)
    return TensorSpec(**values)


def test_tensor_spec_collects_noncontiguous_metadata() -> None:
    tensor = torch.arange(20, dtype=torch.float32).reshape(4, 5).t()

    spec = TensorSpec.from_tensor(tensor)

    assert spec == TensorSpec(
        dtype="torch.float32",
        device_type="cpu",
        device_index=None,
        layout="torch.strided",
        shape=(5, 4),
        stride=(1, 5),
        storage_offset=0,
        requires_grad=False,
    )
    assert all(
        not isinstance(getattr(spec, name), torch.Tensor) for name in spec.__slots__
    )


def test_tensor_spec_reads_fake_cuda_metadata_without_cuda() -> None:
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    with FakeTensorMode():
        tensor = torch.empty_strided(
            (4, 4, 64),
            (768, 64, 1),
            device="cuda:0",
            dtype=torch.float16,
        )
        assert isinstance(tensor, FakeTensor)
        spec = TensorSpec.from_tensor(tensor)

    assert spec.dtype == "torch.float16"
    assert spec.device_type == "cuda"
    assert spec.device_index == 0
    assert spec.shape == (4, 4, 64)
    assert spec.stride == (768, 64, 1)


def test_tensor_spec_preserves_symbolic_fake_tensor_dimension() -> None:
    from torch._dynamo.source import LocalSource
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    symbol = shape_env.create_symintnode(
        shape_env.create_symbol(4, LocalSource("num_tokens")), hint=4
    )
    with FakeTensorMode(shape_env=shape_env):
        tensor = torch.empty_strided(
            (symbol, 8), (8, 1), device="cuda:0", dtype=torch.float16
        )
        spec = TensorSpec.from_tensor(tensor)

    assert isinstance(spec.shape[0], SymbolSpec)
    assert spec.shape[0].expression
    assert spec.shape[1] == 8
    assert spec.stride == (8, 1)


def test_tensor_spec_preserves_v_projection_layout_and_offset() -> None:
    storage = torch.empty(4096, dtype=torch.float16)
    value = storage.as_strided((4, 4, 64), (768, 64, 1), storage_offset=512)

    spec = TensorSpec.from_tensor(value)

    assert spec.shape == (4, 4, 64)
    assert spec.stride == (768, 64, 1)
    assert spec.storage_offset == 512


def test_dynamic_range_distinguishes_ranged_and_exact_artifacts() -> None:
    assert not DynamicRange("num_tokens", 1, 128).is_exact
    assert DynamicRange("num_tokens", 4, 4).is_exact

    with pytest.raises(ValueError, match="non-negative"):
        DynamicRange("num_tokens", -1, 4)
    with pytest.raises(ValueError, match="at least"):
        DynamicRange("num_tokens", 8, 4)


def test_region_spec_represents_residual_mutation() -> None:
    tensor = _tensor_spec()
    spec = RegionSpec(
        inputs=(
            InputSpec("activation", tensor, InputKind.RUNTIME),
            InputSpec("residual", tensor, InputKind.RUNTIME, mutable=True),
            InputSpec("weight", tensor, InputKind.WEIGHT),
        ),
        outputs=(OutputSpec("normalized", tensor),),
        mutations=(MutationSpec("residual"),),
        dynamic_ranges=(DynamicRange("num_tokens", 1, 128),),
    )

    assert spec.mutations == (MutationSpec("residual"),)

    with pytest.raises(ValueError, match="not declared mutable"):
        RegionSpec(
            inputs=(InputSpec("residual", tensor, InputKind.RUNTIME),),
            outputs=(),
            mutations=(MutationSpec("residual"),),
        )
    with pytest.raises(ValueError, match="unknown input"):
        RegionSpec(
            inputs=(),
            outputs=(),
            mutations=(MutationSpec("residual"),),
        )


def test_region_spec_represents_writable_attention_output() -> None:
    tensor = _tensor_spec(shape=(4, 4096), stride=(4096, 1))

    spec = RegionSpec(
        inputs=(InputSpec("hidden_states", tensor, InputKind.RUNTIME),),
        outputs=(
            OutputSpec(
                "attention_output",
                tensor,
                consumer_writable=True,
            ),
        ),
        dynamic_ranges=(DynamicRange("num_tokens", 4, 4),),
    )

    assert spec.outputs[0].consumer_writable
    assert spec.dynamic_ranges[0].is_exact


def test_region_spec_validates_names_and_aliases() -> None:
    tensor = _tensor_spec()
    duplicate = InputSpec("x", tensor, InputKind.RUNTIME)

    with pytest.raises(ValueError, match="input names must be unique"):
        RegionSpec(inputs=(duplicate, duplicate), outputs=())
    with pytest.raises(ValueError, match="aliases unknown input"):
        RegionSpec(
            inputs=(duplicate,),
            outputs=(OutputSpec("view", tensor, aliases_input="missing"),),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": (4, 8), "stride": (8,)},
        {"shape": (-1, 8)},
        {"storage_offset": -1},
    ],
)
def test_tensor_spec_rejects_invalid_concrete_metadata(kwargs) -> None:
    with pytest.raises(ValueError):
        _tensor_spec(**kwargs)
