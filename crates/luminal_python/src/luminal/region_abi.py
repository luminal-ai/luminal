"""Storage-free metadata for a compiled graph region boundary.

The types in this module describe what a region requires without retaining
PyTorch tensors or inspecting their storage.  Compilation may construct these
specifications from FakeTensors; real storage addresses belong to a later
runtime binding phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import torch


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    """A symbolic scalar expression carried by captured graph metadata."""

    expression: str

    def __post_init__(self) -> None:
        if not self.expression.strip():
            raise ValueError("symbol expression must not be empty")


Dimension: TypeAlias = int | SymbolSpec
ScalarValue: TypeAlias = bool | int | float | SymbolSpec


def _dimension(value: int | torch.SymInt) -> Dimension:
    if isinstance(value, torch.SymInt):
        return SymbolSpec(str(value))
    return int(value)


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Tensor metadata that is safe to collect from a FakeTensor."""

    dtype: str
    device_type: str
    device_index: int | None
    layout: str
    shape: tuple[Dimension, ...]
    stride: tuple[Dimension, ...]
    storage_offset: Dimension
    requires_grad: bool

    def __post_init__(self) -> None:
        if not self.dtype:
            raise ValueError("tensor dtype must not be empty")
        if not self.device_type:
            raise ValueError("tensor device type must not be empty")
        if not self.layout:
            raise ValueError("tensor layout must not be empty")
        if len(self.shape) != len(self.stride):
            raise ValueError("tensor shape and stride must have the same rank")
        if any(isinstance(dim, int) and dim < 0 for dim in self.shape):
            raise ValueError("concrete tensor dimensions must be non-negative")
        if isinstance(self.storage_offset, int) and self.storage_offset < 0:
            raise ValueError("tensor storage offset must be non-negative")

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorSpec:
        """Read metadata only; never materialize or inspect tensor storage."""

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"expected a torch.Tensor, got {type(tensor).__name__}")
        return cls(
            dtype=str(tensor.dtype),
            device_type=tensor.device.type,
            device_index=tensor.device.index,
            layout=str(tensor.layout),
            shape=tuple(_dimension(dim) for dim in tensor.shape),
            stride=tuple(_dimension(dim) for dim in tensor.stride()),
            storage_offset=_dimension(tensor.storage_offset()),
            requires_grad=tensor.requires_grad,
        )


class InputKind(str, Enum):
    """How a real value will be supplied after compilation."""

    RUNTIME = "runtime"
    WEIGHT = "weight"


ValueSpec: TypeAlias = TensorSpec | ScalarValue


@dataclass(frozen=True, slots=True)
class InputSpec:
    name: str
    value: ValueSpec
    kind: InputKind
    mutable: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("input name must not be empty")
        if self.mutable and not isinstance(self.value, TensorSpec):
            raise ValueError("only tensor inputs can be mutable")


@dataclass(frozen=True, slots=True)
class OutputSpec:
    name: str
    value: ValueSpec
    aliases_input: str | None = None
    consumer_writable: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("output name must not be empty")
        if self.consumer_writable and not isinstance(self.value, TensorSpec):
            raise ValueError("only tensor outputs can be consumer-writable")


@dataclass(frozen=True, slots=True)
class MutationSpec:
    """A caller-visible input mutation required by region semantics."""

    input_name: str

    def __post_init__(self) -> None:
        if not self.input_name:
            raise ValueError("mutation input name must not be empty")


@dataclass(frozen=True, slots=True)
class DynamicRange:
    """Inclusive bounds for one symbolic runtime value."""

    symbol: str
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("dynamic range symbol must not be empty")
        if self.minimum < 0:
            raise ValueError("dynamic range minimum must be non-negative")
        if self.maximum < self.minimum:
            raise ValueError("dynamic range maximum must be at least its minimum")

    @property
    def is_exact(self) -> bool:
        return self.minimum == self.maximum


@dataclass(frozen=True, slots=True)
class RegionSpec:
    """Semantic boundary contract for one compiler-owned graph region."""

    inputs: tuple[InputSpec, ...]
    outputs: tuple[OutputSpec, ...]
    mutations: tuple[MutationSpec, ...] = ()
    dynamic_ranges: tuple[DynamicRange, ...] = ()

    def __post_init__(self) -> None:
        inputs = {item.name: item for item in self.inputs}
        if len(inputs) != len(self.inputs):
            raise ValueError("region input names must be unique")

        output_names = {item.name for item in self.outputs}
        if len(output_names) != len(self.outputs):
            raise ValueError("region output names must be unique")

        range_symbols = {item.symbol for item in self.dynamic_ranges}
        if len(range_symbols) != len(self.dynamic_ranges):
            raise ValueError("dynamic range symbols must be unique")

        mutated_names = {item.input_name for item in self.mutations}
        if len(mutated_names) != len(self.mutations):
            raise ValueError("mutation input names must be unique")
        for name in mutated_names:
            if name not in inputs:
                raise ValueError(f"mutation refers to unknown input {name!r}")
            if not inputs[name].mutable:
                raise ValueError(f"mutation input {name!r} is not declared mutable")

        for output in self.outputs:
            if output.aliases_input is not None and output.aliases_input not in inputs:
                raise ValueError(
                    f"output {output.name!r} aliases unknown input "
                    f"{output.aliases_input!r}"
                )
