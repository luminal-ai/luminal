"""CUDA entry points for shared PT2 region preparation."""

from luminal_reference.region_compile import compile_region as _compile_region
from luminal_reference.region_compile import (
    load_region_artifact as _load_region_artifact,
)


def compile_region(region, **kwargs):
    return _compile_region(region, device_type="cuda", **kwargs)


def load_region_artifact(data, **kwargs):
    return _load_region_artifact(data, device_type="cuda", **kwargs)
