import os
import tempfile
from typing import Callable, List

import onnx
import torch
import torch._dynamo

import luminal

from .compiled_model import CompiledModel

_pytree_registered = False


def _get_cache_dict(cache):
    """Convert DynamicCache to a dict for pytree flattening."""
    return {
        "key_cache": [layer.keys for layer in cache.layers if layer.keys is not None],
        "value_cache": [
            layer.values for layer in cache.layers if layer.values is not None
        ],
    }


def flatten_dynamic_cache(cache):
    import torch

    return torch.utils._pytree._dict_flatten(_get_cache_dict(cache))


def unflatten_dynamic_cache(values, context):
    import torch
    from transformers.cache_utils import DynamicCache

    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    cache = DynamicCache()
    key_list = dictionary.get("key_cache", [])
    value_list = dictionary.get("value_cache", [])
    for idx in range(max(len(key_list), len(value_list))):
        key = key_list[idx] if idx < len(key_list) else None
        value = value_list[idx] if idx < len(value_list) else None
        cache.update(key, value, idx)
    return cache


def flatten_with_keys_dynamic_cache(cache):
    import torch

    return torch.utils._pytree._dict_flatten_with_keys(_get_cache_dict(cache))


def _register_cache_serialization(verbose: int = 0):
    # Cache serialization: to be moved into appropriate packages
    import torch

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        DynamicCache = None

    # DynamicCache
    unregistered_dynamic_cache = True
    if DynamicCache is not None and DynamicCache in torch.utils._pytree.SUPPORTED_NODES:
        unregistered_dynamic_cache = False
    else:
        if verbose:
            print("[bypass_export_some_errors] register DynamicCache")
        torch.utils._pytree.register_pytree_node(
            DynamicCache,
            flatten_dynamic_cache,
            unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_dynamic_cache,
        )
        torch.fx._pytree.register_pytree_flatten_spec(
            DynamicCache,
            lambda cache, spec: torch.fx._pytree._dict_flatten_spec(
                _get_cache_dict(cache), spec
            ),
        )

    return dict(DynamicCache=unregistered_dynamic_cache)


def luminal_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    _register_cache_serialization()
    device = example_inputs[0].device if example_inputs else torch.device("cpu")
    backend = "cuda" if device.type == "cuda" else "native"

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp_path = tmp.name
    tmp.close()
    _ = gm.eval()
    try:
        _ = torch.onnx.export(
            gm,
            tuple(example_inputs),
            tmp_path,
            opset_version=20,
            input_names=[f"input_{i}" for i in range(len(example_inputs))],
        )

        result = luminal.process_onnx(tmp_path, backend)
    finally:
        os.unlink(tmp_path)
    compiled = CompiledModel(result)
    return compiled
