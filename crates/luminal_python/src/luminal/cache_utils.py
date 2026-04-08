"""Shared DynamicCache pytree serialization for torch.compile compatibility."""


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
    """Register DynamicCache as a pytree node for torch.compile."""
    import torch

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        DynamicCache = None

    if (
        DynamicCache is not None
        and DynamicCache not in torch.utils._pytree.SUPPORTED_NODES
    ):
        if verbose:
            print("[luminal] register DynamicCache pytree serialization")
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
