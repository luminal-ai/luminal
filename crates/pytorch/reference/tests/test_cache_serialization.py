"""Ordinary DynamicCache layer positions survive pytree and PT2 export."""

import torch
import pytest
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)

from luminal_reference.export_utils import _register_cache_serialization


def _config(num_layers=2):
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=num_layers,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    return config


def _cache_with_layers(populated, num_layers=3):
    cache = DynamicCache(config=_config(num_layers)) if num_layers else DynamicCache()
    for index in populated:
        key = torch.full((1, 2, 1, 8), index + 1.0)
        cache.update(key, key + 1, index)
    return cache


def _assert_same_layers(actual, expected):
    assert len(actual.layers) == len(expected.layers)
    for actual_layer, expected_layer in zip(actual.layers, expected.layers):
        if expected_layer.keys is None:
            assert actual_layer.keys is None
            assert actual_layer.values is None
        else:
            torch.testing.assert_close(actual_layer.keys, expected_layer.keys)
            torch.testing.assert_close(actual_layer.values, expected_layer.values)


@pytest.mark.parametrize(
    ("num_layers", "populated"),
    [
        (0, ()),
        (3, ()),
        (3, (2,)),
        (3, (0, 2)),
        (3, (0,)),
        (3, (0, 1, 2)),
    ],
)
def test_dynamic_cache_pytree_preserves_layer_positions(num_layers, populated):
    _register_cache_serialization()
    original = _cache_with_layers(populated, num_layers)
    leaves, spec = torch.utils._pytree.tree_flatten(original)
    restored = torch.utils._pytree.tree_unflatten(leaves, spec)
    _assert_same_layers(restored, original)

    keyed_leaves, keyed_spec = torch.utils._pytree.tree_flatten_with_path(original)
    assert keyed_spec == spec
    assert all(value is leaf for (_, value), leaf in zip(keyed_leaves, leaves))
    fx_leaves = torch.fx._pytree.tree_flatten_spec(original, spec)
    assert len(fx_leaves) == len(leaves)
    assert all(value is leaf for value, leaf in zip(fx_leaves, leaves))
    assert [(path[0].key, path[1].idx) for path, _ in keyed_leaves] == [
        (kind, index)
        for kind in ("key_cache", "value_cache")
        for index in range(num_layers)
    ]


class _SecondLlamaLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = LlamaDecoderLayer(config, layer_idx=1)
        self.rope = LlamaRotaryEmbedding(config)

    def forward(self, hidden, positions, cache):
        embeddings = self.rope(hidden, positions)
        output = self.layer(
            hidden,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=embeddings,
        )
        return output, cache


def test_real_llama_decoder_layer_export_preserves_cache_position(tmp_path):
    _register_cache_serialization()
    torch.manual_seed(7)
    config = _config()
    model = _SecondLlamaLayer(config).eval()
    prefill = torch.randn(1, 3, config.hidden_size)
    prefill_positions = torch.arange(3).unsqueeze(0)
    decode = torch.randn(1, 1, config.hidden_size)
    decode_position = torch.tensor([[3]])
    eager_cache = DynamicCache(config=config)
    export_cache = DynamicCache(config=config)

    with torch.no_grad():
        eager_prefill, eager_cache = model(prefill, prefill_positions, eager_cache)
        exported = torch.export.export(
            model, (prefill, prefill_positions, export_cache), strict=False
        )
        path = tmp_path / "llama-layer.pt2"
        torch.export.save(exported, path)
        with torch.serialization.safe_globals([DynamicCache, DynamicLayer]):
            loaded = torch.export.load(path)
        output, returned_cache = loaded.module()(
            prefill, prefill_positions, export_cache
        )
        eager_decode, eager_cache = model(decode, decode_position, eager_cache)
        decode_exported = torch.export.export(
            model, (decode, decode_position, returned_cache), strict=False
        )
        decode_path = tmp_path / "llama-decode.pt2"
        torch.export.save(decode_exported, decode_path)
        with torch.serialization.safe_globals([DynamicCache, DynamicLayer]):
            loaded_decode = torch.export.load(decode_path)
        resumed_decode, returned_cache = loaded_decode.module()(
            decode, decode_position, returned_cache
        )

    torch.testing.assert_close(output, eager_prefill)
    torch.testing.assert_close(resumed_decode, eager_decode)
    _assert_same_layers(returned_cache, eager_cache)
