"""In-place input mutations (HF StaticCache) through the compiled path.

transformers' StaticLayer.update mutates the cache tensors (and its
cumulative_length counter) in place; torch.export functionalizes those
mutations into extra outputs declared by the graph signature. CompiledModel
applies them back to the caller's tensors and returns only the user outputs,
matching eager semantics and torch.compile's calling contract.
"""

import torch
from luminal import luminal_backend

TINY_LLAMA_KWARGS = dict(
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=128,
    vocab_size=256,
    max_position_embeddings=64,
    use_cache=True,
)


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(**TINY_LLAMA_KWARGS)
    return config, LlamaForCausalLM(config).eval()


def make_static_cache(config, max_cache_len, device):
    from transformers.cache_utils import StaticCache

    cache = StaticCache(config=config, max_cache_len=max_cache_len)
    cache.early_initialization(
        batch_size=1,
        num_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=torch.float32,
        device=device,
    )
    return cache


def static_cache_forward(model, cache, input_ids):
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
    return model(
        input_ids,
        past_key_values=cache,
        cache_position=cache_position,
        position_ids=cache_position.unsqueeze(0),
        logits_to_keep=1,
        use_cache=True,
        return_dict=True,
    )


def test_static_cache_prefill_smoke(device):
    """One StaticCache forward through the compiled path matches eager —
    logits, cache contents, and the in-place mutation semantics."""
    config, model = tiny_llama()
    model = model.to(device)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)

    ref_cache = make_static_cache(config, 16, device)
    lum_cache = make_static_cache(config, 16, device)
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)

    with torch.inference_mode():
        ref = static_cache_forward(model, ref_cache, input_ids)
        lum = static_cache_forward(compiled, lum_cache, input_ids)
    assert torch.allclose(lum.logits, ref.logits, atol=1e-4)
    for ref_layer, lum_layer in zip(ref_cache.layers, lum_cache.layers):
        assert torch.allclose(lum_layer.keys, ref_layer.keys, atol=1e-4)
        assert torch.allclose(lum_layer.values, ref_layer.values, atol=1e-4)


def test_writeback_metadata_exposed(device):
    """The compiled artifact names its write-back outputs and their inputs."""
    config, model = tiny_llama()
    model = model.to(device)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)

    captured = []

    def capturing_backend(gm, example_inputs, options=None):
        compiled = luminal_backend(gm, example_inputs)
        captured.append(compiled)
        return compiled

    cache = make_static_cache(config, 16, device)
    compiled_model = torch.compile(model, backend=capturing_backend, fullgraph=True)
    with torch.inference_mode():
        static_cache_forward(compiled_model, cache, input_ids)

    (compiled,) = captured
    writebacks = compiled.writeback_inputs
    # 2 layers x (keys, values) index_put mutations + 2 cumulative_length
    # counters = 6 write-back outputs, each bound to a distinct input.
    assert len(writebacks) == 2 * config.num_hidden_layers + config.num_hidden_layers
    assert len(set(writebacks.values())) == len(writebacks)
