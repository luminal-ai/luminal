"""Qwen-Image diffusion model integration tests.

Tests the QwenImageTransformer2DModel (MMDiT denoiser) and AutoencoderKLQwenImage (VAE)
through the PyTorch -> ONNX -> luminal pipeline.

The transformer uses complex-valued RoPE (torch.view_as_complex) which isn't ONNX-exportable,
so tests use a wrapper that pre-computes RoPE as real-valued cos/sin and replaces the
attention processor with a real-valued equivalent.

The VAE uses Conv3d, which is supported via the N-dimensional unfold-based conv parser.
"""

import os
import tempfile
import warnings

import onnx
import onnxsim
import pytest
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ============================================================================
# Transformer helpers
# ============================================================================


def _apply_rope_real(x, cos, sin):
    """Apply RoPE using real-valued cos/sin. x: [B, S, H, D], cos/sin: [S, D/2]."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, D/2]
    sin = sin.unsqueeze(0).unsqueeze(2)
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x2 * cos + x1 * sin
    return torch.cat([rotated_x1, rotated_x2], dim=-1)


class RealRoPEAttnProcessor:
    """Attention processor that uses real-valued RoPE for ONNX compatibility.

    Replaces the default QwenDoubleStreamAttnProcessor2_0 which uses
    torch.view_as_complex (not ONNX-exportable).
    """

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_cos, img_sin, txt_cos, txt_sin = image_rotary_emb
            img_query = _apply_rope_real(img_query, img_cos, img_sin)
            img_key = _apply_rope_real(img_key, img_cos, img_sin)
            txt_query = _apply_rope_real(txt_query, txt_cos, txt_sin)
            txt_key = _apply_rope_real(txt_key, txt_cos, txt_sin)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_query = joint_query.transpose(1, 2)
        joint_key = joint_key.transpose(1, 2)
        joint_value = joint_value.transpose(1, 2)
        joint_hidden = torch.nn.functional.scaled_dot_product_attention(
            joint_query, joint_key, joint_value, dropout_p=0.0, is_causal=False
        )
        joint_hidden = joint_hidden.transpose(1, 2)
        joint_hidden = joint_hidden.flatten(2, 3)

        txt_attn = joint_hidden[:, :seq_txt, :]
        img_attn = joint_hidden[:, seq_txt:, :]

        img_attn = attn.to_out[0](img_attn.contiguous())
        if len(attn.to_out) > 1:
            img_attn = attn.to_out[1](img_attn)
        txt_attn = attn.to_add_out(txt_attn.contiguous())

        return img_attn, txt_attn


class TransformerONNXWrapper(nn.Module):
    """Wraps QwenImageTransformer2DModel for ONNX export.

    Pre-computes complex RoPE frequencies as real cos/sin buffers and replaces
    the attention processors with ONNX-friendly real-valued versions.
    """

    def __init__(self, model, img_shapes, txt_seq_len):
        super().__init__()
        self.model = model

        for block in self.model.transformer_blocks:
            block.attn.set_processor(RealRoPEAttnProcessor())

        with torch.no_grad():
            img_freqs, txt_freqs = model.pos_embed(
                img_shapes, max_txt_seq_len=txt_seq_len
            )
        self.register_buffer("img_cos", img_freqs.real.float().contiguous())
        self.register_buffer("img_sin", img_freqs.imag.float().contiguous())
        self.register_buffer("txt_cos", txt_freqs.real.float().contiguous())
        self.register_buffer("txt_sin", txt_freqs.imag.float().contiguous())

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        hidden_states = self.model.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        encoder_hidden_states = self.model.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.model.txt_in(encoder_hidden_states)

        temb = self.model.time_text_embed(timestep, hidden_states)

        rope = (self.img_cos, self.img_sin, self.txt_cos, self.txt_sin)

        for block in self.model.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None,
                temb=temb,
                image_rotary_emb=rope,
            )

        hidden_states = self.model.norm_out(hidden_states, temb)
        output = self.model.proj_out(hidden_states)
        return output


def _make_tiny_transformer_config():
    """Tiny transformer config: ~100K params, 1 layer."""
    return dict(
        patch_size=2,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=4,
        joint_attention_dim=64,
        axes_dims_rope=(4, 6, 6),
    )


def _make_small_transformer_config():
    """Small transformer config: ~1M params, 2 layers."""
    return dict(
        patch_size=2,
        in_channels=16,
        out_channels=16,
        num_layers=2,
        attention_head_dim=32,
        num_attention_heads=8,
        joint_attention_dim=256,
        axes_dims_rope=(8, 12, 12),
    )


def _make_medium_transformer_config():
    """Medium transformer config: ~39M params, 4 layers."""
    return dict(
        patch_size=2,
        in_channels=32,
        out_channels=32,
        num_layers=4,
        attention_head_dim=64,
        num_attention_heads=8,
        joint_attention_dim=512,
        axes_dims_rope=(8, 28, 28),
    )


def _run_transformer_test(config, atol):
    """Compile transformer with luminal backend, compare to PyTorch reference."""
    from diffusers.models import QwenImageTransformer2DModel

    from luminal import luminal_backend

    model = QwenImageTransformer2DModel(**config).eval()
    img_seq_len = 4
    txt_seq_len = 3

    wrapper = TransformerONNXWrapper(model, [(1, 2, 2)], txt_seq_len).eval()
    wrapper_compiled = torch.compile(wrapper, backend=luminal_backend)

    hidden = torch.randn(1, img_seq_len, config["in_channels"])
    encoder_hs = torch.randn(1, txt_seq_len, config["joint_attention_dim"])
    timestep = torch.tensor([1.0])

    with torch.no_grad():
        ref = wrapper(hidden, encoder_hs, timestep)
        out = wrapper_compiled(hidden, encoder_hs, timestep)

    assert torch.allclose(out, ref, atol=atol), (
        f"max_diff={torch.max(torch.abs(out - ref)).item():.2e}"
    )


# ============================================================================
# VAE helpers
# ============================================================================


class _OnnxFriendlyUpsample(nn.Module):
    """Replaces nn.Upsample with repeat_interleave for ONNX compatibility."""

    def __init__(self, scale_factor):
        super().__init__()
        if isinstance(scale_factor, (tuple, list)):
            self.scale_factors = [int(s) for s in scale_factor]
        else:
            sf = int(scale_factor)
            self.scale_factors = [sf]

    def forward(self, x):
        for dim_offset, sf in enumerate(self.scale_factors):
            if sf > 1:
                x = x.repeat_interleave(sf, dim=2 + dim_offset)
        return x


def _make_tiny_vae_config():
    """Tiny VAE config for testing."""
    return dict(
        base_dim=8,
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False],
        dropout=0.0,
        input_channels=3,
    )


def _make_medium_vae_config():
    """Medium VAE config: base_dim=32, z_dim=8."""
    return dict(
        base_dim=32,
        z_dim=8,
        dim_mult=[1, 2, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True],
        dropout=0.0,
        input_channels=3,
    )


def _prepare_vae_for_onnx(vae):
    """Replace non-ONNX-exportable modules in the VAE."""
    import diffusers.models.autoencoders.autoencoder_kl_qwenimage as vae_mod

    def _replace(module):
        for name, child in module.named_children():
            if isinstance(child, vae_mod.QwenImageUpsample):
                setattr(module, name, _OnnxFriendlyUpsample(child.scale_factor))
            else:
                _replace(child)

    _replace(vae)
    return vae


class _VAEDecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z).sample


def _export_and_simplify(wrapper, inputs, input_names, output_names):
    """Export model to ONNX and simplify with onnxsim."""
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        torch.onnx.export(
            wrapper,
            inputs,
            tmp_path,
            opset_version=20,
            input_names=input_names,
            output_names=output_names,
            dynamo=False,
        )
        m = onnx.load(tmp_path)
        m_sim, check = onnxsim.simplify(m)
        assert check, "onnxsim simplification failed"
        onnx.save(m_sim, tmp_path)
        return tmp_path
    except Exception:
        os.unlink(tmp_path)
        raise


def _run_vae_test(config, atol):
    """Export VAE decoder to ONNX, run through luminal, compare."""
    from diffusers import AutoencoderKLQwenImage

    import luminal

    backend = os.environ.get("LUMINAL_BACKEND", "native")
    vae = AutoencoderKLQwenImage(**config).eval()
    vae = _prepare_vae_for_onnx(vae)

    wrapper = _VAEDecoderWrapper(vae).eval()
    latents = torch.randn(1, config["z_dim"], 1, 4, 4)

    with torch.no_grad():
        ref = wrapper(latents)

    onnx_path = _export_and_simplify(wrapper, (latents,), ["latents"], ["output"])
    try:
        graph = luminal.process_onnx(onnx_path, backend)
        graph.set_input("latents", latents.flatten().tolist())
        graph.run()
        out_data = graph.get_output("output")
        out = torch.tensor(out_data, dtype=torch.float32).reshape(ref.shape)
    finally:
        os.unlink(onnx_path)

    assert torch.allclose(out, ref, atol=atol), (
        f"max_diff={torch.max(torch.abs(out - ref)).item():.2e}"
    )


# ============================================================================
# Tests
# ============================================================================


def test_qwen_image_transformer_tiny():
    """Tiny QwenImage transformer: 1 layer, 4 heads, dim=64."""
    _run_transformer_test(_make_tiny_transformer_config(), atol=1e-4)


def test_qwen_image_transformer_small():
    """Small QwenImage transformer: 2 layers, 8 heads, dim=256."""
    _run_transformer_test(_make_small_transformer_config(), atol=1e-4)


def test_qwen_image_transformer_medium():
    """Medium QwenImage transformer: 4 layers, 8 heads, dim=512."""
    _run_transformer_test(_make_medium_transformer_config(), atol=1e-4)


def test_qwen_image_transformer_full():
    """Full QwenImage transformer (production defaults)."""
    from diffusers.models import QwenImageTransformer2DModel

    from luminal import luminal_backend

    model = QwenImageTransformer2DModel().eval()
    config = {k: v for k, v in dict(model.config).items() if not k.startswith("_")}

    wrapper = TransformerONNXWrapper(model, [(1, 2, 2)], txt_seq_len=3).eval()
    wrapper_compiled = torch.compile(wrapper, backend=luminal_backend)

    hidden = torch.randn(1, 4, config["in_channels"])
    encoder_hs = torch.randn(1, 3, config["joint_attention_dim"])
    timestep = torch.tensor([1.0])

    with torch.no_grad():
        ref = wrapper(hidden, encoder_hs, timestep)
        out = wrapper_compiled(hidden, encoder_hs, timestep)

    assert torch.allclose(out, ref, atol=1e-4), (
        f"max_diff={torch.max(torch.abs(out - ref)).item():.2e}"
    )


def test_qwen_image_vae_decoder_tiny():
    """Tiny QwenImage VAE decoder: base_dim=8, z_dim=4."""
    _run_vae_test(_make_tiny_vae_config(), atol=1e-3)


def test_qwen_image_vae_decoder_medium():
    """Medium QwenImage VAE decoder: base_dim=32, z_dim=8."""
    _run_vae_test(_make_medium_vae_config(), atol=1e-3)


@pytest.mark.skip(reason="Full production VAE -- expected to be slow/OOM")
def test_qwen_image_vae_decoder_full():
    """Full QwenImage VAE decoder (production defaults)."""
    from diffusers import AutoencoderKLQwenImage

    config = dict(AutoencoderKLQwenImage().config)
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    _run_vae_test(config, atol=1e-3)
