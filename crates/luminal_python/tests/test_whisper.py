"""Whisper integration tests for the luminal torch.compile backend.

These tests build a PyTorch port of ``openai/whisper-tiny.en`` (the same one
exercised by ``examples/whisper.py``) and verify that running it through
``torch.compile(..., backend=luminal_backend)`` produces logits that match the
eager-mode PyTorch reference, both with random-init small configs and with the
real pretrained tiny.en weights.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch._dynamo

# Reuse the PyTorch port defined in the example script so we test exactly the
# code that runs the demo.
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))
import whisper as whisper_demo  # noqa: E402  (path-modified import)

from luminal import luminal_backend  # noqa: E402


def _make_small_whisper(seed: int = 0) -> whisper_demo.Whisper:
    torch.manual_seed(seed)
    model = whisper_demo.Whisper().eval()
    return model


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.max(torch.abs(a - b)).item()


def test_whisper_attention_forward(device: torch.device):
    """Whisper self-attention: Q/K/V/out projections + scaled dot-product."""
    torch.manual_seed(0)
    attn = whisper_demo.WhisperAttention().eval().to(device)
    compiled: Callable = torch.compile(attn, backend=luminal_backend)
    x = torch.rand((4, whisper_demo.D_MODEL), device=device)
    with torch.no_grad():
        ref = attn(x)
        out = compiled(x)
    if isinstance(out, tuple):
        out = out[0]
    assert torch.allclose(out, ref, atol=1e-4), f"max_diff={_max_diff(out, ref):.2e}"


def test_whisper_encoder_layer(device: torch.device):
    """Single encoder block: pre-norm self-attention + FFN with GELU.

    Tolerance is loose because luminal uses the tanh GELU approximation rather
    than the exact erf form PyTorch uses for ``aten.gelu.default``.
    """
    torch.manual_seed(0)
    layer = whisper_demo.EncoderLayer().eval().to(device)
    compiled: Callable = torch.compile(layer, backend=luminal_backend)
    x = torch.rand((8, whisper_demo.D_MODEL), device=device)
    with torch.no_grad():
        ref = layer(x)
        out = compiled(x)
    if isinstance(out, tuple):
        out = out[0]
    assert torch.allclose(out, ref, atol=1e-3), f"max_diff={_max_diff(out, ref):.2e}"


def test_whisper_decoder_layer(device: torch.device):
    """Single decoder block: causal self-attention + cross-attention + FFN."""
    torch.manual_seed(0)
    layer = whisper_demo.DecoderLayer().eval().to(device)
    compiled: Callable = torch.compile(layer, backend=luminal_backend)
    x = torch.rand((4, whisper_demo.D_MODEL), device=device)
    xa = torch.rand((16, whisper_demo.D_MODEL), device=device)
    with torch.no_grad():
        ref = layer(x, xa)
        out = compiled(x, xa)
    if isinstance(out, tuple):
        out = out[0]
    assert torch.allclose(out, ref, atol=1e-3), f"max_diff={_max_diff(out, ref):.2e}"


@pytest.mark.slow
def test_whisper_encoder_random_init(device: torch.device):
    """Full encoder over a random mel: 2 conv stems + 4 transformer blocks."""
    model = _make_small_whisper().to(device)
    compiled: Callable = torch.compile(model.encoder, backend=luminal_backend)
    mel = torch.rand((whisper_demo.N_MELS, 3000), device=device)
    with torch.no_grad():
        ref = model.encoder(mel)
        out = compiled(mel)
    if isinstance(out, tuple):
        out = out[0]
    assert torch.allclose(out, ref, atol=1e-3), f"max_diff={_max_diff(out, ref):.2e}"


@pytest.mark.slow
def test_whisper_full_random_init_one_step(device: torch.device):
    """End-to-end Whisper forward (encoder + decoder for one step) with random weights.

    Tolerance is loose because errors accumulate across the conv stems plus the
    8 transformer blocks, and luminal uses the tanh GELU approximation rather
    than the exact erf form that PyTorch ``aten.gelu.default`` evaluates.
    """
    model = _make_small_whisper().to(device)
    compiled: Callable = torch.compile(model, backend=luminal_backend)
    mel = torch.rand((whisper_demo.N_MELS, 3000), device=device)
    tokens = torch.tensor(
        [whisper_demo.TOKEN_SOT, whisper_demo.TOKEN_NO_TIMESTAMPS],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        ref = model(mel, tokens)
        out = compiled(mel, tokens)
    if isinstance(out, tuple):
        out = out[0]
    assert torch.allclose(out, ref, atol=5e-2, rtol=1e-3), (
        f"max_diff={_max_diff(out, ref):.2e}"
    )


@pytest.mark.slow
def test_whisper_tiny_en_pretrained_first_token(device: torch.device):
    """Real whisper-tiny.en weights: first generated token must match reference.

    Uses the bundled JFK sample if available; otherwise a zero-mel placeholder
    (the assertion is purely compiled-vs-reference equality, not transcription
    correctness).
    """
    model = whisper_demo.Whisper().eval()
    whisper_demo.load_hf_weights_into(model)
    model = model.to(device)

    # Try to use the real audio so the comparison is on a realistic mel.
    audio_path = whisper_demo.find_default_audio()
    if audio_path is None:
        mel = torch.zeros((whisper_demo.N_MELS, 3000), device=device)
    else:
        from transformers import WhisperFeatureExtractor

        audio = whisper_demo.load_wav_16k_mono(audio_path)
        fe = WhisperFeatureExtractor.from_pretrained(whisper_demo.REPO_ID)
        mel = (
            fe(audio, sampling_rate=16000, return_tensors="pt")
            .input_features[0]
            .to(device)
        )

    tokens = torch.tensor(
        [whisper_demo.TOKEN_SOT, whisper_demo.TOKEN_NO_TIMESTAMPS],
        dtype=torch.long,
        device=device,
    )

    torch._dynamo.reset()
    compiled: Callable = torch.compile(model, backend=luminal_backend)
    with torch.no_grad():
        ref = model(mel, tokens)
        out = compiled(mel, tokens)
    if isinstance(out, tuple):
        out = out[0]
    # Logits diverge slightly due to the GELU approximation; what matters end
    # to end is that the greedy argmax (with whisper's special-token suppression)
    # picks the same token.
    ref_tok = whisper_demo.greedy_decode(ref[-1], suppress_first_eot=True)
    out_tok = whisper_demo.greedy_decode(out[-1], suppress_first_eot=True)
    assert ref_tok == out_tok, (
        f"first token mismatch: ref={ref_tok}, compiled={out_tok}, "
        f"logits max_diff={_max_diff(out, ref):.2e}"
    )
