"""Whisper transcription using the Luminal reference backend.

Implements a small PyTorch port of ``openai/whisper-tiny.en`` that mirrors the
logical Whisper model, loads the official HuggingFace weights, and runs greedy
decoding through ``torch.compile`` with ``luminal_reference.Compiler``.

Usage::

    cd crates/pytorch/reference
    uv run --group dev python examples/whisper.py [path/to/audio.wav]

If no path is provided, downloads the JFK sample from whisper.cpp on first
use and caches it in the gitignored examples/assets directory.
"""

from __future__ import annotations

import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

REPO_ID = "openai/whisper-tiny.en"

# whisper-tiny.en hyperparameters
N_MELS = 80
N_AUDIO_CTX = 1500
D_MODEL = 384
N_HEADS = 6
HEAD_DIM = D_MODEL // N_HEADS
N_AUDIO_LAYER = 4
N_TEXT_LAYER = 4
N_TEXT_CTX = 448
FF_DIM = 4 * D_MODEL
N_VOCAB = 51864
LAYER_NORM_EPS = 1e-5

# Decoder special tokens
TOKEN_SOT = 50257
TOKEN_NO_TIMESTAMPS = 50362
TOKEN_EOT = 50256


# ---------------------------------------------------------------------------
# Model — mirrors the HLIR encoder/decoder in examples/whisper/src/model.rs
# ---------------------------------------------------------------------------


class WhisperAttention(torch.nn.Module):
    """Multi-head attention with separate q/k/v projections (no bias on k_proj)."""

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=True)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=True)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        kv_input: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        # x: (seq, d_model). kv_input is None → self-attn; otherwise cross-attn.
        kv = x if kv_input is None else kv_input
        q = self.q_proj(x)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        seq_q = q.shape[0]
        seq_kv = k.shape[0]

        # (seq, d_model) -> (n_heads, seq, head_dim)
        q = q.reshape(seq_q, self.n_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(seq_kv, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(seq_kv, self.n_heads, self.head_dim).transpose(0, 1)

        scale = 1.0 / (self.head_dim**0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (h, sq, sk)
        if causal:
            # Use a large finite negative instead of -inf so the export pipeline
            # serializes a float instead of the unsupported "-Infinity" sentinel.
            mask = torch.triu(
                torch.full((seq_q, seq_kv), -1e10, device=x.device),
                diagonal=1,
            )
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)  # (h, sq, hd)
        merged = attn.transpose(0, 1).reshape(seq_q, -1)
        return self.out_proj(merged)


class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = WhisperAttention()
        self.self_attn_layer_norm = torch.nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)
        self.fc1 = torch.nn.Linear(D_MODEL, FF_DIM, bias=True)
        self.fc2 = torch.nn.Linear(FF_DIM, D_MODEL, bias=True)
        self.final_layer_norm = torch.nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        h = self.final_layer_norm(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class WhisperEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            N_MELS, D_MODEL, kernel_size=3, padding=1, bias=True
        )
        self.conv2 = torch.nn.Conv1d(
            D_MODEL, D_MODEL, kernel_size=3, stride=2, padding=1, bias=True
        )
        # Position embedding stored as a regular parameter (matches HF layout).
        self.embed_positions = torch.nn.Embedding(N_AUDIO_CTX, D_MODEL)
        self.layers = torch.nn.ModuleList(
            [EncoderLayer() for _ in range(N_AUDIO_LAYER)]
        )
        self.layer_norm = torch.nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (n_mels, 3000) -> add batch dim for conv1d
        x = mel.unsqueeze(0)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        # (1, d_model, 1500) -> (1500, d_model)
        x = x.squeeze(0).transpose(0, 1)
        x = x + self.embed_positions.weight
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = WhisperAttention()
        self.self_attn_layer_norm = torch.nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)
        self.encoder_attn = WhisperAttention()
        self.encoder_attn_layer_norm = torch.nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)
        self.fc1 = torch.nn.Linear(D_MODEL, FF_DIM, bias=True)
        self.fc2 = torch.nn.Linear(FF_DIM, D_MODEL, bias=True)
        self.final_layer_norm = torch.nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)

    def forward(self, x: torch.Tensor, xa: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x), causal=True)
        x = x + self.encoder_attn(self.encoder_attn_layer_norm(x), kv_input=xa)
        h = self.final_layer_norm(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class WhisperDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(N_VOCAB, D_MODEL)
        self.embed_positions = torch.nn.Embedding(N_TEXT_CTX, D_MODEL)
        self.layers = torch.nn.ModuleList([DecoderLayer() for _ in range(N_TEXT_LAYER)])
        self.layer_norm = torch.nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)

    def forward(self, tokens: torch.Tensor, xa: torch.Tensor) -> torch.Tensor:
        # tokens: (seq,) of int64 — absolute positions are 0..seq-1
        seq = tokens.shape[0]
        pos = torch.arange(seq, dtype=torch.long, device=tokens.device)
        x = self.embed_tokens(tokens) + self.embed_positions(pos)
        for layer in self.layers:
            x = layer(x, xa)
        x = self.layer_norm(x)
        # Tied projection
        return torch.matmul(x, self.embed_tokens.weight.transpose(0, 1))


class Whisper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = WhisperEncoder()
        self.decoder = WhisperDecoder()

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        xa = self.encoder(mel)
        return self.decoder(tokens, xa)


# ---------------------------------------------------------------------------
# Weight loading: HF state_dict -> our model
# ---------------------------------------------------------------------------


def load_hf_weights_into(model: Whisper) -> None:
    """Copy HF whisper-tiny.en weights into our matching modules."""
    hf = WhisperForConditionalGeneration.from_pretrained(REPO_ID).eval()
    sd = hf.state_dict()

    def get(name: str) -> torch.Tensor:
        return sd[f"model.{name}"].clone()

    enc = model.encoder
    enc.conv1.weight.data.copy_(get("encoder.conv1.weight"))
    enc.conv1.bias.data.copy_(get("encoder.conv1.bias"))
    enc.conv2.weight.data.copy_(get("encoder.conv2.weight"))
    enc.conv2.bias.data.copy_(get("encoder.conv2.bias"))
    enc.embed_positions.weight.data.copy_(get("encoder.embed_positions.weight"))
    enc.layer_norm.weight.data.copy_(get("encoder.layer_norm.weight"))
    enc.layer_norm.bias.data.copy_(get("encoder.layer_norm.bias"))
    for i, layer in enumerate(enc.layers):
        prefix = f"encoder.layers.{i}"
        layer.self_attn.q_proj.weight.data.copy_(
            get(f"{prefix}.self_attn.q_proj.weight")
        )
        layer.self_attn.q_proj.bias.data.copy_(get(f"{prefix}.self_attn.q_proj.bias"))
        layer.self_attn.k_proj.weight.data.copy_(
            get(f"{prefix}.self_attn.k_proj.weight")
        )
        layer.self_attn.v_proj.weight.data.copy_(
            get(f"{prefix}.self_attn.v_proj.weight")
        )
        layer.self_attn.v_proj.bias.data.copy_(get(f"{prefix}.self_attn.v_proj.bias"))
        layer.self_attn.out_proj.weight.data.copy_(
            get(f"{prefix}.self_attn.out_proj.weight")
        )
        layer.self_attn.out_proj.bias.data.copy_(
            get(f"{prefix}.self_attn.out_proj.bias")
        )
        layer.self_attn_layer_norm.weight.data.copy_(
            get(f"{prefix}.self_attn_layer_norm.weight")
        )
        layer.self_attn_layer_norm.bias.data.copy_(
            get(f"{prefix}.self_attn_layer_norm.bias")
        )
        layer.fc1.weight.data.copy_(get(f"{prefix}.fc1.weight"))
        layer.fc1.bias.data.copy_(get(f"{prefix}.fc1.bias"))
        layer.fc2.weight.data.copy_(get(f"{prefix}.fc2.weight"))
        layer.fc2.bias.data.copy_(get(f"{prefix}.fc2.bias"))
        layer.final_layer_norm.weight.data.copy_(
            get(f"{prefix}.final_layer_norm.weight")
        )
        layer.final_layer_norm.bias.data.copy_(get(f"{prefix}.final_layer_norm.bias"))

    dec = model.decoder
    dec.embed_tokens.weight.data.copy_(get("decoder.embed_tokens.weight"))
    dec.embed_positions.weight.data.copy_(get("decoder.embed_positions.weight"))
    dec.layer_norm.weight.data.copy_(get("decoder.layer_norm.weight"))
    dec.layer_norm.bias.data.copy_(get("decoder.layer_norm.bias"))
    for i, layer in enumerate(dec.layers):
        prefix = f"decoder.layers.{i}"
        layer.self_attn.q_proj.weight.data.copy_(
            get(f"{prefix}.self_attn.q_proj.weight")
        )
        layer.self_attn.q_proj.bias.data.copy_(get(f"{prefix}.self_attn.q_proj.bias"))
        layer.self_attn.k_proj.weight.data.copy_(
            get(f"{prefix}.self_attn.k_proj.weight")
        )
        layer.self_attn.v_proj.weight.data.copy_(
            get(f"{prefix}.self_attn.v_proj.weight")
        )
        layer.self_attn.v_proj.bias.data.copy_(get(f"{prefix}.self_attn.v_proj.bias"))
        layer.self_attn.out_proj.weight.data.copy_(
            get(f"{prefix}.self_attn.out_proj.weight")
        )
        layer.self_attn.out_proj.bias.data.copy_(
            get(f"{prefix}.self_attn.out_proj.bias")
        )
        layer.self_attn_layer_norm.weight.data.copy_(
            get(f"{prefix}.self_attn_layer_norm.weight")
        )
        layer.self_attn_layer_norm.bias.data.copy_(
            get(f"{prefix}.self_attn_layer_norm.bias")
        )
        layer.encoder_attn.q_proj.weight.data.copy_(
            get(f"{prefix}.encoder_attn.q_proj.weight")
        )
        layer.encoder_attn.q_proj.bias.data.copy_(
            get(f"{prefix}.encoder_attn.q_proj.bias")
        )
        layer.encoder_attn.k_proj.weight.data.copy_(
            get(f"{prefix}.encoder_attn.k_proj.weight")
        )
        layer.encoder_attn.v_proj.weight.data.copy_(
            get(f"{prefix}.encoder_attn.v_proj.weight")
        )
        layer.encoder_attn.v_proj.bias.data.copy_(
            get(f"{prefix}.encoder_attn.v_proj.bias")
        )
        layer.encoder_attn.out_proj.weight.data.copy_(
            get(f"{prefix}.encoder_attn.out_proj.weight")
        )
        layer.encoder_attn.out_proj.bias.data.copy_(
            get(f"{prefix}.encoder_attn.out_proj.bias")
        )
        layer.encoder_attn_layer_norm.weight.data.copy_(
            get(f"{prefix}.encoder_attn_layer_norm.weight")
        )
        layer.encoder_attn_layer_norm.bias.data.copy_(
            get(f"{prefix}.encoder_attn_layer_norm.bias")
        )
        layer.fc1.weight.data.copy_(get(f"{prefix}.fc1.weight"))
        layer.fc1.bias.data.copy_(get(f"{prefix}.fc1.bias"))
        layer.fc2.weight.data.copy_(get(f"{prefix}.fc2.weight"))
        layer.fc2.bias.data.copy_(get(f"{prefix}.fc2.bias"))
        layer.final_layer_norm.weight.data.copy_(
            get(f"{prefix}.final_layer_norm.weight")
        )
        layer.final_layer_norm.bias.data.copy_(get(f"{prefix}.final_layer_norm.bias"))


# ---------------------------------------------------------------------------
# Audio loading + decoding
# ---------------------------------------------------------------------------


def load_wav_16k_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)

    if sw == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        samples = (
            np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        ) / 128.0
    else:
        raise ValueError(f"unsupported sample width {sw}")

    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1)

    if sr != 16000:
        ratio = sr / 16000
        out_len = int(len(samples) / ratio)
        idx = np.arange(out_len, dtype=np.float64) * ratio
        lo = idx.astype(np.int64)
        frac = (idx - lo).astype(np.float32)
        hi = np.clip(lo + 1, 0, len(samples) - 1)
        samples = samples[lo] * (1.0 - frac) + samples[hi] * frac

    return samples.astype(np.float32)


def greedy_decode(logits_row: torch.Tensor, suppress_first_eot: bool) -> int:
    masked = logits_row.clone()
    masked[TOKEN_SOT:] = float("-inf")
    if suppress_first_eot:
        masked[TOKEN_EOT] = float("-inf")
    return int(torch.argmax(masked).item())


def download_default_audio() -> Path:
    """Download the JFK sample once, then reuse the ignored local WAV."""
    destination = Path(__file__).resolve().parent / "assets" / "jfk.wav"
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav"
    )
    temporary = None
    try:
        print("Downloading sample audio...", flush=True)
        with (
            urllib.request.urlopen(url, timeout=30) as response,
            tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix=".tmp", delete=False
            ) as output,
        ):
            temporary = Path(output.name)
            while chunk := response.read(65536):
                output.write(chunk)
        load_wav_16k_mono(temporary)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    from luminal_reference import Compiler, DimBucket
    from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec

    device = torch.device("cpu")
    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else download_default_audio()

    print("Loading audio:", audio_path)
    audio = load_wav_16k_mono(audio_path)

    print("Computing log-mel features...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(REPO_ID)
    features = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    mel: torch.Tensor = features.input_features[0].to(device)  # (80, 3000)
    assert mel.shape == (N_MELS, 3000), mel.shape

    print("Building model and loading weights...")
    model = Whisper().eval().to(device)
    load_hf_weights_into(model)
    model = model.to(device)
    tokenizer = WhisperTokenizer.from_pretrained(REPO_ID)

    max_new_tokens = 100

    # 1. Run the encoder once eagerly. The audio doesn't change during decode,
    #    so xa is a constant input to the decoder.
    with torch.no_grad():
        xa = model.encoder(mel)

    # 2. Compile the decoder with a dynamic token length. The encoder output
    #    stays fixed for this utterance but is an ordinary decoder input.
    example_tokens = torch.tensor(
        [TOKEN_SOT, TOKEN_NO_TIMESTAMPS], dtype=torch.long, device=device
    )
    print("Compiling decoder...")
    compile_start = time.time()
    seq = ShapeVar("seq", min=2, max=N_TEXT_CTX, optimization_hint=2)
    compiled_decoder = torch.compile(
        model.decoder,
        backend=Compiler(
            search_iterations=10,
            log=True,
            dim_buckets={
                seq: [
                    DimBucket(min=2, max=32, representative=2),
                    DimBucket(min=33, max=N_TEXT_CTX, representative=64),
                ]
            },
        ),
        fullgraph=True,
        dynamic_shapes=ShapesSpec(
            params={
                "tokens": TensorSpec([seq]),
                "xa": TensorSpec([N_AUDIO_CTX, D_MODEL]),
            }
        ),
    )
    # torch.compile is lazy. Reuse the first call's logits for the first token.
    with torch.no_grad():
        first_logits = compiled_decoder(example_tokens, xa)
    print(f"Compiled in {time.time() - compile_start:.1f}s")

    tokens = [TOKEN_SOT, TOKEN_NO_TIMESTAMPS]

    print("Transcribing", end="", flush=True)
    decode_start = time.time()
    for step in range(max_new_tokens):
        decoder_input_ids = torch.tensor(tokens, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = (
                first_logits if step == 0 else compiled_decoder(decoder_input_ids, xa)
            )

        next_token = greedy_decode(logits[-1], suppress_first_eot=(step == 0))
        if next_token == TOKEN_EOT:
            break
        tokens.append(next_token)
        piece = tokenizer.decode([next_token], skip_special_tokens=False)
        print(piece, end="", flush=True)
    elapsed = time.time() - decode_start
    print()

    transcription = tokenizer.decode(tokens[2:], skip_special_tokens=True)
    print(f"\nFinal transcription: {transcription}")
    print(
        f"Generated {len(tokens) - 2} tokens in {elapsed:.2f}s "
        f"({(len(tokens) - 2) / max(elapsed, 1e-6):.1f} tok/s)"
    )


if __name__ == "__main__":
    main()
