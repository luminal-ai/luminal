"""End-to-end tiny decoder LLM on the reference backend.

A two-block GPT-style transformer is compiled with
``torch.compile(backend=luminal_reference.Compiler())`` and driven through a real prefill
+ greedy-decode loop. The generated token ids and every step's logits must
match Torch eager exactly (within float tolerance). This is the "good PyTorch
citizen" check: the backend has to honour the ``torch.compile`` contract for a
whole generation loop, not just a single forward pass.

The loop keeps a fixed ``[1, MAX_LEN]`` token buffer and a fixed additive
causal mask, so the graph has one static shape and therefore one compilation.
Positions past the current length hold ``PAD_ID``; the causal mask makes them
invisible to the logits we read, so the padded tail never affects the result.
"""

import math

import luminal_reference
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 64
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
MAX_LEN = 16
PROMPT = (3, 1, 4, 1)
NUM_TOKENS = 4  # 1 prefill forward + 3 decode forwards
PAD_ID = 0
ATOL = 1e-5


class CausalSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head_dim = D_MODEL // N_HEADS
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, N_HEADS, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, N_HEADS, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, N_HEADS, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        probs = torch.softmax(scores + causal_mask, dim=-1)

        attended = (probs @ v).transpose(1, 2).reshape(batch, length, D_MODEL)
        return self.out_proj(attended)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.proj = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class LayerNorm(nn.Module):
    """Layer norm with the statistics spelled out.

    ``F.layer_norm`` / ``nn.LayerNorm`` lift ``eps`` to a scalar-tensor
    ``item()`` under dynamic shapes, and ``torch.export`` cannot guard on that
    data-dependent scalar. Computing mean/variance directly keeps the same
    ``aten`` surface with a literal eps.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D_MODEL))
        self.bias = nn.Parameter(torch.zeros(D_MODEL))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        normed = centered * torch.rsqrt(var + 1e-5)
        return normed * self.weight + self.bias


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln_attn = LayerNorm()
        self.attn = CausalSelfAttention()
        self.ln_mlp = LayerNorm()
        self.mlp = MLP()

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_attn(x), causal_mask)
        return x + self.mlp(self.ln_mlp(x))


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.norm = LayerNorm()
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, tokens: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.head(self.norm(x))


def _causal_mask(length: int) -> torch.Tensor:
    """Additive ``[length, length]`` mask: 0 on/below the diagonal, -inf above."""
    mask = torch.zeros(length, length)
    above_diagonal = torch.triu(
        torch.ones(length, length, dtype=torch.bool), diagonal=1
    )
    return mask.masked_fill(above_diagonal, float("-inf"))


@torch.no_grad()
def _generate(
    model: nn.Module, prompt: tuple[int, ...], mask: torch.Tensor
) -> tuple[list[int], torch.Tensor]:
    """Greedy prefill + decode on a fixed-length buffer.

    The first forward is the prefill (the whole prompt at once); each later
    forward is one decode step with the previously chosen token appended.
    """
    tokens = torch.full((1, MAX_LEN), PAD_ID, dtype=torch.long)
    tokens[0, : len(prompt)] = torch.tensor(prompt, dtype=torch.long)

    generated: list[int] = []
    step_logits: list[torch.Tensor] = []
    for step in range(NUM_TOKENS):
        position = len(prompt) - 1 + step
        next_token_logits = model(tokens, mask)[0, position]
        step_logits.append(next_token_logits)
        generated.append(int(next_token_logits.argmax()))
        tokens[0, position + 1] = generated[-1]
    return generated, torch.stack(step_logits)


def test_tiny_llm_prefill_and_decode_match_eager() -> None:
    torch.manual_seed(0)
    model = TinyLM().eval()
    mask = _causal_mask(MAX_LEN)

    eager_tokens, eager_logits = _generate(model, PROMPT, mask)
    compiled = torch.compile(
        model, backend=luminal_reference.Compiler(), fullgraph=True, dynamic=False
    )
    compiled_tokens, compiled_logits = _generate(compiled, PROMPT, mask)

    assert compiled_tokens == eager_tokens, f"{compiled_tokens} != {eager_tokens}"
    assert compiled_logits.shape == eager_logits.shape == (NUM_TOKENS, VOCAB)
    torch.testing.assert_close(compiled_logits, eager_logits, atol=ATOL, rtol=0.0)


@torch.no_grad()
def _growing_decode(
    model: nn.Module, prompt: tuple[int, ...], steps: int
) -> tuple[list[int], torch.Tensor]:
    """Prefill + decode where the context grows one token per step.

    Unlike `_generate`, the sequence tensor changes shape every step, so this
    only runs off a single compilation if the backend honours dynamic dims.
    """
    tokens = torch.tensor([list(prompt)], dtype=torch.long)
    generated: list[int] = []
    step_logits: list[torch.Tensor] = []
    for _ in range(steps):
        logits = model(tokens, _causal_mask(tokens.shape[1]))
        next_token_logits = logits[0, -1]
        step_logits.append(next_token_logits)
        nxt = int(next_token_logits.argmax())
        generated.append(nxt)
        tokens = torch.cat([tokens, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
    return generated, torch.stack(step_logits)


def test_tiny_llm_dynamic_context_reuses_one_compile() -> None:
    torch.manual_seed(0)
    model = TinyLM().eval()
    compiles: list[int] = []

    def backend(gm, example_inputs, **kwargs):
        compiles.append(1)
        return luminal_reference.Compiler(**kwargs)(gm, example_inputs)

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)

    for prompt in ((3, 1, 4, 1), (5, 2, 9)):
        eager_tokens, eager_logits = _growing_decode(model, prompt, NUM_TOKENS)
        tokens, logits = _growing_decode(compiled, prompt, NUM_TOKENS)
        assert tokens == eager_tokens, f"{tokens} != {eager_tokens}"
        torch.testing.assert_close(logits, eager_logits, atol=ATOL, rtol=0.0)

    assert compiles == [1], f"expected one compile, got {len(compiles)}"
