"""PyTorch's opmath contract, checked against eager.

Torch computes f16/bf16 floating math in f32 and rounds once, at the store, so
a chain of half-precision ops rounds once per op. The compiled program states
that same pattern. Two exactness classes:

* ``bits_equal`` where the f32 computation is a single IEEE add/sub/mul or is
  exact regardless of order (a sum of ones), so the compiled result must carry
  eager's bits.
* ``close_half`` where our f32 kernels are not torch's f32 kernels — a division
  is spelled as a multiply by a reciprocal, transcendentals use their own
  approximations, reductions have their own association order — so one f32 ulp
  can flip the final half rounding on a few elements. Those cases check the
  dtype exactly and the value to two half ulps.

Each case's reference is plain eager torch on the same device. These need the
maturin-built extension and a CUDA device; without either they skip cleanly.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("luminal_cuda_lite")

import luminal_cuda_lite  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
halves = pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"]
)


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def bits_equal(a, b):
    """Same dtype, same shape, same bytes."""
    assert a.dtype == b.dtype, f"{a.dtype} vs {b.dtype}"
    assert a.shape == b.shape, f"{tuple(a.shape)} vs {tuple(b.shape)}"
    bits = torch.int16 if a.element_size() == 2 else torch.int32
    return torch.equal(a.contiguous().view(bits), b.contiguous().view(bits))


def close_half(got, ref):
    """Same dtype; values within two ulps of the half dtype (kernel and
    reduction-order class, see the module docstring)."""
    assert got.dtype == ref.dtype, f"{got.dtype} vs {ref.dtype}"
    tol = 2**-9 if ref.dtype == torch.float16 else 2**-7
    torch.testing.assert_close(got, ref, atol=tol, rtol=tol)


# ---------------------------------------------------------------------
# Exact class: one IEEE f32 operation between the widening and the store
# ---------------------------------------------------------------------


@cuda
@halves
@pytest.mark.parametrize("op", [torch.add, torch.sub, torch.mul], ids=["add", "sub", "mul"])
def test_single_op_is_bit_exact(dtype, op):
    def fn(a, b):
        return op(a, b)

    torch.manual_seed(0)
    a = torch.randn(64, 64, device="cuda", dtype=dtype)
    b = torch.randn(64, 64, device="cuda", dtype=dtype)
    ref = fn(a, b)
    got = torch.compile(fn, backend=luminal_cuda_lite)(a, b)
    assert bits_equal(got, ref)


@cuda
@halves
def test_chain_rounds_once_per_op(dtype):
    """The product is stored in the half dtype before the add reads it; a
    chain kept in f32 across both ops differs in the last bit."""

    def fn(x, y):
        return (x * y) + x

    torch.manual_seed(0)
    x = torch.randn(64, 64, device="cuda", dtype=dtype)
    y = torch.randn(64, 64, device="cuda", dtype=dtype)
    ref = fn(x, y)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x, y)
    assert bits_equal(got, ref)


@cuda
@halves
def test_python_scalar_is_read_at_f32(dtype):
    """The scalar enters the add as the f32 value, not rounded to half first."""

    def fn(x):
        return x + 0.1

    x = torch.linspace(-1.0, 1.0, 256, device="cuda").to(dtype)
    ref = fn(x)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    assert bits_equal(got, ref)


@cuda
@halves
def test_int_tensor_operand_is_rounded_to_the_common_dtype_first(dtype):
    """A tensor operand is cast to the result dtype before it is widened, so
    2049, which no half dtype holds, enters the sum rounded."""

    def fn(x, i):
        return x + i

    x = torch.tensor([0.5, 0.25, 1.5, -0.5], device="cuda", dtype=dtype)
    i = torch.tensor([2049, 3, 2049, 7], device="cuda", dtype=torch.int32)
    ref = fn(x, i)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x, i)
    assert bits_equal(got, ref)


@cuda
def test_add_of_bf16_and_f32_is_f32():
    """Mixed operands take torch's promoted result dtype; nothing narrows."""

    def fn(a, b):
        return a + b

    torch.manual_seed(0)
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 64, device="cuda")
    ref = fn(a, b)
    got = torch.compile(fn, backend=luminal_cuda_lite)(a, b)
    assert got.dtype == torch.float32
    assert bits_equal(got, ref)


@cuda
@halves
def test_sum_of_ones_accumulates_in_f32(dtype):
    """3000 ones accumulate exactly in f32 in any order and round once at the
    store: 3000 in f16, 3008 in bf16. A half accumulator would stall long
    before that."""

    def fn(x):
        return x.sum()

    x = torch.ones(3000, device="cuda", dtype=dtype)
    ref = fn(x)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    assert bits_equal(got, ref)
    assert got.item() == (3000.0 if dtype == torch.float16 else 3008.0)


@cuda
@halves
def test_scalar_comparisons_round_the_literal_to_the_half_dtype(dtype):
    """Unlike arithmetic, a comparison rounds its scalar to the tensor's dtype
    first, so the element holding half(0.1) is neither above nor below 0.1."""

    def fn(x):
        return x > 0.1, x < 0.1, x > 0.3, x <= 0.7

    x = torch.tensor(
        [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, -0.1, 0.0], device="cuda", dtype=dtype
    )
    ref = fn(x)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    for g, r in zip(got, ref):
        assert g.dtype == torch.bool
        assert torch.equal(g, r)


@cuda
def test_int_tensor_against_a_float_literal_compares_at_f32():
    def fn(i):
        return i >= 0.5

    i = torch.arange(6, device="cuda", dtype=torch.int32)
    ref = fn(i)
    got = torch.compile(fn, backend=luminal_cuda_lite)(i)
    assert got.dtype == torch.bool
    assert torch.equal(got, ref)


@cuda
@halves
@pytest.mark.parametrize("op", [torch.maximum, torch.minimum], ids=["maximum", "minimum"])
def test_maximum_minimum_are_exact(dtype, op):
    def fn(a, b):
        return op(a, b)

    torch.manual_seed(0)
    a = torch.randn(64, 64, device="cuda", dtype=dtype)
    b = torch.randn(64, 64, device="cuda", dtype=dtype)
    ref = fn(a, b)
    got = torch.compile(fn, backend=luminal_cuda_lite)(a, b)
    assert bits_equal(got, ref)


@cuda
@halves
def test_clamp_and_neg_carry_the_operand_bits(dtype):
    def fn(x):
        return torch.clamp(x, 0.1, 0.7), -x

    x = torch.linspace(-1.0, 1.0, 256, device="cuda").to(dtype)
    ref = fn(x)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    for g, r in zip(got, ref):
        assert bits_equal(g, r)


@cuda
@halves
def test_where_over_a_half_and_an_f32_branch_is_f32(dtype):
    """where takes the promoted dtype of its branches; the condition is never
    cast."""

    def fn(mask, a, b):
        return torch.where(mask, a, b)

    torch.manual_seed(0)
    a = torch.randn(4, 16, device="cuda", dtype=dtype)
    b = torch.randn(4, 16, device="cuda")
    mask = torch.arange(64, device="cuda").reshape(4, 16) % 2 == 0
    ref = fn(mask, a, b)
    got = torch.compile(fn, backend=luminal_cuda_lite)(mask, a, b)
    assert got.dtype == torch.float32
    assert bits_equal(got, ref)


@cuda
@halves
def test_native_layer_norm_statistics_stay_f32(dtype):
    """Every output narrows to its own declared dtype: the normalised tensor
    is half, the mean and rstd are f32."""

    def fn(x, w, b):
        return torch.ops.aten.native_layer_norm(x, [16], w, b, 1e-5)

    torch.manual_seed(0)
    x = torch.randn(4, 16, device="cuda", dtype=dtype)
    w = torch.randn(16, device="cuda", dtype=dtype)
    b = torch.randn(16, device="cuda", dtype=dtype)
    ref = fn(x, w, b)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x, w, b)
    assert got[0].dtype == dtype
    assert got[1].dtype == torch.float32
    assert got[2].dtype == torch.float32
    for g, r in zip(got, ref):
        close_half(g, r)


@cuda
@halves
def test_sum_with_an_explicit_dtype_returns_f32(dtype):
    """sum(dtype=f32) casts first, accumulates in f32 and returns f32."""

    def fn(x):
        return x.sum(-1, dtype=torch.float32)

    torch.manual_seed(0)
    x = torch.randn(4, 16, device="cuda", dtype=dtype)
    ref = fn(x)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, ref)


@cuda
def test_attention_bool_mask_excludes_keys():
    """A Bool mask is a keep-mask: with one key kept per row every output row
    is exactly that key's value."""

    def fn(q, k, v, m):
        # Eager returns the attention as a transposed view; the plan cannot
        # yet write a boundary output at those strides, so hand back a
        # contiguous copy (the value under test is unchanged).
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=m
        ).contiguous()

    torch.manual_seed(0)
    q = torch.randn(2, 2, 4, 8, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    m = torch.tensor([[True, False, False, False]] * 4, device="cuda")
    got = torch.compile(fn, backend=luminal_cuda_lite)(q, k, v, m)
    assert got.dtype == torch.float16
    assert bits_equal(got, v[:, :, :1].expand(2, 2, 4, 8))


# ---------------------------------------------------------------------
# Kernel and order class: dtype exact, value to two half ulps
# ---------------------------------------------------------------------


@cuda
@halves
def test_division_is_the_widened_pattern(dtype):
    # Spelled as a multiply by a reciprocal: two f32 roundings against torch's one.
    def fn(a, b):
        return a / b

    torch.manual_seed(0)
    a = torch.randn(64, 64, device="cuda", dtype=dtype)
    b = torch.randn(64, 64, device="cuda", dtype=dtype).abs() + 0.5
    ref = fn(a, b)
    got = torch.compile(fn, backend=luminal_cuda_lite)(a, b)
    close_half(got, ref)


@cuda
@halves
@pytest.mark.parametrize(
    "op",
    [torch.exp, torch.sigmoid, torch.nn.functional.silu, torch.nn.functional.gelu],
    ids=["exp", "sigmoid", "silu", "gelu"],
)
def test_unary_math_is_the_widened_pattern(dtype, op):
    # Our f32 transcendentals are not torch's f32 routines.
    def fn(x):
        return op(x)

    torch.manual_seed(0)
    x = torch.randn(64, 64, device="cuda", dtype=dtype)
    ref = fn(x)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    close_half(got, ref)


@cuda
@halves
def test_longer_chain_stores_at_every_op(dtype):
    def fn(x, y):
        return torch.sigmoid(x * y) + torch.exp(x) / (y + 2.0)

    torch.manual_seed(0)
    x = torch.randn(64, 64, device="cuda", dtype=dtype)
    y = torch.randn(64, 64, device="cuda", dtype=dtype)
    ref = fn(x, y)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x, y)
    close_half(got, ref)


@cuda
@halves
@pytest.mark.parametrize(
    "op", [torch.sum, torch.mean, torch.var], ids=["sum", "mean", "var"]
)
def test_reductions_accumulate_wide(dtype, op):
    # Reduction order differs from torch's; a half accumulator would be far
    # outside two ulps over 4096 elements.
    def fn(x):
        return op(x, -1)

    torch.manual_seed(0)
    x = torch.randn(4, 4096, device="cuda", dtype=dtype)
    ref = fn(x)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    close_half(got, ref)


@cuda
@halves
@pytest.mark.parametrize(
    "op", [torch.softmax, torch.log_softmax], ids=["softmax", "log_softmax"]
)
def test_softmax_family_computes_wide(dtype, op):
    def fn(x):
        return op(x, -1)

    torch.manual_seed(0)
    x = torch.randn(4, 256, device="cuda", dtype=dtype)
    ref = fn(x)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    close_half(got, ref)


@cuda
@halves
def test_layer_norm_computes_wide(dtype):
    def fn(x, w, b):
        return torch.nn.functional.layer_norm(x, (256,), w, b, 1e-5)

    torch.manual_seed(0)
    x = torch.randn(4, 256, device="cuda", dtype=dtype)
    w = torch.randn(256, device="cuda", dtype=dtype)
    b = torch.randn(256, device="cuda", dtype=dtype)
    ref = fn(x, w, b)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x, w, b)
    close_half(got, ref)


@cuda
@halves
def test_batch_norm_reads_f32_parameters_unrounded(dtype):
    """Inference batch norm on a half input with f32 statistics and affine
    parameters (the autocast layout): the parameters enter the math at f32."""

    def fn(x, rm, rv, w, b):
        return torch.nn.functional.batch_norm(x, rm, rv, w, b, training=False)

    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 4, device="cuda", dtype=dtype)
    rm = torch.randn(3, device="cuda")
    rv = torch.rand(3, device="cuda") + 0.5
    w = torch.randn(3, device="cuda") * 1.001
    b = torch.randn(3, device="cuda") * 1.001
    ref = fn(x, rm, rv, w, b)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x, rm, rv, w, b)
    close_half(got, ref)


@cuda
@halves
def test_matmul_accumulates_wide(dtype):
    # The f32 accumulation order is unspecified on both sides.
    def fn(a, b):
        return a @ b

    torch.manual_seed(0)
    a = torch.randn(4, 16, device="cuda", dtype=dtype)
    b = torch.randn(16, 8, device="cuda", dtype=dtype)
    ref = fn(a, b)
    got = torch.compile(fn, backend=luminal_cuda_lite)(a, b)
    close_half(got, ref)


@cuda
@halves
def test_linear_accumulates_wide(dtype):
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8).to("cuda", dtype).eval()
    compiled = torch.compile(model, backend=luminal_cuda_lite)
    with torch.no_grad():
        x = torch.randn(4, 16, device="cuda", dtype=dtype)
        ref = model(x)
        got = compiled(x)
        close_half(got, ref)
