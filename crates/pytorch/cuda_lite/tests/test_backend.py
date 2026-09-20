"""Smoke tests for the CUDA-lite torch.compile backend.

These need the maturin-built extension and a CUDA device. On a host without a
GPU (or without the extension built) they skip cleanly rather than fail: the
runtime itself builds device-free and is covered by its Rust suites, so a
CUDA-free CI job should not go red over a missing device.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("luminal_cuda_lite")

import luminal_cuda_lite  # noqa: E402


def test_backend_registered():
    assert "luminal_cuda_lite" in torch._dynamo.list_backends()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_linear_roundtrip_repeated_calls():
    """Fresh inputs each call: the per-execution arena is alloc/free'd each time
    and outputs are separate tensors, so every call must still be exact."""
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8).cuda().eval()
    compiled = torch.compile(model, backend=luminal_cuda_lite)
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(4, 16, device="cuda")
            torch.testing.assert_close(compiled(x), model(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_dynamic_batch():
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8).cuda().eval()
    compiled = torch.compile(model, backend=luminal_cuda_lite, dynamic=True)
    with torch.no_grad():
        for n in (1, 3, 7, 2):
            x = torch.randn(n, 16, device="cuda")
            torch.testing.assert_close(compiled(x), model(x), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize(
    "activation",
    [torch.nn.ReLU(), torch.nn.GELU()],
    ids=["relu", "gelu"],
)
def test_select_backed_activations(activation):
    """ReLU/GELU lower through the native ternary select op; before it existed
    their graphs dead-ended extraction and the backend refused to compile."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), activation, torch.nn.Linear(32, 8)
    ).cuda().eval()
    compiled = torch.compile(model, backend=luminal_cuda_lite)
    with torch.no_grad():
        x = torch.randn(4, 16, device="cuda")
        torch.testing.assert_close(compiled(x), model(x), atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize(
    ("dtype", "atol"),
    [
        (torch.float16, 1e-2),
        (torch.bfloat16, 1e-1),
        (torch.float64, 1e-6),
    ],
    ids=["f16", "bf16", "f64"],
)
def test_half_and_double_dtypes(dtype, atol):
    torch.manual_seed(0)
    model = (
        torch.nn.Sequential(
            torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8)
        )
        .to("cuda", dtype)
        .eval()
    )
    compiled = torch.compile(model, backend=luminal_cuda_lite)
    with torch.no_grad():
        x = torch.randn(4, 16, device="cuda", dtype=dtype)
        got = compiled(x)
        assert got.dtype == dtype
        torch.testing.assert_close(got, model(x), atol=atol, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_repeated_calls_reuse_input_tensors():
    """The same input tensor every call: its buffer is re-addressed on each
    execute, so a stale pointer or a dropped binding shows up as a wrong
    result."""
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8).cuda().eval()
    compiled = torch.compile(model, backend=luminal_cuda_lite)
    x = torch.randn(4, 16, device="cuda")
    with torch.no_grad():
        for _ in range(5):
            torch.testing.assert_close(compiled(x), model(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_writeback_after_read():
    """WAR: the returned value reads the input's pre-mutation contents, and
    the writeback lands in the caller's own tensor (one buffer, one pointer)."""

    def fn(x):
        y = x * 2
        x.add_(1)
        return y

    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda")
    expected_y = x * 2
    expected_x = x + 1
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    torch.testing.assert_close(got, expected_y)
    torch.testing.assert_close(x, expected_x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_returned_clone_of_a_mutated_value_is_fresh_storage():
    """Eager's contract: `clone` is fresh storage holding the mutated
    value, and the input is mutated in place. The export names two output
    nodes — the writeback and the clone — and each is its own boundary
    row: the writeback on the input's buffer, the clone on a buffer of its
    own. A clone's only content is that storage identity, so it is never
    merged into the row of the value it copies."""

    def fn(x):
        x.add_(1)
        return x.clone()

    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda")
    expected = x + 1
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(x, expected)
    assert got.data_ptr() != x.data_ptr()
    got.add_(1)
    torch.testing.assert_close(x, expected, msg="the clone is not the input's storage")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_writeback_consumed_downstream():
    """The mutated input is read again after the writeback: the sink and the
    reader are the same storage."""

    def fn(x):
        x.add_(1)
        return x * 2

    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda")
    expected_x = x + 1
    expected = expected_x * 2
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(x, expected_x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.xfail(
    strict=True,
    reason="LUM-830: eager's pointwise output inherits the operand layout; "
    "no strided-destination write yet",
)
def test_transposed_input_binds_zero_copy():
    """A transposed input is bound at the strides it has — (1, 8) — never
    copied and never reinterpreted; that the chain is column-major is the
    e-graph's discovery. Eager gives the pointwise output the operand's
    layout too, so the output is bound at that chain and the search refuses
    by name until a strided-destination write lands."""

    def fn(x):
        return x * 2 + 1

    torch.manual_seed(0)
    x = torch.randn(8, 4, device="cuda").t()
    assert not x.is_contiguous()
    expected = fn(x)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    torch.testing.assert_close(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_output_is_returned_at_eagers_strides():
    """A user-visible output is bound at the strides EAGER gives it and
    allocated at them, so the tensor the caller receives is laid out the way
    the uncompiled program lays it out."""

    def fn(x, y):
        return x * y + 1

    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda")
    y = torch.randn(4, 8, device="cuda")
    expected = fn(x, y)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x, y)
    assert got.stride() == expected.stride()
    torch.testing.assert_close(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_dynamic_batch_output_is_returned_at_eagers_strides():
    """The output's declared extents are the program's own dimensions, so
    each call allocates at the strides eager gives THAT batch size — one
    compile, every extent in the bucket."""
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8).cuda().eval()
    compiled = torch.compile(model, backend=luminal_cuda_lite, dynamic=True)
    with torch.no_grad():
        for n in (3, 7):
            x = torch.randn(n, 16, device="cuda")
            expected = model(x)
            got = compiled(x)
            assert got.stride() == expected.stride()
            torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.xfail(strict=True, reason="LUM-830: no strided-destination write yet")
def test_transposed_output_keeps_eagers_strides():
    """Eager lays this output out the way its operands are laid out, so it
    is column-major, and the output is bound at those exact strides. No
    elected op writes a non-row-major destination today: the search refuses
    by name until the layout-changing copy into the bound output lands."""

    def fn(a, b):
        return a.t() + b.t()

    torch.manual_seed(0)
    a = torch.randn(8, 4, device="cuda")
    b = torch.randn(8, 4, device="cuda")
    expected = fn(a, b)
    got = torch.compile(fn, backend=luminal_cuda_lite)(a, b)
    assert got.stride() == expected.stride()
    torch.testing.assert_close(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_writeback_into_transposed_input_finds_no_plan_naming_the_output():
    """A writeback is bound at its target's chain — here the transposed
    strides — and whether any kernel writes that layout is the SEARCH's
    question, never a prior taken at bind. Today none does, so the search
    plans nothing and says which output, at which layout, it found no plan
    for."""

    def fn(x):
        x.add_(1)
        return x * 2

    torch.manual_seed(0)
    x = torch.randn(8, 4, device="cuda").t()
    with pytest.raises(
        Exception, match=r"no plan writes the bound outputs: v\d+ at Strided"
    ):
        torch.compile(fn, backend=luminal_cuda_lite)(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_read_only_aliasing_binds_two_buffers_on_one_address():
    """Two boundary tensors on one storage, neither written through, are two
    External buffers carrying one address: a fact about the caller's memory,
    not a hazard. ``x`` and ``x[:]`` are distinct objects, so Dynamo hands the
    backend two placeholders rather than deduplicating them."""

    def fn(a, b):
        return a * b

    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda")
    torch.testing.assert_close(
        torch.compile(fn, backend=luminal_cuda_lite)(x, x[:]), x * x
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_aliased_inputs_with_a_writeback_are_refused():
    """The hazardous variant: one binding of the overlapping pair is written
    back into, and one buffer per binding cannot order that write against the
    other's reads. Refused by name."""

    def fn(a, b):
        a.add_(1)
        return a * b

    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda")
    with pytest.raises(Exception, match="share device storage"):
        torch.compile(fn, backend=luminal_cuda_lite)(x, x[:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_int32_and_bool_inputs_bind_external():
    """Integer and boolean boundary tensors cross as the caller's own device
    bytes like any other: no host staging, no reinterpretation. A torch
    boolean is already the byte Bool8 reads."""

    def fn(values, mask):
        return torch.where(mask, values * 2, values)

    values = torch.arange(32, device="cuda", dtype=torch.int32).reshape(4, 8)
    mask = (torch.arange(32, device="cuda") % 2 == 0).reshape(4, 8)
    expected = fn(values, mask)
    got = torch.compile(fn, backend=luminal_cuda_lite)(values, mask)
    assert got.dtype == torch.int32
    torch.testing.assert_close(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.xfail(
    strict=True,
    reason="LUM-830: eager's pointwise output inherits the operand layout; "
    "no strided-destination write yet",
)
def test_transposed_dynamic_batch_input():
    """The transposed input's element stride IS the dynamic batch dimension,
    so the binding states that dimension rather than the number one example
    call had: one compile serves every extent in the bucket. Eager's output
    is column-major here, so the search refuses by name until a
    strided-destination write lands."""

    def fn(x):
        return x * 2 + 1

    torch.manual_seed(0)
    compiled = torch.compile(fn, backend=luminal_cuda_lite, dynamic=True)
    for n in (3, 7, 2):
        x = torch.randn(16, n, device="cuda").t()
        assert not x.is_contiguous()
        torch.testing.assert_close(compiled(x), fn(x), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.xfail(
    strict=True,
    reason="LUM-830: eager's pointwise output inherits the operand layout; "
    "no strided-destination write yet",
)
def test_permuted_dynamic_view_states_its_size_derived_strides():
    """A permuted view is neither row- nor column-major, and every one of
    its strides is a product of the program's own dimensions: the binding
    states those expressions, so one searched plan serves every extent in
    the bucket. Eager's output carries the same permuted strides, so the
    search refuses by name until a strided-destination write lands."""

    def fn(x):
        return x * 2 + 1

    torch.manual_seed(0)
    compiled = torch.compile(fn, backend=luminal_cuda_lite, dynamic=True)
    for a, b, c in ((3, 5, 7), (4, 5, 7), (3, 6, 7)):
        x = torch.randn(a, b, c, device="cuda").permute(2, 0, 1)
        assert x.stride() == (1, b * c, c)
        torch.testing.assert_close(compiled(x), fn(x), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_stride_only_symbol_is_refused_by_name():
    """A stride torch cannot derive from the tensor's own sizes becomes a
    fresh export symbol that appears in no size and carries no range
    constraint. The program states no dimension for it, so the binding
    cannot state it either: refused by name rather than frozen at the
    number this call happened to have."""

    def fn(x):
        return x * 2 + 1

    torch.manual_seed(0)
    compiled = torch.compile(fn, backend=luminal_cuda_lite, dynamic=True)
    x = torch.randn(16, 6, device="cuda").t()[:, ::2]
    with pytest.raises(Exception, match=r"stride on axis 1.*does not declare"):
        compiled(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_offset_input_is_refused_by_name():
    """A binding names a buffer's BASE, and two bindings on one buffer share
    that base: there is no way to say "same buffer, different offset". A
    boundary tensor that starts inside its storage is refused by name rather
    than bound alone with its aliasing left unsayable."""

    def fn(x):
        return x * 2

    torch.manual_seed(0)
    base = torch.randn(5, 8, device="cuda")
    x = base[1:]
    assert x.storage_offset() == 8
    with pytest.raises(Exception, match="storage offset 8"):
        torch.compile(fn, backend=luminal_cuda_lite)(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_expanded_input_is_a_broadcast_read_map():
    """A zero stride is a legitimate READ map: every coordinate of the
    broadcast axis reads one element. The expanded input binds as it is."""

    def fn(x, y):
        return x + y

    torch.manual_seed(0)
    x = torch.randn(1, 8, device="cuda").expand(4, 8)
    assert x.stride() == (0, 1)
    y = torch.randn(4, 8, device="cuda")
    torch.testing.assert_close(
        torch.compile(fn, backend=luminal_cuda_lite)(x, y), fn(x, y)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_stride_zero_mutation_target_finds_no_plan_naming_the_output():
    """The same broadcast view as a WRITE target: it is stated as the
    strided map it is and bound as such, and the e-graph's write gate — a
    destination layout must be injective — leaves the search with nothing
    to install, which it reports by naming the output and its layout."""
    from luminal_cuda_lite.boundary import Strided, boundary_layout

    def fn(t):
        t.add_(1)
        return t * 2

    torch.manual_seed(0)
    x = torch.randn(1, 8, device="cuda").expand(4, 8)
    assert boundary_layout("x", x) == Strided(("Integer(0)", "Integer(1)"))
    with pytest.raises(
        Exception, match=r"no plan writes the bound outputs: v\d+ at Strided"
    ):
        torch.compile(fn, backend=luminal_cuda_lite)(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_a_held_buffer_is_bound_on_the_modules_own_storage():
    """A parameter or buffer is the CALLER's memory too: it is bound at the
    address the module's own tensor has, never at a copy the backend made
    while preparing the graph. Asked of the module Dynamo does not inline,
    which is the configuration that hands the backend a graph holding the
    tensors as attributes."""

    class Counter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.zeros(4, device="cuda"))

        def forward(self, x):
            self.state.add_(1)
            return x + self.state

    held = []

    def capture(gm, example_inputs, **kwargs):
        model = luminal_cuda_lite(gm, example_inputs, **kwargs)
        held.append(model)
        return model

    module = Counter()
    before = module.state.clone()
    with torch._dynamo.config.patch(inline_inbuilt_nn_modules=False):
        compiled = torch.compile(module, backend=capture)
        out = compiled(torch.zeros(4, device="cuda"))

    torch.testing.assert_close(module.state, before + 1)
    torch.testing.assert_close(out, module.state)
    bound = [model for model in held if model._held]
    assert bound, "the backend bound no held tensor"
    for model in bound:
        for name, tensor in model._held.items():
            assert tensor.data_ptr() == module.state.data_ptr(), (
                f"{name} is bound on a copy of the module's storage"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_a_parameter_whose_storage_moved_is_read_at_its_new_address():
    """``p.data = p.data.clone()`` changes no metadata Dynamo guards on, so
    the compiled frame is re-entered with the parameter living somewhere
    else. Held tensors are re-addressed every call, so the numbers stay the
    module's."""
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8).cuda().eval()
    compiled = torch.compile(model, backend=luminal_cuda_lite)
    x = torch.randn(4, 16, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x))
        model.weight.data = model.weight.data.clone() + 1
        torch.cuda.empty_cache()
        torch.testing.assert_close(compiled(x), model(x))
