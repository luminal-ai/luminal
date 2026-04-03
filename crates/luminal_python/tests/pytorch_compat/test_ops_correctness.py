"""Tests for PT2 backend reduction and shape/movement ops."""

import os
import pytest
import torch
import torch._dynamo

# Force PT2 export mode
os.environ["LUMINAL_EXPORT_MODE"] = "pt2"

torch.set_float32_matmul_precision("highest")


@pytest.fixture(autouse=True, scope="function")
def reset_dynamo():
    torch._dynamo.config.cache_size_limit = 1
    torch._dynamo.config.suppress_errors = False
    yield
    torch._dynamo.reset()


def _get_backend():
    import luminal_python

    return luminal_python.luminal_backend


def _run_model(model_fn, *inputs, atol=1e-4, rtol=1e-4):
    """Compile model_fn with luminal and compare against eager PyTorch."""
    torch._dynamo.reset()
    eager_out = model_fn(*inputs)
    compiled = torch.compile(model_fn, backend=_get_backend())
    luminal_out = compiled(*inputs)
    if isinstance(eager_out, torch.Tensor):
        torch.testing.assert_close(luminal_out, eager_out, atol=atol, rtol=rtol)
    elif isinstance(eager_out, (tuple, list)):
        for a, b in zip(luminal_out, eager_out):
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return luminal_out


# ===== Reduction Ops =====


def test_argmax():
    x = torch.randn(3, 5)

    def f(x):
        return x.argmax(dim=1)

    _run_model(f, x)


def test_argmax_keepdim():
    x = torch.randn(3, 5)

    def f(x):
        return x.argmax(dim=1, keepdim=True)

    _run_model(f, x)


def test_argmin():
    x = torch.randn(3, 5)

    def f(x):
        return x.argmin(dim=1)

    _run_model(f, x)


def test_prod():
    x = torch.randn(3, 4).abs() + 0.1  # positive to avoid sign issues

    def f(x):
        return x.prod(dim=1)

    _run_model(f, x, atol=1e-3, rtol=1e-3)


def test_prod_keepdim():
    x = torch.randn(3, 4).abs() + 0.1

    def f(x):
        return x.prod(dim=0, keepdim=True)

    _run_model(f, x, atol=1e-3, rtol=1e-3)


def test_argsort():
    x = torch.randn(4, 5)

    def f(x):
        return x.argsort(dim=1)

    _run_model(f, x)


def test_log_softmax():
    x = torch.randn(3, 5)

    def f(x):
        return torch.nn.functional.log_softmax(x, dim=1)

    _run_model(f, x, atol=1e-3, rtol=1e-3)


def test_std_unbiased():
    x = torch.randn(4, 8)

    def f(x):
        return x.std(dim=1)

    _run_model(f, x, atol=1e-3, rtol=1e-3)


def test_var_unbiased():
    x = torch.randn(4, 8)

    def f(x):
        return x.var(dim=1)

    _run_model(f, x, atol=1e-3, rtol=1e-3)


def test_sort():
    x = torch.randn(3, 5)

    def f(x):
        vals, inds = x.sort(dim=1)
        return vals, inds

    _run_model(f, x)


def test_max_reduction_with_dim():
    x = torch.randn(3, 5)

    def f(x):
        vals, inds = x.max(dim=1)
        return vals, inds

    _run_model(f, x)


def test_min_reduction_with_dim():
    x = torch.randn(3, 5)

    def f(x):
        vals, inds = x.min(dim=1)
        return vals, inds

    _run_model(f, x)


def test_msort():
    x = torch.randn(5, 3)

    def f(x):
        return torch.msort(x)

    _run_model(f, x)


def test_logsumexp():
    x = torch.randn(3, 5)

    def f(x):
        return torch.logsumexp(x, dim=1)

    _run_model(f, x, atol=1e-3, rtol=1e-3)


def test_median():
    x = torch.randn(5, 7)

    def f(x):
        return x.median(dim=1).values

    _run_model(f, x)


# ===== Shape / View / Movement Ops =====


def test_flatten():
    x = torch.randn(2, 3, 4)

    def f(x):
        return x.flatten(1, 2)

    _run_model(f, x)


def test_unflatten():
    x = torch.randn(2, 12)

    def f(x):
        return x.unflatten(1, (3, 4))

    _run_model(f, x)


def test_ravel():
    x = torch.randn(2, 3, 4)

    def f(x):
        return x.ravel()

    _run_model(f, x)


def test_narrow():
    x = torch.randn(4, 6)

    def f(x):
        return x.narrow(1, 1, 3)

    _run_model(f, x)


def test_chunk():
    x = torch.randn(6, 4)

    def f(x):
        chunks = x.chunk(3, dim=0)
        return chunks[0] + chunks[1] + chunks[2]

    _run_model(f, x)


def test_unbind():
    x = torch.randn(3, 4)

    def f(x):
        parts = x.unbind(dim=0)
        return parts[0] + parts[1] + parts[2]

    _run_model(f, x)


def test_constant_pad_nd():
    x = torch.randn(2, 3)

    def f(x):
        return torch.nn.functional.pad(x, (1, 2, 0, 1), value=0.0)

    _run_model(f, x)


def test_repeat():
    x = torch.randn(2, 3)

    def f(x):
        return x.repeat(2, 3)

    _run_model(f, x)


def test_tile():
    x = torch.randn(2, 3)

    def f(x):
        return x.tile(2, 3)

    _run_model(f, x)


def test_fill():
    x = torch.randn(3, 4)

    def f(x):
        return x.fill(42.0)

    _run_model(f, x)


def test_broadcast_to():
    x = torch.randn(1, 3)

    def f(x):
        return x.broadcast_to(4, 3)

    _run_model(f, x)


def test_mT():
    x = torch.randn(2, 3, 4)

    def f(x):
        return x.mT

    _run_model(f, x)


def test_movedim():
    x = torch.randn(2, 3, 4)

    def f(x):
        return x.movedim(0, 2)

    _run_model(f, x)
