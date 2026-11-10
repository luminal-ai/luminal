import pytest
from backend_test_utils import eager_or_skip_unsupported


def test_eager_capability_errors_skip_the_unavailable_device_oracle():
    def unavailable():
        raise RuntimeError(
            "scatter_reduce_cuda_prod_ does not have a deterministic implementation"
        )

    with pytest.raises(pytest.skip.Exception):
        eager_or_skip_unsupported(unavailable)


def test_eager_runtime_errors_remain_failures():
    def broken():
        raise RuntimeError("an unrelated eager execution failure")

    with pytest.raises(RuntimeError, match="unrelated eager execution failure"):
        eager_or_skip_unsupported(broken)
