"""Run PT2 translator parity with the explicitly selected native runtime."""

from backend_test_utils import device, execution_device, reset_dynamo

__all__ = ["device", "execution_device", "reset_dynamo"]


def pytest_configure(config):
    import os

    from backend_test_utils import selected_backend

    os.environ["LUMINAL_TEST_DEVICE"] = (
        "cuda" if selected_backend() == "cuda_lite" else "cpu"
    )
