"""Run PT2 translator parity with the explicitly selected native runtime."""

from backend_test_utils import device, execution_device, reset_dynamo, stage_one_only
from reference_xfails import apply_reference_xfails

__all__ = ["device", "execution_device", "reset_dynamo", "stage_one_only"]


def pytest_configure(config):
    import os

    from backend_test_utils import selected_backend

    os.environ["LUMINAL_TEST_DEVICE"] = (
        "cuda" if selected_backend() == "cuda_lite" else "cpu"
    )


def pytest_collection_modifyitems(config, items):
    apply_reference_xfails(items)
