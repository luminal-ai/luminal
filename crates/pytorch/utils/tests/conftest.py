"""Run PT2 translator parity with the explicitly selected native runtime."""

from backend_test_utils import device, execution_device, reset_dynamo, stage_one_only
from reference_xfails import apply_reference_xfails

__all__ = ["device", "execution_device", "reset_dynamo", "stage_one_only"]


def pytest_configure(config):
    import os

    import torch
    from backend_test_utils import selected_backend

    device = "cuda" if selected_backend() == "cuda_lite" else "cpu"
    os.environ["LUMINAL_TEST_DEVICE"] = device
    # Collection imports each test module before function fixtures run. Put
    # module-level tensor constants on the selected execution device too; the
    # Modal runner isolates this shared tree from backend-owned test roots.
    torch.set_default_device(device)


def pytest_collection_modifyitems(config, items):
    apply_reference_xfails(items)
