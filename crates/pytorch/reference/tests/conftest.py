"""Reference runtime and frontend tests always use CPU inputs."""

import pytest
import torch
from backend_test_utils import reset_dynamo
from reference_xfails import apply_reference_xfails

__all__ = ["reset_dynamo"]


@pytest.fixture
def device():
    return torch.device("cpu")


def pytest_collection_modifyitems(config, items):
    apply_reference_xfails(items)
