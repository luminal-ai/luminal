"""Reference runtime and frontend tests always use CPU inputs."""

import pytest
import torch
from backend_test_utils import reset_dynamo

__all__ = ["reset_dynamo"]


@pytest.fixture
def device():
    return torch.device("cpu")
