import torch


def test_model_loading_uses_an_explicit_cpu_default(device: torch.device):
    assert torch.get_default_device().type == "cpu"
