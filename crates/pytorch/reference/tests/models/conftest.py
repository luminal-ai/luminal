"""Shared model parity cases, also explicitly selected by the CUDA runner.

Model tests state placement on their models and inputs. A suite-wide
``torch.device`` context also wraps HuggingFace ``from_pretrained`` and makes
Transformers infer a ``device_map``, which is a different loading contract and
requires the optional ``accelerate`` package.
"""

from backend_test_utils import device

__all__ = ["device"]
