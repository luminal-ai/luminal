"""FFI-boundary tests for process_pt2's capsule validation.

Deviates from the standard `torch.compile(..., backend=luminal_backend)`
pattern in CLAUDE.md because the thing under test is the capsule-name
check itself, not a feature behavior. Exercising it through torch.compile
would only cover the happy path (`_reference_factory_capsule` produces a
correctly-named capsule, so validation passes trivially).
"""

import ctypes

import pytest

from luminal import process_pt2
from luminal.luminal import _reference_factory_capsule


def _new_capsule(name: bytes):
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    dummy = ctypes.c_void_p(0xDEADBEEF)
    return PyCapsule_New(ctypes.byref(dummy), name, None)


def test_factory_capsule_uses_versioned_abi_name():
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.restype = ctypes.c_char_p
    get_name.argtypes = [ctypes.py_object]

    assert get_name(_reference_factory_capsule()) == b"luminal.backend_factory.v2"


def test_process_pt2_rejects_capsule_with_wrong_name():
    bogus = _new_capsule(b"not.luminal.backend_factory")
    with pytest.raises(ValueError, match="luminal.backend_factory"):
        process_pt2("/dev/null", "/dev/null", 0, bogus, None)


def test_process_pt2_rejects_capsule_with_no_name():
    unnamed = _new_capsule(None)
    with pytest.raises(ValueError, match="luminal.backend_factory"):
        process_pt2("/dev/null", "/dev/null", 0, unnamed, None)
