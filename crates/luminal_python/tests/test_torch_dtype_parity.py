"""Pin luminal's Rust `TorchDType` enum to PyTorch's PT2 schema.

The PT2 export pipeline wire-serializes dtypes as `u32` codes drawn from
`torch._export.serde.schema.ScalarType`. luminal mirrors that enum in
`crates/luminal_python/rust/src/torch_dtype.rs` and depends on the
discriminants matching exactly. If PyTorch renumbers, adds, or removes a
variant, this test fails loudly at CI time — better than a silent
miscompile at runtime.
"""

from torch._export.serde.schema import ScalarType

# `_torch_dtype_codes` is the pyo3-exported map `{variant_name: pt2_code}`.
from luminal.luminal import _torch_dtype_codes


def test_rust_variants_match_pytorch():
    """Every Rust variant must agree with PyTorch's code for the same name."""
    rust = _torch_dtype_codes()
    pt = {v.name: v.value for v in ScalarType}
    mismatches = []
    for name, code in rust.items():
        if name not in pt:
            mismatches.append(f"{name}: luminal={code}, pytorch=<missing variant>")
        elif pt[name] != code:
            mismatches.append(f"{name}: luminal={code}, pytorch={pt[name]}")
    assert not mismatches, (
        "torch_dtype.rs and PyTorch's ScalarType have drifted:\n  "
        + "\n  ".join(mismatches)
    )


def test_no_pytorch_variants_missing_from_rust():
    """Surface new PyTorch variants so we know to extend the Rust enum.

    Failure here doesn't necessarily indicate a bug — it just means
    PyTorch added a dtype (e.g. a new float8 variant) and luminal should
    decide whether to mirror it. Update `TorchDType::ALL` in
    `torch_dtype.rs` plus the `TryFrom` impls to resolve.
    """
    rust = _torch_dtype_codes()
    missing = [v.name for v in ScalarType if v.name not in rust]
    assert not missing, (
        "PyTorch ScalarType variants not mirrored in luminal::TorchDType: "
        f"{missing}. Extend TorchDType::ALL in torch_dtype.rs and decide "
        "whether each maps to a luminal DType variant."
    )
