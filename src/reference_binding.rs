//! M3 Step 1: the reference binding generator — the model/binding split.
//!
//! Models are boundary-free logical structure (the 2026-07-24 ruling);
//! everything about REPRESENTATION at the boundary — layouts, buffers,
//! access, freed-by, dynamic-dim seeds, the Bool8 boundary cast — is
//! binding vocabulary, stated per runtime at load time. This module owns
//! the REFERENCE runtime's binding text: row-major contiguous boundaries
//! at the dtype's own width, byte-code booleans (Bool8), caller-owned
//! buffers. Both the interim translator (until it retires) and the native
//! recorder path emit boundaries through these functions, so the split is
//! a seam, not a fork.

use crate::dtype::DType;

/// The boundary element width for a dtype under the reference binding.
/// Boolean values cross as Bool8 (the two-legal-codes byte type — see the
/// preamble's Dtype contract); everything else at its own bits-of width.
pub fn width_term(dtype: DType) -> String {
    match dtype {
        DType::Bool => "(bits-of (Bool8))".to_string(),
        other => format!("(bits-of ({other:?}))"),
    }
}

/// Input boundary: contiguous row-major storage of the declared dtype,
/// read-only, caller-owned, buffer id = the HLIR node index (their
/// set_data keying). `stem` namespaces the lets ("t{idx}" for the
/// translator, "nat{idx}" for the native path). Returns the binding text;
/// the buffer-tensor let is named `{stem}_buffer_tensor`.
pub fn input_binding(
    stem: &str,
    idx: usize,
    logical_name: &str,
    shape: &str,
    width: &str,
) -> String {
    format!(
        "(let {stem}_layout (RightMajorContiguousElementLayoutLit {shape} {width}))\n\
         (let {stem}_layout_tensor (LayoutTensorLit {logical_name} {stem}_layout))\n\
         (let {stem}_buffer_id (BufferLit {idx}))\n\
         (set (buffer-access-of {stem}_buffer_id) (ReadOnly))\n\
         (set (buffer-freed-by {stem}_buffer_id) (CallerFrees))\n\
         (let {stem}_buffer_tensor (BufferTensorLit {stem}_layout_tensor {stem}_buffer_id))\n\n"
    )
}

/// Output boundary. A Bool value crosses as Bool8: the BINDING states the
/// byte representation by casting (the Bool8 ruling) — the model's output
/// naming stays on the logical value. Buffer id = the output key. Returns
/// the binding text; the buffer-tensor let is named `{stem}_buffer_tensor`.
pub fn output_binding(
    stem: &str,
    key: usize,
    value_name: &str,
    shape: &str,
    dtype: DType,
) -> String {
    let (boundary_name, cast_text) = if dtype == DType::Bool {
        let bool8_name = format!("{stem}_bool8");
        (
            bool8_name.clone(),
            format!("(let {bool8_name} (LogicalCast {value_name} (Bool8)))\n"),
        )
    } else {
        (value_name.to_string(), String::new())
    };
    let width = width_term(dtype);
    format!(
        "{cast_text}\
         (let {stem}_layout (RightMajorContiguousElementLayoutLit {shape} {width}))\n\
         (let {stem}_layout_tensor (LayoutTensorLit {boundary_name} {stem}_layout))\n\
         (let {stem}_buffer_id (BufferLit {key}))\n\
         (set (buffer-access-of {stem}_buffer_id) (ReadWrite))\n\
         (set (buffer-freed-by {stem}_buffer_id) (CallerFrees))\n\
         (let {stem}_buffer_tensor (BufferTensorLit {stem}_layout_tensor {stem}_buffer_id))\n\n"
    )
}

/// The boundary joins: the buffer-tensor lists the bufferizer's
/// BufferInput/BufferOutput nodes are built from.
pub fn boundary_lists(
    input_buffer_tensors: &[String],
    output_buffer_tensors: &[String],
    input_list_name: &str,
    output_list_name: &str,
) -> String {
    let join = |items: &[String]| {
        let mut term = "(BufferTensorNil)".to_string();
        for item in items.iter().rev() {
            term = format!("(BufferTensorCons {item} {term})");
        }
        term
    };
    format!(
        "(let {input_list_name} (BufferInputLit {}))\n(let {output_list_name} (BufferOutputLit {}))\n\n",
        join(input_buffer_tensors),
        join(output_buffer_tensors),
    )
}

/// The standard schedule tail shared by every assembled reference program.
pub const SCHEDULE: &str = "(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))\n\n";
