//! The reference runtime's bindings generator (moved out of core at
//! Step C of the runtime-ownership ruling, 2026-08-17 — core keeps only
//! the `RuntimeBindingsGenerator` trait; every runtime owns its
//! boundary vocabulary).
//!
//! Models are boundary-free logical structure (the 2026-07-24 ruling);
//! everything about REPRESENTATION at the boundary — layouts, buffers,
//! access, freed-by, dynamic-dim seeds, the Bool8 boundary cast — is
//! binding vocabulary, stated per runtime at load time. This module
//! states the REFERENCE runtime's contract: row-major contiguous
//! boundaries at the dtype's own width, byte-code booleans (Bool8),
//! caller-owned buffers.

use luminal::dtype::DType;
use luminal::runtime_binding::RuntimeBindingsGenerator;

/// The reference runtime's binding vocabulary.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReferenceBindings;

impl ReferenceBindings {
    /// The standard schedule tail shared by every assembled reference
    /// program.
    pub const SCHEDULE: &'static str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run cleanup)) (saturate (run fixpoint-invariants)))\n\n";
}

impl RuntimeBindingsGenerator for ReferenceBindings {
    /// The boundary element width for a dtype under the reference
    /// binding. Boolean values cross as Bool8 (the two-legal-codes byte
    /// type — see the preamble's Dtype contract); everything else at
    /// its own bits-of width.
    fn width_term(&self, dtype: DType) -> String {
        match dtype {
            DType::Bool => "(bits-of (Bool8))".to_string(),
            other => format!("(bits-of ({other:?}))"),
        }
    }

    /// Input boundary: contiguous row-major storage of the declared
    /// dtype, read-only, caller-owned, buffer id = the HLIR node index
    /// (their set_data keying). `stem` namespaces the lets; the
    /// buffer-tensor let is named `{stem}_buffer_tensor`.
    fn input_binding(
        &self,
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

    /// Output boundary. A Bool value crosses as Bool8: the BINDING
    /// states the byte representation by casting (the Bool8 ruling) —
    /// the model's output naming stays on the logical value. Buffer id
    /// = the output key; the buffer-tensor let is named
    /// `{stem}_buffer_tensor`.
    fn output_binding(
        &self,
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
        let width = self.width_term(dtype);
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

    fn schedule(&self) -> &str {
        Self::SCHEDULE
    }
}
