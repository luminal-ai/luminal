//! CUDA-lite's OWN bindings generator (Step C of the runtime-ownership
//! ruling, 2026-08-17: each runtime states its boundary vocabulary;
//! core keeps only the trait).
//!
//! Today this is textually near-identical to the reference runtime's
//! binding — row-major contiguous boundaries at the dtype's own width,
//! Bool8 byte booleans, caller-owned buffers — and that duplication is
//! DELIBERATE, not debt: the runtimes share no binding code so this one
//! can diverge freely when the device runtime grows resident geometry
//! and view admission (M4), device-specific dtype widths (half/fp8
//! boundaries), or schedule extensions for CUDA-native match rules
//! (cuBLASLt descriptors). Divergence happens HERE, never in core.

use luminal::dtype::DType;
use luminal::runtime_binding::RuntimeBindingsGenerator;

/// The CUDA-lite runtime's binding vocabulary.
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaBindings;

impl CudaBindings {
    /// The schedule tail this runtime appends to every assembled
    /// program. Identical to the reference schedule today; CUDA-native
    /// rulesets (cuBLASLt matching) will extend it here.
    pub const SCHEDULE: &'static str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run cleanup)) (saturate (run fixpoint-invariants)))\n\n";
}

impl RuntimeBindingsGenerator for CudaBindings {
    /// Boolean values cross as Bool8; everything else at its own
    /// bits-of width. (Half/fp8 boundary widths land with the device
    /// dtype work, and land here.)
    fn width_term(&self, dtype: DType) -> String {
        match dtype {
            DType::Bool => "(bits-of (Bool8))".to_string(),
            other => format!("(bits-of ({other:?}))"),
        }
    }

    /// Input boundary: contiguous row-major storage, read-only,
    /// caller-owned, buffer id = the HLIR node index (set_data keying).
    /// The buffer-tensor let is named `{stem}_buffer_tensor`.
    fn input_binding(
        &self,
        stem: &str,
        idx: usize,
        logical_name: &str,
        shape: &str,
        width: &str,
        mutable: bool,
    ) -> String {
        let access = if mutable { "ReadWrite" } else { "ReadOnly" };
        format!(
            "(let {stem}_layout (RightMajorContiguousElementLayoutLit {shape} {width}))\n\
             (let {stem}_layout_tensor (LayoutTensorLit {logical_name} {stem}_layout))\n\
             (let {stem}_buffer_id (BufferLit {idx}))\n\
             (set (buffer-access-of {stem}_buffer_id) ({access}))\n\
             (set (buffer-freed-by {stem}_buffer_id) (CallerFrees))\n\
             (let {stem}_buffer_tensor (BufferTensorLit {stem}_layout_tensor {stem}_buffer_id))\n\n"
        )
    }

    /// Output boundary; Bool crosses as Bool8 via an explicit boundary
    /// cast (the Bool8 ruling). The buffer-tensor let is named
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
