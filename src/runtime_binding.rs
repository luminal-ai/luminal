//! The runtime bindings generator seam (Step C of the runtime-ownership
//! ruling, 2026-08-17).
//!
//! Models are boundary-free logical structure (the 2026-07-24 ruling);
//! everything about REPRESENTATION at the boundary — layouts, buffers,
//! access, freed-by, dynamic-dim seeds, boundary casts, the run
//! schedule — is binding vocabulary, stated PER RUNTIME at load time.
//! Core owns only this trait; each runtime crate owns its generator
//! (`luminal_reference::ReferenceBindings`, cuda-lite's `CudaBindings`,
//! …) and hands it to [`crate::graph::LogicalGraph::bound_parts`]. The
//! generators may start out near-identical; the seam exists so dtype
//! widths, layout policy, and schedule extensions can diverge per
//! runtime without touching core.

use crate::dtype::DType;
use crate::layout_ir::Access;

/// Per-runtime boundary vocabulary: how a runtime states its
/// representation contract in egglog at load time.
pub trait RuntimeBindingsGenerator {
    /// The boundary element width term for a dtype under this binding
    /// (e.g., whether booleans cross as Bool8).
    fn width_term(&self, dtype: DType) -> String;

    /// Input boundary text. `stem` namespaces the lets; the produced
    /// buffer-tensor let must be named `{stem}_buffer_tensor`. `access`
    /// is `ReadWrite` when some `.output_into()` targets this input (a
    /// caller-storage mutation), `ReadOnly` otherwise.
    fn input_binding(
        &self,
        stem: &str,
        idx: usize,
        logical_name: &str,
        shape: &str,
        width: &str,
        access: Access,
    ) -> String;

    /// Output boundary text. Same `{stem}_buffer_tensor` naming
    /// contract as [`RuntimeBindingsGenerator::input_binding`].
    fn output_binding(
        &self,
        stem: &str,
        key: usize,
        value_name: &str,
        shape: &str,
        dtype: DType,
    ) -> String;

    /// The schedule tail appended to every assembled program.
    fn schedule(&self) -> &str;

    /// The boundary joins: the buffer-tensor lists the bufferizer's
    /// BufferInput/BufferOutput nodes are built from. The cons-list
    /// shape is dictated by the preamble's `BufferInputLit` /
    /// `BufferOutputLit` constructors, not by runtime policy, so a
    /// default body is provided; a runtime with a different boundary
    /// vocabulary may override it.
    fn boundary_lists(
        &self,
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
}
