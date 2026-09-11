//! Metal boundary layouts and saturation schedule. Booleans use byte storage.

use luminal::dtype::DType;
use luminal::layout_ir::Access;
use luminal::runtime_binding::RuntimeBindingsGenerator;

#[derive(Debug, Clone, Copy, Default)]
pub struct MetalBindings;

impl MetalBindings {
    pub const SCHEDULE: &'static str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run cleanup)) (saturate (run fixpoint-invariants)))\n\n";
}

impl RuntimeBindingsGenerator for MetalBindings {
    fn width_term(&self, dtype: DType) -> String {
        match dtype {
            DType::Bool => "(bits-of (Bool8))".to_string(),
            other => format!("(bits-of ({other:?}))"),
        }
    }

    fn input_binding(
        &self,
        stem: &str,
        idx: usize,
        logical_name: &str,
        shape: &str,
        width: &str,
        access: Access,
    ) -> String {
        format!(
            "(let {stem}_layout (RightMajorContiguousElementLayoutLit {shape} {width}))\n\
             (let {stem}_layout_tensor (LayoutTensorLit {logical_name} {stem}_layout))\n\
             (let {stem}_buffer_id (BufferLit {idx}))\n\
             (set (buffer-access-of {stem}_buffer_id) ({access:?}))\n\
             (set (buffer-freed-by {stem}_buffer_id) (CallerFrees))\n\
             (let {stem}_buffer_tensor (BufferTensorLit {stem}_layout_tensor {stem}_buffer_id))\n\n"
        )
    }

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
