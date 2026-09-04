//! A materialize/copy between two distinct layouts of the same
//! logical value, chosen by extraction — CUDA-lite's OWN op (ruling 2026-08-17:
//! every runtime owns its executable ops; the shared crate supplies
//! only the IR traits). Same egglog constructor and label as the
//! reference runtime's copy — assemblies are per-runtime, labels are
//! IR identity — but the structs, matcher, snippets, and codegen all
//! live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelSource, unary};
use anyhow::Result;

/// `CopyGeneric(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterializeLayoutCopy;

impl OpSlotNames for MaterializeLayoutCopy {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MaterializeLayoutCopy {
    fn label(&self) -> &str {
        "CopyGeneric"
    }
}

impl Bufferizable for MaterializeLayoutCopy {}

impl ToDps for MaterializeLayoutCopy {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(MaterializeLayoutCopyDps))
    }
}

impl LayoutIrOp for MaterializeLayoutCopy {}

/// Destination-passing form: `Copy(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterializeLayoutCopyDps;

impl OpSlotNames for MaterializeLayoutCopyDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MaterializeLayoutCopyDps {
    fn label(&self) -> &str {
        "CopyGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for MaterializeLayoutCopyDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for MaterializeLayoutCopyDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for MaterializeLayoutCopyDps {}

/// The CUDA lowering, colocated with its op.
pub(crate) fn codegen(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    unary(ctx, "a[i]")
}

/// Matches `LayoutTensorOpCopyGeneric` and produces this
/// runtime's [`MaterializeLayoutCopy`].
#[derive(Debug, Clone, Copy, Default)]
pub struct MaterializeLayoutCopyMatcher;

impl OpMatcher for MaterializeLayoutCopyMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpCopyGeneric"
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        vec![
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_functional_constructor.egg"),
            },
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("layout", 1)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(MaterializeLayoutCopy)
    }
}
