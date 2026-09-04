//! Elementwise comparison: `out[i] = lhs[i] < rhs[i]`, Bool result — CUDA-lite's OWN op (ruling 2026-08-17:
//! every runtime owns its executable ops; the shared crate supplies
//! only the IR traits). Same egglog constructor and label as the
//! reference runtime's less_than — assemblies are per-runtime, labels are
//! IR identity — but the structs, matcher, snippets, and codegen all
//! live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelSource, binary};
use anyhow::Result;

/// `LessThanGeneric(lhs, rhs) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LessThan;

impl OpSlotNames for LessThan {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for LessThan {
    fn label(&self) -> &str {
        "LessThanGeneric"
    }
}

impl Bufferizable for LessThan {}

impl ToDps for LessThan {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(LessThanDps))
    }
}

impl LayoutIrOp for LessThan {}

/// Destination-passing form: `LessThan(lhs: read, rhs: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LessThanDps;

impl OpSlotNames for LessThanDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for LessThanDps {
    fn label(&self) -> &str {
        "LessThanGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 2 // dest0 is write-only
    }
}

impl Bufferizable for LessThanDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for LessThanDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for LessThanDps {}

/// The CUDA lowering, colocated with its op.
pub(crate) fn codegen(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    binary(ctx, "(a[i] < b[i]) ? 1 : 0")
}

/// Matches `LayoutTensorOpLessThanGeneric` and produces this
/// runtime's [`LessThan`].
#[derive(Debug, Clone, Copy, Default)]
pub struct LessThanMatcher;

impl OpMatcher for LessThanMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpLessThanGeneric"
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
        &[("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(LessThan)
    }
}
