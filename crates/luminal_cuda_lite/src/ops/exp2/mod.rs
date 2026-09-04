//! Elementwise base-2 exponential — CUDA-lite's OWN op (ruling 2026-08-17:
//! every runtime owns its executable ops; the shared crate supplies
//! only the IR traits). Same egglog constructor and label as the
//! reference runtime's exp2 — assemblies are per-runtime, labels are
//! IR identity — but the structs, matcher, snippets, and codegen all
//! live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelSource, unary};
use anyhow::Result;

/// `Exp2FunctionalGeneric(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exp2Functional;

impl OpSlotNames for Exp2Functional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for Exp2Functional {
    fn label(&self) -> &str {
        "Exp2FunctionalGeneric"
    }
}

impl Bufferizable for Exp2Functional {}

impl ToDps for Exp2Functional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(Exp2FunctionalDps))
    }
}

impl LayoutIrOp for Exp2Functional {}

/// Destination-passing form: `Exp2(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exp2FunctionalDps;

impl OpSlotNames for Exp2FunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for Exp2FunctionalDps {
    fn label(&self) -> &str {
        "Exp2FunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for Exp2FunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for Exp2FunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for Exp2FunctionalDps {}

/// The CUDA lowering, colocated with its op.
pub(crate) fn codegen(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    unary(ctx, "exp2f(a[i])")
}

/// Matches `LayoutTensorOpExp2FunctionalGeneric` and produces this
/// runtime's [`Exp2Functional`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Exp2FunctionalMatcher;

impl OpMatcher for Exp2FunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpExp2FunctionalGeneric"
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
        Box::new(Exp2Functional)
    }
}
