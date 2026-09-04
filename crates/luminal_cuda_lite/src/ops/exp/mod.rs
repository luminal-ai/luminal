//! Elementwise exponential — CUDA-lite's OWN op (ruling 2026-08-17:
//! every runtime owns its executable ops; the shared crate supplies
//! only the IR traits). Same egglog constructor and label as the
//! reference runtime's exp — assemblies are per-runtime, labels are
//! IR identity — but the structs, matcher, snippets, and codegen all
//! live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelSource, unary};
use anyhow::Result;

/// `ExpFunctionalGeneric(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpFunctional;

impl OpSlotNames for ExpFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ExpFunctional {
    fn label(&self) -> &str {
        "ExpFunctionalGeneric"
    }
}

impl Bufferizable for ExpFunctional {}

impl ToDps for ExpFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(ExpFunctionalDps))
    }
}

impl LayoutIrOp for ExpFunctional {}

/// Destination-passing form: `Exp(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpFunctionalDps;

impl OpSlotNames for ExpFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ExpFunctionalDps {
    fn label(&self) -> &str {
        "ExpFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for ExpFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for ExpFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for ExpFunctionalDps {}

/// The CUDA lowering, colocated with its op.
pub(crate) fn codegen(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    unary(ctx, "expf(a[i])")
}

/// Matches `LayoutTensorOpExpFunctionalGeneric` and produces this
/// runtime's [`ExpFunctional`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ExpFunctionalMatcher;

impl OpMatcher for ExpFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpExpFunctionalGeneric"
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
        Box::new(ExpFunctional)
    }
}
