//! Truncated integer remainder — CUDA-lite's OWN op (ruling 2026-08-17:
//! every runtime owns its executable ops; the shared crate supplies
//! only the IR traits). Same egglog constructor and label as the
//! reference runtime's trunc_rem — assemblies are per-runtime, labels are
//! IR identity — but the structs, matcher, snippets, and codegen all
//! live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelSource, binary};
use anyhow::Result;

/// `TruncRemFunctionalGeneric(numerator, denominator) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncRemFunctional;

impl OpSlotNames for TruncRemFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "numerator".to_string(),
            1 => "denominator".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for TruncRemFunctional {
    fn label(&self) -> &str {
        "TruncRemFunctionalGeneric"
    }
}

impl Bufferizable for TruncRemFunctional {}

impl ToDps for TruncRemFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(TruncRemFunctionalDps))
    }
}

impl LayoutIrOp for TruncRemFunctional {}

/// Destination-passing form: `TruncRem(numerator: read, denominator: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncRemFunctionalDps;

impl OpSlotNames for TruncRemFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "numerator".to_string(),
            1 => "denominator".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for TruncRemFunctionalDps {
    fn label(&self) -> &str {
        "TruncRemFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 2 // dest0 is write-only
    }
}

impl Bufferizable for TruncRemFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for TruncRemFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for TruncRemFunctionalDps {}

/// The CUDA lowering, colocated with its op.
pub(crate) fn codegen(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    binary(ctx, "a[i] % b[i]") // C remainder carries the dividend's sign
}

/// Matches `LayoutTensorOpTruncRemFunctionalGeneric` and produces this
/// runtime's [`TruncRemFunctional`].
#[derive(Debug, Clone, Copy, Default)]
pub struct TruncRemFunctionalMatcher;

impl OpMatcher for TruncRemFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpTruncRemFunctionalGeneric"
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
        Box::new(TruncRemFunctional)
    }
}
