//! Elementwise addition — CUDA-lite's OWN op (ruling 2026-08-17:
//! every runtime owns its executable ops; the shared crate supplies
//! only the IR traits). Same egglog constructor and label as the
//! reference runtime's add — assemblies are per-runtime, labels are
//! IR identity — but the structs, matcher, snippets, and codegen all
//! live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelSource, binary};
use anyhow::Result;

/// `AddFunctionalGeneric(lhs, rhs) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddFunctional;

impl OpSlotNames for AddFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for AddFunctional {
    fn label(&self) -> &str {
        "AddFunctionalGeneric"
    }
}

impl Bufferizable for AddFunctional {}

impl ToDps for AddFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(AddFunctionalDps))
    }
}

impl LayoutIrOp for AddFunctional {}

/// Destination-passing form: `Add(lhs: read, rhs: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddFunctionalDps;

impl OpSlotNames for AddFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for AddFunctionalDps {
    fn label(&self) -> &str {
        "AddFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand < 2
    }
}

impl Bufferizable for AddFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for AddFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for AddFunctionalDps {}

/// The CUDA lowering, colocated with its op.
pub(crate) fn codegen(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    binary(ctx, "a[i] + b[i]")
}

/// Matches `LayoutTensorOpAddFunctionalGeneric` and produces this
/// runtime's [`AddFunctional`].
#[derive(Debug, Clone, Copy, Default)]
pub struct AddFunctionalMatcher;

impl OpMatcher for AddFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpAddFunctionalGeneric"
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
        Box::new(AddFunctional)
    }
}
