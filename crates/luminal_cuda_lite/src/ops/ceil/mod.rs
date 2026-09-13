//! Elementwise square root — CUDA-lite's OWN op (ruling 2026-08-17:
//! every runtime owns its executable ops; the shared crate supplies
//! only the IR traits). Same egglog constructor and label as the
//! reference runtime's sqrt — assemblies are per-runtime, labels are
//! IR identity — but the structs, matcher, snippets, and codegen all
//! live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, unary};
use anyhow::Result;

/// `CeilFunctionalGeneric(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CeilFunctional;

impl OpSlotNames for CeilFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for CeilFunctional {
    fn label(&self) -> &str {
        "CeilFunctionalGeneric"
    }
}

impl Bufferizable for CeilFunctional {}

impl ToDps for CeilFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(CeilFunctionalDps))
    }
}

impl LayoutIrOp for CeilFunctional {}

/// Destination-passing form: `Ceil(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CeilFunctionalDps;

impl OpSlotNames for CeilFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for CeilFunctionalDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "CeilFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for CeilFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for CeilFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for CeilFunctionalDps {}

/// The CUDA lowering, colocated with its op.
impl KernelOp for CeilFunctionalDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        unary(ctx, "ceilf(a[i])")
    }
}

/// Matches `LayoutTensorOpCeilFunctionalGeneric` and produces this
/// runtime's [`CeilFunctional`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CeilFunctionalMatcher;

impl OpMatcher for CeilFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpCeilFunctionalGeneric"
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
        Box::new(CeilFunctional)
    }
}
