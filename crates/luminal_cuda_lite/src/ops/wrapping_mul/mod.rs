//! Elementwise two's-complement multiplication, modulo 2^w.
//! every runtime owns its executable ops; the shared crate supplies
//! only the IR traits). Same egglog constructor and label as the
//! reference runtime's mul — assemblies are per-runtime, labels are
//! IR identity — but the structs, matcher, snippets, and codegen all
//! live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, binary, wrapping_binary_expr};
use anyhow::Result;

/// `WrappingMulFunctionalGeneric(lhs, rhs) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingMulFunctional;

impl OpSlotNames for WrappingMulFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingMulFunctional {
    fn label(&self) -> &str {
        "WrappingMulFunctionalGeneric"
    }
}

impl Bufferizable for WrappingMulFunctional {}

impl ToDps for WrappingMulFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(WrappingMulFunctionalDps))
    }
}

impl LayoutIrOp for WrappingMulFunctional {}

/// Destination-passing form: `Mul(lhs: read, rhs: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingMulFunctionalDps;

impl OpSlotNames for WrappingMulFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingMulFunctionalDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "WrappingMulFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 2 // dest0 is write-only
    }
}

impl Bufferizable for WrappingMulFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for WrappingMulFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for WrappingMulFunctionalDps {}

/// The CUDA lowering, colocated with its op.
impl KernelOp for WrappingMulFunctionalDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        binary(ctx, &wrapping_binary_expr(ctx, "*")?)
    }
}

/// Matches `LayoutTensorOpWrappingMulFunctionalGeneric` and produces this
/// runtime's [`WrappingMulFunctional`].
#[derive(Debug, Clone, Copy, Default)]
pub struct WrappingMulFunctionalMatcher;

impl OpMatcher for WrappingMulFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpWrappingMulFunctionalGeneric"
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
        Box::new(WrappingMulFunctional)
    }
}
