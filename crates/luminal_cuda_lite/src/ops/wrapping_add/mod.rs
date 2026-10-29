//! Elementwise two's-complement addition, modulo 2^w.
//! every runtime owns its executable ops; the shared crate supplies
//! only the IR traits). Same egglog constructor and label as the
//! reference runtime's add — assemblies are per-runtime, labels are
//! IR identity — but the structs, matcher, snippets, and codegen all
//! live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, binary, wrapping_binary_expr};
use anyhow::Result;

/// `WrappingAddFunctionalGeneric(lhs, rhs) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingAddFunctional;

impl OpSlotNames for WrappingAddFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingAddFunctional {
    fn label(&self) -> &str {
        "WrappingAddFunctionalGeneric"
    }
}

impl Bufferizable for WrappingAddFunctional {}

impl ToDps for WrappingAddFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(WrappingAddFunctionalDps))
    }
}

impl LayoutIrOp for WrappingAddFunctional {}

/// Destination-passing form: `Add(lhs: read, rhs: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingAddFunctionalDps;

impl OpSlotNames for WrappingAddFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingAddFunctionalDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "WrappingAddFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand < 2
    }
}

impl Bufferizable for WrappingAddFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for WrappingAddFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for WrappingAddFunctionalDps {}

/// The CUDA lowering, colocated with its op.
impl KernelOp for WrappingAddFunctionalDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        binary(ctx, &wrapping_binary_expr(ctx, "+")?)
    }
}

/// Matches `LayoutTensorOpWrappingAddFunctionalGeneric` and produces this
/// runtime's [`WrappingAddFunctional`].
#[derive(Debug, Clone, Copy, Default)]
pub struct WrappingAddFunctionalMatcher;

impl OpMatcher for WrappingAddFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpWrappingAddFunctionalGeneric"
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
        Box::new(WrappingAddFunctional)
    }
}
