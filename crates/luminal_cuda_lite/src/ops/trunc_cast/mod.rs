//! The dtype-changing materializer — CUDA-lite's OWN op (ruling
//! 2026-08-17: every runtime owns its executable ops; the shared crate
//! supplies only the IR traits). Same egglog constructor and label as
//! the reference runtime's cast — assemblies are per-runtime, labels
//! are IR identity — but the structs, matcher, snippets, and codegen
//! all live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, cuda_type, unary};
use anyhow::Result;

/// `TruncCastGeneric(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncCast;

impl OpSlotNames for TruncCast {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for TruncCast {
    fn label(&self) -> &str {
        "TruncCastGeneric"
    }
}

impl Bufferizable for TruncCast {}

impl ToDps for TruncCast {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(TruncCastDps))
    }
}

impl LayoutIrOp for TruncCast {}

/// Destination-passing form: `TruncCast(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncCastDps;

impl OpSlotNames for TruncCastDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for TruncCastDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "TruncCastGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for TruncCastDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for TruncCastDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for TruncCastDps {}

/// The CUDA lowering, colocated with its op. The conversion is driven
/// by the buffer dtypes the plan annotated — the op carries no dtype
/// field of its own.
impl KernelOp for TruncCastDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let to = cuda_type(ctx.dest_dtypes[0])?;
        unary(ctx, &format!("({to})truncf(a[i])"))
    }
}

/// Matches `LayoutTensorOpTruncCastGeneric` and produces this runtime's
/// [`TruncCast`].
#[derive(Debug, Clone, Copy, Default)]
pub struct TruncCastMatcher;

impl OpMatcher for TruncCastMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpTruncCastGeneric"
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
        &[("dtype", 1), ("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(TruncCast)
    }
}
