//! Two's-complement sum along one axis, modulo 2^w.
//! operand) — CUDA-lite's OWN op (ruling 2026-08-17: every runtime
//! owns its executable ops; the shared crate supplies only the IR
//! traits). Same egglog constructor and label as the reference
//! runtime's reduce_sum — assemblies are per-runtime, labels are IR
//! identity — but the structs, matcher, snippets, and codegen all
//! live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, reduce, wrapping_fold};
use anyhow::{Context, Result};

/// `WrappingReduceSumGeneric(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingReduceSum {
    /// Reduction axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for WrappingReduceSum {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingReduceSum {
    fn label(&self) -> &str {
        "WrappingReduceSumGeneric"
    }
}

impl Bufferizable for WrappingReduceSum {}

impl ToDps for WrappingReduceSum {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(WrappingReduceSumDps { axis: self.axis }))
    }
}

impl LayoutIrOp for WrappingReduceSum {}

/// Destination-passing form: `WrappingReduceSum(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WrappingReduceSumDps {
    /// Reduction axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for WrappingReduceSumDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingReduceSumDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel_wire::<Self>())
    }

    fn label(&self) -> &str {
        "WrappingReduceSumGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for WrappingReduceSumDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for WrappingReduceSumDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for WrappingReduceSumDps {}

/// The CUDA lowering, colocated with its op.
impl KernelOp for WrappingReduceSumDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let axis = usize::try_from(self.axis).context("negative reduce axis")?;
        reduce(ctx, axis, "0", &wrapping_fold(ctx.operand_dtypes[0], "+")?)
    }
}

/// Matches `LayoutTensorOpWrappingReduceSumGeneric` and produces this
/// runtime's [`WrappingReduceSum`]. Metadata children: `axis` at child 1,
/// `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct WrappingReduceSumMatcher;

impl OpMatcher for WrappingReduceSumMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpWrappingReduceSumGeneric"
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
        &[("axis", 1), ("out_layout", 2)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(WrappingReduceSum {
            axis: site.child_i64(1),
        })
    }
}
