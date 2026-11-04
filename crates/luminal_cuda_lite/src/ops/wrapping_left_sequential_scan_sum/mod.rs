//! Inclusive two's-complement sum scan along one axis, modulo 2^w,
//! evaluated as a left-sequential fold in the same-width unsigned type
//! (signed overflow is undefined behaviour in C++).

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, scan, wrapping_fold};
use anyhow::{Context, Result};

/// `WrappingLeftSequentialScanSum(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingLeftSequentialScanSum {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for WrappingLeftSequentialScanSum {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingLeftSequentialScanSum {
    fn label(&self) -> &str {
        "WrappingLeftSequentialScanSum"
    }
}

impl Bufferizable for WrappingLeftSequentialScanSum {}

impl ToDps for WrappingLeftSequentialScanSum {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(WrappingLeftSequentialScanSumDps {
            axis: self.axis,
        }))
    }
}

impl LayoutIrOp for WrappingLeftSequentialScanSum {}

/// Destination-passing form: `WrappingLeftSequentialScanSum(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingLeftSequentialScanSumDps {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for WrappingLeftSequentialScanSumDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingLeftSequentialScanSumDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "WrappingLeftSequentialScanSum"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for WrappingLeftSequentialScanSumDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for WrappingLeftSequentialScanSumDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for WrappingLeftSequentialScanSumDps {}

/// The CUDA lowering, colocated with its op.
impl KernelOp for WrappingLeftSequentialScanSumDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let axis = usize::try_from(self.axis).context("negative scan axis")?;
        scan(ctx, axis, "0", &wrapping_fold(ctx.operand_dtypes[0], "+")?)
    }
}

/// Matches `LayoutTensorOpWrappingLeftSequentialScanSum` and produces this runtime's
/// [`WrappingLeftSequentialScanSum`]. Metadata children: `axis` at child 1, `out_layout` at
/// child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct WrappingLeftSequentialScanSumMatcher;

impl OpMatcher for WrappingLeftSequentialScanSumMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpWrappingLeftSequentialScanSum"
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
        Box::new(WrappingLeftSequentialScanSum {
            axis: site.child_i64(1),
        })
    }
}
