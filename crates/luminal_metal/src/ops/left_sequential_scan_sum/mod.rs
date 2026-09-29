//! Inclusive sum scan along one axis, evaluated as a left-sequential
//! fold (the axis is op metadata, not an operand).

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, scan};
use anyhow::{Context, Result};

/// `LeftSequentialScanSum(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeftSequentialScanSum {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for LeftSequentialScanSum {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for LeftSequentialScanSum {
    fn label(&self) -> &str {
        "LeftSequentialScanSum"
    }
}

impl Bufferizable for LeftSequentialScanSum {}

impl ToDps for LeftSequentialScanSum {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(LeftSequentialScanSumDps { axis: self.axis }))
    }
}

impl LayoutIrOp for LeftSequentialScanSum {}

/// Destination-passing form: `LeftSequentialScanSum(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeftSequentialScanSumDps {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for LeftSequentialScanSumDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for LeftSequentialScanSumDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::MetalOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "LeftSequentialScanSum"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for LeftSequentialScanSumDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for LeftSequentialScanSumDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for LeftSequentialScanSumDps {}

/// The Metal lowering, colocated with its op.
impl KernelOp for LeftSequentialScanSumDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let axis = usize::try_from(self.axis).context("negative scan axis")?;
        scan(ctx, axis, "0", "acc + v")
    }
}

/// Matches `LayoutTensorOpLeftSequentialScanSum` and produces this runtime's
/// [`LeftSequentialScanSum`]. Metadata children: `axis` at child 1, `out_layout` at
/// child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct LeftSequentialScanSumMatcher;

impl OpMatcher for LeftSequentialScanSumMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpLeftSequentialScanSum"
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
        Box::new(LeftSequentialScanSum {
            axis: site.child_i64(1),
        })
    }
}
