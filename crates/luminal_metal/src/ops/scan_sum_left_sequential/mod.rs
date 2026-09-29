//! Inclusive sum scan along one axis, evaluated as a left-sequential
//! fold (the axis is op metadata, not an operand).

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, scan};
use anyhow::{Context, Result};

/// `ScanSumLeftSequential(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanSumLeftSequential {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for ScanSumLeftSequential {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ScanSumLeftSequential {
    fn label(&self) -> &str {
        "ScanSumLeftSequential"
    }
}

impl Bufferizable for ScanSumLeftSequential {}

impl ToDps for ScanSumLeftSequential {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(ScanSumLeftSequentialDps { axis: self.axis }))
    }
}

impl LayoutIrOp for ScanSumLeftSequential {}

/// Destination-passing form: `ScanSumLeftSequential(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanSumLeftSequentialDps {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for ScanSumLeftSequentialDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ScanSumLeftSequentialDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::MetalOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "ScanSumLeftSequential"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for ScanSumLeftSequentialDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for ScanSumLeftSequentialDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for ScanSumLeftSequentialDps {}

/// The Metal lowering, colocated with its op.
impl KernelOp for ScanSumLeftSequentialDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let axis = usize::try_from(self.axis).context("negative scan axis")?;
        scan(ctx, axis, "0", "acc + v")
    }
}

/// Matches `LayoutTensorOpScanSumLeftSequential` and produces this runtime's
/// [`ScanSumLeftSequential`]. Metadata children: `axis` at child 1, `out_layout` at
/// child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScanSumLeftSequentialMatcher;

impl OpMatcher for ScanSumLeftSequentialMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpScanSumLeftSequential"
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
        Box::new(ScanSumLeftSequential {
            axis: site.child_i64(1),
        })
    }
}
