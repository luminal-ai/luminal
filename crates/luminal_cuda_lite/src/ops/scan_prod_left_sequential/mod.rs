//! Inclusive product scan along one axis, evaluated as a left-sequential
//! fold (the axis is op metadata, not an operand) — CUDA-lite's OWN op:
//! same egglog constructor and label as the reference runtime's, but the
//! structs, matcher, snippets, and codegen all live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, scan};
use anyhow::{Context, Result};

/// `ScanProdLeftSequential(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanProdLeftSequential {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for ScanProdLeftSequential {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ScanProdLeftSequential {
    fn label(&self) -> &str {
        "ScanProdLeftSequential"
    }
}

impl Bufferizable for ScanProdLeftSequential {}

impl ToDps for ScanProdLeftSequential {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(ScanProdLeftSequentialDps { axis: self.axis }))
    }
}

impl LayoutIrOp for ScanProdLeftSequential {}

/// Destination-passing form: `ScanProdLeftSequential(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanProdLeftSequentialDps {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for ScanProdLeftSequentialDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ScanProdLeftSequentialDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "ScanProdLeftSequential"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for ScanProdLeftSequentialDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for ScanProdLeftSequentialDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for ScanProdLeftSequentialDps {}

/// The CUDA lowering, colocated with its op.
impl KernelOp for ScanProdLeftSequentialDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let axis = usize::try_from(self.axis).context("negative scan axis")?;
        scan(ctx, axis, "1", "acc * v")
    }
}

/// Matches `LayoutTensorOpScanProdLeftSequential` and produces this runtime's
/// [`ScanProdLeftSequential`]. Metadata children: `axis` at child 1, `out_layout` at
/// child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScanProdLeftSequentialMatcher;

impl OpMatcher for ScanProdLeftSequentialMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpScanProdLeftSequential"
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
        Box::new(ScanProdLeftSequential {
            axis: site.child_i64(1),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::scan_sum_left_sequential::tests::source_for;

    /// The product scan differs from the sum scan only in its identity and
    /// its fold.
    #[test]
    fn product_scan_folds_with_multiplication() {
        let source = source_for(&ScanProdLeftSequentialDps { axis: 0 }, &[2, 3]);
        for needle in [
            "float acc = 1;",
            "acc = acc * v;",
            "out[outer_index * 3LL * 1LL + r * 1LL + inner_index] = acc;",
        ] {
            assert!(source.contains(needle), "missing `{needle}`:\n{source}");
        }
    }
}
