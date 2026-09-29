//! Inclusive sum scan along one axis, evaluated as a left-sequential
//! fold (the axis is op metadata, not an operand) — CUDA-lite's OWN op:
//! same egglog constructor and label as the reference runtime's, but the
//! structs, matcher, snippets, and codegen all live here.

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
        Some(crate::CudaOpInterface::kernel::<Self>())
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

/// The CUDA lowering, colocated with its op.
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use luminal::dtype::PlanDtype;
    use luminal::layouts::{
        BitWidthTerm, DecodedLayout, IntExprTerm, RightMajorContiguousElementLayout, ShapeTerm,
    };

    /// Dense F32 geometry over `dims` for the operand and the destination —
    /// the right-major spelling the matcher premise admits.
    pub(crate) fn dense_ctx(dims: &[i64]) -> CodegenCtx {
        let layout = DecodedLayout::of(
            RightMajorContiguousElementLayout {
                shape: ShapeTerm(dims.iter().map(|&d| IntExprTerm::Lit(d)).collect()),
                width: BitWidthTerm(32),
            },
            Some(PlanDtype::F32),
        );
        let extents: Vec<_> = dims
            .iter()
            .map(|&d| crate::symbolic::Expr(IntExprTerm::Lit(d)))
            .collect();
        CodegenCtx {
            operand_dims: vec![extents.clone(), extents.clone()],
            operand_dtypes: vec![PlanDtype::F32; 2],
            dest_dims: vec![extents],
            dest_dtypes: vec![PlanDtype::F32],
            operand_layouts: vec![layout.clone(), layout],
        }
    }

    pub(crate) fn source_for(op: &impl KernelOp, dims: &[i64]) -> String {
        let launches = op.codegen(&dense_ctx(dims)).expect("scan codegen succeeds");
        assert_eq!(launches.len(), 1, "a scan is a single-launch op");
        launches.into_iter().next().unwrap().source
    }

    /// A rank-1 scan is one thread folding the whole axis and writing every
    /// prefix as it goes.
    #[test]
    fn rank_one_scan_writes_each_prefix() {
        let source = source_for(&LeftSequentialScanSumDps { axis: 0 }, &[4]);
        for needle in [
            "const unsigned long long n = 1LL;",
            "for (unsigned long long r = 0; r < 4LL; ++r) {",
            "acc = acc + v;",
            "out[outer_index * 4LL * 1LL + r * 1LL + inner_index] = acc;",
        ] {
            assert!(source.contains(needle), "missing `{needle}`:\n{source}");
        }
    }

    /// Scanning the OUTER axis of a `[3,2]` value splits the threads over the
    /// inner axis: two threads, each striding a row per step.
    #[test]
    fn outer_axis_scan_strides_by_the_inner_extent() {
        let source = source_for(&LeftSequentialScanSumDps { axis: 1 }, &[3, 2]);
        for needle in [
            "const unsigned long long n = 2LL;",
            "long long c1 = (long long)(rem % 2LL); rem /= 2LL;",
            "long long c0 = (long long)r;",
            "out[outer_index * 3LL * 2LL + r * 2LL + inner_index] = acc;",
        ] {
            assert!(source.contains(needle), "missing `{needle}`:\n{source}");
        }
    }
}
