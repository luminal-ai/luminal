//! Inclusive running-maximum scan along one axis, evaluated as a
//! left-sequential fold (the axis is op metadata, not an operand).
//! CUDA-lite's own op: same egglog constructor and label as the reference
//! runtime's, but the structs, matcher, snippets, and codegen live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, cuda_f64_literal, scan};
use anyhow::{Context, Result};

/// `LeftSequentialScanMax(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeftSequentialScanMax {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for LeftSequentialScanMax {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for LeftSequentialScanMax {
    fn label(&self) -> &str {
        "LeftSequentialScanMax"
    }
}

impl Bufferizable for LeftSequentialScanMax {}

impl ToDps for LeftSequentialScanMax {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(LeftSequentialScanMaxDps { axis: self.axis }))
    }
}

impl LayoutIrOp for LeftSequentialScanMax {}

/// Destination-passing form: `LeftSequentialScanMax(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeftSequentialScanMaxDps {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for LeftSequentialScanMaxDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for LeftSequentialScanMaxDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "LeftSequentialScanMax"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for LeftSequentialScanMaxDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for LeftSequentialScanMaxDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for LeftSequentialScanMaxDps {}

/// The CUDA lowering, colocated with its op.
impl KernelOp for LeftSequentialScanMaxDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let axis = usize::try_from(self.axis).context("negative scan axis")?;
        // The identity is typed: a float -inf converted to an integer
        // accumulator is undefined, so each integer width names its minimum.
        use luminal::dtype::PlanDtype;
        let init = match ctx.operand_dtypes[0] {
            PlanDtype::Int => "(-2147483647 - 1)".to_string(),
            PlanDtype::Int64 => "(-9223372036854775807LL - 1LL)".to_string(),
            PlanDtype::Bool | PlanDtype::Bool8 => "0".to_string(),
            _ => cuda_f64_literal(f64::NEG_INFINITY),
        };
        scan(ctx, axis, &init, "v > acc ? v : acc")
    }
}

/// Matches `LayoutTensorOpLeftSequentialScanMax` and produces this runtime's
/// [`LeftSequentialScanMax`]. Metadata children: `axis` at child 1, `out_layout` at
/// child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct LeftSequentialScanMaxMatcher;

impl OpMatcher for LeftSequentialScanMaxMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpLeftSequentialScanMax"
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
        Box::new(LeftSequentialScanMax {
            axis: site.child_i64(1),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::left_sequential_scan_sum::tests::{dense_ctx, source_for};
    use luminal::dtype::PlanDtype;

    /// The max scan differs from the sum scan only in its identity and its
    /// fold.
    #[test]
    fn max_scan_folds_with_maximum() {
        let source = source_for(&LeftSequentialScanMaxDps { axis: 0 }, &[2, 3]);
        for needle in [
            "float acc = __uint_as_float(0xff800000u);",
            "acc = v > acc ? v : acc;",
            "out[outer_index * 3LL * 1LL + r * 1LL + inner_index] = acc;",
        ] {
            assert!(source.contains(needle), "missing `{needle}`:\n{source}");
        }
    }

    /// An integer accumulator starts at its own minimum, never at a
    /// converted float infinity.
    #[test]
    fn int_max_scan_seeds_with_the_integer_minimum() {
        let mut ctx = dense_ctx(&[4]);
        ctx.operand_dtypes = vec![PlanDtype::Int; 2];
        ctx.dest_dtypes = vec![PlanDtype::Int];
        let launches = LeftSequentialScanMaxDps { axis: 0 }
            .codegen(&ctx)
            .expect("scan codegen succeeds");
        let source = &launches[0].source;
        assert!(
            source.contains("int acc = (-2147483647 - 1);"),
            "missing the integer identity:\n{source}"
        );
    }
}
