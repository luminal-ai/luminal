//! Inclusive two's-complement prod scan along one axis, modulo 2^w,
//! evaluated as a left-sequential fold in the same-width unsigned type
//! (signed overflow is undefined behaviour in C++).

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, scan, wrapping_fold};
use anyhow::{Context, Result};

/// `WrappingLeftSequentialScanProd(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingLeftSequentialScanProd {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for WrappingLeftSequentialScanProd {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingLeftSequentialScanProd {
    fn label(&self) -> &str {
        "WrappingLeftSequentialScanProd"
    }
}

impl Bufferizable for WrappingLeftSequentialScanProd {}

impl ToDps for WrappingLeftSequentialScanProd {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(WrappingLeftSequentialScanProdDps {
            axis: self.axis,
        }))
    }
}

impl LayoutIrOp for WrappingLeftSequentialScanProd {}

/// Destination-passing form: `WrappingLeftSequentialScanProd(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WrappingLeftSequentialScanProdDps {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for WrappingLeftSequentialScanProdDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingLeftSequentialScanProdDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel_wire::<Self>())
    }

    fn label(&self) -> &str {
        "WrappingLeftSequentialScanProd"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for WrappingLeftSequentialScanProdDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for WrappingLeftSequentialScanProdDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for WrappingLeftSequentialScanProdDps {}

/// The CUDA lowering, colocated with its op.
impl KernelOp for WrappingLeftSequentialScanProdDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let axis = usize::try_from(self.axis).context("negative scan axis")?;
        scan(ctx, axis, "1", &wrapping_fold(ctx.operand_dtypes[0], "*")?)
    }
}

/// Matches `LayoutTensorOpWrappingLeftSequentialScanProd` and produces this runtime's
/// [`WrappingLeftSequentialScanProd`]. Metadata children: `axis` at child 1, `out_layout` at
/// child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct WrappingLeftSequentialScanProdMatcher;

impl OpMatcher for WrappingLeftSequentialScanProdMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpWrappingLeftSequentialScanProd"
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
        Box::new(WrappingLeftSequentialScanProd {
            axis: site.child_i64(1),
        })
    }
}
