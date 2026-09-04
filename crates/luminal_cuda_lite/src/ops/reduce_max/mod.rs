//! Max reduction along one axis (the axis is op metadata, not an
//! operand) — CUDA-lite's OWN op (ruling 2026-08-17: every runtime
//! owns its executable ops; the shared crate supplies only the IR
//! traits). Same egglog constructor and label as the reference
//! runtime's reduce_max — assemblies are per-runtime, labels are IR
//! identity — but the structs, matcher, snippets, and codegen all
//! live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelSource, reduce};
use anyhow::{Context, Result, bail};

/// `ReduceMaxGeneric(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceMax {
    /// Reduction axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for ReduceMax {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ReduceMax {
    fn label(&self) -> &str {
        "ReduceMaxGeneric"
    }
}

impl Bufferizable for ReduceMax {}

impl ToDps for ReduceMax {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(ReduceMaxDps { axis: self.axis }))
    }
}

impl LayoutIrOp for ReduceMax {}

/// Destination-passing form: `ReduceMax(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceMaxDps {
    /// Reduction axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for ReduceMaxDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ReduceMaxDps {
    fn label(&self) -> &str {
        "ReduceMaxGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for ReduceMaxDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for ReduceMaxDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for ReduceMaxDps {}

/// The CUDA lowering, colocated with its op.
pub(crate) fn codegen(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    let Some(r) = op.as_any().downcast_ref::<ReduceMaxDps>() else {
        bail!("reduce_max codegen reached with a non-ReduceMax op");
    };
    let axis = usize::try_from(r.axis).context("negative reduce axis")?;
    // NVRTC compiles the program with no math headers, so the INFINITY macro
    // does not exist there; this intrinsic is IEEE -inf by bit pattern.
    reduce(ctx, axis, "__int_as_float(0xff800000)", "v > acc ? v : acc")
}

/// Matches `LayoutTensorOpReduceMaxGeneric` and produces this
/// runtime's [`ReduceMax`]. Metadata children: `axis` at child 1,
/// `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReduceMaxMatcher;

impl OpMatcher for ReduceMaxMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpReduceMaxGeneric"
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
        Box::new(ReduceMax {
            axis: site.child_i64(1),
        })
    }
}
