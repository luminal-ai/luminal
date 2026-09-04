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

use crate::kernels::{CodegenCtx, KernelSource, cuda_type, unary};
use anyhow::Result;

/// `CastGeneric(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cast;

impl OpSlotNames for Cast {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for Cast {
    fn label(&self) -> &str {
        "CastGeneric"
    }
}

impl Bufferizable for Cast {}

impl ToDps for Cast {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(CastDps))
    }
}

impl LayoutIrOp for Cast {}

/// Destination-passing form: `Cast(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CastDps;

impl OpSlotNames for CastDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for CastDps {
    fn label(&self) -> &str {
        "CastGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for CastDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for CastDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for CastDps {}

/// The CUDA lowering, colocated with its op. The conversion is driven
/// by the buffer dtypes the plan annotated — the op carries no dtype
/// field of its own.
pub(crate) fn codegen(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    let to = cuda_type(ctx.dest_dtypes[0])?;
    unary(ctx, &format!("({to})a[i]"))
}

/// Matches `LayoutTensorOpCastGeneric` and produces this runtime's
/// [`Cast`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CastMatcher;

impl OpMatcher for CastMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpCastGeneric"
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
        Box::new(Cast)
    }
}
