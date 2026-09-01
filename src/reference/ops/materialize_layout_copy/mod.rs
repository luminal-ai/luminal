//! A materialize/copy between two distinct layouts of the same logical value,
//! chosen by extraction. An ordinary op (no special-casing): its cost falls
//! out of the tensor sizes it reads and writes.

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `CopyGeneric(input) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated). Elementwise in the
/// LOGICAL index; physical safety is enforced by the engine's
/// layout-equivalence gate. The Rust type keeps its descriptive name; the
/// label follows the label policy (egglog: `LayoutTensorOpCopyGeneric`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterializeLayoutCopy;

impl OpSlotNames for MaterializeLayoutCopy {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MaterializeLayoutCopy {
    fn label(&self) -> &str {
        "CopyGeneric"
    }
}

impl Bufferizable for MaterializeLayoutCopy {}

impl ToDps for MaterializeLayoutCopy {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(MaterializeLayoutCopyDps))
    }
}

impl LayoutIrOp for MaterializeLayoutCopy {}

/// Destination-passing form of [`MaterializeLayoutCopy`], signature spelled
/// slot by slot:
///
/// ```text
/// CopyGeneric(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterializeLayoutCopyDps;

impl OpSlotNames for MaterializeLayoutCopyDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MaterializeLayoutCopyDps {
    fn label(&self) -> &str {
        "CopyGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // input
            1 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for MaterializeLayoutCopyDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for MaterializeLayoutCopyDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for MaterializeLayoutCopyDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpCopyGeneric` enodes and produces
/// [`MaterializeLayoutCopy`] instances. Metadata children: `layout` at child 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct MaterializeLayoutCopyMatcher;

impl OpMatcher for MaterializeLayoutCopyMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpCopyGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("layout", 1)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(MaterializeLayoutCopy)
    }
}

// ---------------------------------------------------------------------------
// ---- kernel ----
// Reference-runtime execution for this op, dispatched by TypeId from the
// label->fn table in `crate::reference::kernels` (op-folder ruling
// 2026-08-13: everything about an op lives in the op's folder).
// ---------------------------------------------------------------------------

use crate::buffer_tensor_ir::ReferenceKernelCtx;
use crate::reference::kernels::move_gathered;

/// Dense same-geometry copy (CopyGeneric / MaterializeLayoutCopy). The
/// reference runtime holds every buffer dense row-major (no view reads in
/// its allow list), so materializing a copy IS an element copy — but only
/// under identical geometry, which is checked loudly rather than assumed.
pub(in crate::reference) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        ctx.operand_dims[0] == ctx.operand_dims[1],
        "copy kernel: input geometry {:?} vs dest geometry {:?} — a shape-changing \
         copy is not a dense copy and has no reference lowering",
        ctx.operand_dims[0],
        ctx.operand_dims[1]
    );
    anyhow::ensure!(
        ctx.operands[0].len() == ctx.dests[0].len(),
        "copy kernel length mismatch"
    );
    let index_of: Vec<usize> = (0..ctx.dests[0].len()).collect();
    let source = ctx.operands[0].clone();
    move_gathered(&source, &mut ctx.dests[0], &index_of)
}
