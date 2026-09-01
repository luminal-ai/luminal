//! Reinterpret the parent's storage through a composed offset function: the
//! index map, output shape, and composed layout are op metadata, not
//! operands. No bytes move — the planner binds the result into its parent's
//! buffer (the Must tie) and lowering folds the op to a producer redirect.

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `IndexMapApplyViewGeneric(input) -> out`
///
/// The metadata-view form of index-map application. The egglog match admits
/// it only where the output layout IS the composed layout (the parent's
/// offset function precomposed with the index map), so the result is the
/// operand's own bytes by construction: the operand is never read (no bytes
/// observed) and the result is never written (no bytes produced). A rejected
/// tie is repairable — a view over a copy of the parent's buffer is a
/// faithful lowering — so Must is a requirement on WHERE the shared storage
/// lives, never a hard error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexMapApplyView;

impl OpSlotNames for IndexMapApplyView {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for IndexMapApplyView {
    fn label(&self) -> &str {
        "IndexMapApplyViewGeneric"
    }

    fn operand_reads_memory(&self, _operand: usize) -> bool {
        false // metadata op: no bytes observed
    }
    fn result_writes_memory(&self, _result: usize) -> bool {
        false // metadata op: no bytes produced
    }
}

impl Bufferizable for IndexMapApplyView {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for IndexMapApplyView {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // nothing is written: there is no destination to pass
    }
}

impl LayoutIrOp for IndexMapApplyView {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpIndexMapApplyViewGeneric` enodes and produces
/// [`IndexMapApplyView`] instances. Metadata children: `index_map` at child 1, `shape` at child 2, `out_layout` at child 3.
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexMapApplyViewMatcher;

impl OpMatcher for IndexMapApplyViewMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpIndexMapApplyViewGeneric"
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
        &[("index_map", 1), ("shape", 2), ("out_layout", 3)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(IndexMapApplyView)
    }
}
