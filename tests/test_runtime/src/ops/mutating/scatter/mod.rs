//! Mutating spelling of `scatter`.
//!
//! Rehomed from `luminal_reference::ops::scatter` — the reference runtime is
//! functional and out-of-place, and its kernel table has no row for this
//! spelling, so it was registered-but-never-selectable there. The op and
//! its egglog rewrites move together, unchanged.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

// Copied, not imported: this is a generic LayoutTensorCons spine walk over
// ExtractionSite, private to the reference scatter module. The reference
// gather keeps its own copy of the same walk, so a third is the tree's
// existing convention -- and it keeps this crate from reaching into
// another crate's internals for one helper.
/// Walk the LayoutTensorCons spine at `child` counting elements — the
/// shared rank reader for both scatter matchers (same class-resolving walk
/// as gather's; see the OpMatcher validity contract for the panics).
fn coordinate_rank(site: &ExtractionSite<'_>, child: usize) -> usize {
    let mut rank = 0usize;
    let mut class = site.child_class(child);
    loop {
        let spine = site
            .egraph
            .nodes
            .values()
            .find(|node| {
                node.eclass == class
                    && (node.op == "LayoutTensorCons" || node.op == "LayoutTensorNil")
            })
            .unwrap_or_else(|| {
                panic!(
                    "schema drift: coordinate-list class {class} under enode {} has no \
                     LayoutTensorCons/LayoutTensorNil constructor",
                    site.node_id
                )
            });
        if spine.op == "LayoutTensorNil" {
            break;
        }
        rank += 1;
        let tail_id = spine.children.get(1).unwrap_or_else(|| {
            panic!("schema drift: a LayoutTensorCons in class {class} has no tail child")
        });
        class = site
            .egraph
            .nodes
            .get(tail_id)
            .unwrap_or_else(|| panic!("dangling list tail node {tail_id}"))
            .eclass
            .clone();
    }
    rank
}

/// `ScatterMutatingGeneric(init: read+write, src: read, coord0..: read) -> out`
///
/// Mutating form: the kernel reads and overwrites ONE storage — init's.
/// Matched in egglog only when the output layout IS init's layout (the
/// precondition, discharged at match time). The Must tie is relocatable as
/// always: a rejected mutation copies init into the tied result's fresh
/// buffer and mutates there. No write-map injectivity gate exists or
/// can — scatter's writes are data-dependent, and duplicate coordinates
/// are UB by ruling. No reference kernel exists for this form (the
/// reference runtime is out-of-place only, ruling 2026-08-05/06).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScatterMutating {
    pub rank: usize,
}

impl OpSlotNames for ScatterMutating {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "init".to_string(),
            1 => "src".to_string(),
            n if n < 2 + self.rank => format!("coord{}", n - 2),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ScatterMutating {
    fn label(&self) -> &str {
        "ScatterMutatingGeneric"
    }
}

impl Bufferizable for ScatterMutating {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for ScatterMutating {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already destination-form: the destination IS operand 0
    }
}

impl LayoutIrOp for ScatterMutating {}

/// Matches `LayoutTensorOpScatterMutatingGeneric` enodes and produces
/// [`ScatterMutating`] instances. No metadata children: the output layout
/// IS init's, by the match rule's precondition.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScatterMutatingMatcher;

impl OpMatcher for ScatterMutatingMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpScatterMutatingGeneric"
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        vec![
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_mutating_constructor.egg"),
            },
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_mutating.egg"),
            },
        ]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(ScatterMutating {
            rank: coordinate_rank(site, 2),
        })
    }
}
