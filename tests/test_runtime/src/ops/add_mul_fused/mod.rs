//! Fused elementwise add + multiply over one operand pair, two results.
//!
//! REHOMED (ruling 2026-08-13): the reference runtime implements only plain
//! spellings, so the AddMulFused family was deleted from
//! the reference registry (now `luminal_reference::ops`) at commit aff22598 and restored here verbatim
//! (paths adjusted `crate::` → `luminal::`; the `.egg` texts live under this
//! crate's `src/egg/`). It exists to exercise runtime MACHINERY — fused
//! multi-output claiming, genome dedup, waste destinations — not to compute.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `AddMulFusedGeneric(lhs, rhs) -> (add_out, mul_out)`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, both results freshly allocated). Elementwise: element
/// `i` of each input is read before element `i` of either output is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddMulFused;

impl OpSlotNames for AddMulFused {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for AddMulFused {
    fn label(&self) -> &str {
        "AddMulFusedGeneric"
    }
}

impl Bufferizable for AddMulFused {}

impl ToDps for AddMulFused {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(AddMulFusedDps))
    }
}

impl LayoutIrOp for AddMulFused {}

/// Destination-passing form of [`AddMulFused`] — two results, so two
/// destinations, each tied to its own result, spelled slot by slot:
///
/// ```text
/// AddMulFusedGeneric(lhs: read, rhs: read,
///                    dest0: write-only ↔ out0 (add),
///                    dest1: write-only ↔ out1 (mul)) -> (out0, out1)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddMulFusedDps;

impl OpSlotNames for AddMulFusedDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            3 => "dest1".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for AddMulFusedDps {
    fn label(&self) -> &str {
        "AddMulFusedGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // lhs
            1 => true,  // rhs
            2 => false, // dest0: write-only destination
            3 => false, // dest1: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for AddMulFusedDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![
            AliasInfo {
                operand: 2,
                result: 0,
                sharing: Sharing::Must,
            },
            AliasInfo {
                operand: 3,
                result: 1,
                sharing: Sharing::Must,
            },
        ]
    }
}

impl ToDps for AddMulFusedDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for AddMulFusedDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpAddMulFusedGeneric` enodes and produces
/// [`AddMulFused`] instances. Metadata children: `add_out_layout` at child 2, `mul_out_layout` at child 3.
#[derive(Debug, Clone, Copy, Default)]
pub struct AddMulFusedMatcher;

impl OpMatcher for AddMulFusedMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpAddMulFusedGeneric"
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
        &[("add_out_layout", 2), ("mul_out_layout", 3)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(AddMulFused)
    }
}
