//! FORKED from `luminal_reference::ops::mul` — the TestRuntime owns its
//! whole op set outright and depends on no other runtime.
//!
//! The KERNEL is deliberately not carried over. This runtime is
//! plan-level: it asserts on `ExtractedGraph`s and `BufferIrGraph`s and
//! never executes, so a kernel here would be dead code demanding a
//! dispatch table to sit in. What it needs is the matcher, the instance
//! and the DPS form — the declarations the bufferizer reads.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `MulFunctionalGeneric(lhs, rhs) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated). Elementwise: element
/// `i` of each input is read before element `i` of `out` is written (op-level
/// all-pairs claim — documented by `bufferizes_to_elementwise_access`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulFunctional;

impl OpSlotNames for MulFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MulFunctional {
    fn label(&self) -> &str {
        "MulFunctionalGeneric"
    }
}

impl Bufferizable for MulFunctional {}

impl ToDps for MulFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(MulFunctionalDps))
    }
}

impl LayoutIrOp for MulFunctional {}

/// Destination-passing form of [`MulFunctional`], signature spelled slot by slot:
///
/// ```text
/// MulGeneric(lhs: read, rhs: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulFunctionalDps;

impl OpSlotNames for MulFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MulFunctionalDps {
    fn label(&self) -> &str {
        "MulFunctionalGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // lhs
            1 => true,  // rhs
            2 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for MulFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for MulFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for MulFunctionalDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpMulFunctionalGeneric` enodes and produces
/// [`MulFunctional`] instances. Metadata children: `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct MulFunctionalMatcher;

impl OpMatcher for MulFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpMulFunctionalGeneric"
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
        &[("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(MulFunctional)
    }
}
