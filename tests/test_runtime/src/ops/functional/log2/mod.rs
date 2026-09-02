//! FORKED from `luminal_reference::ops::log2` — the TestRuntime owns its
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

/// `Log2FunctionalGeneric(input) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated). Elementwise: element
/// `i` of `input` is read before element `i` of `out` is written (op-level
/// all-pairs claim — documented by `bufferizes_to_elementwise_access`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Log2Functional;

impl OpSlotNames for Log2Functional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for Log2Functional {
    fn label(&self) -> &str {
        "Log2FunctionalGeneric"
    }
}

impl Bufferizable for Log2Functional {}

impl ToDps for Log2Functional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(Log2FunctionalDps))
    }
}

impl LayoutIrOp for Log2Functional {}

/// Destination-passing form of [`Log2Functional`], signature spelled slot by slot:
///
/// ```text
/// Log2Generic(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Log2FunctionalDps;

impl OpSlotNames for Log2FunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for Log2FunctionalDps {
    fn label(&self) -> &str {
        "Log2FunctionalGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // input
            1 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for Log2FunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for Log2FunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for Log2FunctionalDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpLog2FunctionalGeneric` enodes and produces
/// [`Log2Functional`] instances. Metadata children: `layout` at child 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct Log2FunctionalMatcher;

impl OpMatcher for Log2FunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpLog2FunctionalGeneric"
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
        &[("layout", 1)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(Log2Functional)
    }
}
