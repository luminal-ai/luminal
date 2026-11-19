//! Mutating cosine for plan-level tests.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `CosMutatingGeneric(x: read+write) -> out`
///
/// Reads and overwrites one buffer. The matcher requires identical input
/// and output layouts and an injective output. If the input's storage is
/// unavailable for mutation, the bufferizer copies it to the destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CosMutating;

impl OpSlotNames for CosMutating {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for CosMutating {
    fn label(&self) -> &str {
        "CosMutatingGeneric"
    }
}

impl Bufferizable for CosMutating {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for CosMutating {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already destination-form: the destination IS operand 0
    }
}

impl LayoutIrOp for CosMutating {}

/// Matches `LayoutTensorOpCosMutatingGeneric` enodes and produces
/// [`CosMutating`] instances. No metadata children: the output layout IS
/// the mutated operand's, by the match rule's precondition.
#[derive(Debug, Clone, Copy, Default)]
pub struct CosMutatingMatcher;

impl OpMatcher for CosMutatingMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpCosMutatingGeneric"
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

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(CosMutating)
    }
}
