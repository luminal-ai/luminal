//! Elementwise round-half-to-even: `out[i] = round_ties_even(in[i])`.
//! Functional (out-of-place) with a DPS form; NaN and ±inf propagate in
//! the element type.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `RoundFunctionalGeneric(input) -> out`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundFunctional;

impl OpSlotNames for RoundFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for RoundFunctional {
    fn label(&self) -> &str {
        "RoundFunctionalGeneric"
    }
}

impl Bufferizable for RoundFunctional {}

impl ToDps for RoundFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(RoundFunctionalDps))
    }
}

impl LayoutIrOp for RoundFunctional {}

/// Destination-passing form of [`RoundFunctional`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundFunctionalDps;

impl OpSlotNames for RoundFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for RoundFunctionalDps {
    fn label(&self) -> &str {
        "RoundFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,
            1 => false,
            _ => true,
        }
    }
}

impl Bufferizable for RoundFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for RoundFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for RoundFunctionalDps {}

/// Matches `LayoutTensorOpRoundFunctionalGeneric` enodes. Metadata slot:
/// `layout` at child 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoundFunctionalMatcher;

impl OpMatcher for RoundFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpRoundFunctionalGeneric"
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
        Box::new(RoundFunctional)
    }
}

use crate::typed_buffer::ReferenceKernelCtx;

pub(crate) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    ctx.unary_elementwise_typed(|x| x.round_ties_even(), |x| x.round_ties_even())
}
