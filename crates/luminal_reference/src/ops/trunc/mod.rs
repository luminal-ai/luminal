//! Elementwise truncation toward zero: `out[i] = trunc(in[i])`. Functional
//! (out-of-place) with a DPS form; NaN and ±inf propagate in the element
//! type.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `TruncFunctionalGeneric(input) -> out`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncFunctional;

impl OpSlotNames for TruncFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for TruncFunctional {
    fn label(&self) -> &str {
        "TruncFunctionalGeneric"
    }
}

impl Bufferizable for TruncFunctional {}

impl ToDps for TruncFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(TruncFunctionalDps))
    }
}

impl LayoutIrOp for TruncFunctional {}

/// Destination-passing form of [`TruncFunctional`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncFunctionalDps;

impl OpSlotNames for TruncFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for TruncFunctionalDps {
    fn label(&self) -> &str {
        "TruncFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,
            1 => false,
            _ => true,
        }
    }
}

impl Bufferizable for TruncFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for TruncFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for TruncFunctionalDps {}

/// Matches `LayoutTensorOpTruncFunctionalGeneric` enodes. Metadata slot:
/// `layout` at child 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct TruncFunctionalMatcher;

impl OpMatcher for TruncFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpTruncFunctionalGeneric"
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
        Box::new(TruncFunctional)
    }
}

use crate::typed_buffer::ReferenceKernelCtx;

pub(crate) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    ctx.unary_elementwise_typed(|x| x.trunc(), |x| x.trunc())
}
