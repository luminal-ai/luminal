//! Elementwise ceil: `out[i] = ceil(in[i])`. Functional (out-of-place)
//! with a DPS form; NaN and ±inf propagate in the element type.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `CeilFunctionalGeneric(input) -> out`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CeilFunctional;

impl OpSlotNames for CeilFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for CeilFunctional {
    fn label(&self) -> &str {
        "CeilFunctionalGeneric"
    }
}

impl Bufferizable for CeilFunctional {}

impl ToDps for CeilFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(CeilFunctionalDps))
    }
}

impl LayoutIrOp for CeilFunctional {}

/// Destination-passing form of [`CeilFunctional`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CeilFunctionalDps;

impl OpSlotNames for CeilFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for CeilFunctionalDps {
    fn label(&self) -> &str {
        "CeilFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,
            1 => false,
            _ => true,
        }
    }
}

impl Bufferizable for CeilFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for CeilFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for CeilFunctionalDps {}

/// Matches `LayoutTensorOpCeilFunctionalGeneric` enodes. Metadata slot:
/// `layout` at child 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct CeilFunctionalMatcher;

impl OpMatcher for CeilFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpCeilFunctionalGeneric"
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
        Box::new(CeilFunctional)
    }
}

use crate::typed_buffer::ReferenceKernelCtx;

pub(crate) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    ctx.unary_elementwise_typed(|x| x.ceil(), |x| x.ceil())
}
