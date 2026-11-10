use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HypotFunctional;

impl OpSlotNames for HypotFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for HypotFunctional {
    fn label(&self) -> &str {
        "HypotFunctionalGeneric"
    }
}

impl Bufferizable for HypotFunctional {}

impl ToDps for HypotFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(HypotFunctionalDps))
    }
}

impl LayoutIrOp for HypotFunctional {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HypotFunctionalDps;

impl OpSlotNames for HypotFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for HypotFunctionalDps {
    fn label(&self) -> &str {
        "HypotFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand < 2
    }
}

impl Bufferizable for HypotFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for HypotFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for HypotFunctionalDps {}

#[derive(Debug, Clone, Copy, Default)]
pub struct HypotFunctionalMatcher;

impl OpMatcher for HypotFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpHypotFunctionalGeneric"
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
        Box::new(HypotFunctional)
    }
}

use crate::typed_buffer::{ReferenceKernelCtx, TypedBuffer};

pub(crate) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    match &ctx.operands[0] {
        TypedBuffer::F16(_) => {
            ctx.binary_elementwise_f16(|a, b| Ok(half::f16::from_f32(a.to_f32().hypot(b.to_f32()))))
        }
        TypedBuffer::Bf16(lhs) => {
            let lhs = lhs.clone();
            let rhs = ctx.operands[1].as_bf16()?.clone();
            let dest = ctx.dests[0].as_bf16_mut()?;
            anyhow::ensure!(
                lhs.len() == rhs.len() && lhs.len() == dest.len(),
                "binary kernel length mismatch"
            );
            for (index, out) in dest.iter_mut().enumerate() {
                *out = half::bf16::from_f32(lhs[index].to_f32().hypot(rhs[index].to_f32()));
            }
            Ok(())
        }
        TypedBuffer::F32(_) => ctx.binary_elementwise(f32::hypot),
        TypedBuffer::F64(_) => ctx.binary_elementwise_f64(|a, b| Ok(a.hypot(b))),
        other => anyhow::bail!("hypot has no {} arm", other.type_name()),
    }
}
