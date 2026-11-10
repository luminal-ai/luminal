use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundDecimalsFunctional {
    pub decimals: i64,
}

impl OpSlotNames for RoundDecimalsFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for RoundDecimalsFunctional {
    fn label(&self) -> &str {
        "RoundDecimalsFunctionalGeneric"
    }
}

impl Bufferizable for RoundDecimalsFunctional {}

impl ToDps for RoundDecimalsFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(RoundDecimalsFunctionalDps {
            decimals: self.decimals,
        }))
    }
}

impl LayoutIrOp for RoundDecimalsFunctional {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundDecimalsFunctionalDps {
    pub decimals: i64,
}

impl OpSlotNames for RoundDecimalsFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for RoundDecimalsFunctionalDps {
    fn label(&self) -> &str {
        "RoundDecimalsFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1
    }
}

impl Bufferizable for RoundDecimalsFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for RoundDecimalsFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for RoundDecimalsFunctionalDps {}

#[derive(Debug, Clone, Copy, Default)]
pub struct RoundDecimalsFunctionalMatcher;

impl OpMatcher for RoundDecimalsFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpRoundDecimalsFunctionalGeneric"
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
        &[("decimals", 1), ("layout", 2)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(RoundDecimalsFunctional {
            decimals: site.child_i64(1),
        })
    }
}

use crate::typed_buffer::ReferenceKernelCtx;
use crate::typed_buffer::TypedBuffer;

pub(crate) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = op
        .as_any()
        .downcast_ref::<RoundDecimalsFunctionalDps>()
        .expect("round-decimals kernel received the wrong op type");
    let magnitude_f32 = 10.0_f32.powf(op.decimals.unsigned_abs() as f32);
    let magnitude_f64 = 10.0_f64.powf(op.decimals.unsigned_abs() as f64);
    let round_f32 = |x: f32| {
        if op.decimals < 0 {
            (x / magnitude_f32).round_ties_even() * magnitude_f32
        } else {
            (x * magnitude_f32).round_ties_even() / magnitude_f32
        }
    };
    let round_f64 = |x: f64| {
        if op.decimals < 0 {
            (x / magnitude_f64).round_ties_even() * magnitude_f64
        } else {
            (x * magnitude_f64).round_ties_even() / magnitude_f64
        }
    };

    match &ctx.operands[0] {
        TypedBuffer::F16(_) => {
            ctx.unary_elementwise_f16(|x| half::f16::from_f32(round_f32(x.to_f32())))
        }
        TypedBuffer::Bf16(input) => {
            let input = input.clone();
            let dest = ctx.dests[0].as_bf16_mut()?;
            anyhow::ensure!(input.len() == dest.len(), "unary kernel length mismatch");
            for (out, x) in dest.iter_mut().zip(input) {
                *out = half::bf16::from_f32(round_f32(x.to_f32()));
            }
            Ok(())
        }
        TypedBuffer::F32(_) | TypedBuffer::F64(_) => {
            ctx.unary_elementwise_typed(round_f32, round_f64)
        }
        other => anyhow::bail!(
            "round.decimals requires a floating operand, found {}",
            other.type_name()
        ),
    }
}
