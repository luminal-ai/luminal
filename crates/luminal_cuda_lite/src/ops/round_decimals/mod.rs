use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, cuda_f64_literal, cuda_type, unary};
use anyhow::{Result, bail};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel_wire::<Self>())
    }

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

impl KernelOp for RoundDecimalsFunctionalDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let dtype = ctx.operand_dtypes[0];
        let ty = cuda_type(dtype)?;
        let magnitude = 10.0_f64.powf(self.decimals.unsigned_abs() as f64);
        let scale = cuda_f64_literal(magnitude);
        let (round, argument, rounded) = match dtype {
            PlanDtype::F16 | PlanDtype::Bf16 => (
                "rintf",
                if self.decimals < 0 {
                    "(float)(x / scale)"
                } else {
                    "(float)(x * scale)"
                },
                format!("({ty})"),
            ),
            PlanDtype::F32 => (
                "rintf",
                if self.decimals < 0 {
                    "x / scale"
                } else {
                    "x * scale"
                },
                String::new(),
            ),
            PlanDtype::F64 => (
                "rint",
                if self.decimals < 0 {
                    "x / scale"
                } else {
                    "x * scale"
                },
                String::new(),
            ),
            other => bail!("cuda-lite round.decimals has no {other:?} arm"),
        };
        let combine = if self.decimals < 0 { "*" } else { "/" };
        let helper = format!(
            "__device__ __forceinline__ {ty} luminal_round_decimals({ty} x) {{\n\
             \x20   volatile {ty} scale_storage = ({ty})({scale});\n\
             \x20   {ty} scale = scale_storage;\n\
             \x20   {ty} rounded = {rounded}{round}({argument});\n\
             \x20   return ({ty})(rounded {combine} scale);\n\
             }}\n"
        );
        let mut kernels = unary(ctx, "luminal_round_decimals(a[i])")?;
        for kernel in &mut kernels {
            kernel.source.insert_str(0, &helper);
        }
        Ok(kernels)
    }
}

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
        &[("decimals", 1), ("out_layout", 2)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(RoundDecimalsFunctional {
            decimals: site.child_i64(1),
        })
    }
}
