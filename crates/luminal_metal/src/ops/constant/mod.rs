use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, metal_f64_literal, metal_type, numel};
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constant {
    pub value: f64,
}

impl OpSlotNames for Constant {}

impl BufferTensorIrOp for Constant {
    fn label(&self) -> &str {
        "ConstantGeneric"
    }
}

impl Bufferizable for Constant {}

impl ToDps for Constant {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(ConstantDps { value: self.value }))
    }
}

impl LayoutIrOp for Constant {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantDps {
    pub value: f64,
}

impl OpSlotNames for ConstantDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ConstantDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::MetalOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "ConstantGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 0 // dest0 is write-only; there are no other operands
    }
}

impl Bufferizable for ConstantDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for ConstantDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for ConstantDps {}

impl KernelOp for ConstantDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let to = metal_type(ctx.dest_dtypes[0])?;
        let n = numel(&ctx.dest_dims[0]);
        let value = metal_f64_literal(self.value);
        let source = format!(
            r#"kernel void k(device {to}* out, device const long* params, uint gid [[thread_position_in_grid]]) {{
    const ulong n = {n};
    ulong i = gid;
    if (i < n) out[i] = ({to}){value};
}}"#
        );
        Ok(vec![KernelSource::plain(source, n)])
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ConstantMatcher;

impl OpMatcher for ConstantMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpConstantGeneric"
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
        &[("value", 0), ("out_layout", 1)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(Constant {
            value: site.child_f64(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::dtype::PlanDtype;

    fn f32_dest_ctx() -> CodegenCtx {
        CodegenCtx {
            operand_dims: vec![],
            operand_dtypes: vec![],
            dest_dims: vec![vec![4usize.into()]],
            dest_dtypes: vec![PlanDtype::F32],
            operand_layouts: vec![],
        }
    }

    fn source_for(value: f64) -> String {
        let op = ConstantDps { value };
        let launches = op
            .codegen(&f32_dest_ctx())
            .expect("constant codegen succeeds");
        assert_eq!(launches.len(), 1, "constant is a single-launch op");
        launches.into_iter().next().unwrap().source
    }

    fn longest_digit_run(source: &str) -> usize {
        let (mut best, mut run) = (0usize, 0usize);
        for c in source.chars() {
            if c.is_ascii_digit() {
                run += 1;
                best = best.max(run);
            } else {
                run = 0;
            }
        }
        best
    }

    #[test]
    fn metal_f64_literal_is_a_valid_c_token_for_extreme_and_non_finite_values() {
        assert_eq!(metal_f64_literal(3.0), "3e0");
        assert_eq!(metal_f64_literal(f32::MIN as f64), "-3.4028234663852886e38");
        assert_eq!(metal_f64_literal(1e30), "1e30");
        assert_eq!(metal_f64_literal(-0.0), "-0e0");
        assert_eq!(
            metal_f64_literal(f64::NEG_INFINITY),
            "as_type<float>(0xff800000u)"
        );
        assert_eq!(
            metal_f64_literal(f64::INFINITY),
            "as_type<float>(0x7f800000u)"
        );
        assert_eq!(metal_f64_literal(f64::NAN), "as_type<float>(0x7fc00000u)");
    }

    #[test]
    fn constant_literal_is_a_valid_c_token_for_extreme_and_non_finite_values() {
        let extreme = source_for(f32::MIN as f64);
        assert!(
            extreme.contains("out[i] = (float)-3.4028234663852886e38;"),
            "extreme finite constant lost its exponent form:\n{extreme}"
        );

        let neg_inf = source_for(f64::NEG_INFINITY);
        assert!(
            neg_inf.contains("out[i] = (float)as_type<float>(0xff800000u);"),
            "-inf constant did not lower to its bit pattern:\n{neg_inf}"
        );

        for source in [&extreme, &neg_inf] {
            assert!(
                !source.contains("inf"),
                "Rust's `inf` spelling reached the kernel source:\n{source}"
            );
            assert!(
                !source.contains("NaN"),
                "Rust's `NaN` spelling reached the kernel source:\n{source}"
            );
            assert!(
                longest_digit_run(source) < 20,
                "a bare {}-digit run reached the kernel source (C reads it as an \
                 out-of-range integer literal):\n{source}",
                longest_digit_run(source)
            );
        }
    }
}
