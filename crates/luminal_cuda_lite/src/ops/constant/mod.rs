//! The scalar constant (a zero-input source writing one value into a
//! rank-0 output) — CUDA-lite's OWN op (ruling 2026-08-17: every
//! runtime owns its executable ops; the shared crate supplies only the
//! IR traits). Same egglog constructor and label as the reference
//! runtime's constant — assemblies are per-runtime, labels are IR
//! identity — but the structs, matcher, snippets, and codegen all
//! live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelSource, cuda_f64_literal, cuda_type, numel};
use anyhow::{Result, bail};

/// `ConstantGeneric() -> out` — pure dataflow source form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constant {
    /// The literal value (the term's f64 metadata).
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

/// Destination-passing form: `Constant(dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantDps {
    /// The literal value (the term's f64 metadata).
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

/// The CUDA lowering, colocated with its op.
pub(crate) fn codegen(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    let Some(constant) = op.as_any().downcast_ref::<ConstantDps>() else {
        bail!("constant codegen reached with a non-Constant op");
    };
    // Dest-only signature: the check that used to stand here was the
    // write fence — see `kernels::CodegenCtx::from_descriptors`.
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let n = numel(&ctx.dest_dims[0]);
    let value = cuda_f64_literal(constant.value);
    let source = format!(
        r#"extern "C" __global__ void k({to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ({to}){value};
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

/// Matches `LayoutTensorOpConstantGeneric` and produces this runtime's
/// [`Constant`]. Metadata children: `value` at child 0, `out_layout`
/// at child 1 — all metadata, no tensor operands.
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

    /// The dest-only geometry a constant lowers against: no operands,
    /// one F32 destination.
    fn f32_dest_ctx() -> CodegenCtx {
        CodegenCtx {
            operand_dims: vec![],
            operand_dtypes: vec![],
            dest_dims: vec![vec![4]],
            dest_dtypes: vec![PlanDtype::F32],
            operand_layouts: vec![],
        }
    }

    fn source_for(value: f64) -> String {
        let op = ConstantDps { value };
        let launches = codegen(&op, &f32_dest_ctx()).expect("constant codegen succeeds");
        assert_eq!(launches.len(), 1, "constant is a single-launch op");
        launches.into_iter().next().unwrap().source
    }

    /// The longest run of consecutive ASCII digits — a decimal-point-free,
    /// exponent-free digit string is exactly the shape C misreads as an
    /// out-of-range integer literal.
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
    fn cuda_f64_literal_is_a_valid_c_token_for_extreme_and_non_finite_values() {
        // Finite: `{:e}` is the shortest round-trip form and always
        // carries an exponent, so it is always a `double` literal.
        assert_eq!(cuda_f64_literal(3.0), "3e0");
        assert_eq!(cuda_f64_literal(f32::MIN as f64), "-3.4028234663852886e38");
        assert_eq!(cuda_f64_literal(1e30), "1e30");
        assert_eq!(cuda_f64_literal(-0.0), "-0e0");
        // Non-finite: bit patterns, because NVRTC has no math headers
        // and so no `INFINITY`/`NAN` macros.
        assert_eq!(
            cuda_f64_literal(f64::NEG_INFINITY),
            "__uint_as_float(0xff800000u)"
        );
        assert_eq!(
            cuda_f64_literal(f64::INFINITY),
            "__uint_as_float(0x7f800000u)"
        );
        assert_eq!(cuda_f64_literal(f64::NAN), "__uint_as_float(0x7fc00000u)");
    }

    /// The defect this closes: `Display` formatted `f32::MIN as f64` as
    /// `-340282346638528860000000000000000000000` (an integer literal
    /// too large for any C type) and `-inf` as the non-token `-inf`.
    /// The frontend reaches both — `cummax` seeds with `f32::MIN`
    /// (`src/frontend/unary.rs`), attention masks fill with `-inf`.
    #[test]
    fn constant_literal_is_a_valid_c_token_for_extreme_and_non_finite_values() {
        let extreme = source_for(f32::MIN as f64);
        assert!(
            extreme.contains("out[i] = (float)-3.4028234663852886e38;"),
            "extreme finite constant lost its exponent form:\n{extreme}"
        );

        let neg_inf = source_for(f64::NEG_INFINITY);
        assert!(
            neg_inf.contains("out[i] = (float)__uint_as_float(0xff800000u);"),
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

    /// An ordinary value keeps the kernel text it always had, save for
    /// the literal itself now carrying an exponent.
    #[test]
    fn constant_kernel_text_is_otherwise_unchanged() {
        assert_eq!(
            source_for(3.0),
            r#"extern "C" __global__ void k(float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)3e0;
}"#
        );
    }
}
