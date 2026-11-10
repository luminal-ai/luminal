//! Every direct logical math kernel names the CUDA function and conversions
//! for its operands' dtype: F32 calls the `f` function, F64 calls the double
//! function, and F16/BF16 explicitly use F32 opmath before rounding to the
//! declared destination. Codegen runs on any host, so this needs no device.

use crate::kernels::{CodegenCtx, KernelOp};
use luminal::dtype::PlanDtype;
use luminal::layouts::{
    BitWidthTerm, DecodedLayout, IntExprTerm, RightMajorContiguousElementLayout, ShapeTerm,
};

fn dense_layout(dtype: PlanDtype) -> DecodedLayout {
    let width = match dtype {
        PlanDtype::F64 | PlanDtype::Int64 => 64,
        PlanDtype::F16 | PlanDtype::Bf16 => 16,
        _ => 32,
    };
    DecodedLayout::of(
        RightMajorContiguousElementLayout {
            shape: ShapeTerm(vec![IntExprTerm::Lit(4)]),
            width: BitWidthTerm(width),
        },
        Some(dtype),
    )
}

/// A dense `[4]` geometry: `arity` value operands of `dtype`, then the
/// destination-passing slot, and one `dest` result.
fn dense_ctx(arity: usize, dtype: PlanDtype, dest: PlanDtype) -> CodegenCtx {
    let extents = vec![crate::symbolic::Expr(IntExprTerm::Lit(4))];
    let mut operand_dtypes = vec![dtype; arity];
    operand_dtypes.push(dest);
    let mut operand_layouts = vec![dense_layout(dtype); arity];
    operand_layouts.push(dense_layout(dest));
    CodegenCtx {
        operand_dims: vec![extents.clone(); arity + 1],
        operand_dtypes,
        dest_dims: vec![extents],
        dest_dtypes: vec![dest],
        operand_layouts,
        dest_layouts: vec![dense_layout(dest)],
    }
}

/// One math kernel: the op, its operand count, the float and double
/// function names, and the destination dtype for an operand dtype.
struct MathKernel {
    op: Box<dyn KernelOp>,
    arity: usize,
    f32_name: &'static str,
    f64_name: &'static str,
    dest: fn(PlanDtype) -> PlanDtype,
}

fn same(dtype: PlanDtype) -> PlanDtype {
    dtype
}

fn to_int64(_: PlanDtype) -> PlanDtype {
    PlanDtype::Int64
}

fn math_kernels() -> Vec<MathKernel> {
    let unary = |op: Box<dyn KernelOp>, f32_name, f64_name| MathKernel {
        op,
        arity: 1,
        f32_name,
        f64_name,
        dest: same,
    };
    vec![
        unary(Box::new(super::sqrt::SqrtFunctionalDps), "sqrtf", "sqrt"),
        unary(Box::new(super::exp::ExpFunctionalDps), "expf", "exp"),
        unary(Box::new(super::exp2::Exp2FunctionalDps), "exp2f", "exp2"),
        unary(Box::new(super::log2::Log2FunctionalDps), "log2f", "log2"),
        unary(Box::new(super::sin::SinFunctionalDps), "sinf", "sin"),
        unary(Box::new(super::tanh::TanhFunctionalDps), "tanhf", "tanh"),
        unary(
            Box::new(super::floor::FloorFunctionalDps),
            "floorf",
            "floor",
        ),
        unary(Box::new(super::ceil::CeilFunctionalDps), "ceilf", "ceil"),
        unary(
            Box::new(super::trunc::TruncFunctionalDps),
            "truncf",
            "trunc",
        ),
        unary(Box::new(super::round::RoundFunctionalDps), "rintf", "rint"),
        MathKernel {
            op: Box::new(super::modulo::ModFunctionalDps),
            arity: 2,
            f32_name: "fmodf",
            f64_name: "fmod",
            dest: same,
        },
        MathKernel {
            op: Box::new(super::trunc_cast::TruncCastDps),
            arity: 1,
            f32_name: "truncf",
            f64_name: "trunc",
            dest: to_int64,
        },
    ]
}

fn source(kernel: &MathKernel, dtype: PlanDtype) -> anyhow::Result<String> {
    let ctx = dense_ctx(kernel.arity, dtype, (kernel.dest)(dtype));
    let launches = kernel.op.codegen(&ctx)?;
    assert_eq!(launches.len(), 1, "a math kernel is a single launch");
    Ok(launches.into_iter().next().unwrap().source)
}

#[test]
fn f32_operands_call_the_float_function() {
    for kernel in math_kernels() {
        let source = source(&kernel, PlanDtype::F32).expect("F32 codegen");
        let call = format!("{}(a[", kernel.f32_name);
        assert!(source.contains(&call), "missing `{call}`:\n{source}");
    }
}

/// An F64 operand calls the double function, so its value is never
/// converted to float: before this, `floor(0.99999999999)` returned 1.
#[test]
fn f64_operands_call_the_double_function() {
    for kernel in math_kernels() {
        let source = source(&kernel, PlanDtype::F64).expect("F64 codegen");
        let call = format!("{}(a[", kernel.f64_name);
        assert!(source.contains(&call), "missing `{call}`:\n{source}");
        let float_call = format!("{}(", kernel.f32_name);
        assert!(
            !source.contains(&float_call),
            "F64 kernel calls `{float_call}`:\n{source}"
        );
    }
}

#[test]
fn narrow_floats_use_explicit_f32_opmath_and_round_to_the_destination() {
    for kernel in math_kernels() {
        for dtype in [PlanDtype::F16, PlanDtype::Bf16] {
            let source = source(&kernel, dtype).expect("narrow-float codegen");
            let call = format!("{}((float)(a[i])", kernel.f32_name);
            assert!(
                source.contains(&call),
                "{} {dtype:?}: missing `{call}`:\n{source}",
                kernel.f32_name,
            );
            let dest = crate::kernels::cuda_type((kernel.dest)(dtype)).unwrap();
            assert!(
                source.contains(&format!("({dest})({}", kernel.f32_name)),
                "{} {dtype:?}: result is not explicitly converted to {dest}:\n{source}",
                kernel.f32_name,
            );
            assert!(
                !source.contains(&format!("{}(", kernel.f64_name)),
                "{} {dtype:?}: narrow kernel calls the F64 function:\n{source}",
                kernel.f32_name,
            );
        }
    }
}

#[test]
fn non_float_math_is_refused_by_name() {
    for kernel in math_kernels() {
        let error = source(&kernel, PlanDtype::Int).expect_err("integer math must refuse");
        let message = error.to_string();
        assert!(
            message.contains("has no Int arm") && message.contains(kernel.f64_name),
            "{}: unexpected refusal `{message}`",
            kernel.f64_name
        );
    }
}

#[test]
fn mixed_operand_dtypes_are_refused() {
    let mut ctx = dense_ctx(2, PlanDtype::F32, PlanDtype::F32);
    ctx.operand_dtypes[1] = PlanDtype::F64;
    let error = super::modulo::ModFunctionalDps
        .codegen(&ctx)
        .expect_err("fmod over F32 and F64 must refuse");
    assert!(error.to_string().contains("disagree"), "{error}");
}
