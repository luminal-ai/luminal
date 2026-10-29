//! Every kernel that calls a CUDA math-library function names the
//! function for its operands' dtype — `sqrtf` for F32, `sqrt` for F64 —
//! and refuses every other dtype instead of letting C++ convert the
//! value at the call. Codegen runs on any host, so this needs no device.

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

/// A narrow float is storage: the kernel refuses it rather than computing
/// in float behind the graph's back, and so does every non-float dtype.
#[test]
fn every_other_dtype_is_refused_by_name() {
    for kernel in math_kernels() {
        for dtype in [PlanDtype::F16, PlanDtype::Bf16, PlanDtype::Int] {
            let error = source(&kernel, dtype).expect_err("non-F32/F64 math must refuse");
            let message = error.to_string();
            assert!(
                message.contains(&format!("has no {dtype:?} arm"))
                    && message.contains(kernel.f64_name),
                "{}: unexpected refusal `{message}`",
                kernel.f64_name
            );
        }
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
