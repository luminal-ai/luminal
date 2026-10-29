//! The wrapping family computes in the same-width unsigned type: signed
//! overflow is undefined behaviour in C++, unsigned overflow wraps modulo
//! 2^w. Codegen runs on any host, so this needs no device.

use crate::kernels::{CodegenCtx, KernelOp};
use luminal::dtype::PlanDtype;
use luminal::layouts::{
    BitWidthTerm, DecodedLayout, IntExprTerm, RightMajorContiguousElementLayout, ShapeTerm,
};

fn layout(dims: &[i64], dtype: PlanDtype) -> DecodedLayout {
    let width = match dtype {
        PlanDtype::Int64 | PlanDtype::UInt64 | PlanDtype::F64 => 64,
        _ => 32,
    };
    DecodedLayout::of(
        RightMajorContiguousElementLayout {
            shape: ShapeTerm(dims.iter().map(|&d| IntExprTerm::Lit(d)).collect()),
            width: BitWidthTerm(width),
        },
        Some(dtype),
    )
}

/// `inputs` value operands of `in_dims`, then the destination slot of
/// `out_dims`, all of `dtype`.
fn ctx(inputs: usize, in_dims: &[i64], out_dims: &[i64], dtype: PlanDtype) -> CodegenCtx {
    let extents = |dims: &[i64]| {
        dims.iter()
            .map(|&d| crate::symbolic::Expr(IntExprTerm::Lit(d)))
            .collect::<Vec<_>>()
    };
    let mut operand_dims = vec![extents(in_dims); inputs];
    operand_dims.push(extents(out_dims));
    let mut operand_layouts = vec![layout(in_dims, dtype); inputs];
    operand_layouts.push(layout(out_dims, dtype));
    CodegenCtx {
        operand_dims,
        operand_dtypes: vec![dtype; inputs + 1],
        dest_dims: vec![extents(out_dims)],
        dest_dtypes: vec![dtype],
        operand_layouts,
    }
}

fn source(op: &dyn KernelOp, ctx: &CodegenCtx) -> anyhow::Result<String> {
    Ok(op.codegen(ctx)?.into_iter().next().unwrap().source)
}

#[test]
fn signed_wrapping_arithmetic_runs_unsigned() {
    let add = super::wrapping_add::WrappingAddFunctionalDps;
    let mul = super::wrapping_mul::WrappingMulFunctionalDps;
    for (dtype, signed, unsigned) in [
        (PlanDtype::Int, "int", "unsigned int"),
        (PlanDtype::Int64, "long long", "unsigned long long"),
    ] {
        let c = ctx(2, &[4], &[4], dtype);
        let add_src = source(&add, &c).unwrap();
        let expected = format!("({signed})(({unsigned})a[i] + ({unsigned})b[i])");
        assert!(
            add_src.contains(&expected),
            "missing `{expected}`:\n{add_src}"
        );
        let mul_src = source(&mul, &c).unwrap();
        let expected = format!("({signed})(({unsigned})a[i] * ({unsigned})b[i])");
        assert!(
            mul_src.contains(&expected),
            "missing `{expected}`:\n{mul_src}"
        );

        let sum = super::wrapping_reduce_sum::WrappingReduceSumDps { axis: 0 };
        let sum_src = source(&sum, &ctx(1, &[2, 3], &[2], dtype)).unwrap();
        let expected = format!("acc = ({signed})(({unsigned})acc + ({unsigned})v);");
        assert!(
            sum_src.contains(&expected),
            "missing `{expected}`:\n{sum_src}"
        );

        let c = ctx(1, &[2, 3], &[2, 3], dtype);
        let cumsum =
            super::wrapping_left_sequential_scan_sum::WrappingLeftSequentialScanSumDps { axis: 0 };
        let cumprod =
            super::wrapping_left_sequential_scan_prod::WrappingLeftSequentialScanProdDps {
                axis: 0,
            };
        for (scan_src, op) in [
            (source(&cumsum, &c).unwrap(), "+"),
            (source(&cumprod, &c).unwrap(), "*"),
        ] {
            let expected = format!("({signed})(({unsigned})acc {op} ({unsigned})v)");
            assert!(
                scan_src.contains(&expected),
                "missing `{expected}`:\n{scan_src}"
            );
        }
    }
}

#[test]
fn unsigned_wrapping_arithmetic_is_native_and_floats_are_refused() {
    let add = super::wrapping_add::WrappingAddFunctionalDps;
    let src = source(&add, &ctx(2, &[4], &[4], PlanDtype::UInt64)).unwrap();
    assert!(src.contains("a[i] + b[i]"), "{src}");
    let err = source(&add, &ctx(2, &[4], &[4], PlanDtype::F32)).unwrap_err();
    assert!(err.to_string().contains("no F32 arm"), "{err}");
}
