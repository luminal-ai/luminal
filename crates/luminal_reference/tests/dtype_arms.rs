//! The wide-dtype kernel arms (LUM-1096): F64 everywhere F32 is
//! implemented, I64 everywhere I32 is, Bool8 and the narrow floats for
//! comparison. Each case is chosen so the obvious wrong implementation —
//! an f32 detour, an i32 accumulator, comparing raw float codes — fails.

use luminal::buffer_tensor_ir::BufferTensorIrOp;
use luminal::prelude::f16;
use luminal_reference::{ReferenceKernelCtx, TypedBuffer, kernels, ops};

fn run(
    op: &dyn BufferTensorIrOp,
    operands: Vec<TypedBuffer>,
    dest: TypedBuffer,
) -> anyhow::Result<TypedBuffer> {
    let operand_dims = operands.iter().map(|b| vec![b.len()]).collect();
    let mut ctx = ReferenceKernelCtx {
        operands,
        operand_dims,
        dests: vec![dest],
        dims: Default::default(),
    };
    (kernels::kernel_for(op).expect("registered kernel").execute)(op, &mut ctx)?;
    Ok(ctx.dests.pop().unwrap())
}

fn less_than(lhs: TypedBuffer, rhs: TypedBuffer) -> Vec<u8> {
    let n = lhs.len();
    let out = run(
        &ops::LessThanDps,
        vec![lhs, rhs],
        TypedBuffer::Bool8(vec![0; n]),
    )
    .unwrap();
    out.as_bool8().unwrap().clone()
}

#[test]
fn f64_less_than_does_not_collapse_through_f32() {
    let (a, b) = (1.0, 1.0 + f64::EPSILON);
    assert_eq!(a as f32, b as f32, "precondition: equal in f32");
    let out = less_than(
        TypedBuffer::F64(vec![a, b, f64::NAN]),
        TypedBuffer::F64(vec![b, a, 0.0]),
    );
    assert_eq!(out, vec![1, 0, 0]);
}

#[test]
fn bool8_less_than_is_false_before_true() {
    let out = less_than(
        TypedBuffer::Bool8(vec![0, 0, 1, 1]),
        TypedBuffer::Bool8(vec![0, 1, 0, 1]),
    );
    assert_eq!(out, vec![0, 1, 0, 0]);
}

#[test]
fn narrow_float_less_than_compares_values_not_codes() {
    let f = |v: &[f32]| TypedBuffer::F16(v.iter().map(|x| f16::from_f32(*x)).collect());
    // -2.0 is code 0xC000, above 1.0's 0x3C00: a code comparison inverts it.
    let lhs = f(&[-2.0, -0.0, 1.0, f32::NAN]);
    let rhs = f(&[1.0, 0.0, -3.0, 1.0]);
    assert_eq!(less_than(lhs, rhs), vec![1, 0, 0, 0]);
}

#[test]
fn f64_reduce_sum_accumulates_in_f64() {
    let out = run(
        &ops::ReduceSumDps { axis: 0 },
        vec![TypedBuffer::F64(vec![0.1, 0.2])],
        TypedBuffer::F64(vec![0.0]),
    )
    .unwrap();
    assert_eq!(out.as_f64().unwrap()[0], 0.1 + 0.2);
    assert_ne!(0.1f64 + 0.2, f64::from(0.1f32 + 0.2f32));
}

#[test]
fn f64_max_propagates_nan() {
    let input = || vec![TypedBuffer::F64(vec![1.0, f64::NAN, 3.0])];
    let out = run(
        &ops::ReduceMaxDps { axis: 0 },
        input(),
        TypedBuffer::F64(vec![0.0]),
    )
    .unwrap();
    assert!(out.as_f64().unwrap()[0].is_nan());
    let out = run(
        &ops::LeftSequentialScanMaxDps { axis: 0 },
        input(),
        TypedBuffer::F64(vec![0.0; 3]),
    )
    .unwrap();
    let out = out.as_f64().unwrap();
    assert_eq!(out[0], 1.0);
    assert!(out[1].is_nan() && out[2].is_nan());
}

#[test]
fn i64_scans_and_sums_exceed_the_i32_range() {
    let big = 3_000_000_000i64;
    let scan = |op: &dyn BufferTensorIrOp, values: Vec<i64>| {
        let n = values.len();
        run(
            op,
            vec![TypedBuffer::I64(values)],
            TypedBuffer::I64(vec![0; n]),
        )
        .map(|b| b.as_i64().unwrap().clone())
    };
    assert_eq!(
        scan(&ops::LeftSequentialScanSumDps { axis: 0 }, vec![big, 1, -2]).unwrap(),
        vec![big, big + 1, big - 1]
    );
    assert_eq!(
        scan(
            &ops::LeftSequentialScanProdDps { axis: 0 },
            vec![100_000; 3]
        )
        .unwrap(),
        vec![100_000, 10_000_000_000, 1_000_000_000_000_000]
    );
    assert_eq!(
        scan(
            &ops::LeftSequentialScanMaxDps { axis: 0 },
            vec![-big, big, 0]
        )
        .unwrap(),
        vec![-big, big, big]
    );
    let sum = run(
        &ops::ReduceSumDps { axis: 0 },
        vec![TypedBuffer::I64(vec![big, big])],
        TypedBuffer::I64(vec![0]),
    )
    .unwrap();
    assert_eq!(sum.as_i64().unwrap()[0], 2 * big);
}

#[test]
fn i64_accumulation_refuses_overflow() {
    // Ints are non-wrapping (ruling 2026-08-11): an overflow is a loud error.
    let overflow = vec![TypedBuffer::I64(vec![i64::MAX, 1])];
    assert!(
        run(
            &ops::LeftSequentialScanSumDps { axis: 0 },
            overflow.clone(),
            TypedBuffer::I64(vec![0; 2])
        )
        .is_err()
    );
    assert!(
        run(
            &ops::ReduceSumDps { axis: 0 },
            overflow,
            TypedBuffer::I64(vec![0])
        )
        .is_err()
    );
}

#[test]
fn f64_mod_is_fmod() {
    let out = run(
        &ops::ModFunctionalDps,
        vec![
            TypedBuffer::F64(vec![5.5, -5.5, 1e300]),
            TypedBuffer::F64(vec![-2.0, 2.0, 7.0]),
        ],
        TypedBuffer::F64(vec![0.0; 3]),
    )
    .unwrap();
    assert_eq!(*out.as_f64().unwrap(), vec![1.5, -1.5, 1e300 % 7.0]);
}

#[test]
fn integer_mod_stays_refused() {
    // Integer remainder is TruncRem; the frontend no longer records it as Mod.
    let ints = vec![TypedBuffer::I64(vec![5]), TypedBuffer::I64(vec![2])];
    assert!(run(&ops::ModFunctionalDps, ints, TypedBuffer::I64(vec![0])).is_err());
}
