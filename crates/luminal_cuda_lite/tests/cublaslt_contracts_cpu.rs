//! Capacity checks on exact vendor descriptors, with no destination rebinding.
use luminal::dtype::PlanDtype;
use luminal_cuda_lite::ops::cublaslt::{
    CublasLtForm,
    exec::{
        LtActivation, LtCMatrixInput, LtCall, LtDesc, assert_bias_destination_order,
        validate_ld_bounds,
    },
};

fn call(form: CublasLtForm) -> LtCall {
    let d = if form.has_bias() {
        LtDesc::col(4, 3, 4)
    } else {
        LtDesc::row(4, 3, 3)
    };
    LtCall {
        computation: luminal_cuda_lite::ops::cublaslt::exec::LtComputation::Float32,
        form,
        m: 4,
        n: 3,
        k: 5,
        trans_a: false,
        trans_b: false,
        a_dtype: PlanDtype::F32,
        b_dtype: PlanDtype::F32,
        d_dtype: PlanDtype::F32,
        a: LtDesc::row(4, 5, 5),
        b: LtDesc::row(5, 3, 3),
        c: form.has_c().then_some(LtCMatrixInput {
            descriptor: d,
            data_type: PlanDtype::F32,
            operand: 2,
        }),
        d,
        scale_dtype: PlanDtype::F32,
        scalars: luminal_cuda_lite::ops::cublaslt::exec::LtScalars::host(
            1.0,
            if form.has_c() { 1.0 } else { 0.0 },
        ),
        activation: LtActivation::Identity,
        bias: form
            .has_bias()
            .then_some(luminal_cuda_lite::ops::cublaslt::exec::LtBiasInput {
                operand: if form.has_c() { 3 } else { 2 },
                data_type: luminal::dtype::PlanDtype::F32,
            }),
    }
}
#[test]
fn bounds_check_every_order_including_single_major_line_and_overflow() {
    for rows in [1, 2, 4] {
        for cols in [1, 3, 5] {
            for pad in [0, 3] {
                for d in [
                    LtDesc::row(rows, cols, cols + pad),
                    LtDesc::col(rows, cols, rows + pad),
                ] {
                    let span = match d.order {
                        luminal_cuda_lite::ops::cublaslt::exec::LtOrder::Row => {
                            (rows - 1) * d.ld + cols
                        }
                        _ => (cols - 1) * d.ld + rows,
                    } as usize;
                    validate_ld_bounds("test", &d, span).unwrap();
                    assert!(validate_ld_bounds("test", &d, span - 1).is_err());
                }
            }
        }
    }
    for d in [
        LtDesc::row(0, 3, 3),
        LtDesc::col(2, 0, 2),
        LtDesc::row(1, 3, 0),
        LtDesc::row(3, 2, -1),
        LtDesc::row(i64::MAX, 2, i64::MAX),
    ] {
        assert!(validate_ld_bounds("bad", &d, usize::MAX).is_err());
    }
}
#[test]
fn every_operand_destination_and_optional_argument_is_capacity_checked() {
    for form in CublasLtForm::ALL {
        let call = call(form);
        let mut sizes = vec![20, 15];
        if form.has_c() {
            sizes.push(12);
        }
        if form.has_bias() {
            sizes.push(4);
        }
        call.validate_against(&sizes, 12).unwrap();
        for i in 0..sizes.len() {
            let mut bad = sizes.clone();
            bad[i] -= 1;
            assert!(
                call.validate_against(&bad, 12).is_err(),
                "{form:?} operand {i}"
            );
        }
        assert!(call.validate_against(&sizes, 11).is_err());
        assert!(
            call.validate_against(&sizes[..sizes.len() - 1], 12)
                .is_err()
        );
        assert_eq!(
            call.scalars.host_values(),
            Some((1.0, if form.has_c() { 1.0 } else { 0.0 }))
        );
        assert_eq!(call.c.map(|c| c.operand), form.has_c().then_some(2));
        assert_bias_destination_order(&call, "test").unwrap();
    }
}
#[test]
fn separate_c_and_d_pitches_are_preserved_and_checked_independently() {
    let mut call = call(CublasLtForm::Accumulate);
    call.c.as_mut().unwrap().descriptor = LtDesc::row(4, 3, 8);
    call.d = LtDesc::row(4, 3, 5);
    call.validate_against(&[20, 15, 27], 18).unwrap();
    assert!(call.validate_against(&[20, 15, 26], 18).is_err());
    assert!(call.validate_against(&[20, 15, 27], 17).is_err());
    assert_eq!(call.c.unwrap().descriptor.ld, 8);
    assert_eq!(call.d.ld, 5);
}
#[test]
fn bias_order_is_checked_even_for_direct_vendor_test_calls() {
    for form in [CublasLtForm::Bias, CublasLtForm::AccumulateBias] {
        let mut call = call(form);
        call.d = LtDesc::row(4, 3, 3);
        assert!(assert_bias_destination_order(&call, "test").is_err());
    }
}

#[test]
fn zero_leading_dimension_is_a_read_only_c_permission_with_vector_bounds() {
    for (descriptor, span) in [(LtDesc::row(4, 3, 0), 3), (LtDesc::col(4, 3, 0), 4)] {
        let mut c = call(CublasLtForm::Accumulate);
        c.c.as_mut().unwrap().descriptor = descriptor;
        c.validate_against(&[20, 15, span], 12).unwrap();
        assert!(c.validate_against(&[20, 15, span - 1], 12).is_err());
        c.a.ld = 0;
        assert!(c.validate_against(&[20, 15, span], 12).is_err());
        c.a.ld = 5;
        c.b.ld = 0;
        assert!(c.validate_against(&[20, 15, span], 12).is_err());
        c.b.ld = 3;
        c.d.ld = 0;
        assert!(c.validate_against(&[20, 15, span], 12).is_err());
    }
}

#[test]
fn scalar_dtype_and_device_scalar_capacity_are_checked_before_dispatch() {
    use luminal_cuda_lite::ops::cublaslt::exec::LtScalars;
    let mut op = call(CublasLtForm::Base);
    for dtype in [
        PlanDtype::F16,
        PlanDtype::Bf16,
        PlanDtype::F64,
        PlanDtype::Int,
    ] {
        op.scale_dtype = dtype;
        assert!(op.validate_against(&[100, 100], 100).is_err());
    }
    op.scale_dtype = PlanDtype::F32;
    op.scalars = LtScalars::Device {
        alpha_operand: 2,
        beta_operand: 3,
    };
    op.validate_against(&[100, 100, 1, 1], 100).unwrap();
    for elems in [[100, 100, 0, 1], [100, 100, 1, 0]] {
        assert!(op.validate_against(&elems, 100).is_err());
    }
    assert!(op.validate_against(&[100, 100, 1], 100).is_err());
    op.scalars = LtScalars::host(1.25, -0.5);
    assert!(
        op.validate_against(&[100, 100], 100).is_err(),
        "absent C requires beta zero"
    );
}
