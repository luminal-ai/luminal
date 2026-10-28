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
        beta_is_one: form.has_c(),
        activation: LtActivation::Identity,
        bias_operand: form.has_bias().then_some(if form.has_c() { 3 } else { 2 }),
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
        assert_eq!(call.beta_is_one, form.has_c());
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
