use super::test_support::*;
use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

fn run(t: &super::Translation, buffers: Vec<TypedBuffer>) -> ReferenceRuntime {
    let ids = t.outputs.iter().map(|o| o.tensor).collect::<Vec<_>>();
    let bindings = ReferenceBindings::dense(&t.graph.logical, &ids);
    let mut rt = ReferenceRuntime::load_with(&t.graph, bindings).unwrap();
    let data: FxHashMap<_, _> = t
        .inputs
        .iter()
        .zip(buffers)
        .map(|(i, b)| (i.tensor, b))
        .collect();
    rt.search(
        &Default::default(),
        &Default::default(),
        &data,
        &harness_search_options(),
    )
    .unwrap();
    for (id, b) in data {
        rt.set_data(id, b);
    }
    rt.execute().unwrap();
    rt
}

#[test]
fn nextafter_observes_ieee_encodings_including_subnormals_and_signed_zero() {
    let a = vec![
        0.,
        -0.,
        1.,
        -1.,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MAX,
        f32::NAN,
    ];
    let b = vec![-0., 1., 2., 0., 0., 0., 0., 0., f32::INFINITY, 1.];
    let expected = a
        .iter()
        .zip(&b)
        .map(|(&a, &b)| {
            if a.is_nan() || b.is_nan() {
                a + b
            } else if a == b {
                b
            } else if b > a {
                a.next_up()
            } else {
                a.next_down()
            }
        })
        .collect::<Vec<_>>();
    let t = translate_one(
        node(
            "torch.ops.aten.nextafter.default",
            vec![tensor_input("self", "a"), tensor_input("other", "b")],
            &["y"],
        ),
        &[
            ("a", FLOAT, &[10]),
            ("b", FLOAT, &[10]),
            ("y", FLOAT, &[10]),
        ],
        &["a", "b"],
        &["y"],
    );
    let rt = run(&t, vec![TypedBuffer::F32(a), TypedBuffer::F32(b)]);
    for (&actual, &expected) in rt
        .get_f32(t.outputs[0].tensor)
        .unwrap()
        .iter()
        .zip(&expected)
    {
        assert!(
            actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()),
            "{actual:?} != {expected:?}"
        );
    }
}

#[test]
fn kthvalue_and_mode_keep_int64_precision_and_ties() {
    let values = vec![
        (1i64 << 54) + 3,
        (1i64 << 54) + 1,
        (1i64 << 54) + 3,
        (1i64 << 54) + 2,
        -9,
        -3,
        -3,
        -7,
    ];
    for (target, extra, expected, indices) in [
        (
            "kthvalue.default",
            vec![scalar_int("k", 2)],
            vec![(1i64 << 54) + 2, -7],
            vec![3, 3],
        ),
        (
            "mode.default",
            vec![],
            vec![(1i64 << 54) + 3, -3],
            vec![2, 2],
        ),
    ] {
        let mut args = vec![tensor_input("self", "x")];
        args.extend(extra);
        args.push(scalar_int("dim", -1));
        let t = translate_one(
            node(&format!("torch.ops.aten.{target}"), args, &["v", "i"]),
            &[("x", 5, &[2, 4]), ("v", 5, &[2]), ("i", 5, &[2])],
            &["x"],
            &["v", "i"],
        );
        let rt = run(&t, vec![TypedBuffer::I64(values.clone())]);
        assert_eq!(rt.get_i64(t.outputs[0].tensor).unwrap(), &expected);
        assert_eq!(rt.get_i64(t.outputs[1].tensor).unwrap(), &indices);
    }
}

#[test]
fn selection_keeps_nans_out_of_finite_frequency_counts() {
    for (target, args, expected, index) in [
        ("mode.default", vec![], 1.0, 2),
        ("kthvalue.default", vec![scalar_int("k", 3)], f32::NAN, 0),
    ] {
        let mut inputs = vec![tensor_input("self", "x")];
        inputs.extend(args);
        let t = translate_one(
            node(&format!("torch.ops.aten.{target}"), inputs, &["v", "i"]),
            &[("x", FLOAT, &[3]), ("v", FLOAT, &[]), ("i", 5, &[])],
            &["x"],
            &["v", "i"],
        );
        let rt = run(&t, vec![TypedBuffer::F32(vec![f32::NAN, 1.0, 1.0])]);
        let actual = rt.get_f32(t.outputs[0].tensor).unwrap()[0];
        assert!(actual == expected || (actual.is_nan() && expected.is_nan()));
        assert_eq!(rt.get_i64(t.outputs[1].tensor).unwrap(), &vec![index]);
    }
}

#[test]
fn lu_unpack_observes_pivots_and_rectangular_triangular_parts() {
    let t = translate_one(
        node(
            "torch.ops.aten.lu_unpack.default",
            vec![tensor_input("LU_data", "a"), tensor_input("LU_pivots", "p")],
            &["permutation", "lower", "upper"],
        ),
        &[
            ("a", FLOAT, &[2, 2]),
            ("p", INT, &[2]),
            ("permutation", FLOAT, &[2, 2]),
            ("lower", FLOAT, &[2, 2]),
            ("upper", FLOAT, &[2, 2]),
        ],
        &["a", "p"],
        &["permutation", "lower", "upper"],
    );
    let rt = run(
        &t,
        vec![
            TypedBuffer::F32(vec![2., 3., 4., 5.]),
            TypedBuffer::I32(vec![2, 2]),
        ],
    );
    for (i, expected) in [
        vec![0., 1., 1., 0.],
        vec![1., 0., 4., 1.],
        vec![2., 3., 0., 5.],
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(rt.get_f32(t.outputs[i].tensor).unwrap(), expected);
    }
}

#[test]
fn logaddexp_handles_equal_infinities_and_finite_cancellation() {
    let t = translate_one(
        node(
            "torch.ops.aten.logaddexp.default",
            vec![tensor_input("self", "a"), tensor_input("other", "b")],
            &["y"],
        ),
        &[("a", FLOAT, &[4]), ("b", FLOAT, &[4]), ("y", FLOAT, &[4])],
        &["a", "b"],
        &["y"],
    );
    let rt = run(
        &t,
        vec![
            TypedBuffer::F32(vec![f32::INFINITY, f32::NEG_INFINITY, 100., 0.]),
            TypedBuffer::F32(vec![f32::INFINITY, f32::NEG_INFINITY, -100., 0.]),
        ],
    );
    let y = rt.get_f32(t.outputs[0].tensor).unwrap();
    assert_eq!(&y[..3], &[f32::INFINITY, f32::NEG_INFINITY, 100.]);
    assert!((y[3] - std::f32::consts::LN_2).abs() < 1e-6);
}

/// ATen's `logit_kernel`, in f32: a negative or NaN `eps` clamps nothing,
/// otherwise `x < lo ? lo : (x > hi ? hi : x)` with `lo = eps`,
/// `hi = 1 - eps`, then `x == 1 ? inf : log(x / (1 - x))`.
fn aten_logit(x: f32, eps: Option<f32>) -> f32 {
    let x = match eps {
        Some(lo) if lo >= 0. => {
            let hi = 1. - lo;
            if x < lo {
                lo
            } else if x > hi {
                hi
            } else {
                x
            }
        }
        _ => x,
    };
    if x == 1. {
        f32::INFINITY
    } else {
        (x / (1. - x)).ln()
    }
}

#[test]
fn logit_clamps_low_bound_first_when_eps_crosses_the_bounds() {
    let x = vec![
        0.,
        0.2,
        0.5,
        0.7,
        0.9,
        1.,
        1.5,
        -0.5,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    // eps > 0.5 makes lo > hi; eps < 0 is the kernel's no-clamp branch.
    for eps in [
        None,
        Some(0.),
        Some(0.1),
        Some(0.5),
        Some(0.6),
        Some(0.9),
        Some(-0.1),
    ] {
        let mut inputs = vec![tensor_input("self", "x")];
        if let Some(eps) = eps {
            inputs.push(scalar_float("eps", eps));
        }
        let t = translate_one(
            node("torch.ops.aten.logit.default", inputs, &["y"]),
            &[("x", FLOAT, &[11]), ("y", FLOAT, &[11])],
            &["x"],
            &["y"],
        );
        let rt = run(&t, vec![TypedBuffer::F32(x.clone())]);
        let actual = rt.get_f32(t.outputs[0].tensor).unwrap();
        for (i, (&actual, &input)) in actual.iter().zip(&x).enumerate() {
            let expected = aten_logit(input, eps.map(|eps| eps as f32));
            let same = if expected.is_nan() {
                actual.is_nan()
            } else if expected.is_infinite() {
                actual == expected
            } else {
                (actual - expected).abs() <= 1e-6 * expected.abs().max(1.)
            };
            assert!(
                same,
                "logit(x[{i}] = {input}, eps = {eps:?}): expected {expected}, got {actual}"
            );
        }
    }
}

#[test]
fn scalar_predicates_broadcast_with_the_other_value_branch() {
    let t = translate_one(
        node(
            "torch.ops.aten.xlogy.Tensor",
            vec![tensor_input("self", "a"), tensor_input("other", "b")],
            &["y"],
        ),
        &[("a", FLOAT, &[]), ("b", FLOAT, &[3]), ("y", FLOAT, &[3])],
        &["a", "b"],
        &["y"],
    );
    let rt = run(
        &t,
        vec![
            TypedBuffer::F32(vec![0.]),
            TypedBuffer::F32(vec![1., 2., f32::NAN]),
        ],
    );
    let result = rt.get_f32(t.outputs[0].tensor).unwrap();
    assert_eq!(&result[..2], &[0., 0.]);
    assert!(result[2].is_nan());
}

#[test]
fn empty_dot_returns_its_additive_identity() {
    for code in [FLOAT, 5, 8] {
        let t = translate_one(
            node(
                "torch.ops.aten.dot.default",
                vec![tensor_input("self", "a"), tensor_input("other", "b")],
                &["y"],
            ),
            &[("a", code, &[0]), ("b", code, &[0]), ("y", code, &[])],
            &["a", "b"],
            &["y"],
        );
        let buffer = match code {
            FLOAT => TypedBuffer::F32(vec![]),
            5 => TypedBuffer::I64(vec![]),
            _ => TypedBuffer::F64(vec![]),
        };
        let rt = run(&t, vec![buffer.clone(), buffer]);
        match code {
            FLOAT => assert_eq!(rt.get_f32(t.outputs[0].tensor).unwrap(), &vec![0.]),
            5 => assert_eq!(rt.get_i64(t.outputs[0].tensor).unwrap(), &vec![0]),
            _ => assert_eq!(rt.get_f64(t.outputs[0].tensor).unwrap(), &vec![0.]),
        }
    }
}

#[test]
fn nansum_removes_nan_before_casting_to_boolean_output() {
    let t = translate_one(
        node(
            "torch.ops.aten.nansum.default",
            vec![
                tensor_input("self", "x"),
                ints("dim", &[0]),
                scalar_type("dtype", BOOL),
            ],
            &["y"],
        ),
        &[("x", FLOAT, &[3]), ("y", BOOL, &[])],
        &["x"],
        &["y"],
    );
    let rt = run(&t, vec![TypedBuffer::F32(vec![0., f32::NAN, 0.])]);
    assert_eq!(rt.get_bool8(t.outputs[0].tensor).unwrap(), &vec![0]);
}

#[test]
fn resize_nearest_exact_uses_half_pixel_coordinates() {
    let t = translate_one(
        node(
            "torch.ops.aten._upsample_nearest_exact1d.default",
            vec![tensor_input("self", "x"), ints("output_size", &[3])],
            &["y"],
        ),
        &[("x", FLOAT, &[1, 1, 5]), ("y", FLOAT, &[1, 1, 3])],
        &["x"],
        &["y"],
    );
    let rt = run(&t, vec![TypedBuffer::F32(vec![0., 1., 2., 3., 4.])]);
    assert_eq!(rt.get_f32(t.outputs[0].tensor).unwrap(), &vec![0., 2., 4.]);
}

#[test]
fn rms_norm_uses_the_trailing_axes_without_mean_centering() {
    let t = translate_one(
        node(
            "torch.ops.aten.rms_norm.default",
            vec![
                tensor_input("input", "x"),
                ints("normalized_shape", &[2]),
                scalar_float("eps", 0.),
            ],
            &["y"],
        ),
        &[("x", FLOAT, &[2, 2]), ("y", FLOAT, &[2, 2])],
        &["x"],
        &["y"],
    );
    let input = vec![3., 4., -3., -4.];
    let rt = run(&t, vec![TypedBuffer::F32(input.clone())]);
    for (a, b) in rt.get_f32(t.outputs[0].tensor).unwrap().iter().zip(input) {
        assert!((a - b / 12.5f32.sqrt()).abs() < 1e-5);
    }
}

#[test]
fn half_rms_norm_default_epsilon_uses_float32_opmath() {
    for code in [HALF, BFLOAT16] {
        let t = translate_one(
            node(
                "torch.ops.aten.rms_norm.default",
                vec![tensor_input("input", "x"), ints("normalized_shape", &[2])],
                &["y"],
            ),
            &[("x", code, &[2]), ("y", code, &[2])],
            &["x"],
            &["y"],
        );
        assert!(t.graph.logical.petgraph().node_weights().any(|n| {
            matches!(n.op, luminal::prelude::LogicalOp::Constant(eps)
                if eps == f32::EPSILON as f64)
        }));
    }
}

#[test]
fn fused_rms_norm_returns_reciprocal_standard_deviation() {
    let t = translate_one(
        node(
            "torch.ops.aten._fused_rms_norm.default",
            vec![
                tensor_input("input", "x"),
                ints("normalized_shape", &[2]),
                scalar_float("eps", 0.),
            ],
            &["y", "rstd"],
        ),
        &[
            ("x", FLOAT, &[2, 2]),
            ("y", FLOAT, &[2, 2]),
            ("rstd", FLOAT, &[2, 1]),
        ],
        &["x"],
        &["y", "rstd"],
    );
    let rt = run(&t, vec![TypedBuffer::F32(vec![3., 4., 0., 2.])]);
    for (&actual, expected) in rt
        .get_f32(t.outputs[1].tensor)
        .unwrap()
        .iter()
        .zip([1.0 / 12.5f32.sqrt(), 1.0 / 2.0f32.sqrt()])
    {
        assert!((actual - expected).abs() < 1e-6);
    }
}
