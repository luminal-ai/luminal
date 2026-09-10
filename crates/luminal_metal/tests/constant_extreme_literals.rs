#![cfg(target_os = "macos")]

use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_metal::HostBuffer;
use luminal_metal::MetalRuntime;
use luminal_metal::ops::constant::ConstantDps;

fn walked_dense(rt: &MetalRuntime, out: NodeIndex) -> Vec<f32> {
    let (data, binding) = rt.fetch(out).expect("escape-and-disclose fetch");
    let bytes = data
        .as_f32()
        .unwrap_or_else(|err| panic!("output is not f32: {err}"));
    luminal_metal::layouts::dense_f32(&bytes, &binding.layout)
        .expect("the returned layout reads dense over its backing buffer")
}

fn assert_constant_reaches_the_device(rt: &MetalRuntime, expected: f64, what: &str) {
    let plan = rt.plan().expect("plan loaded");
    let elected: Vec<f64> = plan
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::Compute { op, .. } => op
                .as_any()
                .downcast_ref::<ConstantDps>()
                .map(|constant| constant.value),
            _ => None,
        })
        .collect();
    assert!(
        elected.iter().any(|v| v.to_bits() == expected.to_bits()),
        "{what}: no ConstantGeneric compute node carries {expected:?} \
         (the plan's constants are {elected:?}) — the literal never \
         reached MSL, so this test would prove nothing"
    );
}

fn run_on_device(
    cx: &Graph,
    inputs: &[(NodeIndex, Vec<f32>)],
    out: NodeIndex,
    expected_constant: f64,
    what: &str,
) -> Vec<f32> {
    let mut rt = MetalRuntime::load(cx).expect("metal load");
    let data: FxHashMap<NodeIndex, HostBuffer> = inputs
        .iter()
        .map(|(id, v)| (*id, v.clone().into()))
        .collect();
    rt.search(&data, &luminal_metal::harness_search_options())
        .expect("metal search");
    assert_constant_reaches_the_device(&rt, expected_constant, what);
    for (id, v) in inputs {
        rt.set_data(*id, v.clone());
    }
    rt.execute()
        .expect("device execute (MSL compiles the constant kernel here)");
    walked_dense(&rt, out)
}

fn constant_plus_zero(value: f32) -> Vec<f32> {
    const N: usize = 8;
    let mut cx = Graph::new();
    let a = cx.tensor(N, DType::F32);
    let c = cx.constant_float(value).expand_rhs(a.dims());
    let out = (a + c).output();
    let got = run_on_device(
        &cx,
        &[(a.id, vec![0.0f32; N])],
        out.id,
        value as f64,
        &format!("constant {value:?}"),
    );
    assert_eq!(got.len(), N, "constant {value:?}: wrong output length");
    got
}

#[test]
fn extreme_finite_constant_survives_msl_bit_for_bit() {
    for (i, g) in constant_plus_zero(f32::MIN).iter().enumerate() {
        assert_eq!(
            g.to_bits(),
            f32::MIN.to_bits(),
            "element {i}: expected f32::MIN ({:e}), got {g:e}",
            f32::MIN
        );
    }
}

#[test]
fn negative_infinity_constant_survives_msl() {
    for (i, g) in constant_plus_zero(f32::NEG_INFINITY).iter().enumerate() {
        assert!(
            g.is_infinite() && g.is_sign_negative(),
            "element {i}: expected -inf, got {g}"
        );
    }
}

#[test]
fn nan_constant_survives_msl() {
    for (i, g) in constant_plus_zero(f32::NAN).iter().enumerate() {
        assert!(g.is_nan(), "element {i}: expected NaN, got {g}");
    }
}

#[test]
fn cummax_seed_constant_survives_msl() {
    let input = vec![-5.0f32, -3., -9., -1., -7., -2., -8., -4.];
    let mut cx = Graph::new();
    let a = cx.tensor(input.len(), DType::F32);
    let out = a.cummax(0).output();
    let got = run_on_device(
        &cx,
        &[(a.id, input.clone())],
        out.id,
        f32::MIN as f64,
        "cummax seed",
    );

    let mut running = f32::NEG_INFINITY;
    let want: Vec<f32> = input
        .iter()
        .map(|v| {
            running = running.max(*v);
            running
        })
        .collect();
    assert_eq!(got, want, "cummax over {input:?}");
}
