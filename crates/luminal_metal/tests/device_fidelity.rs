#![cfg(target_os = "macos")]

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_metal::HostBuffer;
use luminal_metal::MetalRuntime;
use luminal_reference::TypedBuffer;

fn walked_dense(rt: &MetalRuntime, out: NodeIndex) -> Vec<f32> {
    let (data, binding) = rt.fetch(out).expect("escape-and-disclose fetch");
    let bytes = data
        .as_f32()
        .unwrap_or_else(|err| panic!("output is not f32: {err}"));
    luminal_metal::layouts::dense_f32(&bytes, &binding.layout)
        .expect("the returned layout reads dense over its backing buffer")
}

fn run_both(cx: &Graph, inputs: &[(NodeIndex, Vec<f32>)], out: NodeIndex) -> (Vec<f32>, Vec<f32>) {
    let staged: Vec<(NodeIndex, TypedBuffer)> = inputs
        .iter()
        .map(|(id, v)| (*id, v.clone().into()))
        .collect();
    let reference = luminal_reference::harness::run_reference(cx, &staged);
    let want = reference.get_f32(out).expect("reference output").clone();

    let mut rt = MetalRuntime::load(cx).expect("metal load");
    let data: FxHashMap<NodeIndex, HostBuffer> = inputs
        .iter()
        .map(|(id, v)| (*id, v.clone().into()))
        .collect();
    rt.search(&data, &luminal_metal::harness_search_options())
        .expect("metal search");
    for (id, v) in inputs {
        rt.set_data(*id, v.clone());
    }
    rt.execute().expect("device execute");
    let got = walked_dense(&rt, out);
    (want, got)
}

fn assert_close(want: &[f32], got: &[f32], what: &str) {
    assert_eq!(want.len(), got.len(), "{what}: length mismatch");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        let tol = 1e-5f32.max(w.abs() * 1e-5);
        assert!(
            (w - g).abs() <= tol,
            "{what}: element {i} diverges — reference {w} vs device {g}"
        );
    }
}

#[test]
fn elementwise_chain() {
    let mut cx = Graph::new();
    let a = cx.tensor((2usize, 3usize), DType::F32);
    let b = cx.tensor((2usize, 3usize), DType::F32);
    let out = ((a + b) * a).sqrt().exp().output();
    let (want, got) = run_both(
        &cx,
        &[
            (a.id, vec![1.0, 2., 3., 4., 5., 6.]),
            (b.id, vec![0.5, 1., 1.5, 2., 2.5, 3.]),
        ],
        out.id,
    );
    assert_close(&want, &got, "elementwise chain");
}

#[test]
fn reduce_and_broadcast() {
    let mut cx = Graph::new();
    let a = cx.tensor((3usize, 4usize), DType::F32);
    let e = a.exp();
    let out = (e / e.sum(1).expand_dim(1, 4)).output();
    let (want, got) = run_both(
        &cx,
        &[(a.id, (0..12).map(|i| i as f32 * 0.25).collect())],
        out.id,
    );
    assert_close(&want, &got, "softmax-ish");
}

#[test]
fn movement_materialize() {
    let mut cx = Graph::new();
    let a = cx.tensor((4usize, 5usize), DType::F32);
    let out = a
        .slice((1..3, 1..4))
        .pad(((1usize, 0usize), (0usize, 2usize)), 0.)
        .output();
    let (want, got) = run_both(&cx, &[(a.id, (0..20).map(|i| i as f32).collect())], out.id);
    assert_close(&want, &got, "slice+pad materialize");
}

#[test]
fn iota_arange() {
    let mut cx = Graph::new();
    let idx = cx.arange(6usize);
    let a = cx.tensor(6usize, DType::F32);
    let out = (a * idx.cast(luminal::dtype::DType::F32)).output();
    let (want, got) = run_both(&cx, &[(a.id, vec![2.0; 6])], out.id);
    assert_close(&want, &got, "arange*x");
}

#[test]
fn gather_rows() {
    let mut cx = Graph::new();
    let table = cx.tensor((5usize, 3usize), DType::F32);
    let rows = cx.arange(2usize); // rows 0 and 1
    let out = table.gather1d(rows).output();
    let (want, got) = run_both(
        &cx,
        &[(table.id, (0..15).map(|i| i as f32).collect())],
        out.id,
    );
    assert_close(&want, &got, "gather1d");
}

#[test]
fn scatter_write() {
    let mut cx = Graph::new();
    let init = cx.tensor(6usize, DType::F32);
    let src = cx.tensor(2usize, DType::F32);
    let coords = cx.arange(2usize); // write positions 0 and 1
    let out = init.scatter(&[coords], src).output();
    let (want, got) = run_both(
        &cx,
        &[(init.id, vec![10.0; 6]), (src.id, vec![-1.0, -2.0])],
        out.id,
    );
    assert_close(&want, &got, "scatter");
}
