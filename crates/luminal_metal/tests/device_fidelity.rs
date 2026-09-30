#![cfg(target_os = "macos")]

mod support;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_metal::HostBuffer;
use luminal_metal::MetalRuntime;
use luminal_reference::TypedBuffer;
#[cfg(target_os = "macos")]
use support::TestTransfers;

fn walked_dense(rt: &MetalRuntime, arena_rt: &support::Allocation, out: NodeIndex) -> Vec<f32> {
    let (data, binding) = rt
        .download(arena_rt, out)
        .expect("escape-and-disclose fetch");
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
    rt.search(
        &Default::default(),
        &Default::default(),
        &data,
        &luminal_metal::harness_search_options(),
    )
    .expect("metal search");
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    for (id, v) in inputs {
        rt.upload(&mut arena_rt, *id, v.clone()).unwrap();
    }
    rt.execute(arena_rt.buffer()).expect("device execute");
    let got = walked_dense(&rt, &arena_rt, out);
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
    let out = ((a + b) * a).sqrt().exp();
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
    let out = e / e.sum(1).expand_dim(1, 4);
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
        .pad(((1usize, 0usize), (0usize, 2usize)), 0.);
    let (want, got) = run_both(&cx, &[(a.id, (0..20).map(|i| i as f32).collect())], out.id);
    assert_close(&want, &got, "slice+pad materialize");
}

#[test]
fn iota_arange() {
    let mut cx = Graph::new();
    let idx = cx.arange(6usize);
    let a = cx.tensor(6usize, DType::F32);
    let out = a * idx.cast(luminal::dtype::DType::F32);
    let (want, got) = run_both(&cx, &[(a.id, vec![2.0; 6])], out.id);
    assert_close(&want, &got, "arange*x");
}

#[test]
fn gather_rows() {
    let mut cx = Graph::new();
    let table = cx.tensor((5usize, 3usize), DType::F32);
    let rows = cx.arange(2usize); // rows 0 and 1
    let out = table.gather1d(rows);
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
    let out = init.scatter(&[coords], src);
    let (want, got) = run_both(
        &cx,
        &[(init.id, vec![10.0; 6]), (src.id, vec![-1.0, -2.0])],
        out.id,
    );
    assert_close(&want, &got, "scatter");
}

/// Scan results are integers in f32 here, so both the reference and the
/// device must hit them exactly — a fused or reassociated fold would not.
fn assert_exact_both(want: &[f32], got: &[f32], expected: &[f32], what: &str) {
    assert_eq!(want, expected, "{what}: reference diverges from the fact");
    assert_eq!(got, expected, "{what}: device diverges from the fact");
}

#[test]
fn cumsum_rank1() {
    let mut cx = Graph::new();
    let a = cx.tensor(4usize, DType::F32);
    let out = a.cumsum(0);
    let (want, got) = run_both(&cx, &[(a.id, vec![1.0, 2., 3., 4.])], out.id);
    assert_exact_both(&want, &got, &[1.0, 3., 6., 10.], "cumsum [1,2,3,4]");
}

#[test]
fn cumprod_carries_signs() {
    let mut cx = Graph::new();
    let a = cx.tensor(4usize, DType::F32);
    let out = a.cumprod(0);
    let (want, got) = run_both(&cx, &[(a.id, vec![-1.0, 2., -3., 4.])], out.id);
    assert_exact_both(&want, &got, &[-1.0, -2., 6., 24.], "cumprod [-1,2,-3,4]");
}

#[test]
fn cummax_is_the_running_maximum() {
    let mut cx = Graph::new();
    let a = cx.tensor(8usize, DType::F32);
    let out = a.cummax(0);
    let (want, got) = run_both(
        &cx,
        &[(a.id, vec![-5.0, -3., -9., -1., -7., -2., -8., -4.])],
        out.id,
    );
    assert_exact_both(
        &want,
        &got,
        &[-5.0, -3., -3., -1., -1., -1., -1., -1.],
        "cummax [-5,-3,-9,-1,-7,-2,-8,-4]",
    );
}

#[test]
fn prod_along_rows() {
    let mut cx = Graph::new();
    let a = cx.tensor((2usize, 2usize), DType::F32);
    let out = a.prod(1);
    let (want, got) = run_both(&cx, &[(a.id, vec![-2.0, 3., -2., -3.])], out.id);
    assert_exact_both(&want, &got, &[-6.0, 6.], "prod over rows");
}

/// Axis 0 of a rank-2 value exercises the kernel's outer/inner split: the
/// scanned axis is not the innermost, so each thread strides by the row.
#[test]
fn cumsum_along_the_outer_axis() {
    let mut cx = Graph::new();
    let a = cx.tensor((3usize, 2usize), DType::F32);
    let out = a.cumsum(0);
    let (want, got) = run_both(&cx, &[(a.id, vec![1.0, 2., 3., 4., 5., 6.])], out.id);
    assert_exact_both(
        &want,
        &got,
        &[1.0, 2., 4., 6., 9., 12.],
        "cumsum along axis 0",
    );
}

/// The scanned axis is the outer one, so each thread strides by the row.
#[test]
fn cummax_along_the_outer_axis() {
    let mut cx = Graph::new();
    let a = cx.tensor((3usize, 2usize), DType::F32);
    let out = a.cummax(0);
    let (want, got) = run_both(&cx, &[(a.id, vec![-1.0, 6., 3., -2., 2., 5.])], out.id);
    assert_exact_both(
        &want,
        &got,
        &[-1.0, 6., 3., 6., 3., 6.],
        "cummax along axis 0",
    );
}

/// All-negative integers separate a correct identity from a zero or a
/// converted float one: any seed above the inputs would reach the output.
#[test]
fn int_cummax_over_negatives() {
    let input = vec![-7i32, -9, -3, -5];
    let mut cx = Graph::new();
    let a = cx.tensor(4usize, DType::Int);
    let out = a.cummax(0);
    let mut rt = MetalRuntime::load(&cx).expect("device load");
    let data: FxHashMap<NodeIndex, HostBuffer> =
        [(a.id, input.clone().into())].into_iter().collect();
    rt.search(
        &Default::default(),
        &Default::default(),
        &data,
        &luminal_metal::harness_search_options(),
    )
    .expect("device search");
    let mut arena = support::allocate_arena(&rt).expect("allocate execution arena");
    rt.upload(&mut arena, a.id, input).expect("upload input");
    rt.execute(arena.buffer()).expect("device execute");
    assert_eq!(rt.read_i32(&arena, out.id).unwrap(), vec![-7, -7, -3, -3]);
}

/// Exact equality position by position, where a NaN in `expected` demands a
/// NaN on both sides.
fn assert_exact_both_with_nan(want: &[f32], got: &[f32], expected: &[f32], what: &str) {
    assert_eq!(want.len(), expected.len(), "{what}: reference length");
    assert_eq!(got.len(), expected.len(), "{what}: device length");
    for (i, e) in expected.iter().enumerate() {
        if e.is_nan() {
            assert!(
                want[i].is_nan(),
                "{what}: reference element {i} is {} not NaN",
                want[i]
            );
            assert!(
                got[i].is_nan(),
                "{what}: device element {i} is {} not NaN",
                got[i]
            );
        } else {
            assert_eq!(want[i], *e, "{what}: reference element {i}");
            assert_eq!(got[i], *e, "{what}: device element {i}");
        }
    }
}

/// `max` is IEEE 754-2019 `maximum`: a NaN in a slice is that slice's maximum.
#[test]
fn max_propagates_nan() {
    let mut cx = Graph::new();
    let a = cx.tensor((2usize, 2usize), DType::F32);
    let out = a.max(1);
    let (want, got) = run_both(&cx, &[(a.id, vec![1.0, f32::NAN, 3., 2.])], out.id);
    assert_exact_both_with_nan(&want, &got, &[f32::NAN, 3.0], "max over rows with a NaN");
}

/// Once the running maximum meets a NaN it stays NaN.
#[test]
fn cummax_propagates_nan() {
    let mut cx = Graph::new();
    let a = cx.tensor(4usize, DType::F32);
    let out = a.cummax(0);
    let (want, got) = run_both(&cx, &[(a.id, vec![1.0, f32::NAN, 2., 3.])], out.id);
    assert_exact_both_with_nan(
        &want,
        &got,
        &[1.0, f32::NAN, f32::NAN, f32::NAN],
        "cummax after a NaN",
    );
}

/// Scan loops and row strides use the live dimension while storage covers the
/// entire declared domain. The test owns and reuses that device allocation.
#[test]
fn scan_chain_runs_across_one_declared_domain() {
    let mut graph = Graph::new();
    let x = graph.tensor(('n', 2), DType::F32);
    let out = x.cumsum(0).cumprod(0).cummax(0);
    let bounds = luminal::shape::SymbolBounds::from_ranges([('n'.into(), (1, 6))]).unwrap();
    let dims = [('n'.into(), 3)].into_iter().collect();
    let data = [(x.id, HostBuffer::from([-1f32, 2.].repeat(3)))]
        .into_iter()
        .collect();
    let mut rt = MetalRuntime::load(&graph).unwrap();
    rt.search(
        &bounds,
        &dims,
        &data,
        &luminal_metal::harness_search_options(),
    )
    .unwrap();
    let mut arena = support::allocate_arena(&rt).unwrap();
    for extent in [1, 6, 2, 5] {
        let input = [-1f32, 2.].repeat(extent);
        let mut sum = [0f32; 2];
        let mut prod = [1f32; 2];
        let mut max = [f32::NEG_INFINITY; 2];
        let mut expected = Vec::new();
        for row in input.chunks_exact(2) {
            for col in 0..2 {
                sum[col] += row[col];
                prod[col] *= sum[col];
                max[col] = max[col].max(prod[col]);
                expected.push(max[col]);
            }
        }
        rt.set_dim('n', extent);
        rt.upload(&mut arena, x.id, input).unwrap();
        rt.execute(arena.buffer()).unwrap();
        assert_eq!(walked_dense(&rt, &arena, out.id), expected);
    }
}
