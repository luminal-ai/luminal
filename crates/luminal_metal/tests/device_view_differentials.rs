#![cfg(target_os = "macos")]

use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_metal::CompileOptions;
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

fn view_search_options() -> CompileOptions {
    CompileOptions {
        generations: 4,
        generation_size: 8,
        mutations: 4,
        trials: 1,
        seed: 0,
        search_log: false,
        ..Default::default()
    }
}

fn run_differential(
    cx: &Graph,
    inputs: &[(NodeIndex, Vec<f32>)],
    out: NodeIndex,
    what: &str,
) -> (Vec<f32>, Vec<f32>) {
    let staged: Vec<(NodeIndex, TypedBuffer)> = inputs
        .iter()
        .map(|(id, v)| (*id, v.clone().into()))
        .collect();
    let reference = luminal_reference::harness::run_reference(cx, &staged);
    let want = reference.get_f32(out).expect("reference output").clone();

    let mut rt =
        MetalRuntime::load_with_registry(cx, luminal_metal::metal_registry()).expect("metal load");
    let data: FxHashMap<NodeIndex, HostBuffer> = inputs
        .iter()
        .map(|(id, v)| (*id, v.clone().into()))
        .collect();
    rt.search(&data, &view_search_options())
        .expect("metal search");

    let plan = rt.plan().expect("plan loaded");
    let mut folded_slots = 0usize;
    for node in plan.dag.node_weights() {
        if let BufferNode::Compute {
            op, operand_info, ..
        } = node
        {
            let label = op.label();
            if label == "BufferAlloc" || label == "BufferFree" {
                continue;
            }
            assert_ne!(
                label,
                "IndexMapApplyMaterialize",
                "{what}: foldable movement was materialized:\n{}",
                plan.summary()
            );
            folded_slots += operand_info
                .iter()
                .filter(|s| s.layout != plan.buffers[&s.buffer].layout)
                .count();
        }
    }
    assert!(
        folded_slots > 0,
        "{what}: no consumer reads through a folded view:\n{}",
        plan.summary()
    );

    for (id, v) in inputs {
        rt.set_data(*id, v.clone());
    }
    rt.execute().expect("device execute");
    let got = walked_dense(&rt, out);
    (want, got)
}

fn assert_bytes_equal(want: &[f32], got: &[f32], what: &str) {
    assert_eq!(want.len(), got.len(), "{what}: length mismatch");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        assert_eq!(
            w.to_bits(),
            g.to_bits(),
            "{what}: element {i} diverges bitwise — reference {w} vs device {g}"
        );
    }
}

#[test]
fn transpose_consumer_byte_matches_materialize_route() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 3usize), DType::F32);
    let c = cx.tensor((3usize, 2usize), DType::F32);
    let out = (x.permute((1, 0)) * c).output();
    let (want, got) = run_differential(
        &cx,
        &[
            (x.id, vec![1.5, -2.25, 3.125, 4.0, 5.5, -6.75]),
            (c.id, vec![0.5, 1.25, -2.0, 3.5, -4.75, 6.0]),
        ],
        out.id,
        "transpose consumer",
    );
    assert_bytes_equal(&want, &got, "transpose consumer");
}

#[test]
fn slice_consumer_byte_matches_materialize_route() {
    let mut cx = Graph::new();
    let x = cx.tensor((4usize, 6usize), DType::F32);
    let c = cx.tensor((2usize, 6usize), DType::F32);
    let out = (x.slice((1..3, ..)) * c).output();
    let (want, got) = run_differential(
        &cx,
        &[
            (x.id, (0..24).map(|v| (v as f32) * 1.375 - 7.0).collect()),
            (c.id, (0..12).map(|v| (v as f32) * -0.625 + 2.0).collect()),
        ],
        out.id,
        "slice consumer",
    );
    assert_bytes_equal(&want, &got, "slice consumer");
}

#[test]
fn broadcast_consumer_byte_matches_materialize_route() {
    let mut cx = Graph::new();
    let x = cx.tensor(3usize, DType::F32);
    let c = cx.tensor((2usize, 3usize), DType::F32);
    let out = (x.expand_dim(0, 2) * c).output();
    let (want, got) = run_differential(
        &cx,
        &[
            (x.id, vec![1.125, -2.5, 3.75]),
            (c.id, vec![0.25, 1.5, -2.75, 3.0, -4.25, 5.5]),
        ],
        out.id,
        "broadcast consumer",
    );
    assert_bytes_equal(&want, &got, "broadcast consumer");
}

#[test]
fn chained_matmul_byte_matches_materialize_route() {
    let mut cx = Graph::new();
    let a = cx.tensor((2usize, 3usize), DType::F32);
    let b = cx.tensor((3usize, 4usize), DType::F32);
    let c = cx.tensor((4usize, 2usize), DType::F32);
    let out = a.matmul(b).matmul(c).output();
    let (want, got) = run_differential(
        &cx,
        &[
            (a.id, (0..6).map(|v| (v as f32) * 0.875 - 1.5).collect()),
            (b.id, (0..12).map(|v| (v as f32) * -0.375 + 2.25).collect()),
            (c.id, (0..8).map(|v| (v as f32) * 1.0625 - 3.0).collect()),
        ],
        out.id,
        "chained matmul",
    );
    assert_bytes_equal(&want, &got, "chained matmul");
}
