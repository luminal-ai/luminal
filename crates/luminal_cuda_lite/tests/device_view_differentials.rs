//! M4 PHASE 5 GATE (c), `device` feature only: a REAL searched plan
//! elects a folded view on CUDA-lite, and the device output BYTE-MATCHES
//! the reference runtime's materialize-route output on identical inputs.
//!
//! Both sides run the full ladder from the same graph and data:
//!  * reference — its allow list is permanently materialize-only
//!    (ruling aff22598), so movement lowers to materialize kernels;
//!  * CUDA-lite — the view op is electable, the plan is ASSERTED to
//!    fold it (zero materialize computes + a consumer slot carrying
//!    composed access), and the kernel reads through the fold on
//!    device.
//!
//! Byte-match is the honest bar here: every fixture's arithmetic is
//! elementwise f32 mul and/or the serial per-output reduce loop, and
//! both executors accumulate in the same order over the same values —
//! IEEE-deterministic, so the routes must agree BIT-FOR-BIT, not just
//! within tolerance.
#![cfg(feature = "device")]

use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::CudaRuntime;

/// Read the device output DENSELY through its RETURNED LAYOUT
/// (escape-and-disclose + the corrected contract, 2026-08-31): a
/// view-elected output returns its BACKING buffer's bytes (possibly
/// parent-sized) plus the elected layout `L`, so the honest comparison
/// EVALUATES that layout at each coordinate down to a flat element
/// index. The hop-chain walker is gone with the hop machinery; the
/// reader is this runtime evaluating its OWN vocabulary
/// (`layouts::dense_f32`). The canonical cross-runtime version lives in
/// the testing crate as `test_runtime::test_equality::dense_f32` — CL
/// cannot depend on it (`test_runtime` depends on CL).
///
/// A dense election evaluates the identity, so this stays the universal
/// readback: no fixture assumes dense.
fn walked_dense(rt: &CudaRuntime, out: NodeIndex) -> Vec<f32> {
    let (data, binding) = rt.fetch(out).expect("escape-and-disclose fetch");
    let bytes = match data {
        TypedBuffer::F32(values) => values,
        other => panic!("output is {}, not f32", other.type_name()),
    };
    luminal_cuda_lite::layouts::dense_f32(bytes, &binding.layout)
        .expect("the returned layout reads dense over its backing buffer")
}

fn view_search_options() -> ImplementationSearchOptions {
    ImplementationSearchOptions {
        generations: 4,
        generation_size: 8,
        mutations: 4,
        trials: 1,
        seed: 0,
        search_log: false,
    }
}

/// Reference (materialize route) vs CUDA (folded-view route) from one
/// graph + inputs; asserts the CUDA plan actually folded before
/// executing it.
fn run_differential(
    cx: &Graph,
    inputs: &[(NodeIndex, Vec<f32>)],
    out: NodeIndex,
    what: &str,
) -> (Vec<f32>, Vec<f32>) {
    // Reference side: the materialize route by construction.
    let staged: Vec<(NodeIndex, TypedBuffer)> = inputs
        .iter()
        .map(|(id, v)| (*id, v.clone().into()))
        .collect();
    let reference = luminal_reference::harness::run_reference(cx, &staged);
    let want = reference.get_f32(out).expect("reference output").clone();

    // CUDA side: search under the CL allow list (view electable).
    let mut rt = CudaRuntime::load(cx).expect("cuda load");
    let data: FxHashMap<NodeIndex, TypedBuffer> = inputs
        .iter()
        .map(|(id, v)| (*id, v.clone().into()))
        .collect();
    rt.search(&data, &view_search_options())
        .expect("cuda search");

    // The plan must have ELECTED AND FOLDED the view: no materialize
    // computes, and at least one consumer READS THROUGH A LAYOUT THAT IS
    // NOT the one its buffer was allocated for (the corrected contract's
    // fold discriminator — the plan does not label folds).
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

/// BIT-FOR-BIT: the two routes read the same values in the same order
/// through IEEE-deterministic ops.
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

/// TRANSPOSE CONSUMER: mul reads x through the folded swap map.
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

/// SLICE CONSUMER: mul reads rows 1..3 of a (4,6) through the folded
/// offset map.
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

/// BROADCAST CONSUMER: mul reads a (3,) row through the folded
/// stride-0 map (a legal non-injective view read).
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

/// CHAINED-MATMUL-SHAPED: (a·b)·c — both matmul stages' broadcast and
/// permute movement folds; the muls read through composed access and
/// the reduces run the same serial order on both executors.
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
