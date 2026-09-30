#![cfg(target_os = "macos")]
mod support;
use luminal::{dtype::DType, graph::Graph};
use luminal_metal::{MetalRuntime, harness_search_options};
#[cfg(target_os = "macos")]
use support::TestTransfers;

/// floor / ceil / trunc / round (half-to-even) execute on device with the
/// same semantics as the reference runtime: NaN/±inf propagate for the
/// dtype-preserving ops.
#[test]
fn rounding_ops_execute_on_device() {
    let mut g = Graph::new();
    let x = g.tensor(9, DType::F32);
    let floor = x.floor();
    let ceil = x.ceil();
    let trunc = x.trunc();
    let round = x.round();
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(
        &Default::default(),
        &Default::default(),
        &[(x.id, vec![1f32; 9].into())].into_iter().collect(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");

    let values: Vec<f32> = vec![1.9, 1.5, 0.5, -0.5, -1.5, -1.9, 2.5, 3.5, -2.5];
    rt.upload(&mut arena_rt, x.id, values.clone()).unwrap();
    rt.execute(arena_rt.buffer()).unwrap();

    assert_eq!(
        rt.read_f32(&arena_rt, floor.id).unwrap(),
        values.iter().map(|v| v.floor()).collect::<Vec<_>>()
    );
    assert_eq!(
        rt.read_f32(&arena_rt, ceil.id).unwrap(),
        values.iter().map(|v| v.ceil()).collect::<Vec<_>>()
    );
    assert_eq!(
        rt.read_f32(&arena_rt, trunc.id).unwrap(),
        values.iter().map(|v| v.trunc()).collect::<Vec<_>>()
    );
    assert_eq!(
        rt.read_f32(&arena_rt, round.id).unwrap(),
        values
            .iter()
            .map(|v| v.round_ties_even())
            .collect::<Vec<_>>()
    );
}

/// `trunc_cast` is the explicit lossy float -> int conversion: truncate
/// toward zero, in the integer target dtype.
#[test]
fn trunc_cast_executes_on_device() {
    let mut g = Graph::new();
    let x = g.tensor(6, DType::F32);
    let out = x.trunc_cast(DType::Int);
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(
        &Default::default(),
        &Default::default(),
        &[(x.id, vec![1f32; 6].into())].into_iter().collect(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");

    rt.upload(&mut arena_rt, x.id, vec![1.9, 1.5, 0.5, -0.5, -1.5, -1.9])
        .unwrap();
    rt.execute(arena_rt.buffer()).unwrap();
    assert_eq!(
        rt.read_i32(&arena_rt, out.id).unwrap(),
        vec![1, 1, 0, 0, -1, -1]
    );
}
