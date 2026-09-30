#![cfg(target_os = "macos")]

mod support;
use luminal::{dtype::DType, graph::Graph};
use luminal_metal::{MetalRuntime, harness_search_options};
#[cfg(target_os = "macos")]
use support::TestTransfers;

#[test]
fn select_preserves_branch_bits_through_strided_views() {
    let mut graph = Graph::new();
    let condition = graph.tensor((3, 2), DType::F32);
    let a = graph.tensor((3, 2), DType::F32);
    let b = graph.tensor((2, 3), DType::F32);
    let output = condition
        .permute((1, 0))
        .lt(graph.constant_f32(0.).expand_rhs((2, 3)))
        .select(a.permute((1, 0)), b);
    let condition_values = vec![-1f32, 1., 1., -1., -1., 1.];
    let a_values = vec![-0f32, 99., f32::NAN, f32::INFINITY, f32::NAN, 99.];
    let b_values = vec![f32::NAN, 2., 99., -3., f32::NAN, f32::NEG_INFINITY];
    let mut runtime = MetalRuntime::load(&graph).unwrap();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &[
                (condition.id, condition_values.clone().into()),
                (a.id, a_values.clone().into()),
                (b.id, b_values.clone().into()),
            ]
            .into_iter()
            .collect(),
            &harness_search_options(),
        )
        .unwrap();
    let mut arena_runtime = support::allocate_arena(&runtime).expect("allocate execution arena");
    runtime
        .upload(&mut arena_runtime, condition.id, condition_values.clone())
        .unwrap();
    runtime
        .upload(&mut arena_runtime, a.id, a_values.clone())
        .unwrap();
    runtime
        .upload(&mut arena_runtime, b.id, b_values.clone())
        .unwrap();
    runtime.execute(arena_runtime.buffer()).unwrap();
    let actual = runtime.read_f32(&arena_runtime, output.id).unwrap();
    for row in 0..2 {
        for col in 0..3 {
            let src = col * 2 + row;
            let dst = row * 3 + col;
            let expected = if condition_values[src] < 0. {
                a_values[src]
            } else {
                b_values[dst]
            };
            assert_eq!(actual[dst].to_bits(), expected.to_bits(), "element {dst}");
        }
    }
}
