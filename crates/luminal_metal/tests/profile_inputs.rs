#![cfg(target_os = "macos")]
mod support;
use luminal::prelude::*;
use luminal::shape::DimensionBounds;
use luminal_metal::{CompileOptions, HostBuffer, MetalRuntime};
#[cfg(target_os = "macos")]
use support::TestTransfers;

#[test]
fn explicit_profile_payload_and_assignment() {
    let mut graph = Graph::new();
    let input = graph.tensor('q', DType::F32);
    let output = input + 1.;
    let bounds = DimensionBounds::from_ranges([('q'.into(), (1, 128))]).unwrap();
    let dims = [('q'.into(), 128)].into_iter().collect();
    let options = CompileOptions {
        generations: 1,
        generation_size: 1,
        trials: 1,
        search_log: false,
        ..Default::default()
    };
    let data: FxHashMap<_, HostBuffer> =
        [(input.id, vec![128f32; 128].into())].into_iter().collect();
    let mut runtime = MetalRuntime::load(&graph).unwrap();
    runtime.search(&bounds, &dims, &data, &options).unwrap();
    let mut arena_runtime = support::allocate_arena(&runtime).expect("allocate execution arena");
    assert_eq!(runtime.graph_stats().unwrap().launches, 3);
    for q in [128usize, 1, 7] {
        runtime.set_dim('q', q);
        runtime
            .upload(&mut arena_runtime, input.id, vec![q as f32; q])
            .unwrap();
        runtime.execute(arena_runtime.buffer()).unwrap();
        assert_eq!(
            runtime.read_f32(&arena_runtime, output.id).unwrap(),
            vec![q as f32 + 1.; q]
        );
    }
    let too_short = [(input.id, vec![1f32].into())].into_iter().collect();
    let error = MetalRuntime::load(&graph)
        .unwrap()
        .search(&bounds, &dims, &too_short, &options)
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("profiling shape requires"),
        "{error:#}"
    );
    assert!(
        MetalRuntime::load(&graph)
            .unwrap()
            .search(
                &bounds,
                &[('q'.into(), 129)].into_iter().collect(),
                &data,
                &options
            )
            .is_err()
    );
}
