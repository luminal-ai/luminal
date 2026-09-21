#![cfg(target_os = "macos")]
use luminal::layout_ir::{Access, FreedBy};
use luminal::prelude::*;
use luminal_metal::bindings::MetalBindings;
use luminal_metal::{HostBuffer, MetalRuntime, harness_search_options, metal_registry};
fn dense(rt: &MetalRuntime, t: GraphTensor) -> Vec<f32> {
    let (data, binding) = rt.fetch(t.id).unwrap();
    luminal_metal::layouts::dense_f32(&data.as_f32().unwrap(), &binding.layout).unwrap()
}
#[test]
fn resident_weights_and_in_place_state_survive_bucket_changes_and_explicit_updates() {
    let mut cx = Graph::new();
    let weights = cx.tensor(4, DType::F32);
    let state = cx.tensor(4, DType::F32);
    let input = cx.tensor('n', DType::F32);
    let previous = state.sum(0);
    let next = state + weights * input.sum(0).expand_dim(0, 4);
    // `next` writes the state input's storage, and the weights and the
    // state stay in the device arena between executions: both are
    // statements of the binding, not of the model.
    let mut bindings = MetalBindings::dense(&cx.logical, &[previous.id]);
    let home = bindings.buffer_of_input(state.id).unwrap();
    bindings.declare(home, Access::ReadWrite, FreedBy::Caller);
    bindings.output_on(next.id, home);
    bindings.resident(weights.id).unwrap();
    bindings.resident(state.id).unwrap();
    let mut rt = MetalRuntime::load_with(&cx, bindings, metal_registry()).unwrap();
    rt.bind_dim_buckets(
        'n',
        vec![
            luminal::graph::DimBucket::new(1, 1),
            luminal::graph::DimBucket::new(2, 4),
        ],
    )
    .unwrap();
    rt.set_dim('n', 1);
    let data: FxHashMap<NodeIndex, HostBuffer> = [
        (weights.id, vec![1f32, 2., 3., 4.].into()),
        (state.id, vec![0f32; 4].into()),
        (input.id, vec![2f32].into()),
    ]
    .into_iter()
    .collect();
    rt.search(&data, &harness_search_options()).unwrap();
    let search_stats = rt.graph_stats().unwrap();
    for (id, v) in data {
        rt.set_data(id, v);
    }
    rt.execute().unwrap();
    assert_eq!(dense(&rt, previous), vec![0.]);
    assert!(rt.fetch(next.id).is_err());
    let first = rt.graph_stats().unwrap();
    assert_eq!(
        first.resident_upload_bytes - search_stats.resident_upload_bytes,
        32
    );
    assert!(
        rt.bind_dyn_range("other", 1, 1).is_err(),
        "rebinding must preserve existing resident data"
    );
    rt.set_dim('n', 3);
    rt.set_data(input.id, vec![1f32; 3]);
    rt.execute().unwrap();
    assert_eq!(dense(&rt, previous), vec![20.]);
    let second = rt.graph_stats().unwrap();
    assert_eq!(
        second.resident_upload_bytes - search_stats.resident_upload_bytes,
        32
    );
    assert_eq!(second.arena_generation, first.arena_generation);
    rt.set_dim('n', 1);
    rt.set_data(input.id, vec![1f32]);
    rt.execute().unwrap();
    assert_eq!(dense(&rt, previous), vec![50.]);
    rt.set_data(state.id, vec![0f32; 4]);
    rt.set_data(weights.id, vec![2f32; 4]);
    rt.execute().unwrap();
    assert_eq!(dense(&rt, previous), vec![0.]);
    rt.execute().unwrap();
    assert_eq!(dense(&rt, previous), vec![8.]);
    assert_eq!(
        rt.graph_stats().unwrap().resident_upload_bytes - search_stats.resident_upload_bytes,
        64
    );
}
