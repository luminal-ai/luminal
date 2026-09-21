#![cfg(feature = "device")]
use luminal::layout_ir::{Access, FreedBy};
use luminal::prelude::*;
use luminal_cuda_lite::{CudaBindings, CudaRuntime, HostBuffer, harness_search_options};
fn dense(rt: &CudaRuntime, t: GraphTensor) -> Vec<f32> {
    let (data, binding) = rt.fetch(t.id).unwrap();
    luminal_cuda_lite::layouts::dense_f32(&data.as_f32().unwrap(), &binding.layout).unwrap()
}
#[test]
fn resident_weights_and_in_place_state_survive_bucket_changes_and_explicit_updates() {
    let mut cx = Graph::new();
    let weights = cx.tensor(4, DType::F32);
    let state = cx.tensor(4, DType::F32);
    let input = cx.tensor('n', DType::F32);
    let previous = state.sum(0);
    let next = state + weights * input.sum(0).expand_dim(0, 4);
    // THE BINDING IS THE STATEMENT: the weights and the state live in the
    // device arena across executions, and `next` writes the state's own
    // buffer — a mutation sink, so it is never read back.
    let mut bindings = CudaBindings::new();
    bindings.input_resident(weights.id);
    let state_buffer = bindings.input_resident(state.id);
    bindings.declare(state_buffer, Access::ReadWrite, FreedBy::Caller);
    bindings.input(input.id);
    bindings.output(previous.id);
    bindings.output_on(next.id, state_buffer);
    let mut rt = CudaRuntime::load_with(&cx, bindings, luminal_cuda_lite::cuda_registry()).unwrap();
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
        rt.set_data(id, v).unwrap();
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
    rt.set_data(input.id, vec![1f32; 3]).unwrap();
    rt.execute().unwrap();
    assert_eq!(dense(&rt, previous), vec![20.]);
    let second = rt.graph_stats().unwrap();
    assert_eq!(
        second.resident_upload_bytes - search_stats.resident_upload_bytes,
        32
    );
    assert_eq!(second.arena_base, first.arena_base);
    rt.set_dim('n', 1);
    rt.set_data(input.id, vec![1f32]).unwrap();
    rt.execute().unwrap();
    assert_eq!(dense(&rt, previous), vec![50.]);
    rt.set_data(state.id, vec![0f32; 4]).unwrap();
    rt.set_data(weights.id, vec![2f32; 4]).unwrap();
    rt.execute().unwrap();
    assert_eq!(dense(&rt, previous), vec![0.]);
    rt.execute().unwrap();
    assert_eq!(dense(&rt, previous), vec![8.]);
    assert_eq!(
        rt.graph_stats().unwrap().resident_upload_bytes - search_stats.resident_upload_bytes,
        64
    );
}
