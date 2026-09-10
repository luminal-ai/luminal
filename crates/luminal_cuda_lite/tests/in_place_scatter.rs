//! IN-PLACE KV WRITES (the serving landing, 2026-09-10): a scatter whose
//! init is a mutable input and whose value is an output delivered INTO
//! that input (`output_into`) lands in the input's own buffer — the plan
//! shows the scatter writing the buffer it reads — and the resident copy
//! carries the written rows into the NEXT execute with no copy. A
//! `ReadOnly` init keeps the functional, out-of-place scatter.
#![cfg(feature = "device")]

use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::{CudaRuntime, HostBuffer};

const SLOTS: usize = 6;
const WIDTH: usize = 4;

/// Build `cache' = scatter_rows(rows, slots, cache)`; `sum = Σ cache'`
/// is the only declared output, so the cache value is consumed and dies
/// inside the program.
fn build(mutable: bool) -> (Graph, NodeIndex, NodeIndex, NodeIndex, NodeIndex) {
    let mut cx = Graph::new();
    let cache = cx.tensor((SLOTS, WIDTH), DType::F32);
    let rows = cx.tensor((2, WIDTH), DType::F32);
    let slots = cx.tensor(2, DType::Int);
    if mutable {
        cx.logical.mark_input_mutable(cache.id);
    }
    let written = luminal_nn::scatter_rows(rows, slots, cache);
    if mutable {
        // Delivered INTO the cache's own buffer: the in-place designation.
        written.output_into(cache, "cache");
    }
    let sum = written.sum(0).output();
    (cx, cache.id, rows.id, slots.id, sum.id)
}

fn scatter_writes_its_own_init(rt: &CudaRuntime, cache: NodeIndex) -> bool {
    let plan = rt.plan().expect("plan");
    let cache_lit = plan
        .buffers
        .values()
        .find(|b| {
            b.label.contains("nat")
                && b.lit.is_some()
                && b.owner == luminal::bufferize::Owner::Caller
        })
        .map(|_| ());
    let _ = (cache, cache_lit);
    plan.dag.node_weights().any(|node| {
        matches!(node, BufferNode::Compute { op, reads, writes, .. }
            if op.label() == "ScatterFunctionalGeneric" && reads.first() == writes.first())
    })
}

#[test]
fn a_mutable_cache_is_written_in_place_and_persists_across_executes() {
    let (cx, cache, rows, slots, sum) = build(true);
    let data: FxHashMap<NodeIndex, HostBuffer> = [
        (cache, HostBuffer::from(vec![0f32; SLOTS * WIDTH])),
        (rows, HostBuffer::from(vec![1f32; 2 * WIDTH])),
        (slots, HostBuffer::from(vec![0i32, 1])),
    ]
    .into_iter()
    .collect();
    let mut rt = CudaRuntime::load(&cx).expect("load");
    rt.search(&data, &luminal_cuda_lite::harness_search_options())
        .unwrap_or_else(|e| panic!("search: {e:#}"));
    assert!(
        scatter_writes_its_own_init(&rt, cache),
        "a ReadWrite init must be scattered in place"
    );
    rt.set_data(cache, vec![0f32; SLOTS * WIDTH]);
    rt.set_data(rows, vec![1f32; 2 * WIDTH]);
    rt.set_data(slots, vec![0i32, 1]);
    rt.execute().expect("execute 1");
    assert_eq!(rt.get_f32(sum).unwrap(), vec![2.0; WIDTH]);
    // Second tick: different slots, the resident cache still holds rows 0
    // and 1 from the first tick — no set_data on the cache in between.
    rt.set_data(rows, vec![3f32; 2 * WIDTH]);
    rt.set_data(slots, vec![2i32, 3]);
    rt.execute().expect("execute 2");
    assert_eq!(rt.get_f32(sum).unwrap(), vec![2.0 + 6.0; WIDTH]);
    // Re-staging the cache resets it.
    rt.set_data(cache, vec![0f32; SLOTS * WIDTH]);
    rt.set_data(rows, vec![5f32; 2 * WIDTH]);
    rt.set_data(slots, vec![4i32, 5]);
    rt.execute().expect("execute 3");
    assert_eq!(rt.get_f32(sum).unwrap(), vec![10.0; WIDTH]);
}

#[test]
fn a_read_only_cache_stays_out_of_place() {
    let (cx, cache, rows, slots, sum) = build(false);
    let data: FxHashMap<NodeIndex, HostBuffer> = [
        (cache, HostBuffer::from(vec![0f32; SLOTS * WIDTH])),
        (rows, HostBuffer::from(vec![1f32; 2 * WIDTH])),
        (slots, HostBuffer::from(vec![0i32, 1])),
    ]
    .into_iter()
    .collect();
    let mut rt = CudaRuntime::load(&cx).expect("load");
    rt.search(&data, &luminal_cuda_lite::harness_search_options())
        .unwrap_or_else(|e| panic!("search: {e:#}"));
    assert!(
        !scatter_writes_its_own_init(&rt, cache),
        "a ReadOnly init must never be written"
    );
    rt.set_data(cache, vec![0f32; SLOTS * WIDTH]);
    rt.set_data(rows, vec![1f32; 2 * WIDTH]);
    rt.set_data(slots, vec![0i32, 1]);
    rt.execute().expect("execute 1");
    rt.set_data(rows, vec![3f32; 2 * WIDTH]);
    rt.set_data(slots, vec![2i32, 3]);
    rt.execute().expect("execute 2");
    // The cache input was never mutated: only this tick's rows count.
    assert_eq!(rt.get_f32(sum).unwrap(), vec![6.0; WIDTH]);
}
