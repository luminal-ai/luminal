//! BUCKETS ON THE CUDA LADDER (D7, 2026-09-03), CPU-side.
//!
//! The same bucket model as the reference runtime's
//! (`luminal_reference::runtime::tests::bucketed_search_validates_searches_and_selects`),
//! duplicated here because the ladders are duplicated: bind disjoint
//! intervals per dim, search one Cartesian combination at a time — each
//! validated bucket-wide over its whole range before its representative
//! is searched — and select the covering plan from the runtime's dims.
//!
//! Everything here is device-free: this runtime ranks candidates with
//! `luminal_cuda_lite::heuristic`, so a bucketed search needs no data and
//! no device. Only `execute` would, and the static-plan refusal below
//! fires before any of that.

use luminal::dtype::DType;
use luminal::graph::{DimBucket, Graph};
use luminal::shape::{DynMap, Symbol};
use luminal_cuda_lite::{harness_search_options, CudaRuntime};

fn elementwise_graph() -> (Graph, luminal::prelude::GraphTensor) {
    let mut cx = Graph::new();
    cx.set_dim('a', 3);
    let x = cx.tensor(('a', 2), DType::F32);
    let y = cx.tensor(('a', 2), DType::F32);
    let out = (x * y).output();
    (cx, out)
}

/// Two buckets over 'a': one plan each, both validated bucket-wide,
/// selection covers runtime dims, and out-of-range dims select nothing.
#[test]
fn bucketed_search_validates_searches_and_selects() {
    let (cx, _out) = elementwise_graph();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4), DimBucket::new(5, 9)])
        .expect("disjoint sorted buckets bind");

    let outcome = rt
        .search(&Default::default(), &harness_search_options())
        .expect("bucketed search completes");
    assert!(
        outcome.plans_profiled > 0,
        "the returned outcome is the FIRST bucket's, and it must be a real search"
    );
    assert_eq!(rt.bucket_plans().len(), 2, "one plan per bucket");

    let mut dims = DynMap::default();
    dims.insert(Symbol::from('a'), 3usize);
    assert_eq!(
        luminal_cuda_lite::search::select_bucket(rt.bucket_plans(), &dims)
            .expect("bucket [2,4] covers a = 3")
            .ranges[&Symbol::from('a')],
        (2, 4)
    );
    dims.insert(Symbol::from('a'), 7usize);
    assert_eq!(
        luminal_cuda_lite::search::select_bucket(rt.bucket_plans(), &dims)
            .expect("bucket [5,9] covers a = 7")
            .ranges[&Symbol::from('a')],
        (5, 9)
    );
    dims.insert(Symbol::from('a'), 20usize);
    assert!(
        luminal_cuda_lite::search::select_bucket(rt.bucket_plans(), &dims).is_none(),
        "a = 20 is outside every bucket"
    );

    // Each plan is a real bufferized plan at its own representative.
    for plan in rt.bucket_plans() {
        let rep = plan.representative[&Symbol::from('a')];
        assert!(
            plan.ranges[&Symbol::from('a')].0 <= rep && rep <= plan.ranges[&Symbol::from('a')].1,
            "the representative is inside its bucket"
        );
        assert!(
            plan.outcome.best_plan.dag.node_count() > 0,
            "bucket {:?} produced an empty plan",
            plan.ranges
        );
    }
}

/// THE PHASE 1 LIMITATION, pinned: each bucket's plan is STATIC at its
/// representative (plan spans are literals), so executing it at another
/// value inside the same bucket refuses by name and points at the
/// symbolic-plan open item. This runtime cannot execute at all without a
/// device, but the refusal is raised by plan SELECTION, before any
/// device work — which is exactly the point: the wrong-geometry run is
/// never even attempted.
#[test]
fn a_non_representative_pin_refuses_loudly() {
    let (cx, _out) = elementwise_graph();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4)])
        .expect("buckets bind");
    rt.search(&Default::default(), &harness_search_options())
        .expect("bucketed search completes");

    // 3 is the representative of [2, 4]; 4 is inside the bucket and is
    // NOT the pin the plan was searched at.
    rt.set_dim('a', 4);
    let err = rt
        .execute()
        .expect_err("a non-representative pin must refuse");
    let text = format!("{err:#}");
    assert!(text.contains("STATIC at that pin"), "{text}");
    assert!(
        text.contains("a = 3"),
        "the message names the representative: {text}"
    );
    assert!(text.contains("symbolic plans"), "{text}");
}

/// Buckets must partition: overlap is refused, not resolved first-wins.
#[test]
fn overlapping_or_unsorted_buckets_are_refused() {
    let (cx, _out) = elementwise_graph();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");

    let err = rt
        .bind_dim_buckets('a', vec![DimBucket::new(2, 6), DimBucket::new(5, 9)])
        .expect_err("overlapping buckets must refuse");
    assert!(
        format!("{err:#}").contains("sorted and disjoint"),
        "{err:#}"
    );

    let err = rt
        .bind_dim_buckets('a', vec![DimBucket::new(5, 9), DimBucket::new(2, 4)])
        .expect_err("unsorted buckets must refuse");
    assert!(
        format!("{err:#}").contains("sorted and disjoint"),
        "{err:#}"
    );

    let err = rt
        .bind_dim_buckets('a', vec![])
        .expect_err("an empty bucket list must refuse");
    assert!(format!("{err:#}").contains("no buckets"), "{err:#}");
}

/// A dim cannot be both pinned and bucketed — the two would emit
/// conflicting bounds seeds for the same variable.
#[test]
fn a_dim_cannot_be_both_pinned_and_bucketed() {
    let (cx, _out) = elementwise_graph();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    rt.bind_dyn_range('a', 3, 3).expect("pin binds");
    let err = rt
        .bind_dim_buckets('a', vec![DimBucket::new(2, 4)])
        .expect_err("a pinned dim must not take buckets");
    assert!(format!("{err:#}").contains("already pinned"), "{err:#}");

    let (cx, _out) = elementwise_graph();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4)])
        .expect("buckets bind");
    let err = rt
        .bind_dyn_range('a', 3, 3)
        .expect_err("a bucketed dim must not take a range binding");
    assert!(format!("{err:#}").contains("has buckets bound"), "{err:#}");
}
