//! Bucket selection and range-valid compilation without a device.

use luminal::dtype::DType;
use luminal::graph::{DimBucket, Graph};
use luminal::shape::{DynMap, Symbol};
use luminal_cuda_lite::{CudaRuntime, harness_search_options};

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

/// The selected physical plan retains the range variable instead of freezing
/// geometry to the representative. GPU replay is covered in dynamic_graphs.
#[test]
fn bucket_plan_keeps_symbolic_capacity() {
    let (cx, _) = elementwise_graph();
    let mut rt = CudaRuntime::load(&cx).unwrap();
    rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4)])
        .unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();
    assert!(
        rt.bucket_plans()[0]
            .plan
            .buffers
            .values()
            .any(|b| b.layout.literal_extents().is_none())
    );
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
    assert!(
        format!("{err:#}").contains("already carries a range binding [3, 3]"),
        "{err:#}"
    );

    // A NON-TIGHT range is just as exclusive: it seeds the same IntVar's
    // bounds, and `lower-bound-of`/`upper-bound-of` MERGE (max/min) rather
    // than error, so a second seed set would silently intersect with the
    // per-bucket one instead of refusing.
    let (cx, _out) = elementwise_graph();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    rt.bind_dyn_range('a', 2, 8).expect("range binds");
    let err = rt
        .bind_dim_buckets('a', vec![DimBucket::new(5, 9)])
        .expect_err("a range-bound dim must not take buckets");
    assert!(
        format!("{err:#}").contains("already carries a range binding [2, 8]"),
        "{err:#}"
    );

    let (cx, _out) = elementwise_graph();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4)])
        .expect("buckets bind");
    let err = rt
        .bind_dyn_range('a', 3, 3)
        .expect_err("a bucketed dim must not take a range binding");
    assert!(format!("{err:#}").contains("has buckets bound"), "{err:#}");
}

/// A BOUNDS-DEPENDENT AUTHORING CHECK UNDER BUCKETS (review C3/C10).
/// `reduce_max` emits `require_extent_at_least` on its axis, which needs
/// `lower-bound-of` for that dim. A bucketed dim is seeded PER BUCKET,
/// never in a base render — so a base render of this graph carries no
/// bounds for 'a' at all and the check refuses. The bucketed ladder must
/// never render one: every per-bucket render seeds the interval and
/// passes.
#[test]
fn a_bucketed_dim_may_carry_a_bounds_dependent_check() {
    let mut cx = Graph::new();
    cx.set_dim('a', 3);
    let x = cx.tensor(('a', 2), DType::F32);
    let _out = x.max(0).output();

    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4), DimBucket::new(5, 9)])
        .expect("disjoint sorted buckets bind");
    rt.search(&Default::default(), &harness_search_options())
        .expect("a bucketed search never renders the unseeded base program");
    assert_eq!(rt.bucket_plans().len(), 2, "one plan per bucket");
}
