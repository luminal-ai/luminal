//! Single-program dimension contracts, independent of application bucketing.
use luminal::prelude::*;
use luminal_cuda_lite::{CudaRuntime, harness_search_options};

#[test]
fn invalid_search_domains_are_rejected_before_device_work() {
    let mut graph = Graph::new();
    let x = graph.tensor(('a', 2), DType::F32);
    let _ = x + 1.;
    for (bounds, dims, message) in [
        (SymbolBounds::default(), DynMap::default(), "missing"),
        (
            SymbolBounds::from_ranges([('a'.into(), (2, 9)), ('b'.into(), (1, 1))]).unwrap(),
            [('a'.into(), 3), ('b'.into(), 1)].into_iter().collect(),
            "unused",
        ),
        (
            SymbolBounds::from_ranges([('a'.into(), (2, 9))]).unwrap(),
            [('a'.into(), 10)].into_iter().collect(),
            "outside",
        ),
    ] {
        let error = CudaRuntime::load(&graph)
            .unwrap()
            .search(
                &bounds,
                &dims,
                &Default::default(),
                &harness_search_options(),
            )
            .unwrap_err();
        assert!(format!("{error:#}").contains(message), "{error:#}");
    }
}

#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn bounded_plan_retains_symbolic_capacity_and_satisfies_authoring_checks() {
    let mut graph = Graph::new();
    let x = graph.tensor(('a', 2), DType::F32);
    let _ = x.max(0);
    let bounds = SymbolBounds::from_ranges([('a'.into(), (2, 9))]).unwrap();
    let dims = [('a'.into(), 3)].into_iter().collect();
    let mut runtime = CudaRuntime::load(&graph).unwrap();
    let outcome = runtime
        .search(
            &bounds,
            &dims,
            &Default::default(),
            &harness_search_options(),
        )
        .unwrap();
    assert!(outcome.plans_profiled > 0);
    assert_eq!(runtime.bounds(), &bounds);
    assert!(
        runtime
            .plan()
            .unwrap()
            .buffers
            .values()
            .any(|b| b.layout.literal_extents().is_none())
    );
}
