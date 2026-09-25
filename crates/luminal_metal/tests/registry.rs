use luminal::{dtype::DType, graph::Graph};
use luminal_metal::{
    MetalRuntime, harness_search_options, metal_registry, metal_registry_filtered,
};

#[test]
fn registry_matches_claims_and_decodes_its_live_schema() {
    let mut g = Graph::new();
    let x = g.tensor(3, DType::F32);
    let _out = x + 1.;
    let rows = metal_registry();
    for row in &rows {
        assert_eq!(row.label(), row.prototype.label());
    }
    let runtime = MetalRuntime::load_with_registry(&g, rows).unwrap();
    assert_eq!(runtime.active_allow_list(), MetalRuntime::allow_list());
    runtime.saturated_egraph().unwrap();
}

#[test]
#[cfg_attr(
    not(target_os = "macos"),
    ignore = "candidate search requires a Metal device"
)]
fn filtered_vocabulary_cannot_elect_an_unregistered_operation() {
    let mut g = Graph::new();
    let x = g.tensor(3, DType::F32);
    let y = g.tensor(3, DType::F32);
    let _out = x + y;
    let mut runtime = MetalRuntime::load_with_registry(
        &g,
        metal_registry_filtered(|row| row.label() != "AddFunctionalGeneric"),
    )
    .unwrap();
    assert!(
        !runtime
            .active_allow_list()
            .contains(&"LayoutTensorOpAddFunctionalGeneric")
    );
    assert!(
        runtime
            .search(
                &[(x.id, vec![1f32; 3].into()), (y.id, vec![2f32; 3].into())]
                    .into_iter()
                    .collect(),
                &harness_search_options()
            )
            .is_err()
    );
}

#[test]
#[cfg_attr(
    not(target_os = "macos"),
    ignore = "candidate search requires a Metal device"
)]
fn arena_budget_rejects_a_plan_set_that_cannot_fit() {
    let mut g = Graph::new();
    let x = g.tensor(3, DType::F32);
    let _out = x + 1.;
    let mut runtime = MetalRuntime::load(&g).unwrap();
    let mut options = harness_search_options();
    options.device_budget_bytes = Some(0);
    let error = runtime
        .search(
            &[(x.id, vec![1f32; 3].into())].into_iter().collect(),
            &options,
        )
        .unwrap_err();
    // The memory pass now rejects an impossible boundary before profiling
    // candidates or assembling a device plan.
    assert!(
        error.to_string().contains("0-byte arena budget"),
        "{error:#}"
    );
    assert!(
        error
            .to_string()
            .contains("required BufferInputLit boundary"),
        "{error:#}"
    );
    assert!(runtime.plan().is_none());
}
