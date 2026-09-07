//! THE OP REGISTRY IS AN INSTANCE CHOICE (#420/#422 rejoin Phase 2,
//! ruling 2026-09-03: *"you should select the allowed ops when you
//! initialize the runtime ... You should not need to edit CL in order to
//! modify this. It should be configurable."*).
//!
//! Everything here runs on the host: the board is `load` (which registry
//! an instance holds), `active_allow_list` (what it derives from that
//! registry), and `search` (what it will and will not elect). No device.
//!
//! The pins locate ops STRUCTURALLY — by egglog constructor and by the
//! `RegisteredOp` API — never by class id or registry position.

use luminal::dtype::DType;
use luminal::prelude::FxHashMap;
use luminal_cuda_lite::ops::cublaslt::{CublasLt, CublasLtForm, CublasLtMarkerMatcher};
use luminal_cuda_lite::{
    CudaRuntime, RegisteredOp, cuda_registry, cuda_registry_filtered,
    cuda_registry_without_cublaslt, harness_search_options,
};

const ADD: &str = "LayoutTensorOpAddFunctionalGeneric";

/// The smallest graph that CANNOT be planned without the add op.
fn add_graph() -> (
    luminal::graph::Graph,
    luminal::prelude::GraphTensor,
    luminal::prelude::GraphTensor,
) {
    let mut cx = luminal::graph::Graph::new();
    let a = cx.tensor((2usize, 3usize), DType::F32);
    let b = cx.tensor((2usize, 3usize), DType::F32);
    let _out = (a + b).output();
    (cx, a, b)
}

fn payloads(
    a: luminal::prelude::GraphTensor,
    b: luminal::prelude::GraphTensor,
) -> FxHashMap<luminal::prelude::NodeIndex, luminal_cuda_lite::HostBuffer> {
    [
        (a.id, vec![1.0f32, 2., 3., 4., 5., 6.].into()),
        (b.id, vec![10.0f32, 20., 30., 40., 50., 60.].into()),
    ]
    .into_iter()
    .collect()
}

/// (b) The two shipped presets are ordinary registry VALUES: `load` is
/// `load_with_registry(cuda_registry())`, and `cuda_registry()` is the
/// FULL one — markers included, since the 2026-09-04 always-on ruling.
/// Asserted in the claim set each instance actually derives, not merely
/// by construction.
#[test]
fn the_presets_are_just_registries() {
    let (cx, _a, _b) = add_graph();

    let default = CudaRuntime::load(&cx).expect("load");
    let explicit = CudaRuntime::load_with_registry(&cx, cuda_registry()).expect("load explicit");
    assert_eq!(
        default.active_allow_list(),
        explicit.active_allow_list(),
        "`load` must be `load_with_registry(cuda_registry())`"
    );
    assert_eq!(
        default.active_allow_list(),
        CudaRuntime::allow_list(),
        "the instance claim set must be the default preset's static one"
    );

    let decomposed = CudaRuntime::load_with_registry(&cx, cuda_registry_without_cublaslt())
        .expect("load decomposed");

    // The default is the decomposed preset plus the four host-call
    // contracts, and nothing is lost on the way.
    for claimed in decomposed.active_allow_list() {
        assert!(
            default.active_allow_list().contains(claimed),
            "the default preset dropped {claimed}"
        );
    }
    for form in CublasLtForm::ALL {
        assert!(
            default
                .active_allow_list()
                .contains(&form.constructor_name()),
            "the DEFAULT preset does not claim {} — the marker vocabulary is \
             on by default (ruling 2026-09-04)",
            form.constructor_name()
        );
        assert!(
            !decomposed
                .active_allow_list()
                .contains(&form.constructor_name()),
            "`cuda_registry_without_cublaslt` claims {} — that preset is exactly \
             the one with no marker in it",
            form.constructor_name()
        );
    }
}

/// (a) A REGISTRY CHOSEN AT INITIALIZATION IS THE SEARCH'S VOCABULARY:
/// drop the add row and the same graph, searched by the same runtime
/// type with the same options, refuses — loudly, naming the blockage —
/// while the default registry plans it.
#[test]
fn a_filtered_registry_withholds_the_op_and_the_search_refuses() {
    let (cx, a, b) = add_graph();
    let data = payloads(a, b);

    let without_add = cuda_registry_filtered(|op| op.label() != "AddFunctionalGeneric");
    // The narrowing is REAL, not a predicate that matched nothing.
    assert_eq!(
        without_add.len() + 1,
        cuda_registry().len(),
        "the filter must have removed exactly the add row"
    );

    let mut narrowed = CudaRuntime::load_with_registry(&cx, without_add).expect("load");
    assert!(
        !narrowed.active_allow_list().contains(&ADD),
        "a withheld op must not be claimable: {:?}",
        narrowed.active_allow_list()
    );
    let err = narrowed
        .search(&data, &harness_search_options())
        .expect_err("a graph that needs add must not plan without the add op");
    let text = err.to_string();
    assert!(
        text.contains("no candidate genome produced an executable plan"),
        "the refusal must be the search's exhaustion, got: {text}"
    );
    assert!(
        text.contains("dead-ends: 1") && text.contains("choice-cycles: 0"),
        "the blockage must be diagnosed as a DEAD END (nothing produces the sum), \
         not a choice cycle, got: {text}"
    );

    // The SAME graph, the default registry: green.
    let mut default = CudaRuntime::load(&cx).expect("load");
    assert!(default.active_allow_list().contains(&ADD));
    let outcome = default
        .search(&data, &harness_search_options())
        .expect("the default registry plans a + b");
    assert!(outcome.plans_profiled > 0, "no plans profiled");
}

/// (c) A ROW REGISTERED FROM OUTSIDE THIS CRATE joins the instance's
/// claim set: `RegisteredOp::new` over a matcher/prototype pair the
/// caller assembles itself. The row here is a matcher-only one (a
/// cuBLASLt marker: host-dispatchable, no kernel-table row), which is
/// exactly the class an external caller can add today — a kernel-bearing
/// row still needs a codegen entry inside CL (the punted "cuda heavy"
/// composition story).
#[test]
fn a_row_registered_from_outside_joins_the_claim_set() {
    let (cx, _a, _b) = add_graph();

    // The BASE this row is added to is the decomposed preset: the
    // default already ships all four marker rows, so adding one there
    // would be a duplicate, not an outside registration.
    let mut registry = cuda_registry_without_cublaslt();
    registry.push(RegisteredOp::new(
        Box::new(CublasLtMarkerMatcher {
            form: CublasLtForm::Base,
        }),
        Box::new(CublasLt {
            form: CublasLtForm::Base,
            spec: None,
        }),
    ));

    let rt = CudaRuntime::load_with_registry(&cx, registry).expect("load");
    let default = CudaRuntime::load_with_registry(&cx, cuda_registry_without_cublaslt())
        .expect("load decomposed");

    assert!(
        rt.active_allow_list()
            .contains(&CublasLtForm::Base.constructor_name()),
        "the hand-registered row is not claimed: {:?}",
        rt.active_allow_list()
    );
    // Exactly one row more than the base, and the other three marker
    // forms stayed out — the caller chose the vocabulary row by row.
    assert_eq!(
        rt.active_allow_list().len(),
        default.active_allow_list().len() + 1
    );
    for form in [
        CublasLtForm::Bias,
        CublasLtForm::Accumulate,
        CublasLtForm::AccumulateBias,
    ] {
        assert!(!rt.active_allow_list().contains(&form.constructor_name()));
    }
}

/// `RegisteredOp::label` is the house label — the constructor minus the
/// `LayoutTensorOp` prefix and nothing else — so it is the same string
/// the registered PROTOTYPE answers with. Callers filter on it; the two
/// must not drift.
#[test]
fn registry_labels_agree_with_the_prototypes() {
    for entry in cuda_registry() {
        assert!(
            entry.constructor().starts_with("LayoutTensorOp"),
            "{} is not a LayoutTensorOp constructor",
            entry.constructor()
        );
        assert_eq!(
            entry.label(),
            entry.prototype.label(),
            "row {} disagrees with its prototype's label",
            entry.constructor()
        );
    }
}

/// THE ONE ESTATE THAT IS NOT ROW-BY-ROW: only the Base cuBLASLt matcher
/// emits egglog snippets, and that one snippet set declares all four
/// marker constructors and every minting rule. A registry holding a
/// non-Base marker WITHOUT Base would claim an op the assembled program
/// never declares and never mints — un-electable, while
/// `active_allow_list()` says it is available. `load_with_registry`
/// refuses that configuration by name.
#[test]
fn a_non_base_cublaslt_row_without_base_is_refused_at_load() {
    let (cx, _a, _b) = add_graph();

    let mut registry = cuda_registry_without_cublaslt();
    registry.push(RegisteredOp::new(
        Box::new(CublasLtMarkerMatcher {
            form: CublasLtForm::Bias,
        }),
        Box::new(CublasLt {
            form: CublasLtForm::Bias,
            spec: None,
        }),
    ));
    let err = match CudaRuntime::load_with_registry(&cx, registry) {
        Ok(_) => panic!("a Bias row without Base must be refused"),
        Err(err) => err,
    };
    let msg = format!("{err:#}");
    assert!(msg.contains("LayoutTensorOpCublasLtBias"), "{msg}");
    assert!(msg.contains("without the Base row"), "{msg}");

    // Dropping Base out of the WHOLE preset is the same configuration,
    // however the caller arrives at it.
    let err = match CudaRuntime::load_with_registry(
        &cx,
        cuda_registry_filtered(|op| op.constructor() != CublasLtForm::Base.constructor_name()),
    ) {
        Ok(_) => panic!("the preset minus Base must be refused"),
        Err(err) => err,
    };
    assert!(
        format!("{err:#}").contains("without the Base row"),
        "{err:#}"
    );

    // Base alone, and all four together, both load.
    let (cx, _a, _b) = add_graph();
    CudaRuntime::load_with_registry(&cx, cuda_registry()).expect("the whole marker estate loads");
    let mut base_only = cuda_registry_without_cublaslt();
    base_only.push(RegisteredOp::new(
        Box::new(CublasLtMarkerMatcher {
            form: CublasLtForm::Base,
        }),
        Box::new(CublasLt {
            form: CublasLtForm::Base,
            spec: None,
        }),
    ));
    CudaRuntime::load_with_registry(&cx, base_only).expect("Base alone loads");
}

/// THE ASSEMBLY TRIPWIRE, over the CUDA-lite registry's own assembled
/// program: every constructor of a decoded sort that the marker estate's
/// snippets add to core's preamble has exactly one registered decoder,
/// and every registered decoder names a constructor the program really
/// declares under that sort.
///
/// The estate declares no `Layout` constructor today (its datatypes are
/// `CublasLt*` and its four `LayoutTensorOp*` rows), so the check passes
/// on core's five. The point of the pin is that it is CHECKED, not
/// assumed: the day a matcher declares one, `OpMatcher::decoders` must
/// carry the struct that reads it back or this fails by name.
#[test]
fn the_cuda_registry_declares_no_constructor_it_cannot_decode() {
    let (cx, _a, _b) = add_graph();
    let rt = CudaRuntime::load(&cx).expect("load with the default registry");
    // `saturated_egraph` runs the tripwire internally; this asserts on
    // the registry directly too, so the failure names the constructor.
    rt.saturated_egraph()
        .expect("the assembled program saturates and every constructor decodes");
    let decoders = rt.decoders();
    assert!(
        decoders
            .constructors_of("Layout")
            .any(|name| name == "LeftMajorContiguousElementLayoutLit"),
        "the Layout sort is decoded by this registry"
    );
}
