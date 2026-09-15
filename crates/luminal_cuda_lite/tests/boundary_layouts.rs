//! THE BOUNDARY CARRIES A LAYOUT (CUDA-lite's divergence from the
//! reference binding): a caller hands this runtime the storage it
//! already has — column-major, or strided — and says so in the binding
//! rather than repacking on the host.
//!
//! Asked of the SATURATED E-GRAPH the search reads and of the PLAN it
//! installs, never of an election: which spelling a class holds and what
//! a decoded layout computes are facts; which genome won is not.

use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::egglog_utils::eclass::EGraphView;
use luminal::graph::Graph;
use luminal::layout_ir::{Access, FreedBy};
use luminal::prelude::egraph_serialize::{ClassId, EGraph};
use luminal_cuda_lite::bindings::BoundaryLayout;
use luminal_cuda_lite::{
    CudaBindings, CudaRuntime, cuda_registry_without_cublaslt, harness_search_options,
};

/// The class of the input declared under `name` — an input's name lives
/// in its own `LogicalTensorInputLit` declaration.
fn input_class(egraph: &EGraph, name: &str) -> ClassId {
    let named = |class: &ClassId| {
        egraph
            .nodes
            .values()
            .filter(|n| n.eclass == *class && n.op == "LogicalIdLit")
            .filter_map(|n| n.children.first())
            .any(|child| egraph.nodes[child].op.trim_matches('"') == name)
    };
    egraph
        .nodes
        .values()
        .filter(|n| n.op == "LogicalTensorInputLit")
        .find(|n| named(&egraph.nodes[&n.children[0]].eclass))
        .map(|n| n.eclass.clone())
        .unwrap_or_else(|| panic!("no LogicalTensorInputLit declares {name:?}"))
}

/// Every layout class some `LayoutTensorLit` gives that logical value.
fn layout_classes(egraph: &EGraph, logical: &ClassId) -> Vec<ClassId> {
    egraph
        .nodes
        .values()
        .filter(|n| n.op == "LayoutTensorLit" && egraph.nodes[&n.children[0]].eclass == *logical)
        .map(|n| egraph.nodes[&n.children[1]].eclass.clone())
        .collect()
}

/// Does some layout class of this input hold the given spelling?
fn holds_spelling(view: &EGraphView<'_>, logical: &ClassId, constructor: &str) -> bool {
    layout_classes(view.egraph(), logical)
        .iter()
        .any(|class| view.class(class).nodes_named(constructor).next().is_some())
}

/// A COLUMN-MAJOR INPUT: the boundary the binding stated is the boundary
/// the e-graph holds.
#[test]
fn a_column_major_input_carries_the_left_major_spelling() {
    let mut cx = Graph::new();
    let x = cx.named_tensor("x", (2usize, 3usize), DType::F32);
    let c = cx.tensor((2usize, 3usize), DType::F32);
    let out = x * c;

    let mut bindings = CudaBindings::new();
    bindings.input_with(x.id, BoundaryLayout::ColumnMajor);
    bindings.input(c.id);
    bindings.output(out.id);
    let rt = CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt())
        .expect("column-major load");

    let egraph = rt.saturated_egraph().expect("saturation");
    let view = EGraphView::new(&egraph, rt.decoders());
    let x_class = input_class(&egraph, "x");
    assert!(
        holds_spelling(&view, &x_class, "LeftMajorContiguousElementLayoutLit"),
        "the column-major input's layout class holds no LeftMajorContiguousElementLayoutLit"
    );
}

/// A STRIDED INPUT: the spelling is minted, and the plan the search
/// installs reads the caller's storage through exactly that map.
#[test]
fn a_strided_input_is_spelled_and_planned_at_its_own_strides() {
    // (2,3) at element strides [1, 2] — the first axis is the fast one,
    // so out (i, j) lives at flat i + 2j.
    let strides = vec![1i64, 2];
    let mut cx = Graph::new();
    let x = cx.named_tensor("x", (2usize, 3usize), DType::F32);
    let c = cx.tensor((2usize, 3usize), DType::F32);
    let out = x * c;

    let mut bindings = CudaBindings::new();
    bindings.input_with(
        x.id,
        BoundaryLayout::Strided {
            strides: strides.clone(),
        },
    );
    bindings.input(c.id);
    bindings.output(out.id);
    let mut rt =
        CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt()).expect("load");

    let egraph = rt.saturated_egraph().expect("saturation");
    let view = EGraphView::new(&egraph, rt.decoders());
    let x_class = input_class(&egraph, "x");
    // The spelling alone does not pin the map — the preamble equates a
    // contiguous layout with its own strided spelling — so the decoded
    // read below is the discriminating fact.
    assert!(
        holds_spelling(&view, &x_class, "StridedElementLayoutLit"),
        "the strided input's layout class holds no StridedElementLayoutLit"
    );

    // The host search (heuristic ranking, no device) installs a plan; the
    // buffer the caller stages `x` into carries the layout it stated.
    rt.search(&Default::default(), &harness_search_options())
        .expect("host search");
    let lit = rt.input_buffer(x.id).expect("x has an input buffer");
    let plan = rt.plan().expect("the search installed a plan");
    let buffer = plan
        .buffers
        .values()
        .find(|buffer| buffer.lit == Some(lit))
        .expect("the plan holds the bound input's buffer");
    assert_eq!(buffer.layout.literal_extents(), Some(vec![2, 3]));
    for i in 0..2usize {
        for j in 0..3usize {
            assert_eq!(
                buffer.layout.element_index(&[i, j]).expect("decoded read"),
                i * strides[0] as usize + j * strides[1] as usize,
                "strided input: ({i},{j}) must read flat {}",
                i + 2 * j
            );
        }
    }
}

/// A REFUSED STRIDED BINDING: the stride count is the value's rank, and
/// the strides are positive — stated by name at load, not discovered at
/// saturation.
#[test]
fn a_strided_binding_states_one_positive_stride_per_axis() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 3usize), DType::F32);
    let out = x + 1.;

    let refusal = |layout: BoundaryLayout| {
        let mut bindings = CudaBindings::new();
        bindings.input_with(x.id, layout);
        bindings.output(out.id);
        match CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt()) {
            Ok(_) => panic!("the binding must be refused"),
            Err(refusal) => refusal.to_string(),
        }
    };
    assert!(
        refusal(BoundaryLayout::Strided {
            strides: vec![1, 2, 3],
        })
        .contains("rank"),
        "a rank mismatch must be named"
    );
    assert!(
        refusal(BoundaryLayout::Strided {
            strides: vec![0, 1]
        })
        .contains("positive"),
        "a non-positive stride must be named"
    );
}

/// AN OUTPUT ON AN INPUT'S BUFFER: aliasing has exactly one spelling —
/// two bindings naming one buffer id — and the plan puts both boundary
/// values on that one buffer.
#[test]
fn an_output_bound_on_a_read_write_input_shares_its_buffer() {
    let mut cx = Graph::new();
    let state = cx.tensor(4usize, DType::F32);
    let delta = cx.tensor(4usize, DType::F32);
    let next = state + delta;

    let mut bindings = CudaBindings::new();
    let state_buffer = bindings.input(state.id);
    bindings.declare(state_buffer, Access::ReadWrite, FreedBy::Caller);
    bindings.input(delta.id);
    bindings.output_on(next.id, state_buffer);
    let mut rt =
        CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt()).expect("load");
    rt.search(&Default::default(), &harness_search_options())
        .expect("host search");

    assert_eq!(
        rt.input_buffer(state.id).expect("state binding"),
        state_buffer
    );
    let slot_index = rt.output_slot_index(next.id).expect("next binding");
    let plan = rt.plan().expect("the search installed a plan");
    let input_buffer = plan
        .buffers
        .values()
        .find(|buffer| buffer.lit == Some(state_buffer))
        .expect("the plan holds the bound buffer");
    let slot = plan
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::BufferOutput { slots } => Some(slots),
            _ => None,
        })
        .flatten()
        .find(|slot| slot.index == slot_index)
        .expect("the plan delivers the bound output slot");
    assert_eq!(
        slot.buffer, input_buffer.id,
        "the output slot must write the input's own buffer, not a copy of it"
    );
}

/// ONE VALUE DELIVERED TWICE: legal (two bindings, two buffers), but
/// there is then no tensor-keyed answer to "which slot" — the ambiguity
/// is refused by name rather than resolved first-wins.
#[test]
fn a_value_bound_on_two_buffers_has_no_tensor_keyed_slot() {
    let mut cx = Graph::new();
    let x = cx.tensor(4usize, DType::F32);
    let out = x + 1.;

    let mut bindings = CudaBindings::new();
    bindings.input(x.id);
    bindings.output(out.id);
    bindings.output(out.id);
    let rt = CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt()).expect("load");
    let refusal = match rt.output_slot_index(out.id) {
        Ok(slot) => panic!("a doubly-bound output must not resolve to slot {slot}"),
        Err(refusal) => refusal.to_string(),
    };
    assert!(
        refusal.contains("2 buffers"),
        "the refusal must name the ambiguity: {refusal}"
    );
}
