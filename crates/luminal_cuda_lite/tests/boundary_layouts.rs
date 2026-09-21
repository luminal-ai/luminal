//! THE BOUNDARY CARRIES A LAYOUT (CUDA-lite's divergence from the
//! reference binding): a caller hands this runtime the storage it
//! already has — column-major, or strided — and says so in the binding
//! rather than repacking on the host.
//!
//! Asked of the SATURATED E-GRAPH the search reads and of the PLAN it
//! installs, never of an election: which spelling a class holds and what
//! a decoded layout computes are facts; which genome won is not.

use luminal::bufferize::{BufferIrGraph, BufferNode};
use luminal::dtype::DType;
use luminal::egglog_utils::eclass::EGraphView;
use luminal::graph::{DimBucket, Graph};
use luminal::layout_ir::{Access, FreedBy};
use luminal::layouts::DecodedLayout;
use luminal::prelude::egraph_serialize::{ClassId, EGraph};
use luminal::shape::{DynMap, IntExpr, Symbol};
use luminal_cuda_lite::bindings::BoundaryLayout;
use luminal_cuda_lite::kernels::{self, Coords};
use luminal_cuda_lite::symbolic::{self, Expr};
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

/// Every operand layout an elected `Compute` node reads one of these
/// bound buffers through, paired with the buffer it reads: the
/// production read path's own view of the caller's storage, which is
/// where a boundary layout ends up if it survives the search intact.
fn read_layouts<'a>(
    plan: &'a BufferIrGraph<DecodedLayout>,
    lits: &[i64],
) -> Vec<(i64, &'a DecodedLayout)> {
    let mut layouts = Vec::new();
    for node in plan.dag.node_weights() {
        let BufferNode::Compute {
            reads,
            operand_info,
            ..
        } = node
        else {
            continue;
        };
        for (slot, read) in reads.iter().enumerate() {
            if let Some(lit) = plan.buffers[read].lit.filter(|lit| lits.contains(lit)) {
                layouts.push((lit, &operand_info[slot].layout));
            }
        }
    }
    layouts
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
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn a_strided_input_is_spelled_and_planned_at_its_own_strides() {
    // (2,3) at element strides [1, 2] — the first axis is the fast one,
    // so out (i, j) lives at flat i + 2j.
    let strides = vec![1i64, 2];
    let mut cx = Graph::new();
    let x = cx.named_tensor("x", (2usize, 3usize), DType::F32);
    let c = cx.tensor((2usize, 3usize), DType::F32);
    let out = x * c;

    let mut bindings = CudaBindings::new();
    bindings.input_with(x.id, BoundaryLayout::strided_literal(strides.clone()));
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

/// A RENDERED CHAIN IS DISCOVERED CONTIGUOUS: the caller states element
/// strides and nothing else — (2,4) at [4, 1] — and the e-graph mints
/// the right-major spelling from the chain itself, so recognition is
/// the preamble's and never the caller's.
#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn a_right_major_strides_chain_is_discovered_contiguous() {
    let mut cx = Graph::new();
    let x = cx.named_tensor("x", (2usize, 4usize), DType::F32);
    let c = cx.tensor((2usize, 4usize), DType::F32);
    let out = x * c;

    let mut bindings = CudaBindings::new();
    bindings.input_with(x.id, BoundaryLayout::strided_literal([4, 1]));
    bindings.input(c.id);
    bindings.output(out.id);
    let mut rt = CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt())
        .expect("a rendered strides chain loads");

    let egraph = rt.saturated_egraph().expect("saturation");
    let view = EGraphView::new(&egraph, rt.decoders());
    let x_class = input_class(&egraph, "x");
    assert!(
        holds_spelling(&view, &x_class, "RightMajorContiguousElementLayoutLit"),
        "the row-major chain's layout class holds no RightMajorContiguousElementLayoutLit"
    );
    rt.search(&Default::default(), &harness_search_options())
        .expect("host search over a discovered-contiguous boundary");
}

/// THE SAME DISCOVERY AT A SYMBOLIC DIM: shape (n, 4) at strides
/// [4, 1] is the right-major chain for every n, so the spelling is
/// minted without the dim having a value, and the search plans each
/// bucket.
#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn a_symbolic_shape_at_right_major_strides_is_discovered_contiguous() {
    let mut cx = Graph::new();
    let x = cx.named_tensor("x", ('n', 4usize), DType::F32);
    let c = cx.named_tensor("c", ('n', 4usize), DType::F32);
    let out = x * c;

    let mut bindings = CudaBindings::new();
    bindings.input_with(x.id, BoundaryLayout::strided_literal([4, 1]));
    bindings.input(c.id);
    bindings.output(out.id);
    let mut rt = CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt())
        .expect("a symbolic-shape strides chain loads");

    let egraph = rt.saturated_egraph().expect("saturation");
    let view = EGraphView::new(&egraph, rt.decoders());
    let x_class = input_class(&egraph, "x");
    assert!(
        holds_spelling(&view, &x_class, "RightMajorContiguousElementLayoutLit"),
        "the symbolic shape's row-major chain holds no RightMajorContiguousElementLayoutLit"
    );

    rt.bind_dim_buckets('n', vec![DimBucket::new(2, 4), DimBucket::new(5, 9)])
        .expect("disjoint sorted buckets bind");
    rt.search(&Default::default(), &harness_search_options())
        .expect("host search over a discovered-contiguous symbolic boundary");
    assert_eq!(rt.bucket_plans().len(), 2, "one plan per bucket");
    for n in [3usize, 7] {
        let mut dims = DynMap::default();
        dims.insert(Symbol::from('n'), n);
        assert!(
            luminal_cuda_lite::search::select_bucket(rt.bucket_plans(), &dims).is_some(),
            "no plan covers n = {n}"
        );
    }
}

/// A DEGENERATE AXIS CARRIES NO STRIDE: torch's stride on a size-1 axis
/// is arbitrary, so shape (1,4) arrives at [7, 1]. That axis's
/// coordinate is zero, so the summand the 7 multiplies is zero and the
/// chain the caller rendered IS the contiguous chain — the class holds
/// BOTH contiguous spellings, which at this shape are one map, and the
/// read the plan installs never reaches the 7.
#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn an_arbitrary_stride_on_a_degenerate_axis_is_discovered_contiguous() {
    let mut cx = Graph::new();
    let x = cx.named_tensor("x", (1usize, 4usize), DType::F32);
    let c = cx.tensor((1usize, 4usize), DType::F32);
    let out = x * c;

    let mut bindings = CudaBindings::new();
    bindings.input_with(x.id, BoundaryLayout::strided_literal([7, 1]));
    bindings.input(c.id);
    bindings.output(out.id);
    let mut rt = CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt())
        .expect("a degenerate-axis strides chain loads");

    let egraph = rt.saturated_egraph().expect("saturation");
    let view = EGraphView::new(&egraph, rt.decoders());
    let x_class = input_class(&egraph, "x");
    assert!(
        holds_spelling(&view, &x_class, "RightMajorContiguousElementLayoutLit"),
        "the degenerate-axis chain holds no RightMajorContiguousElementLayoutLit"
    );
    assert!(
        holds_spelling(&view, &x_class, "LeftMajorContiguousElementLayoutLit"),
        "the degenerate-axis chain holds no LeftMajorContiguousElementLayoutLit"
    );

    rt.search(&Default::default(), &harness_search_options())
        .expect("host search over a degenerate-axis boundary");
    let lit = rt.input_buffer(x.id).expect("x has an input buffer");
    let plan = rt.plan().expect("the search installed a plan");
    let buffer = plan
        .buffers
        .values()
        .find(|buffer| buffer.lit == Some(lit))
        .expect("the plan holds the bound input's buffer");
    for j in 0..4usize {
        assert_eq!(
            buffer.layout.element_index(&[0, j]).expect("decoded read"),
            j,
            "the stride stated on the size-1 axis reached an address"
        );
    }
}

/// A REFUSED STRIDED BINDING: the stride count is the value's rank and
/// no literal stride is negative — stated by name at load, not
/// discovered at saturation.
#[test]
fn a_strided_binding_states_one_non_negative_stride_per_axis() {
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
        refusal(BoundaryLayout::strided_literal([1, 2, 3])).contains("rank"),
        "a rank mismatch must be named"
    );
    assert!(
        refusal(BoundaryLayout::strided_literal([-1, 1])).contains("never negative"),
        "a negative stride must be named"
    );
}

/// A ZERO STRIDE IS A BROADCAST READ MAP: a legitimate read of the
/// caller's storage, planned and lowered as one row rather than widened
/// into a materialized copy.
#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn a_zero_stride_input_plans_as_a_broadcast_read() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 3usize), DType::F32);
    let delta = cx.tensor((2usize, 3usize), DType::F32);
    let out = x + delta;

    let mut bindings = CudaBindings::new();
    bindings.input_with(x.id, BoundaryLayout::strided_literal([0, 1]));
    bindings.input(delta.id);
    bindings.output(out.id);
    let mut rt = CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt())
        .expect("a broadcast read map binds");
    rt.search(&Default::default(), &harness_search_options())
        .expect("host search over a broadcast read map");

    // THE BROADCAST SURVIVES TO THE READ: the map an elected node reads
    // `x` through spans one row — 1 + (2-1)*0 + (3-1)*1 = 3 — rather than
    // the six elements a materialized copy would need, and the production
    // read path lowers it.
    let lit = rt.input_buffer(x.id).expect("x has an input buffer");
    let plan = rt.plan().expect("the search installed a plan");
    let layouts = read_layouts(plan, &[lit]);
    assert!(
        !layouts.is_empty(),
        "no elected node reads the broadcast input's buffer"
    );
    for (_, layout) in layouts {
        assert_eq!(
            symbolic::span(layout)
                .expect("the broadcast read map has a span")
                .eval(&DynMap::default())
                .expect("a literal span evaluates"),
            3,
            "the zero stride was widened into a materialized row"
        );
        let dims: Vec<Expr> = layout.shape().0.iter().cloned().map(Expr).collect();
        kernels::layout_read_index("boundary", layout, &dims, Coords::FlatIndex { prefix: "c" })
            .expect("the broadcast read map lowers");
    }
}

/// WRITABILITY IS THE SEARCH'S QUESTION, never bind's: a mutation sink
/// binds at the layout its target has, and a layout no kernel writes —
/// here a broadcast, where every coordinate would land on one element —
/// is answered by a search that finds no plan and names the output.
#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn a_zero_stride_sink_binds_and_the_search_names_it() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 3usize), DType::F32);
    let delta = cx.tensor((2usize, 3usize), DType::F32);
    let out = x + delta;

    let mut bindings = CudaBindings::new();
    let home = bindings.input_with(x.id, BoundaryLayout::strided_literal([0, 1]));
    bindings.declare(home, Access::ReadWrite, FreedBy::Caller);
    bindings.input(delta.id);
    bindings.output_on_with(out.id, home, BoundaryLayout::strided_literal([0, 1]));
    let mut rt = CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt())
        .expect("a stride-0 sink binds");

    let refusal = match rt.search(&Default::default(), &harness_search_options()) {
        Ok(_) => panic!("a broadcast write map must plan nothing"),
        Err(refusal) => format!("{refusal:#}"),
    };
    assert!(
        refusal.contains(&format!("v{}", out.id.index())) && refusal.contains("Strided"),
        "the refusal must name the output and its bound layout: {refusal}"
    );
}

/// A COLUMN-MAJOR WRITEBACK TARGET binds — the sink takes the layout its
/// target has — and the search answers: today no kernel writes a
/// left-major destination, so the refusal names the output and the
/// layout instead of a bind-time prior about writability.
#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn a_column_major_sink_binds_and_the_search_names_it() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 3usize), DType::F32);
    let delta = cx.tensor((2usize, 3usize), DType::F32);
    let out = x + delta;

    let mut bindings = CudaBindings::new();
    let home = bindings.input_with(x.id, BoundaryLayout::ColumnMajor);
    bindings.declare(home, Access::ReadWrite, FreedBy::Caller);
    bindings.input(delta.id);
    bindings.output_on_with(out.id, home, BoundaryLayout::ColumnMajor);
    let mut rt = CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt())
        .expect("a column-major sink binds");
    let refusal = match rt.search(&Default::default(), &harness_search_options()) {
        Ok(_) => panic!("no kernel writes a left-major destination today"),
        Err(refusal) => format!("{refusal:#}"),
    };
    assert!(
        refusal.contains(&format!("v{}", out.id.index())) && refusal.contains("ColumnMajor"),
        "the refusal must name the output and its bound layout: {refusal}"
    );
}

/// A USER-VISIBLE OUTPUT IS BOUND AT THE CALLER'S OWN STRIDES, and
/// whether a kernel writes them is the search's question: today no
/// elected op writes a left-major destination, so the search plans
/// nothing and names the output and the layout it is bound at. LUM-830
/// — a layout-changing copy into the bound output layout — is the flip
/// that makes this one plan.
#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn a_column_major_output_binds_and_the_search_names_it() {
    let mut cx = Graph::new();
    let a = cx.tensor((2usize, 3usize), DType::F32);
    let b = cx.tensor((2usize, 3usize), DType::F32);
    let out = a + b;

    let mut bindings = CudaBindings::new();
    bindings.input_external(a.id);
    bindings.input_external(b.id);
    bindings.output_external_with(out.id, BoundaryLayout::ColumnMajor);
    let mut rt = CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt())
        .expect("a column-major output binds");

    let refusal = match rt.search(&Default::default(), &harness_search_options()) {
        Ok(_) => panic!("no kernel writes a left-major destination today"),
        Err(refusal) => format!("{refusal:#}"),
    };
    assert!(
        refusal.contains(&format!("v{}", out.id.index())) && refusal.contains("ColumnMajor"),
        "the refusal must name the output and its bound layout: {refusal}"
    );
}

/// THE SAME PROGRAM AT THE OTHER BOUND LAYOUT PLANS: a row-major output
/// is written straight into the caller's allocation, and the guard reads
/// that the elected slot sits on the bound buffer rather than on a view
/// of it.
#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn a_row_major_output_bound_external_plans() {
    let mut cx = Graph::new();
    let a = cx.tensor((2usize, 3usize), DType::F32);
    let b = cx.tensor((2usize, 3usize), DType::F32);
    let out = a + b;

    let mut bindings = CudaBindings::new();
    bindings.input_external(a.id);
    bindings.input_external(b.id);
    let buffer = bindings.output_external_with(out.id, BoundaryLayout::RowMajor);
    let mut rt = CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt())
        .expect("a row-major output binds");
    rt.search(&Default::default(), &harness_search_options())
        .expect("host search over a row-major bound output");

    assert_eq!(rt.output_buffer(out.id).expect("out has a buffer"), buffer);
    rt.check_external_outputs()
        .expect("the elected slot sits on the bound buffer");
}

/// A ZERO EXTENT AT TORCH'S OWN CONTIGUOUS CHAIN: torch's running
/// product multiplies by `max(size, 1)`, so a contiguous (4, 0) arrives
/// at [1, 1]; the preamble's right-major fold multiplies by the raw
/// extent and renders [0, 1]. They differ on the axis OUTSIDE the empty
/// one, whose coordinate is not zero, so the bound chain is no
/// contiguous spelling — and only a contiguous spelling certifies a
/// destination injective. The search answers by naming the output and
/// the layout it is bound at, exactly as it does for a left-major
/// destination; at the chain the preamble itself renders, the same
/// empty output plans.
#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn a_zero_extent_output_at_torchs_contiguous_chain_is_named_by_the_search() {
    let plan_at = |strides: [i64; 2]| {
        let mut cx = Graph::new();
        let a = cx.tensor((4usize, 0usize), DType::F32);
        let b = cx.tensor((4usize, 0usize), DType::F32);
        let out = a + b;

        let mut bindings = CudaBindings::new();
        bindings.input_external(a.id);
        bindings.input_external(b.id);
        bindings.output_external_with(out.id, BoundaryLayout::strided_literal(strides));
        let mut rt = CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt())
            .expect("an empty bound output loads");
        let result = rt
            .search(&Default::default(), &harness_search_options())
            .map(|_| ());
        (out.id, result)
    };

    let (out, refused) = plan_at([1, 1]);
    let refusal = match refused {
        Ok(()) => panic!("torch's zero-extent chain is no contiguous spelling here"),
        Err(refusal) => format!("{refusal:#}"),
    };
    assert!(
        refusal.contains(&format!("v{}", out.index())) && refusal.contains("[1, 1]"),
        "the refusal must name the output and its bound chain: {refusal}"
    );

    let (_, rendered) = plan_at([0, 1]);
    rendered.expect("the preamble's own right-major rendering of (4, 0) plans");
}

/// A SYMBOLIC STRIDE ON A USER-VISIBLE OUTPUT ROUND-TRIPS THROUGH BIND:
/// the output's element layout reaches the preamble as the dim itself,
/// so the caller's declared strides stay a statement over the program's
/// dimensions rather than the numbers one call happened to have.
#[test]
fn a_symbolic_strided_output_binds_at_its_own_dim() {
    let mut cx = Graph::new();
    let x = cx.tensor(('n', 4usize), DType::F32);
    let out = x + 1.;

    let mut bindings = CudaBindings::new();
    bindings.input_external(x.id);
    bindings.output_external_with(
        out.id,
        BoundaryLayout::Strided {
            strides: vec![IntExpr::from(1i64), IntExpr::from('n')],
        },
    );
    let bound = bindings
        .bind(&cx.logical)
        .expect("a symbolic strided output binds");
    let shape = cx
        .logical
        .value_shape_term(out.id)
        .expect("the output states its shape");
    let expected = format!(
        "(StridedElementLayoutLit {shape} \
         (IntAffineExprCons (IntMul (CoordVar {shape} 1) (IntLit 1)) \
         (IntAffineExprCons (IntMul (CoordVar {shape} 0) (IntVar \"n\")) \
         (IntAffineExprNil))) (bits-of (F32)))"
    );
    assert!(
        bound.prefix.contains(&expected),
        "the output's declared stride was frozen: {}",
        bound.prefix
    );
}

/// AN OUTPUT ON AN INPUT'S BUFFER: aliasing has exactly one spelling —
/// two bindings naming one buffer id — and the plan puts both boundary
/// values on that one buffer.
#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
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

/// Does the term rooted at this class reach an `(IntVar "name")`?
fn reaches_int_var(egraph: &EGraph, root: &ClassId, name: &str) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![root.clone()];
    while let Some(class) = stack.pop() {
        if !seen.insert(class.clone()) {
            continue;
        }
        for node in egraph.nodes.values().filter(|n| n.eclass == class) {
            if node.op == "IntVar"
                && node
                    .children
                    .iter()
                    .any(|child| egraph.nodes[child].op.trim_matches('"') == name)
            {
                return true;
            }
            stack.extend(
                node.children
                    .iter()
                    .map(|child| egraph.nodes[child].eclass.clone()),
            );
        }
    }
    false
}

/// A SYMBOLIC STRIDE IS A BINDING, not a number the caller must have.
/// Three shape-`(n, 4)` inputs whose element strides name `n` itself:
/// `x` is the transposed (column-major) view, `w` is a four-column slice
/// of an `(n, n)` buffer, which at symbolic `n` is no contiguous form at
/// all and can only be read through its chain, and `v` is every other
/// column of a column-major `(n, 8)` buffer, whose fast-axis stride is a
/// COMPOUND expression over the dim. The e-graph holds all three
/// spellings with the dim in them, the search plans them over whole
/// buckets, and the reads the elected nodes lower go through the dim
/// parameter instead of a baked stride.
#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn symbolic_strided_inputs_are_spelled_planned_and_lowered_through_their_dim() {
    let mut cx = Graph::new();
    let x = cx.named_tensor("x", ('n', 4usize), DType::F32);
    let w = cx.named_tensor("w", ('n', 4usize), DType::F32);
    let v = cx.named_tensor("v", ('n', 4usize), DType::F32);
    let out = x * w * v;

    let dim = IntExpr::from('n');
    let one = IntExpr::from(1i64);
    let mut bindings = CudaBindings::new();
    bindings.input_with(
        x.id,
        BoundaryLayout::Strided {
            strides: vec![one, dim],
        },
    );
    bindings.input_with(
        w.id,
        BoundaryLayout::Strided {
            strides: vec![dim, one],
        },
    );
    bindings.input_with(
        v.id,
        BoundaryLayout::Strided {
            strides: vec![one, dim * IntExpr::from(2i64)],
        },
    );
    bindings.output(out.id);
    let mut rt = CudaRuntime::load_with(&cx, bindings, cuda_registry_without_cublaslt())
        .expect("a symbolic strided boundary loads");

    // (a) THE SPELLING: the strided literal is in the e-graph, and its
    // chain carries the dim itself, not a number.
    let egraph = rt.saturated_egraph().expect("saturation");
    let view = EGraphView::new(&egraph, rt.decoders());
    for name in ["x", "w", "v"] {
        let class = input_class(&egraph, name);
        assert!(
            holds_spelling(&view, &class, "StridedElementLayoutLit"),
            "{name}'s layout class holds no StridedElementLayoutLit"
        );
        let chain = layout_classes(&egraph, &class)
            .iter()
            .flat_map(|class| {
                view.class(class)
                    .nodes_named("StridedElementLayoutLit")
                    .filter_map(|node| node.child(1).map(|chain| chain.id().clone()))
                    .collect::<Vec<_>>()
            })
            .next()
            .unwrap_or_else(|| panic!("{name}'s strided spelling names no chain"));
        assert!(
            reaches_int_var(&egraph, &chain, "n"),
            "{name}'s strided chain holds no (IntVar \"n\") — the stride was frozen"
        );
    }

    // (b) A PLAN AT BOTH DIMS: one search per bucket, each valid over its
    // whole interval, and both dims select one.
    rt.bind_dim_buckets('n', vec![DimBucket::new(2, 4), DimBucket::new(5, 9)])
        .expect("disjoint sorted buckets bind");
    rt.search(&Default::default(), &harness_search_options())
        .expect("host search over the symbolic strided boundary");
    assert_eq!(rt.bucket_plans().len(), 2, "one plan per bucket");
    for n in [3usize, 7] {
        let mut dims = DynMap::default();
        dims.insert(Symbol::from('n'), n);
        assert!(
            luminal_cuda_lite::search::select_bucket(rt.bucket_plans(), &dims).is_some(),
            "no plan covers n = {n}"
        );
    }

    // (c) THE LOWERED READ: ask the production read path for each node
    // that reads a caller buffer. The index must name the dim parameter
    // — a literal there would be the bucket representative baked in.
    let lits: Vec<i64> = [x.id, w.id, v.id]
        .iter()
        .map(|id| rt.input_buffer(*id).expect("the input has a buffer"))
        .collect();
    let parameter = symbolic::variable("n");
    for bucket in rt.bucket_plans() {
        let layouts = read_layouts(&bucket.plan, &lits);
        for (_, layout) in &layouts {
            let dims: Vec<Expr> = layout.shape().0.iter().cloned().map(Expr).collect();
            let (code, index) = kernels::layout_read_index(
                "boundary",
                layout,
                &dims,
                Coords::FlatIndex { prefix: "c" },
            )
            .expect("the symbolic strided boundary lowers");
            assert!(
                code.contains(&parameter) || index.contains(&parameter),
                "a read of the caller's storage baked a literal stride: {code}{index}"
            );
        }
        for (name, lit) in ["x", "w", "v"].iter().zip(&lits) {
            assert!(
                layouts.iter().any(|(read, _)| read == lit),
                "bucket {:?}: no elected node reads {name}'s own storage",
                bucket.ranges
            );
        }
    }
}
