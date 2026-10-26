//! Empty domains must size to zero through compilation and through externally
//! loaded strided plans. A scalar domain still has one element.
use luminal::bufferize::{
    Buffer, BufferEdge, BufferId, BufferIrGraph, BufferNode, EdgeKind, InputBinding, OutputBinding,
    Owner,
};
use luminal::dtype::PlanDtype;
use luminal::egglog_utils::eclass::Spellings;
use luminal::layout_ir::{Access, FreedBy};
use luminal::layouts::{
    BitWidthTerm, DecodedLayout, IntExprTerm, Layout, ShapeTerm, StridedElementLayout,
};
use luminal::prelude::{DType, Graph, SymbolBounds};
use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer, harness_search_options};
use petgraph::graph::DiGraph;
use rustc_hash::FxHashMap;
use std::sync::Arc;

#[test]
fn empty_and_pitched_spans_execute_through_reference_runtime() {
    // Exercise the ordinary recorder, matcher, extractor, bufferizer and kernels.
    for shape in [vec![0, 4], vec![3, 0], vec![2, 0, 3]] {
        let mut graph = Graph::new();
        let input = graph.tensor(shape.clone(), DType::F32);
        let empty_output = input + input;
        let scalar_output = empty_output.sum((0..shape.len()).collect::<Vec<_>>());
        let mut runtime = ReferenceRuntime::load_with(
            &graph,
            ReferenceBindings::dense(&graph.logical, &[empty_output.id, scalar_output.id]),
        )
        .unwrap();
        let data: FxHashMap<_, _> = [(input.id, TypedBuffer::F32(vec![]))].into_iter().collect();
        runtime
            .search(
                &Default::default(),
                &Default::default(),
                &data,
                &harness_search_options(),
            )
            .unwrap();
        runtime.set_data(input.id, Vec::<f32>::new());
        runtime.execute().unwrap();
        assert!(runtime.get_f32(empty_output.id).unwrap().is_empty());
        assert_eq!(runtime.get_f32(scalar_output.id).unwrap(), &vec![0.0]);
    }

    // An authored buffer-plan fixture exercises affine span sizing itself,
    // without a contiguous spelling masking a bad strided span. No op selection
    // or graph rewriting is performed here: the program copies one full buffer.
    let layout = DecodedLayout {
        class: "pitched-layout".into(),
        dtype: Some(PlanDtype::F32),
        spellings: Spellings::<Layout>::from_decoded(vec![Arc::new(StridedElementLayout {
            shape: ShapeTerm(vec![IntExprTerm::Var("m".into()), IntExprTerm::Lit(4)]),
            chain: vec![
                IntExprTerm::Mul(
                    Box::new(IntExprTerm::Coord { axis_from_end: 1 }),
                    Box::new(IntExprTerm::Lit(8)),
                ),
                IntExprTerm::Coord { axis_from_end: 0 },
            ],
            width: BitWidthTerm(32),
        })]),
    };
    let source = BufferId::Boundary("source".into());
    let destination = BufferId::Boundary("destination".into());
    let mut dag = DiGraph::new();
    let input = dag.add_node(BufferNode::BufferInput {
        slots: vec![InputBinding {
            value: "input".into(),
            buffer: source.clone(),
        }],
    });
    let copy = dag.add_node(BufferNode::BufferCopy {
        src: source.clone(),
        dst: destination.clone(),
    });
    let output = dag.add_node(BufferNode::BufferOutput {
        slots: vec![OutputBinding {
            index: 0,
            value: "output".into(),
            buffer: destination.clone(),
            layout: layout.clone(),
        }],
    });
    dag.add_edge(
        input,
        copy,
        BufferEdge {
            buffer: source.clone(),
            port: "source".into(),
            kind: EdgeKind::Data,
        },
    );
    dag.add_edge(
        copy,
        output,
        BufferEdge {
            buffer: destination.clone(),
            port: "output".into(),
            kind: EdgeKind::Data,
        },
    );
    let buffers = [(0, source), (1, destination)]
        .into_iter()
        .map(|(lit, id)| {
            (
                id.clone(),
                Buffer {
                    id,
                    access: Access::ReadWrite,
                    freed_by: FreedBy::Caller,
                    owner: Owner::Caller,
                    label: format!("buffer {lit}"),
                    lit: Some(lit),
                    backs: format!("value {lit}").into(),
                    layout: layout.clone(),
                },
            )
        })
        .collect();
    let plan = BufferIrGraph {
        dag,
        buffers,
        value_buffer: Default::default(),
        outputs: vec![output],
    };
    let mut runtime = ReferenceRuntime::default();
    runtime.load_plan(
        plan,
        SymbolBounds::from_ranges([('m'.into(), (0, 3))]).unwrap(),
    );
    for rows in [0, 1, 3, 0] {
        runtime.set_dim('m', rows);
        let elements = if rows == 0 { 0 } else { (rows - 1) * 8 + 4 };
        let data: Vec<f32> = (0..elements).map(|i| i as f32).collect();
        runtime.set_data_buffer(0, TypedBuffer::F32(data.clone()));
        runtime.execute().unwrap();
        let TypedBuffer::F32(actual) = runtime.get_buffer(1).unwrap() else {
            panic!("expected F32")
        };
        assert_eq!(actual, &data);
    }
}
