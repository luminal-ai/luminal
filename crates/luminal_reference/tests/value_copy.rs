//! A value delivery has an explicit span; whole-storage repairs remain exact-size.
use luminal::bufferize::{
    Buffer, BufferEdge, BufferId, BufferIrGraph, BufferNode, EdgeKind, InputBinding, OutputBinding,
    Owner,
};
use luminal::dtype::PlanDtype;
use luminal::egglog_utils::eclass::Spellings;
use luminal::layout_ir::{Access, FreedBy};
use luminal::layouts::{
    BitWidthTerm, DecodedLayout, IntExprTerm, Layout, RightMajorContiguousElementLayout, ShapeTerm,
    StridedElementLayout,
};
use luminal::prelude::SymbolBounds;
use luminal_reference::{ReferenceRuntime, TypedBuffer};
use petgraph::graph::DiGraph;
use std::sync::Arc;

fn plan(value_copy: bool) -> BufferIrGraph<DecodedLayout> {
    let shape = ShapeTerm(vec![IntExprTerm::Var("m".into()), IntExprTerm::Lit(4)]);
    let dense = DecodedLayout {
        class: "dense".into(),
        dtype: Some(PlanDtype::F32),
        spellings: Spellings::<Layout>::from_decoded(vec![Arc::new(
            RightMajorContiguousElementLayout {
                shape: shape.clone(),
                width: BitWidthTerm(32),
            },
        )]),
    };
    let pitched = DecodedLayout {
        class: "pitched".into(),
        dtype: Some(PlanDtype::F32),
        spellings: Spellings::<Layout>::from_decoded(vec![Arc::new(StridedElementLayout {
            shape,
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
        slots: vec![
            InputBinding {
                value: "source".into(),
                buffer: source.clone(),
            },
            InputBinding {
                value: "old-destination".into(),
                buffer: destination.clone(),
            },
        ],
    });
    let copy = dag.add_node(if value_copy {
        BufferNode::BufferCopyValue {
            src: source.clone(),
            dst: destination.clone(),
            layout: dense.clone(),
        }
    } else {
        BufferNode::BufferCopy {
            src: source.clone(),
            dst: destination.clone(),
        }
    });
    let output = dag.add_node(BufferNode::BufferOutput {
        slots: vec![OutputBinding {
            index: 0,
            value: "delivered".into(),
            buffer: destination.clone(),
            layout: dense.clone(),
        }],
    });
    for (from, to, buffer) in [
        (input, copy, source.clone()),
        (input, copy, destination.clone()),
        (copy, output, destination.clone()),
    ] {
        dag.add_edge(
            from,
            to,
            BufferEdge {
                buffer,
                port: "transport".into(),
                kind: EdgeKind::Data,
            },
        );
    }
    let buffers = [(0, source, dense), (1, destination, pitched)]
        .into_iter()
        .map(|(lit, id, layout)| {
            (
                id.clone(),
                Buffer {
                    id,
                    access: Access::ReadWrite,
                    freed_by: FreedBy::Caller,
                    owner: Owner::Caller,
                    label: format!("buffer {lit}"),
                    lit: Some(lit),
                    backs: format!("backing {lit}").into(),
                    layout,
                },
            )
        })
        .collect();
    BufferIrGraph {
        dag,
        buffers,
        value_buffer: Default::default(),
        outputs: vec![output],
    }
}
#[test]
fn value_delivery_preserves_capacity_tail_and_round_trips_in_compiled_artifacts() {
    use luminal_reference::compiled_artifact::{self, CompiledProgram};
    let bounds = SymbolBounds::from_ranges([('m'.into(), (0, 3))]).unwrap();
    let bytes = compiled_artifact::serialize(&CompiledProgram {
        bounds: bounds.clone(),
        input_buffers: vec![0, 1],
        output_buffers: vec![1],
        plan: plan(true),
    })
    .unwrap();
    let decoded = compiled_artifact::deserialize(&bytes).unwrap();
    let mut runtime = ReferenceRuntime::default();
    runtime.load_plan(decoded.plan, decoded.bounds);
    for rows in [0, 1, 2, 3, 0] {
        runtime.set_dim('m', rows);
        let n = rows * 4;
        let capacity = if rows == 0 { 0 } else { (rows - 1) * 8 + 4 };
        let source: Vec<_> = (0..n).map(|i| i as f32 + 0.25).collect();
        runtime.set_data_buffer(0, TypedBuffer::F32(source.clone()));
        runtime.set_data_buffer(1, TypedBuffer::F32(vec![-777.0; capacity]));
        runtime.execute().unwrap();
        let TypedBuffer::F32(actual) = runtime.get_buffer(1).unwrap() else {
            panic!("expected F32")
        };
        assert_eq!(&actual[..n], source);
        assert!(actual[n..].iter().all(|value| *value == -777.0));
    }
    let mut runtime = ReferenceRuntime::default();
    runtime.load_plan(plan(false), bounds);
    runtime.set_dim('m', 3);
    runtime.set_data_buffer(0, TypedBuffer::F32(vec![1.0; 12]));
    runtime.set_data_buffer(1, TypedBuffer::F32(vec![0.0; 20]));
    assert!(
        format!("{:#}", runtime.execute().unwrap_err()).contains("copy length mismatch"),
        "whole-storage copies must remain exact-size"
    );
}
