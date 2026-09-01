//! The destination-passing-style rewrite (value-SSA → value-SSA).
//!
//! For every op whose [`ToDps`](crate::layout_ir::ToDps) declares a DPS form,
//! replace it with that form and append one **write-only, poison-valued
//! destination operand per result** (a consecutive trailing range — MLIR's tie
//! invariant). Each destination is produced by a synthesized
//! [`Poison`](crate::layout_ir::Poison) node carrying the tied result's shape,
//! dtype, and — critically — its **layout** (the physical-equivalence gate keys
//! on layout e-class identity).
//!
//! The interior stays pure value-SSA: a poison destination is just a value with
//! undefined contents. No buffer notion is introduced here — which storage the
//! destination denotes is decided entirely inside `bufferize()` (fresh
//! allocation today; a seeded output buffer once destination seeding lands).
//!
//! Idempotent by construction: DPS forms answer `to_dps() = None`.

use egraph_serialize::ClassId;

use crate::layout_ir::{
    ExtractedEdge, ExtractedGraph, ExtractedNode, LayoutTensorInfo, LogicalInfo, OpInput, OpNode,
    Provenance,
};
use crate::reference::ops::Poison;

/// Rewrite every DPS-capable op into its destination-passing form. Out of
/// place: the source graph is untouched.
pub fn dps_rewrite(graph: &ExtractedGraph) -> ExtractedGraph {
    let mut out = graph.clone();
    // Poison identities must stay unique ACROSS invocations (a later pass may
    // insert new DPS-capable ops and re-run the rewrite): continue from the
    // highest synthesized id already in the graph.
    let mut synth: u32 = out
        .dag
        .node_weights()
        .filter_map(|node| match node {
            ExtractedNode::LayoutOp(op) => match op.provenance {
                Provenance::Synthesized { id } => Some(id),
                Provenance::Extracted { .. } => None,
            },
            _ => None,
        })
        .max()
        .unwrap_or(0);

    let op_indices: Vec<_> = out
        .dag
        .node_indices()
        .filter(|&idx| matches!(&out.dag[idx], ExtractedNode::LayoutOp(_)))
        .collect();

    for idx in op_indices {
        let (dps_op, results) = {
            let ExtractedNode::LayoutOp(op) = &out.dag[idx] else {
                unreachable!()
            };
            let Some(dps_op) = op.op.to_dps() else {
                continue;
            };
            (dps_op, op.outputs.clone())
        };

        // The tie invariant — one trailing destination per result — holds by
        // construction: this loop IS its definition. The data/destination
        // operand split is likewise not bookkept here; the DPS form's own
        // `alias_info()` and per-operand read declarations are its single
        // definition.
        for result in &results {
            synth += 1;
            // The destination value: fresh identity, fresh POISON logical
            // (never the result's logical — the values are different things),
            // but the result's shape/dtype/layout, so the tied pair is
            // physically coincident wherever they share storage.
            let poison_value = LayoutTensorInfo {
                eclass: ClassId::from(format!("dps$poison${synth}")),
                // Display label only — identity lives in the eclass; the
                // number would leak into buffer labels (alloc#N[poisonNN])
                // where alloc#N already disambiguates.
                label: "Poison".to_string(),
                tooltip: "poison destination (contents undefined)".to_string(),
                shape: result.shape.clone(),
                dtype: result.dtype.clone(),
                dtype_enum: result.dtype_enum,
                // The poison destination is physically coincident with its
                // tied result, geometry included.
                dims: result.dims.clone(),
                element_bits: result.element_bits,
                logical: LogicalInfo {
                    eclass: ClassId::from(format!("dps$poison_logical${synth}")),
                    // Display label only — identity lives in the eclass, so
                    // every poison shows plainly as "Poison", unnumbered.
                    label: "Poison".to_string(),
                    tooltip: "undefined contents".to_string(),
                    op: None,
                    children: Vec::new(),
                },
                layout: result.layout.clone(),
            };
            let poison_eclass = poison_value.eclass.clone();
            let poison_node = out.dag.add_node(ExtractedNode::LayoutOp(OpNode {
                op: Box::new(Poison),
                provenance: Provenance::Synthesized { id: synth },
                inputs: Vec::new(),
                outputs: vec![poison_value],
                tooltip: "synthesized by dps_rewrite".to_string(),
                heuristic_cost: 0,
            }));

            let ExtractedNode::LayoutOp(op) = &mut out.dag[idx] else {
                unreachable!()
            };
            // The slot name comes from the DPS form itself (`OpSlotNames`,
            // the single source of slot naming): the dest for result j sits
            // at signature index data_len + j.
            let port = dps_op.operand_name(op.inputs.len());
            op.inputs.push(OpInput {
                port: port.clone(),
                value: poison_eclass.clone(),
            });
            out.dag.add_edge(
                poison_node,
                idx,
                ExtractedEdge {
                    value: poison_eclass,
                    port,
                },
            );
        }

        let ExtractedNode::LayoutOp(op) = &mut out.dag[idx] else {
            unreachable!()
        };
        op.op = dps_op;
    }

    out
}
