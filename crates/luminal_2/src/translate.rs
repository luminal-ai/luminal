use crate::{
    CompatKernel, Diff, GraphTerm,
    codegen::{GRID_DIMS, THREADBLOCK_DIMS},
    utils::{loop_in, loop_out},
};
use luminal::prelude::{
    petgraph::{Directed, algo::toposort, prelude::StableGraph},
    *,
};
use rustc_hash::FxHashMap;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub enum InitData {
    Expr(Expression),
    Data(Vec<f32>),
}

impl InitData {
    pub fn to_vec(self, dyn_map: &FxHashMap<char, usize>) -> Vec<f32> {
        match self {
            InitData::Data(d) => d,
            InitData::Expr(e) => vec![e.exec(dyn_map).unwrap() as f32],
        }
    }
}

pub type CrossSubGraphTensorIndexes = (NodeIndex, NodeIndex);
pub type MetaGraphNodeIndex = NodeIndex;
pub type SubGraphNodeIndex = NodeIndex;
pub type OrigGraphNodeIndex = NodeIndex;
pub type OptimalGraphNodeIndex = NodeIndex;

pub type SubGraph = StableGraph<GraphTerm, (), Directed>;
pub type MetaGraph = StableGraph<SubGraph, CrossSubGraphTensorIndexes, Directed>;

fn collect_ancestors(graph: &Graph, n: NodeIndex) -> HashSet<NodeIndex> {
    let mut seen = HashSet::new();
    let mut stack = vec![n];
    while let Some(cur) = stack.pop() {
        for (src, _out, _shape) in graph.get_sources(cur) {
            if seen.insert(src) {
                stack.push(src);
            }
        }
    }
    seen
}

pub fn translate_graph(
    graph: &Graph,
) -> (
    MetaGraph,
    FxHashMap<NodeIndex, (NodeIndex /*meta*/, NodeIndex /*inner*/)>,
    Vec<(String /*orig*/, InitData)>,
) {
    let mut meta: MetaGraph = MetaGraph::new();

    // --- current slice state ---
    let mut g: SubGraph = SubGraph::new();
    let mut orig_to_subgraph_node_map: FxHashMap<(OrigGraphNodeIndex, usize), SubGraphNodeIndex> =
        FxHashMap::default();
    let mut key_to_subgraph_buffer: FxHashMap<NodeIndex, SubGraphNodeIndex> = FxHashMap::default(); // can hold original or synthetic nodeIndexes
    let mut inits: Vec<(String, InitData)> = vec![];
    let mut simplify_cache = FxHashMap::default();

    // meta-level outputs
    let mut global_map: FxHashMap<OrigGraphNodeIndex, (MetaGraphNodeIndex, SubGraphNodeIndex)> =
        FxHashMap::default();

    // For the CURRENT slice being built, remember meta-edge stubs that must be wired
    // when this slice is finalized: (src_meta, src_inner_out, dst_inner_placeholder)
    let mut pending_in_edges: Vec<(NodeIndex, NodeIndex, NodeIndex)> = vec![];

    // finalize current slice into the meta-graph and return its meta node
    let finalize_current_slice = |meta: &mut MetaGraph,
                                  g: &mut SubGraph,
                                  old_to_new: &mut FxHashMap<NodeIndex, NodeIndex>,
                                  pending_in_edges: &mut Vec<(NodeIndex, NodeIndex, NodeIndex)>,
                                  global_map: &mut FxHashMap<NodeIndex, (NodeIndex, NodeIndex)>|
     -> NodeIndex {
        // move inner graph into meta node
        let meta_node = meta.add_node(std::mem::take(g));

        // wire all deferred inbound edges to this meta node
        for (src_meta, src_inner, dst_inner) in pending_in_edges.drain(..) {
            meta.add_edge(src_meta, meta_node, (src_inner, dst_inner));
        }

        // publish inits and global mapping for nodes in the slice we just sealed
        for (old, inner) in old_to_new.drain() {
            global_map.insert(old, (meta_node, inner));
        }

        // clear per-slice maps (node_map cleared by caller if needed)
        meta_node
    };

    for old_node in toposort(&graph.graph, None).unwrap() {
        let node_weight = graph.node_weight(old_node).unwrap();
        let op_name_full = format!("{node_weight:?}");
        let op = op_name_full
            .split('|')
            .next()
            .unwrap_or(&op_name_full)
            .trim();
        let mut sources = graph.get_sources(old_node);
        match op {
            // ---- GRAPH BREAK ----
            "GraphBreak" => {
                // 1) identify producer's inner node (no scope_in/out needed)
                let (src_old, out_idx, _shape_unused) = sources.pop().unwrap();
                let producer_inner_out = orig_to_subgraph_node_map[&(src_old, out_idx as usize)];

                // 2) seal current slice into meta
                let producer_meta = finalize_current_slice(
                    &mut meta,
                    &mut g,
                    &mut key_to_subgraph_buffer,
                    &mut pending_in_edges,
                    &mut global_map,
                );

                // 3) start fresh slice; create placeholder fed by the producer
                let ancestors = collect_ancestors(graph, old_node);
                orig_to_subgraph_node_map.retain(|(orig, _out), _| !ancestors.contains(orig));
                simplify_cache = FxHashMap::default();

                let placeholder = g.add_node(GraphTerm::GMEM {
                    label: format!("break_{}", old_node.index()),
                });

                // defer wiring: (producer_meta, producer_inner_out) -> (this_meta (TBD), placeholder)
                pending_in_edges.push((producer_meta, producer_inner_out, placeholder));

                // map Break node to the placeholder (passthrough)
                orig_to_subgraph_node_map.insert((old_node, 0), placeholder);
                key_to_subgraph_buffer.insert(old_node, placeholder);
            }

            // ---- UNARY ELEMENTWISE ----
            "Sqrt" | "Exp2" | "Log2" | "Sin" | "Contiguous" | "Recip" => {
                let (s0, i0, shape0) = sources.pop().unwrap();
                let base0 = orig_to_subgraph_node_map[&(s0, i0 as usize)];
                let (base0, ranges) = scope_in(
                    base0,
                    shape0,
                    None,
                    &mut g,
                    &mut inits,
                    &mut simplify_cache,
                    &mut key_to_subgraph_buffer,
                    graph.node_count(),
                );

                let mut out = if op == "Contiguous" {
                    base0
                } else {
                    let r = g.add_node(match op {
                        "Sqrt" => GraphTerm::Sqrt,
                        "Exp2" => GraphTerm::Exp2,
                        "Log2" => GraphTerm::Log2,
                        "Sin" => GraphTerm::Sin,
                        "Recip" => GraphTerm::Recip,
                        _ => unreachable!(),
                    });
                    g.add_edge(base0, r, ());
                    r
                };
                out = scope_out(out, ranges, None, &mut g);
                key_to_subgraph_buffer.insert(old_node, out);
                orig_to_subgraph_node_map.insert((old_node, 0), out);
            }

            // ---- BINARY ELEMENTWISE ----
            "Add" | "Mul" | "Mod" | "LessThan" => {
                let (sa, ia, shape_a) = sources.pop().unwrap();
                let (sb, ib, shape_b) = sources.pop().unwrap();
                let (ain, ranges) = scope_in(
                    orig_to_subgraph_node_map[&(sa, ia as usize)],
                    shape_a,
                    None,
                    &mut g,
                    &mut inits,
                    &mut simplify_cache,
                    &mut key_to_subgraph_buffer,
                    graph.node_count(),
                );
                let (bin, _) = scope_in(
                    orig_to_subgraph_node_map[&(sb, ib as usize)],
                    shape_b,
                    None,
                    &mut g,
                    &mut inits,
                    &mut simplify_cache,
                    &mut key_to_subgraph_buffer,
                    graph.node_count(),
                );

                let mut opn = g.add_node(match op {
                    "Add" => GraphTerm::Add,
                    "Mul" => GraphTerm::Mul,
                    "Mod" => GraphTerm::Mod,
                    "LessThan" => GraphTerm::LessThan,
                    _ => unreachable!(),
                });
                g.add_edge(ain, opn, ());
                g.add_edge(bin, opn, ());
                opn = scope_out(opn, ranges, None, &mut g);
                key_to_subgraph_buffer.insert(old_node, opn);
                orig_to_subgraph_node_map.insert((old_node, 0), opn);
            }

            // ---- REDUCTIONS ----
            s if s.starts_with("SumReduce") || s.starts_with("MaxReduce") => {
                let (start_val, term, reduce_dim) = match op {
                    s if s.starts_with("SumReduce") => {
                        (0.0, GraphTerm::Add, graph.get_op::<SumReduce>(old_node).0)
                    }
                    s if s.starts_with("MaxReduce") => (
                        f32::NEG_INFINITY,
                        GraphTerm::Max,
                        graph.get_op::<MaxReduce>(old_node).0,
                    ),
                    _ => unreachable!(),
                };

                let acc_name = format!("acc_{}", inits.len());
                let mut acc = g.add_node(GraphTerm::GMEM {
                    label: acc_name.clone(),
                });
                let orig_acc = acc;

                let (source, output_index, shape) = sources.pop().unwrap();
                let (new_source, ranges) = scope_in(
                    orig_to_subgraph_node_map[&(source, output_index as usize)],
                    shape,
                    Some(reduce_dim),
                    &mut g,
                    &mut inits,
                    &mut simplify_cache,
                    &mut key_to_subgraph_buffer,
                    graph.node_count(),
                );

                let mut rm_strides = ranges
                    .iter()
                    .rev()
                    .scan(Expression::from(1), |i, (s, _)| {
                        let r = *i;
                        *i *= s;
                        Some(r)
                    })
                    .collect::<Vec<_>>();
                rm_strides.reverse();
                for (i, ((range, name), acc_stride)) in ranges.iter().zip(rm_strides).enumerate() {
                    let stride = if i == reduce_dim {
                        Expression::from(Term::Acc('a'))
                    } else if i > GRID_DIMS + THREADBLOCK_DIMS {
                        Expression::from('z') * acc_stride
                    } else {
                        Expression::from(0)
                    };
                    let new_acc = g.add_node(GraphTerm::LoopIn {
                        range: *range,
                        stride,
                        marker: name.to_string(),
                    });
                    g.add_edge(acc, new_acc, ());
                    acc = new_acc;
                }
                key_to_subgraph_buffer
                    .insert(NodeIndex::new(graph.node_count() + inits.len()), orig_acc);
                inits.push((acc_name, InitData::Data(vec![start_val])));

                let mut opn = g.add_node(term);
                g.add_edge(new_source, opn, ());
                g.add_edge(acc, opn, ());
                opn = scope_out(opn, ranges, Some(reduce_dim), &mut g);
                key_to_subgraph_buffer.insert(old_node, opn);
                orig_to_subgraph_node_map.insert((old_node, 0), opn);
            }

            // ---- CONSTANTS / CUSTOM / DIFF / LOADS ----
            _ => {
                if let Some(constant) = node_weight.as_any().downcast_ref::<Constant>() {
                    let newn = g.add_node(GraphTerm::GMEM {
                        label: op.to_string(),
                    });
                    key_to_subgraph_buffer.insert(old_node, newn);
                    orig_to_subgraph_node_map.insert((old_node, 0), newn);
                    key_to_subgraph_buffer
                        .insert(NodeIndex::new(graph.node_count() + inits.len()), newn);
                    inits.push((
                        op.to_string(),
                        match constant.0 {
                            ConstantValue::Expression(e) => InitData::Expr(e),
                            ConstantValue::Float(f) => InitData::Data(vec![f]),
                        },
                    ));
                } else if let Some(kernel) = node_weight.as_any().downcast_ref::<CompatKernel>() {
                    let custom = g.add_node(GraphTerm::Custom(kernel.0.clone()));
                    for (source, ind, _) in sources {
                        g.add_edge(
                            orig_to_subgraph_node_map[&(source, ind as usize)],
                            custom,
                            (),
                        );
                    }
                    for i in 0..kernel.0.outputs.len() {
                        orig_to_subgraph_node_map.insert((old_node, i), custom);
                    }
                    key_to_subgraph_buffer.insert(old_node, custom);
                } else if let Some(diff) = node_weight.as_any().downcast_ref::<Diff>() {
                    let custom = g.add_node(GraphTerm::Diff(diff.name.clone()));
                    for (source, ind, _) in sources {
                        g.add_edge(
                            orig_to_subgraph_node_map[&(source, ind as usize)],
                            custom,
                            (),
                        );
                    }
                    orig_to_subgraph_node_map.insert((old_node, 0), custom);
                    key_to_subgraph_buffer.insert(old_node, custom);
                } else {
                    // Assume a load
                    let newn = g.add_node(GraphTerm::GMEM {
                        label: op.to_string(),
                    });
                    // println!("LOAD: {:?} | {:?}", old_node, newn);
                    orig_to_subgraph_node_map.insert((old_node, 0), newn);
                    key_to_subgraph_buffer.insert(old_node, newn);
                }
            }
        }
    }

    // seal trailing slice if non-empty or if it has pending inbound edges
    if g.node_count() > 0 || !pending_in_edges.is_empty() {
        finalize_current_slice(
            &mut meta,
            &mut g,
            &mut key_to_subgraph_buffer,
            &mut pending_in_edges,
            &mut global_map,
        );
    }

    (meta, global_map, inits)
}

fn scope_in(
    mut src: NodeIndex,
    shape: ShapeTracker,
    reduce_dim: Option<usize>,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
    inits: &mut Vec<(String, InitData)>,
    simplify_cache: &mut FxHashMap<Expression, Expression>,
    translation_mapping: &mut FxHashMap<NodeIndex, NodeIndex>,
    n_orig_nodes: usize,
) -> (NodeIndex, Vec<(Expression, String)>) {
    // Loop in through all dimensions, handle padding
    let strides = shape.strides();
    let mut ranges = vec![];
    let mut pad_mask = None; // mask for pads
    for i in 0..shape.len() {
        let mut range = shape.dims()[i];
        let mut stride = strides[i] * 'z';
        let (left_pad, right_pad) = shape.padding[shape.indexes[i]];
        let (left_slice, right_slice) = shape.mask[shape.indexes[i]];
        if reduce_dim == Some(i) {
            assert!(
                left_pad == 0 && right_pad == 0,
                "pad on a reduce dim not implemented!"
            );
            // for z in i..THREADBLOCK_DIMS + GRID_DIMS {
            //     ranges.push((Expression::from(1), format!("pad{z}")));
            //     src = loop_in(src, 1, 0, format!("pad{z}"), graph);
            // }
            ranges.push((range, i.to_string()));
            src = loop_in(
                src,
                range.simplify_cache(simplify_cache),
                stride.simplify_cache(simplify_cache),
                i,
                graph,
            ); // Problem: acc stride only is ever 'a', which doesn't work if there is multiple accs!
        } else if left_pad != 0 {
            // Pad left
            stride = stride.substitute('z', (Expression::from('z') - left_pad).max(0));
            // Bring in mask
            let mut mask = graph.add_node(GraphTerm::GMEM {
                label: "Mask".to_string(),
            });
            translation_mapping.insert(NodeIndex::new(n_orig_nodes + inits.len()), mask);
            inits.push(("Mask".to_string(), InitData::Data(vec![0., 1.])));
            // Loop mask in
            for level in 0..i {
                mask = loop_in(
                    mask,
                    shape.dims()[level].simplify_cache(simplify_cache),
                    0,
                    level,
                    graph,
                );
            }
            mask = loop_in(
                mask,
                range.simplify_cache(simplify_cache),
                Expression::from('z')
                    .gte(left_pad)
                    .simplify_cache(simplify_cache),
                i,
                graph,
            );
            for level in (i + 1)..shape.len() {
                mask = loop_in(
                    mask,
                    shape.dims()[level].simplify_cache(simplify_cache),
                    0,
                    level,
                    graph,
                );
            }
            pad_mask = Some(mask);
            ranges.push((range, i.to_string()));
            src = loop_in(
                src,
                range.simplify_cache(simplify_cache),
                stride.simplify_cache(simplify_cache),
                i,
                graph,
            );
        } else if right_pad != 0 {
            // Pad right
            stride =
                stride.substitute('z', Expression::from('z').min(shape.dims[shape.indexes[i]]));
            // Bring in mask
            let mut mask = graph.add_node(GraphTerm::GMEM {
                label: "Mask".to_string(),
            });
            translation_mapping.insert(NodeIndex::new(n_orig_nodes + inits.len()), mask);
            inits.push(("Mask".to_string(), InitData::Data(vec![0., 1.])));
            // Loop mask in
            for level in 0..i {
                mask = loop_in(
                    mask,
                    shape.dims()[level].simplify_cache(simplify_cache),
                    0,
                    level,
                    graph,
                );
            }
            mask = loop_in(
                mask,
                range.simplify_cache(simplify_cache),
                Expression::from('z').lt(right_pad),
                i,
                graph,
            );
            for level in (i + 1)..shape.len() {
                mask = loop_in(
                    mask,
                    shape.dims()[level].simplify_cache(simplify_cache),
                    0,
                    level,
                    graph,
                );
            }
            pad_mask = Some(mask);
            ranges.push((range, i.to_string()));
            src = loop_in(
                src,
                range.simplify_cache(simplify_cache),
                stride.simplify_cache(simplify_cache),
                i,
                graph,
            );
        } else if left_slice != 0 || right_slice != i32::MAX {
            range = (right_slice.min(shape.dims[shape.indexes[i]]) - left_slice).max(0);
            stride = stride.substitute('z', Expression::from('z') + left_slice);
            ranges.push((range, i.to_string()));
            src = loop_in(
                src,
                range.simplify_cache(simplify_cache),
                stride.simplify_cache(simplify_cache),
                i,
                graph,
            );
        } else {
            // No pads or reduces
            ranges.push((range, i.to_string()));
            src = loop_in(
                src,
                range.simplify_cache(simplify_cache),
                stride.simplify_cache(simplify_cache),
                i,
                graph,
            );
        }
    }
    src = if let Some(mask) = pad_mask {
        crate::utils::binary(src, mask, GraphTerm::Mul, graph)
    } else {
        src
    };
    (src, ranges)
}

fn scope_out(
    mut src: NodeIndex,
    ranges: Vec<(Expression, String)>,
    reduce: Option<usize>,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
) -> NodeIndex {
    for (stride, range, loop_name) in ranges.into_iter().enumerate().rev().scan(
        Expression::from('z'),
        |i, (ind, (range, loop_name))| {
            if reduce == Some(ind) {
                Some((Expression::from(Term::Acc('a')), range, loop_name)) // THIS IS WRONG, IT SHOULD MIRROR THE INCOMING ACC AND BE RANDOMLY GENERATED
            } else {
                let r = *i;
                *i *= range;
                Some((r, range, loop_name))
            }
        },
    ) {
        src = loop_out(src, range, stride, loop_name, graph);
    }
    src
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph {
        let mut cx = Graph::new();

        // Create some basic tensors for testing
        let a = cx.tensor((2, 3)).set([[1., 2., 3.], [4., 5., 6.]]);
        let b = cx.tensor((2, 3)).set([[7., 8., 9.], [10., 11., 12.]]);

        // Add some operations that will be translated
        let _c = (a + b).retrieve();

        cx
    }

    #[test]
    fn test_node_indexes_consistency() {
        let cx = create_test_graph();
        let (meta_graph, global_map, _inits) = translate_graph(&cx);

        // Test that all original nodes are mapped to (meta_node, sub_node) pairs
        for orig_node in cx.graph.node_indices() {
            assert!(
                global_map.contains_key(&orig_node),
                "Original node {:?} not found in global mapping",
                orig_node
            );

            let (meta_node, sub_node) = global_map[&orig_node];

            // Verify meta node exists in meta graph
            assert!(
                meta_graph.node_weight(meta_node).is_some(),
                "Meta node {:?} not found in meta graph",
                meta_node
            );

            // Verify sub node exists in the corresponding subgraph
            let subgraph = meta_graph.node_weight(meta_node).unwrap();
            assert!(
                subgraph.node_weight(sub_node).is_some(),
                "Sub node {:?} not found in subgraph of meta node {:?}",
                sub_node,
                meta_node
            );
        }
    }

    #[test]
    fn test_unary_sqrt_translation() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([1., 4., 9.]);
        let _b = a.sqrt().retrieve();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);

        // Find the sqrt operation in the translated graph
        let mut found_sqrt = false;
        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            for sub_node_idx in subgraph.node_indices() {
                if let Some(GraphTerm::Sqrt) = subgraph.node_weight(sub_node_idx) {
                    found_sqrt = true;
                    break;
                }
            }
        }

        assert!(found_sqrt, "Sqrt operation not found in translated graph");
    }

    #[test]
    fn test_unary_exp2_translation() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([1., 2., 3.]);
        let _b = a.exp2().retrieve();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);

        let mut found_exp2 = false;
        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            for sub_node_idx in subgraph.node_indices() {
                if let Some(GraphTerm::Exp2) = subgraph.node_weight(sub_node_idx) {
                    found_exp2 = true;
                    break;
                }
            }
        }

        assert!(found_exp2, "Exp2 operation not found in translated graph");
    }

    #[test]
    fn test_unary_log2_translation() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([1., 2., 4.]);
        let _b = a.log2().retrieve();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);

        let mut found_log2 = false;
        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            for sub_node_idx in subgraph.node_indices() {
                if let Some(GraphTerm::Log2) = subgraph.node_weight(sub_node_idx) {
                    found_log2 = true;
                    break;
                }
            }
        }

        assert!(found_log2, "Log2 operation not found in translated graph");
    }

    #[test]
    fn test_unary_sin_translation() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([0., 1.5708, 3.14159]);
        let _b = a.sin().retrieve();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);

        let mut found_sin = false;
        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            for sub_node_idx in subgraph.node_indices() {
                if let Some(GraphTerm::Sin) = subgraph.node_weight(sub_node_idx) {
                    found_sin = true;
                    break;
                }
            }
        }

        assert!(found_sin, "Sin operation not found in translated graph");
    }

    #[test]
    fn test_unary_recip_translation() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([1., 2., 4.]);
        let _b = a.reciprocal().retrieve();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);

        let mut found_recip = false;
        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            for sub_node_idx in subgraph.node_indices() {
                if let Some(GraphTerm::Recip) = subgraph.node_weight(sub_node_idx) {
                    found_recip = true;
                    break;
                }
            }
        }

        assert!(found_recip, "Recip operation not found in translated graph");
    }

    #[test]
    fn test_unary_contiguous_translation() {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 3)).set([[1., 2., 3.], [4., 5., 6.]]);
        let _b = a.permute((1, 0)).contiguous().retrieve();

        let (_meta_graph, global_map, _inits) = translate_graph(&cx);

        // Contiguous should be handled correctly in the translation
        // It doesn't create a new GraphTerm but passes through the base node
        assert!(
            !global_map.is_empty(),
            "Global map should not be empty after contiguous operation"
        );
    }

    #[test]
    fn test_binary_add_translation() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([1., 2., 3.]);
        let b = cx.tensor(3).set([4., 5., 6.]);
        let _c = (a + b).retrieve();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);

        let mut found_add = false;
        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            for sub_node_idx in subgraph.node_indices() {
                if let Some(GraphTerm::Add) = subgraph.node_weight(sub_node_idx) {
                    found_add = true;
                    break;
                }
            }
        }

        assert!(found_add, "Add operation not found in translated graph");
    }

    #[test]
    fn test_binary_mul_translation() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([1., 2., 3.]);
        let b = cx.tensor(3).set([4., 5., 6.]);
        let _c = (a * b).retrieve();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);

        let mut found_mul = false;
        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            for sub_node_idx in subgraph.node_indices() {
                if let Some(GraphTerm::Mul) = subgraph.node_weight(sub_node_idx) {
                    found_mul = true;
                    break;
                }
            }
        }

        assert!(found_mul, "Mul operation not found in translated graph");
    }

    #[test]
    fn test_binary_mod_translation() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([5., 7., 9.]);
        let b = cx.tensor(3).set([2., 3., 4.]);
        let _c = (a % b).retrieve();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);

        let mut found_mod = false;
        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            for sub_node_idx in subgraph.node_indices() {
                if let Some(GraphTerm::Mod) = subgraph.node_weight(sub_node_idx) {
                    found_mod = true;
                    break;
                }
            }
        }

        assert!(found_mod, "Mod operation not found in translated graph");
    }

    #[test]
    fn test_binary_less_than_translation() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([1., 5., 3.]);
        let b = cx.tensor(3).set([2., 4., 6.]);
        let _c = a.gt(b).retrieve();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);

        let mut found_less_than = false;
        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            for sub_node_idx in subgraph.node_indices() {
                if let Some(GraphTerm::LessThan) = subgraph.node_weight(sub_node_idx) {
                    found_less_than = true;
                    break;
                }
            }
        }

        assert!(
            found_less_than,
            "LessThan operation not found in translated graph"
        );
    }

    #[test]
    fn test_sum_reduce_translation() {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 3)).set([[1., 2., 3.], [4., 5., 6.]]);
        let _b = a.sum(1).retrieve();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);

        // Sum reduction should create Add nodes in the translated graph
        let mut found_reduce_add = false;
        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            for sub_node_idx in subgraph.node_indices() {
                if let Some(GraphTerm::Add) = subgraph.node_weight(sub_node_idx) {
                    // Check if this Add node has the structure of a reduction
                    // (should have LoopIn nodes feeding into it)
                    let predecessors: Vec<_> = subgraph
                        .neighbors_directed(sub_node_idx, petgraph::Direction::Incoming)
                        .collect();
                    if predecessors.len() == 2 {
                        found_reduce_add = true;
                        break;
                    }
                }
            }
        }

        assert!(found_reduce_add, "Sum reduction not properly translated");

        // Check that init data was created for the accumulator
        assert!(
            !_inits.is_empty(),
            "No init data created for reduction accumulator"
        );
    }

    #[test]
    fn test_max_reduce_translation() {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 3)).set([[1., 2., 3.], [4., 5., 6.]]);
        let _b = a.max(1).retrieve();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);

        // Max reduction should create Max nodes in the translated graph
        let mut found_reduce_max = false;
        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            for sub_node_idx in subgraph.node_indices() {
                if let Some(GraphTerm::Max) = subgraph.node_weight(sub_node_idx) {
                    let predecessors: Vec<_> = subgraph
                        .neighbors_directed(sub_node_idx, petgraph::Direction::Incoming)
                        .collect();
                    if predecessors.len() == 2 {
                        found_reduce_max = true;
                        break;
                    }
                }
            }
        }

        assert!(found_reduce_max, "Max reduction not properly translated");

        // Check that init data was created with NEG_INFINITY
        let has_neg_inf_init = _inits.iter().any(|(_, init_data)| {
            matches!(init_data, InitData::Data(data) if !data.is_empty() && data[0] == f32::NEG_INFINITY)
        });
        assert!(
            has_neg_inf_init,
            "NEG_INFINITY init data not found for max reduction"
        );
    }

    #[test]
    fn test_constant_translation() {
        let mut cx = Graph::new();
        let a = cx.constant(5.0);
        let _b = a.retrieve();

        let (meta_graph, _global_map, inits) = translate_graph(&cx);

        // Constants should be translated to GMEM nodes with appropriate init data
        let mut found_constant_gmem = false;
        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            for sub_node_idx in subgraph.node_indices() {
                if let Some(GraphTerm::GMEM { label }) = subgraph.node_weight(sub_node_idx) {
                    if label.contains("Constant(5.0)") {
                        found_constant_gmem = true;
                        break;
                    }
                }
            }
        }

        assert!(found_constant_gmem, "Constant not translated to GMEM node");
        assert!(!inits.is_empty(), "No init data created for constant");

        // Check that init data contains the constant value
        let has_constant_init = inits.iter().any(|(_, init_data)| {
            matches!(init_data, InitData::Data(data) if !data.is_empty() && data[0] == 5.0)
        });
        assert!(has_constant_init, "Constant value not found in init data");
    }

    #[test]
    fn test_graph_break_translation() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([1., 2., 3.]);
        let b = cx.tensor(3).set([1., 2., 3.]);
        let c = (a + b).graph_break();
        let _d = c * 10.0;
        cx.execute_debug();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);

        // Graph break should create multiple meta nodes
        assert!(
            meta_graph.node_count() >= 2,
            "GraphBreak should create multiple meta nodes, found {}",
            meta_graph.node_count()
        );

        // There should be edges between meta nodes due to the break
        assert!(
            meta_graph.edge_count() > 0,
            "GraphBreak should create edges between meta nodes"
        );

        // Check that the break node created a placeholder GMEM node
        let mut found_break_placeholder = false;
        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            for sub_node_idx in subgraph.node_indices() {
                if let Some(GraphTerm::GMEM { label }) = subgraph.node_weight(sub_node_idx) {
                    if label.starts_with("break_") {
                        found_break_placeholder = true;
                        break;
                    }
                }
            }
        }

        assert!(found_break_placeholder, "GraphBreak placeholder not found");
    }

    #[test]
    fn test_complex_graph_node_mapping() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([1., 2., 3.]);
        let b = cx.tensor(3).set([4., 5., 6.]);
        let c = (a + b).sqrt();
        let d = c * a;
        let _e = d.sum(0).retrieve();

        let (meta_graph, global_map, _inits) = translate_graph(&cx);

        // Verify all original nodes are mapped
        for orig_node in cx.graph.node_indices() {
            assert!(
                global_map.contains_key(&orig_node),
                "Original node {:?} missing from global mapping",
                orig_node
            );
        }

        // Verify meta graph structure makes sense
        assert!(meta_graph.node_count() > 0, "Meta graph should have nodes");

        // Verify each subgraph has reasonable structure
        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            assert!(subgraph.node_count() > 0, "Each subgraph should have nodes");
        }
    }

    #[test]
    fn test_scope_in_out_structure() {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 3)).set([[1., 2., 3.], [4., 5., 6.]]);
        let b = cx.tensor((2, 3)).set([[7., 8., 9.], [10., 11., 12.]]);
        let _c = (a + b).retrieve();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);

        // Check that LoopIn and LoopOut nodes are created properly
        let mut found_loop_in = false;
        let mut found_loop_out = false;

        for meta_node_idx in meta_graph.node_indices() {
            let subgraph = meta_graph.node_weight(meta_node_idx).unwrap();
            for sub_node_idx in subgraph.node_indices() {
                match subgraph.node_weight(sub_node_idx) {
                    Some(GraphTerm::LoopIn { .. }) => found_loop_in = true,
                    Some(GraphTerm::LoopOut { .. }) => found_loop_out = true,
                    _ => {}
                }
            }
        }

        assert!(found_loop_in, "LoopIn nodes should be created for scoping");
        assert!(
            found_loop_out,
            "LoopOut nodes should be created for scoping"
        );
    }
}
