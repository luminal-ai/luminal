//! Loop materialization: expanding the rolled loop regions the auto-roll
//! prepass placed in an LLIR graph into the fully unrolled deployment graph
//! runtimes execute.

use std::collections::hash_map::Entry;

use itertools::Itertools;
use petgraph::{Direction, stable_graph::NodeIndex};
use rustc_hash::{FxHashMap, FxHashSet};

use super::packed::PackedLLIRGraph;
use crate::graph::LLIRGraph;
use crate::op::LLIROp;

/// Marker nodes and per-slot metadata of one rolled loop region, grouped
/// from the LLIR graph by `loop_id`.
#[derive(Default)]
struct LoopRegion {
    /// slot_idx → LoopStart.
    starts: std::collections::BTreeMap<usize, NodeIndex>,
    /// slot_idx → LoopEnd.
    ends: std::collections::BTreeMap<usize, NodeIndex>,
    /// stream_id → LoopInput.
    inputs: std::collections::BTreeMap<usize, NodeIndex>,
    /// stream_id → LoopOutput. Each stream has one LoopOutput.
    outputs: std::collections::BTreeMap<usize, NodeIndex>,
    /// LoopOutputSelect NodeIndex → (stream_id, iter).
    output_selects: FxHashMap<NodeIndex, (usize, usize)>,
    iters: usize,
    /// Every marker node of this region.
    markers: FxHashSet<NodeIndex>,
}

fn collect_loop_regions(llir: &LLIRGraph) -> std::collections::BTreeMap<usize, LoopRegion> {
    use crate::hlir::{LoopEnd, LoopInput, LoopOutput, LoopOutputSelect, LoopStart};

    let mut regions: std::collections::BTreeMap<usize, LoopRegion> =
        std::collections::BTreeMap::new();
    for n in llir.node_indices() {
        let op = &llir[n];
        let loop_id = if let Some(ls) = op.to_op::<LoopStart>() {
            let region = regions.entry(ls.loop_id).or_default();
            region.iters = region.iters.max(ls.iters.to_usize().unwrap_or(1));
            region.starts.insert(ls.slot_idx, n);
            ls.loop_id
        } else if let Some(le) = op.to_op::<LoopEnd>() {
            regions
                .entry(le.loop_id)
                .or_default()
                .ends
                .insert(le.slot_idx, n);
            le.loop_id
        } else if let Some(li) = op.to_op::<LoopInput>() {
            regions
                .entry(li.loop_id)
                .or_default()
                .inputs
                .insert(li.stream_id, n);
            li.loop_id
        } else if let Some(los) = op.to_op::<LoopOutputSelect>() {
            regions
                .entry(los.loop_id)
                .or_default()
                .output_selects
                .insert(n, (los.stream_id, los.iter));
            los.loop_id
        } else if let Some(lo) = op.to_op::<LoopOutput>() {
            regions
                .entry(lo.loop_id)
                .or_default()
                .outputs
                .insert(lo.stream_id, n);
            lo.loop_id
        } else {
            continue;
        };
        regions.entry(loop_id).or_default().markers.insert(n);
    }
    regions
}

/// Forward-reachable body of one region: successors of its entry markers,
/// stopping at `Output` ops and at any loop marker of any region. Also
/// reports whether a marker belonging to a *different* region was reached —
/// i.e. that region is nested inside this one.
fn loop_region_body(
    llir: &LLIRGraph,
    region: &LoopRegion,
    marker_owner: &FxHashMap<NodeIndex, usize>,
    self_id: usize,
) -> (FxHashSet<NodeIndex>, std::collections::BTreeSet<usize>) {
    use crate::hlir::Output;

    let mut body_nodes: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut foreign: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let mut worklist: Vec<NodeIndex> = region
        .starts
        .values()
        .chain(region.inputs.values())
        .flat_map(|n| {
            llir.neighbors_directed(*n, Direction::Outgoing)
                .collect::<Vec<_>>()
        })
        .collect();
    while let Some(n) = worklist.pop() {
        if body_nodes.contains(&n) {
            continue;
        }
        if let Some(&owner) = marker_owner.get(&n) {
            if owner != self_id {
                foreign.insert(owner);
            }
            continue;
        }
        if llir[n].to_op::<Output>().is_some() {
            continue;
        }
        body_nodes.insert(n);
        for succ in llir
            .neighbors_directed(n, Direction::Outgoing)
            .collect::<Vec<_>>()
        {
            worklist.push(succ);
        }
    }
    (body_nodes, foreign)
}

/// Remove and return an innermost loop region — one whose body contains no
/// other region's markers — together with that body. Marker ops are unique,
/// so a repeated occurrence can never contain one and regions are always
/// strictly nested or disjoint.
fn take_innermost_region(llir: &LLIRGraph) -> Option<(LoopRegion, FxHashSet<NodeIndex>)> {
    let mut regions = collect_loop_regions(llir);
    if regions.is_empty() {
        return None;
    }
    let marker_owner: FxHashMap<NodeIndex, usize> = regions
        .iter()
        .flat_map(|(&id, region)| region.markers.iter().map(move |&n| (n, id)))
        .collect();
    let (id, body_nodes) = regions
        .iter()
        .find_map(|(&id, region)| {
            let (body_nodes, foreign) = loop_region_body(llir, region, &marker_owner, id);
            foreign.is_empty().then_some((id, body_nodes))
        })
        .unwrap_or_else(|| {
            let contains: Vec<String> = regions
                .iter()
                .map(|(&id, region)| {
                    let (body_nodes, foreign) = loop_region_body(llir, region, &marker_owner, id);
                    // Re-walk with parent tracking to show the exact bridge
                    // edges into foreign markers.
                    let mut bridges: Vec<String> = Vec::new();
                    let mut seen: FxHashSet<NodeIndex> = FxHashSet::default();
                    let mut worklist: Vec<(NodeIndex, Option<NodeIndex>)> = region
                        .starts
                        .values()
                        .chain(region.inputs.values())
                        .flat_map(|m| {
                            llir.neighbors_directed(*m, Direction::Outgoing)
                                .map(|s| (s, Some(*m)))
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    while let Some((n, pred)) = worklist.pop() {
                        if seen.contains(&n) {
                            continue;
                        }
                        if let Some(&owner) = marker_owner.get(&n) {
                            if owner != id && bridges.len() < 4 {
                                let pred_desc = pred
                                    .map(|p| format!("{:?}", llir[p]))
                                    .unwrap_or_else(|| "<entry>".to_string());
                                bridges.push(format!(
                                    "      {} -> {:?}",
                                    &pred_desc[..pred_desc.len().min(160)],
                                    llir[n]
                                ));
                            }
                            continue;
                        }
                        seen.insert(n);
                        for succ in llir
                            .neighbors_directed(n, Direction::Outgoing)
                            .collect::<Vec<_>>()
                        {
                            worklist.push((succ, Some(n)));
                        }
                    }
                    format!(
                        "loop {id}: body={} starts={} inputs={} reaches markers of {foreign:?}\n{}",
                        body_nodes.len(),
                        region.starts.len(),
                        region.inputs.len(),
                        bridges.join("\n"),
                    )
                })
                .collect();
            panic!(
                "loop regions must be strictly nested or disjoint; none is innermost:\n  {}",
                contains.join("\n  ")
            )
        });
    Some((regions.remove(&id).unwrap(), body_nodes))
}

/// Inline every iteration-invariant input marker: rewire its consumers to
/// its single shared source and delete it. `LoopInputStatic` is invariant by
/// definition (one source), and a `LoopInput` whose per-iteration sources are
/// all the same node is invariant in fact. Egglog unions invariant markers
/// with their source value, which lets extraction elect a marker node as the
/// representative of a value class consumed far outside its region (an inner
/// region's invariant input is often exactly an enclosing region's value).
/// Region walks and rewiring must never see that cross-region aliasing, so
/// both `unroll_loops_in_llir` and `collapse_loops_to_first_iter` inline them
/// before touching any region.
fn inline_static_loop_inputs(llir: &mut LLIRGraph) {
    use crate::hlir::{LoopInput, LoopInputStatic};
    use petgraph::visit::EdgeRef;

    // One marker at a time with live edge reads: invariant markers chain
    // (an inner region's invariant input is often an enclosing region's
    // marker), so a source captured up front can be deleted before its
    // dependent marker is processed.
    while let Some((marker, source)) = llir.node_indices().find_map(|n| {
        if llir[n].to_op::<LoopInputStatic>().is_none() && llir[n].to_op::<LoopInput>().is_none() {
            return None;
        }
        let mut sources = llir.neighbors_directed(n, Direction::Incoming);
        let first = sources.next()?;
        sources.all(|s| s == first).then_some((n, first))
    }) {
        // Per-edge remove+add to keep each consumer's edge-id ordering via
        // LIFO reuse — the runtime reads inputs sorted by edge id.
        let consumers: Vec<(petgraph::graph::EdgeIndex, NodeIndex)> = llir
            .edges_directed(marker, Direction::Outgoing)
            .sorted_by_key(|e| e.id())
            .map(|e| (e.id(), e.target()))
            .collect();
        for (eid, consumer) in consumers {
            llir.remove_edge(eid);
            llir.add_edge(source, consumer, ());
        }
        llir.remove_node(marker);
        crate::mask_events::INVARIANT_MARKER_INLINED.record();
    }
}

#[derive(Clone)]
struct MaterializeLoopRegion {
    iters: usize,
}

#[derive(Clone)]
enum MaterializeLoopMarker {
    Start {
        region: usize,
        initial: usize,
        body_producer: usize,
    },
    End {
        region: usize,
        body_producer: usize,
    },
    Input {
        region: usize,
        sources: Vec<usize>,
    },
    StaticInput {
        source: usize,
    },
    Output {
        region: usize,
        body_producer: usize,
    },
    OutputSelect {
        region: usize,
        iter: usize,
        body_producer: usize,
    },
}

struct MaterializeContextLayout {
    membership: u128,
    /// Region-local mixed-radix multipliers. Unlike the previous global
    /// multiplier, this includes only loops that actually contain the node.
    region_multipliers: Vec<(usize, usize)>,
    count: usize,
}

impl MaterializeContextLayout {
    fn new(mut membership: u128, regions: &[MaterializeLoopRegion]) -> Option<Self> {
        let original_membership = membership;
        let mut region_multipliers = Vec::with_capacity(membership.count_ones() as usize);
        let mut count = 1usize;
        while membership != 0 {
            let region = membership.trailing_zeros() as usize;
            region_multipliers.push((region, count));
            count = count.checked_mul(regions[region].iters)?;
            membership &= membership - 1;
        }
        Some(Self {
            membership: original_membership,
            region_multipliers,
            count,
        })
    }

    fn decode(&self, context: usize, regions: &[MaterializeLoopRegion], iterations: &mut [usize]) {
        debug_assert!(context < self.count);
        for &(region, multiplier) in &self.region_multipliers {
            iterations[region] = (context / multiplier) % regions[region].iters;
        }
    }

    fn encode(
        &self,
        assigned_membership: u128,
        regions: &[MaterializeLoopRegion],
        iterations: &[usize],
    ) -> Option<usize> {
        if self.membership & !assigned_membership != 0 {
            return None;
        }
        let mut context = 0usize;
        for &(region, multiplier) in &self.region_multipliers {
            let iter = iterations[region];
            if iter >= regions[region].iters {
                return None;
            }
            context = context.checked_add(iter.checked_mul(multiplier)?)?;
        }
        Some(context)
    }
}

struct MaterializeAdjacency {
    offsets: Vec<usize>,
    neighbors: Vec<usize>,
}

impl MaterializeAdjacency {
    fn incoming(llir: &LLIRGraph, node_bound: usize) -> Self {
        let mut offsets = vec![0usize; node_bound + 1];
        for edge in llir.edge_indices() {
            let (_, destination) = llir.edge_endpoints(edge).unwrap();
            offsets[destination.index() + 1] += 1;
        }
        for index in 1..offsets.len() {
            offsets[index] += offsets[index - 1];
        }
        let mut cursor = offsets[..node_bound].to_vec();
        let mut neighbors = vec![usize::MAX; llir.edge_count()];
        // StableGraph edge indices iterate in ascending slot order. Filling
        // each destination's range in that order preserves positional inputs
        // without sorting or allocating one Vec per node.
        for edge in llir.edge_indices() {
            let (source, destination) = llir.edge_endpoints(edge).unwrap();
            neighbors[cursor[destination.index()]] = source.index();
            cursor[destination.index()] += 1;
        }
        Self { offsets, neighbors }
    }

    fn outgoing(llir: &LLIRGraph, node_bound: usize) -> Self {
        let mut offsets = vec![0usize; node_bound + 1];
        for edge in llir.edge_indices() {
            let (source, _) = llir.edge_endpoints(edge).unwrap();
            offsets[source.index() + 1] += 1;
        }
        for index in 1..offsets.len() {
            offsets[index] += offsets[index - 1];
        }
        let mut cursor = offsets[..node_bound].to_vec();
        let mut neighbors = vec![usize::MAX; llir.edge_count()];
        for edge in llir.edge_indices() {
            let (source, destination) = llir.edge_endpoints(edge).unwrap();
            neighbors[cursor[source.index()]] = destination.index();
            cursor[source.index()] += 1;
        }
        Self { offsets, neighbors }
    }

    fn get(&self, node: usize) -> &[usize] {
        &self.neighbors[self.offsets[node]..self.offsets[node + 1]]
    }
}

trait MaterializeLLIRView {
    fn node_bound(&self) -> usize;
    fn op(&self, node: usize) -> Option<&LLIROp>;
    fn incoming(&self, node: usize) -> &[usize];
    fn outgoing(&self, node: usize) -> &[usize];
}

struct StableMaterializeView<'a> {
    graph: &'a LLIRGraph,
    incoming: MaterializeAdjacency,
    outgoing: MaterializeAdjacency,
    node_bound: usize,
}

impl<'a> StableMaterializeView<'a> {
    fn new(graph: &'a LLIRGraph) -> Self {
        let node_bound = petgraph::visit::NodeIndexable::node_bound(graph);
        Self {
            graph,
            incoming: MaterializeAdjacency::incoming(graph, node_bound),
            outgoing: MaterializeAdjacency::outgoing(graph, node_bound),
            node_bound,
        }
    }
}

impl MaterializeLLIRView for StableMaterializeView<'_> {
    fn node_bound(&self) -> usize {
        self.node_bound
    }

    fn op(&self, node: usize) -> Option<&LLIROp> {
        self.graph.node_weight(NodeIndex::new(node))
    }

    fn incoming(&self, node: usize) -> &[usize] {
        self.incoming.get(node)
    }

    fn outgoing(&self, node: usize) -> &[usize] {
        self.outgoing.get(node)
    }
}

impl MaterializeLLIRView for PackedLLIRGraph {
    fn node_bound(&self) -> usize {
        self.ops.len()
    }

    fn op(&self, node: usize) -> Option<&LLIROp> {
        self.ops.get(node)
    }

    fn incoming(&self, node: usize) -> &[usize] {
        &self.incoming_sources[self.incoming_offsets[node]..self.incoming_offsets[node + 1]]
    }

    fn outgoing(&self, node: usize) -> &[usize] {
        &self.outgoing_targets[self.outgoing_offsets[node]..self.outgoing_offsets[node + 1]]
    }
}

#[derive(Default)]
struct MaterializeCollectedRegion {
    starts: std::collections::BTreeMap<usize, usize>,
    ends: std::collections::BTreeMap<usize, usize>,
    inputs: std::collections::BTreeMap<usize, usize>,
    outputs: std::collections::BTreeMap<usize, usize>,
    output_selects: FxHashMap<usize, (usize, usize)>,
    iters: usize,
    markers: FxHashSet<usize>,
}

fn collect_materialize_loop_regions(
    llir: &impl MaterializeLLIRView,
) -> std::collections::BTreeMap<usize, MaterializeCollectedRegion> {
    use crate::hlir::{LoopEnd, LoopInput, LoopOutput, LoopOutputSelect, LoopStart};

    let mut regions: std::collections::BTreeMap<usize, MaterializeCollectedRegion> =
        std::collections::BTreeMap::new();
    for node in 0..llir.node_bound() {
        let Some(op) = llir.op(node) else {
            continue;
        };
        let loop_id = if let Some(marker) = op.to_op::<LoopStart>() {
            let region = regions.entry(marker.loop_id).or_default();
            region.iters = region.iters.max(marker.iters.to_usize().unwrap_or(1));
            region.starts.insert(marker.slot_idx, node);
            marker.loop_id
        } else if let Some(marker) = op.to_op::<LoopEnd>() {
            regions
                .entry(marker.loop_id)
                .or_default()
                .ends
                .insert(marker.slot_idx, node);
            marker.loop_id
        } else if let Some(marker) = op.to_op::<LoopInput>() {
            regions
                .entry(marker.loop_id)
                .or_default()
                .inputs
                .insert(marker.stream_id, node);
            marker.loop_id
        } else if let Some(marker) = op.to_op::<LoopOutputSelect>() {
            regions
                .entry(marker.loop_id)
                .or_default()
                .output_selects
                .insert(node, (marker.stream_id, marker.iter));
            marker.loop_id
        } else if let Some(marker) = op.to_op::<LoopOutput>() {
            regions
                .entry(marker.loop_id)
                .or_default()
                .outputs
                .insert(marker.stream_id, node);
            marker.loop_id
        } else {
            continue;
        };
        regions.entry(loop_id).or_default().markers.insert(node);
    }
    regions
}

/// Materialize a rolled LLIR into one fully-unrolled StableGraph without ever
/// mutating a graph containing loop markers.  Each surviving `(node, loop
/// context)` pair is allocated exactly once and its ordered incoming edges are
/// emitted directly into the final graph.  This avoids StableGraph free-list
/// edge reuse and therefore needs no repair/compaction pass.
fn materialize_unrolled_view(llir: &impl MaterializeLLIRView) -> Option<LLIRGraph> {
    use crate::hlir::{
        LoopEnd, LoopInput, LoopInputStatic, LoopOutput, LoopOutputSelect, LoopStart, Output,
    };
    let node_bound = llir.node_bound();

    let mut loop_regions = collect_materialize_loop_regions(llir);
    let mut markers: Vec<Option<MaterializeLoopMarker>> = vec![None; node_bound];

    // Invariant LoopInput variants are transparent and must not seed a loop
    // body. This mirrors `inline_static_loop_inputs` before region discovery.
    for (node, marker_slot) in markers.iter_mut().enumerate() {
        let Some(op) = llir.op(node) else {
            continue;
        };
        if op.to_op::<LoopInputStatic>().is_some() {
            let &source = llir.incoming(node).first()?;
            *marker_slot = Some(MaterializeLoopMarker::StaticInput { source });
        } else if let Some(input) = op.to_op::<LoopInput>() {
            let sources = llir.incoming(node);
            if let Some(&first) = sources.first()
                && sources.iter().all(|source| *source == first)
            {
                *marker_slot = Some(MaterializeLoopMarker::StaticInput { source: first });
                if let Some(region) = loop_regions.get_mut(&input.loop_id) {
                    region.inputs.remove(&input.stream_id);
                    region.markers.remove(&node);
                }
            }
        }
    }

    if loop_regions.len() > 128 {
        return None;
    }

    let mut region_by_id: FxHashMap<usize, usize> = FxHashMap::default();
    let mut regions = Vec::with_capacity(loop_regions.len());
    for (&loop_id, region) in &loop_regions {
        if region.iters <= 1 || region.starts.is_empty() {
            return None;
        }
        region_by_id.insert(loop_id, regions.len());
        regions.push(MaterializeLoopRegion {
            iters: region.iters,
        });
    }

    let mut marker_owner = vec![None; node_bound];
    for (&loop_id, region) in &loop_regions {
        let region_index = region_by_id[&loop_id];
        for &marker in &region.markers {
            marker_owner[marker] = Some(region_index);
        }

        for (&slot, &start) in &region.starts {
            let &end = region.ends.get(&slot)?;
            let &initial = llir.incoming(start).first()?;
            let &body_producer = llir.incoming(end).first()?;
            markers[start] = Some(MaterializeLoopMarker::Start {
                region: region_index,
                initial,
                body_producer,
            });
            markers[end] = Some(MaterializeLoopMarker::End {
                region: region_index,
                body_producer,
            });
        }

        for &input in region.inputs.values() {
            let sources = llir.incoming(input).to_vec();
            if sources.len() != region.iters {
                return None;
            }
            markers[input] = Some(MaterializeLoopMarker::Input {
                region: region_index,
                sources,
            });
        }

        let mut output_producers = FxHashMap::default();
        for (&stream, &output) in &region.outputs {
            let &body_producer = llir.incoming(output).first()?;
            output_producers.insert(stream, body_producer);
            markers[output] = Some(MaterializeLoopMarker::Output {
                region: region_index,
                body_producer,
            });
        }
        for (&select, &(stream, iter)) in &region.output_selects {
            if iter >= region.iters {
                return None;
            }
            markers[select] = Some(MaterializeLoopMarker::OutputSelect {
                region: region_index,
                iter,
                body_producer: *output_producers.get(&stream)?,
            });
        }
    }

    // Reject malformed marker graphs rather than silently materializing a
    // marker as an executable op. Callers treat this as an invariant failure.
    for (node, marker) in markers.iter().enumerate() {
        let Some(op) = llir.op(node) else {
            continue;
        };
        let is_marker = op.to_op::<LoopStart>().is_some()
            || op.to_op::<LoopEnd>().is_some()
            || op.to_op::<LoopInput>().is_some()
            || op.to_op::<LoopInputStatic>().is_some()
            || op.to_op::<LoopOutput>().is_some()
            || op.to_op::<LoopOutputSelect>().is_some();
        if is_marker && marker.is_none() {
            return None;
        }
    }

    // Region membership is a transitive scope: foreign (nested) markers are
    // transparent, while this region's own exit markers are boundaries. This
    // directly describes the nodes that must be cloned for each enclosing
    // iteration.
    let mut membership = vec![0u128; node_bound];
    for (&loop_id, region) in &loop_regions {
        let region_index = region_by_id[&loop_id];
        let mut seen = vec![false; node_bound];
        let mut worklist: Vec<usize> = region
            .starts
            .values()
            .chain(region.inputs.values())
            .flat_map(|marker| llir.outgoing(*marker).iter().copied())
            .collect();
        while let Some(node) = worklist.pop() {
            if seen[node] {
                continue;
            }
            seen[node] = true;
            if markers[node].is_some() {
                if marker_owner[node] != Some(region_index) {
                    worklist.extend(llir.outgoing(node));
                }
                continue;
            }
            if llir.op(node)?.to_op::<Output>().is_some() {
                continue;
            }
            membership[node] |= 1u128 << region_index;
            worklist.extend(llir.outgoing(node));
        }
    }

    // Nodes only need instances for the loops that contain them. Independent
    // regions therefore remain independent instead of contributing to a
    // graph-wide Cartesian product. A membership containing multiple regions
    // represents genuine nesting, where the product is semantically required.
    let mut contexts_by_membership: FxHashMap<u128, MaterializeContextLayout> =
        FxHashMap::default();
    contexts_by_membership.insert(0, MaterializeContextLayout::new(0, &regions)?);
    let mut node_instance_offsets = vec![usize::MAX; node_bound];
    let mut final_node_count = 0usize;
    let mut final_edge_count = 0usize;
    for node in 0..node_bound {
        if llir.op(node).is_none() || markers[node].is_some() {
            continue;
        }
        let context_count = match contexts_by_membership.entry(membership[node]) {
            Entry::Occupied(entry) => entry.get().count,
            Entry::Vacant(entry) => {
                entry
                    .insert(MaterializeContextLayout::new(membership[node], &regions)?)
                    .count
            }
        };
        node_instance_offsets[node] = final_node_count;
        final_node_count = final_node_count.checked_add(context_count)?;
        final_edge_count =
            final_edge_count.checked_add(context_count.checked_mul(llir.incoming(node).len())?)?;
    }

    let mut materialized = LLIRGraph::with_capacity(final_node_count, final_edge_count);

    for node in 0..node_bound {
        let Some(op) = llir.op(node) else {
            continue;
        };
        if markers[node].is_some() {
            continue;
        }
        let layout = &contexts_by_membership[&membership[node]];
        for context in 0..layout.count {
            let materialized_node = materialized.add_node(op.clone());
            debug_assert_eq!(
                materialized_node.index(),
                node_instance_offsets[node] + context
            );
        }
    }

    let mut context_iterations = vec![0usize; regions.len()];
    for node in 0..node_bound {
        if llir.op(node).is_none() || markers[node].is_some() {
            continue;
        }
        let node_membership = membership[node];
        let node_layout = &contexts_by_membership[&node_membership];
        for context in 0..node_layout.count {
            let target = NodeIndex::new(node_instance_offsets[node] + context);
            for &original_source in llir.incoming(node) {
                node_layout.decode(context, &regions, &mut context_iterations);
                let mut assigned_membership = node_membership;
                let mut source = original_source;
                let mut marker_hops = 0usize;
                while let Some(marker) = markers[source].as_ref() {
                    marker_hops += 1;
                    if marker_hops > node_bound {
                        return None;
                    }
                    match marker {
                        MaterializeLoopMarker::Start {
                            region,
                            initial,
                            body_producer,
                        } => {
                            let region_bit = 1u128 << *region;
                            if assigned_membership & region_bit == 0 {
                                return None;
                            }
                            let iter = context_iterations[*region];
                            if iter == 0 {
                                source = *initial;
                            } else {
                                source = *body_producer;
                                context_iterations[*region] = iter - 1;
                            }
                        }
                        MaterializeLoopMarker::End {
                            region,
                            body_producer,
                        } => {
                            source = *body_producer;
                            assigned_membership |= 1u128 << *region;
                            context_iterations[*region] = regions[*region].iters - 1;
                        }
                        MaterializeLoopMarker::Input { region, sources } => {
                            let region_bit = 1u128 << *region;
                            if assigned_membership & region_bit == 0 {
                                return None;
                            }
                            source = sources[context_iterations[*region]];
                        }
                        MaterializeLoopMarker::StaticInput {
                            source: static_source,
                        } => {
                            source = *static_source;
                        }
                        MaterializeLoopMarker::Output {
                            region,
                            body_producer,
                        } => {
                            source = *body_producer;
                            let region_bit = 1u128 << *region;
                            if assigned_membership & region_bit == 0 {
                                assigned_membership |= region_bit;
                                context_iterations[*region] = 0;
                            }
                        }
                        MaterializeLoopMarker::OutputSelect {
                            region,
                            iter,
                            body_producer,
                        } => {
                            source = *body_producer;
                            assigned_membership |= 1u128 << *region;
                            context_iterations[*region] = *iter;
                        }
                    }
                }
                if llir.op(source).is_none() || node_instance_offsets[source] == usize::MAX {
                    return None;
                }
                let source_layout = contexts_by_membership.get(&membership[source])?;
                let source_context =
                    source_layout.encode(assigned_membership, &regions, &context_iterations)?;
                let materialized_source =
                    NodeIndex::new(node_instance_offsets[source] + source_context);
                materialized.add_edge(materialized_source, target, ());
            }
        }
    }

    Some(materialized)
}

pub(crate) fn materialize_unrolled_llir(llir: &LLIRGraph) -> Option<LLIRGraph> {
    materialize_unrolled_view(&StableMaterializeView::new(llir))
}

/// Complete the indexed hot path by materializing its dense rolled graph
/// directly into the public StableGraph representation expected by runtimes.
pub fn unroll_packed_llir(llir: PackedLLIRGraph) -> LLIRGraph {
    let started_at = std::time::Instant::now();
    let rolled_nodes = llir.ops.len();
    let graph = materialize_unrolled_view(&llir)
        .expect("rolled LLIR must satisfy the loop materialization invariants");
    if crate::egglog_utils::llir_profile_enabled() {
        eprintln!(
            "LLIR_UNROLL_PROFILE total_ms={:.3} rolled_nodes={} unrolled_nodes={} unrolled_edges={}",
            started_at.elapsed().as_secs_f64() * 1e3,
            rolled_nodes,
            graph.node_count(),
            graph.edge_count(),
        );
    }
    graph
}

/// Expand all loop-region markers in an LLIR graph into fully unrolled bodies.
///
/// Reads `LoopStart` / `LoopEnd` / `LoopInput` / `LoopOutput` metadata placed
/// by the auto-roll prepass, clones the loop body `iters-1` additional times,
/// threads loop-carried state between clones, routes per-iteration inputs and
/// per-iteration outputs, and removes the four marker op types.
///
/// Incoming-edge ORDER is preserved for every affected node — ops read their
/// inputs by edge-id order, so edges are rebuilt in position.
pub fn unroll_loops_in_llir(llir: &mut LLIRGraph) {
    let started_at = std::time::Instant::now();
    let rolled_nodes = llir.node_count();
    *llir = materialize_unrolled_llir(llir)
        .expect("rolled LLIR must satisfy the loop materialization invariants");
    if crate::egglog_utils::llir_profile_enabled() {
        eprintln!(
            "LLIR_UNROLL_PROFILE total_ms={:.3} rolled_nodes={} unrolled_nodes={} unrolled_edges={}",
            started_at.elapsed().as_secs_f64() * 1e3,
            rolled_nodes,
            llir.node_count(),
            llir.edge_count(),
        );
    }
}

/// Collapse all loop markers in an LLIR graph down to a SINGLE iteration's
/// body, with first-iteration inputs and outputs only. This is the cheap
/// per-candidate form used by the genetic search — profiling one transformer
/// block instead of N×block makes the search ~N× faster, and the relative
/// cost of any extraction choice is preserved on the body shape.
///
/// LoopStart consumers re-route to the initial value, LoopInput consumers
/// re-route to `sources[0]`, LoopEnd's post-loop consumers re-route to the body producer
/// directly, and each `LoopOutput` is replaced with a single `Output { node:
/// targets[0] }`. After collapse the LLIR has no marker ops left and contains
/// exactly the iter-0 body plus the surrounding non-loop graph.
pub fn collapse_loops_to_first_iter(llir: &mut LLIRGraph) {
    inline_static_loop_inputs(llir);
    // Innermost first: collapsing an inner region leaves its iter-0 body as
    // plain nodes inside the enclosing region, which then collapses over them
    // in turn.
    while let Some((region, body)) = take_innermost_region(llir) {
        if region.starts.is_empty() {
            eprintln!(
                "[loop-debug] collapse abandoned on degenerate region: ends={} inputs={} outputs={} selects={}",
                region.ends.len(),
                region.inputs.len(),
                region.outputs.len(),
                region.output_selects.len(),
            );
            return;
        }
        collapse_loop_region(llir, &region, &body);
        let compacted = compact_llir_preserving_input_order(llir);
        *llir = compacted;
    }
}

fn collapse_loop_region(
    llir: &mut LLIRGraph,
    region: &LoopRegion,
    body_nodes: &FxHashSet<NodeIndex>,
) {
    use petgraph::visit::EdgeRef;

    let LoopRegion {
        starts,
        ends,
        inputs,
        outputs,
        output_selects,
        markers: loop_markers,
        ..
    } = region;

    // Initial value per LoopStart, body producer per LoopEnd / LoopOutput.
    let mut start_initial: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    for &start_node in starts.values() {
        let initial = llir
            .neighbors_directed(start_node, Direction::Incoming)
            .next()
            .expect("LoopStart must have an initial-value producer");
        start_initial.insert(start_node, initial);
    }
    let mut input_first_source: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    for input_node in inputs.values() {
        let first = llir
            .edges_directed(*input_node, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .next()
            .expect("LoopInput must have at least one source");
        input_first_source.insert(*input_node, first);
    }

    // Resolve a source reference to its iter-0 equivalent.
    let resolve_src = |src: NodeIndex| -> NodeIndex {
        if let Some(&initial) = start_initial.get(&src) {
            initial
        } else if let Some(&first) = input_first_source.get(&src) {
            first
        } else {
            src
        }
    };

    // Rewrite every body node's incoming edges. Per-edge remove+add to keep
    // edge-id ordering via LIFO reuse — runtime reads inputs sorted by edge
    // id so position must be preserved.
    for &b in body_nodes {
        let pairs: Vec<(NodeIndex, petgraph::graph::EdgeIndex)> = llir
            .edges_directed(b, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| (e.source(), e.id()))
            .collect();
        for (src, eid) in pairs {
            let new_src = resolve_src(src);
            llir.remove_edge(eid);
            llir.add_edge(new_src, b, ());
        }
    }

    // Per LoopOutput stream, find the body producer (its single incoming edge).
    let mut output_body_producer: FxHashMap<usize, NodeIndex> = FxHashMap::default();
    for (&stream_id, &output_node) in outputs {
        let body_producer = llir
            .neighbors_directed(output_node, Direction::Incoming)
            .next()
            .expect("LoopOutput missing body producer during rewire");
        output_body_producer.insert(stream_id, body_producer);
    }

    // Post-loop consumers reading from LoopEnd / LoopOutputSelect must
    // instead read from the body producer (iter-0's value) directly. In the
    // collapsed form every Select(i) — regardless of i — re-routes to iter-0's
    // body producer; iter > 0 Selects don't have a real value to forward, so
    // they alias iter 0's. This keeps post-loop graph topology unchanged.
    let mut marker_post_sub: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    for &end_node in ends.values() {
        let body_producer = llir
            .neighbors_directed(end_node, Direction::Incoming)
            .next()
            .expect("LoopEnd missing body producer during rewire");
        marker_post_sub.insert(end_node, body_producer);
    }
    for (&select_node, &(stream_id, _)) in output_selects {
        if let Some(&body_producer) = output_body_producer.get(&stream_id) {
            marker_post_sub.insert(select_node, body_producer);
        }
    }
    // Entry markers can also have consumers outside the walked body:
    // extraction may elect a marker as the representative of a value class
    // read anywhere in the graph. Resolve them to their iter-0 values —
    // never leave a consumer pointing at a marker about to be removed.
    for (&input_node, &first) in &input_first_source {
        marker_post_sub.insert(input_node, first);
    }
    for (&start_node, &initial) in &start_initial {
        marker_post_sub.insert(start_node, initial);
    }
    let post_loop_consumers: FxHashSet<NodeIndex> = loop_markers
        .iter()
        .flat_map(|n| {
            llir.neighbors_directed(*n, Direction::Outgoing)
                .collect::<Vec<_>>()
        })
        .filter(|n| !loop_markers.contains(n) && !body_nodes.contains(n))
        .collect();
    for &consumer in &post_loop_consumers {
        let pairs: Vec<(NodeIndex, petgraph::graph::EdgeIndex)> = llir
            .edges_directed(consumer, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| (e.source(), e.id()))
            .collect();
        for (src, eid) in pairs {
            let new_src = marker_post_sub.get(&src).copied().unwrap_or(src);
            llir.remove_edge(eid);
            llir.add_edge(new_src, consumer, ());
        }
    }

    for &n in loop_markers {
        llir.remove_node(n);
    }
}

/// Rebuild an LLIR graph into a fresh StableGraph, copying nodes and edges
/// such that edge IDs are sequential in the insertion order we choose
/// (per-node incoming edges in their original edge-id order). This erases
/// any free-list reuse artifacts from prior `remove_edge` / `remove_node`
/// calls.
fn compact_llir_preserving_input_order(old: &LLIRGraph) -> LLIRGraph {
    use petgraph::visit::EdgeRef;
    let mut new_graph = LLIRGraph::default();
    let mut old_to_new: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    // Topo sort to add nodes in a deterministic order. If the graph has
    // cycles (shouldn't for LLIR), fall back to node_indices order.
    let topo = match petgraph::algo::toposort(old, None) {
        Ok(v) => v,
        Err(_) => old.node_indices().collect(),
    };
    for n in &topo {
        let new_n = new_graph.add_node(old[*n].clone());
        old_to_new.insert(*n, new_n);
    }
    // Add edges in topo order, per-node incoming sorted by old edge id.
    // This reassigns new edge indices sequentially so sort-by-id matches
    // the intended input position.
    for n in &topo {
        let incoming: Vec<NodeIndex> = old
            .edges_directed(*n, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect();
        for src in incoming {
            if let (Some(&new_src), Some(&new_dst)) = (old_to_new.get(&src), old_to_new.get(n)) {
                new_graph.add_edge(new_src, new_dst, ());
            }
        }
    }
    new_graph
}
