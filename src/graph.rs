use crate::egglog_utils::{
    egglog_to_llir, extract_generation, hash_choice_set, hlir_to_egglog, random_initial_choice,
    run_egglog,
};
use crate::{
    egglog_utils::SerializedEGraph,
    op::{EgglogOp, IntoEgglogOp, LLIROp},
};
use crate::{hlir::CustomOpKind, op::*, prelude::*};
use colored::Colorize;
use itertools::Itertools;
use petgraph::{Direction, algo::toposort, stable_graph::StableGraph, visit::EdgeRef};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    any::TypeId,
    fmt::Debug,
    io::Write,
    ops::{Deref, DerefMut},
    sync::Arc,
};
use tracing;

pub type LLIRGraph = StableGraph<LLIROp, ()>;
pub type HLIRGraph = StableGraph<Box<dyn HLIROp>, ()>;

#[derive(Debug, Clone)]
struct RollingOccurrence {
    nodes: Vec<NodeIndex>,
    boundary_inputs: Vec<NodeIndex>,
    output_nodes: Vec<NodeIndex>,
}

#[derive(Debug, Clone)]
struct RollingCandidate {
    occurrences: Vec<RollingOccurrence>,
    state_param_indices: Vec<usize>,
    savings: usize,
}

#[derive(Debug, Clone)]
struct RollingRun {
    occurrences: Vec<RollingOccurrence>,
    starts: Vec<usize>,
    window: usize,
}

#[derive(Debug, Clone, Default)]
struct RollingSearchDiagnostics {
    windows_probed: usize,
    adjacent_hash_matches: usize,
    repeated_signature_runs: usize,
    rejected_zero_state_params: usize,
    best_rejected: Option<RollingRejectedCandidate>,
    top_runs: Vec<String>,
}

#[derive(Debug, Clone)]
struct RollingRejectedCandidate {
    window: usize,
    repetitions: usize,
    boundary_inputs: usize,
    state_params: usize,
    savings: usize,
}

#[derive(Debug, Clone)]
struct RollingSearchReport {
    candidate: Option<RollingCandidate>,
    diagnostics: RollingSearchDiagnostics,
}

/// A compiled bucket: (bucket_indices, representative_dyn_map, stitched_llir).
pub type BucketLLIR = (FxHashMap<char, usize>, FxHashMap<char, usize>, LLIRGraph);

/// A bucket for a dynamic dimension, defining a range of valid values.
/// For an exact value, use `min == max` (zero-length range).
#[derive(Debug, Clone)]
pub struct DimBucket {
    pub min: usize,
    pub max: usize,
    representative_override: Option<usize>,
}

impl DimBucket {
    /// Create a new bucket covering `[min, max]` inclusive.
    /// For an exact value, pass `min == max`.
    pub fn new(min: usize, max: usize) -> Self {
        assert!(min <= max, "DimBucket min ({min}) must be <= max ({max})");
        DimBucket {
            min,
            max,
            representative_override: None,
        }
    }

    /// Override the representative value used during search profiling.
    /// Must be within `[min, max]`.
    pub fn representative(mut self, val: usize) -> Self {
        assert!(
            val >= self.min && val <= self.max,
            "Representative {val} must be in [{}, {}]",
            self.min,
            self.max
        );
        self.representative_override = Some(val);
        self
    }

    /// The representative value used during search profiling.
    /// Defaults to midpoint `(min + max) / 2`.
    pub fn representative_value(&self) -> usize {
        self.representative_override
            .unwrap_or((self.min + self.max) / 2)
    }

    /// Check if `val` falls within this bucket's range.
    pub fn contains(&self, val: usize) -> bool {
        val >= self.min && val <= self.max
    }
}

/// Options for controlling the genetic search algorithm.
///
/// Use the builder pattern to configure search parameters:
/// ```
/// use luminal::prelude::SearchOptions;
/// let opts = SearchOptions::new(5)
///     .generation_size(50)
///     .mutations(40)
///     .trials(15);
/// ```
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Maximum number of graphs to evaluate
    pub limit: usize,
    /// Number of offspring per generation (default: 30)
    pub generation_size: usize,
    /// Number of mutations applied to each offspring (default: 30)
    pub mutations: usize,
    /// Number of profiling trials per candidate (default: 10)
    pub trials: usize,
    /// Number of best genomes to keep as parents per generation (default: 1)
    pub keep_best: usize,
    /// Optional per-candidate profiling timeout.
    pub profile_timeout: Option<std::time::Duration>,
    /// Optional per-group search timeout.
    pub group_timeout: Option<std::time::Duration>,
    /// Optional profiling dimension overrides.
    pub profile_dims: FxHashMap<char, usize>,
}

impl SearchOptions {
    /// Create new search options with the given limit. Other fields use defaults.
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            generation_size: 30,
            mutations: 30,
            trials: 10,
            keep_best: 1,
            profile_timeout: None,
            group_timeout: None,
            profile_dims: FxHashMap::default(),
        }
    }

    /// Set the number of offspring per generation.
    pub fn generation_size(mut self, generation_size: usize) -> Self {
        self.generation_size = generation_size;
        self
    }

    /// Set the number of mutations per offspring.
    pub fn mutations(mut self, mutations: usize) -> Self {
        self.mutations = mutations;
        self
    }

    /// Set the number of profiling trials per candidate.
    pub fn trials(mut self, trials: usize) -> Self {
        self.trials = trials;
        self
    }

    /// Set the number of best genomes to keep as parents per generation.
    pub fn keep_best(mut self, keep_best: usize) -> Self {
        self.keep_best = keep_best;
        self
    }

    /// Set an optional per-candidate profiling timeout.
    pub fn profile_timeout(mut self, profile_timeout: std::time::Duration) -> Self {
        self.profile_timeout = Some(profile_timeout);
        self
    }

    /// Set an optional per-group search timeout.
    pub fn group_timeout(mut self, group_timeout: std::time::Duration) -> Self {
        self.group_timeout = Some(group_timeout);
        self
    }

    /// Override a dynamic dimension value used during search profiling.
    pub fn profile_dim(mut self, dim: char, value: usize) -> Self {
        self.profile_dims.insert(dim, value);
        self
    }
}

/// A Luminal compute graph.
///
/// All computation is represented as a directed acyclic graph.
#[derive(Debug, Default)]
pub struct Graph {
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: FxHashMap<char, usize>,
    /// Edge weights: (Input index, Output index, Input shape)
    pub graph: HLIRGraph,
    /// E-Graph search space. Always exactly one e-graph; the `Vec` is kept
    /// for the public `Graph::egraph()` accessor's `Option<&...>` shape.
    egraphs: Vec<SerializedEGraph>,
    /// Available ops
    pub ops: Option<Vec<Arc<Box<dyn EgglogOp>>>>,
    /// Custom ops
    pub custom_ops: Vec<Box<dyn CustomOp>>,
    /// Bucket definitions per dynamic dimension. Dimensions without buckets use a
    /// single implicit bucket (current behavior). When set, search compiles a
    /// separate LLIR per bucket combination and runtime dispatches automatically.
    pub dim_buckets: FxHashMap<char, Vec<DimBucket>>,
    /// Metadata for Input nodes: NodeIndex -> (label, dtype).
    /// Stored as plain data so it survives cross-binary type identity mismatches
    /// when external backend plugins are compiled separately.
    pub input_meta: FxHashMap<NodeIndex, (String, DType)>,
}

impl Graph {
    /// Create a new graph
    pub fn new() -> Graph {
        Graph::default()
    }

    fn run_auto_loop_rolling_prepass(&mut self) {
        let before = self.graph.node_count();
        let inserted = self.auto_roll_loops_prepass();
        if inserted == 0 {
            println!(
                "   {:>6}  no loop regions found (max body={})",
                "Rolled".cyan().bold(),
                before / 2,
            );
        }
    }

    /// Mutate the HLIR graph in place to fold N repeated body occurrences into
    /// a single body plus loop-marker ops. See `auto_roll_loops_prepass`.
    fn insert_loop_region_ops(&mut self, candidate: RollingCandidate) -> usize {
        use crate::hlir::{LoopEnd, LoopInput, LoopOutput, LoopOutputSelect, LoopStart, Output};
        use petgraph::visit::EdgeRef;

        let nodes_before = self.graph.node_count();
        let n_iters = candidate.occurrences.len();
        let loop_id = 0usize;

        // Build the body-node sets EXCLUDING `Output` HLIR nodes. An Output
        // inside a rolled occurrence is a graph-external sink for that
        // iteration's value, not body computation; we treat it as a cross-
        // region consumer so each iteration's Output survives all the way
        // through and gets rewired to its `LoopOutputSelect(i)` below.
        let body_nodes: FxHashSet<NodeIndex> = candidate.occurrences[0]
            .nodes
            .iter()
            .copied()
            .filter(|&n| self.try_get_op::<Output>(n).is_none())
            .collect();
        let mut duplicate_body_nodes: FxHashSet<NodeIndex> = FxHashSet::default();
        for occ in &candidate.occurrences[1..] {
            for &n in &occ.nodes {
                if self.try_get_op::<Output>(n).is_none() {
                    duplicate_body_nodes.insert(n);
                }
            }
        }

        let n_boundary = candidate.occurrences[0].boundary_inputs.len();
        let state_set: FxHashSet<usize> = candidate.state_param_indices.iter().copied().collect();

        let mut state_out_pos_per_slot: Vec<usize> =
            Vec::with_capacity(candidate.state_param_indices.len());
        let mut state_output_positions: FxHashSet<usize> = FxHashSet::default();
        for &p in &candidate.state_param_indices {
            let next_val = candidate.occurrences[1].boundary_inputs[p];
            let pos = candidate.occurrences[0]
                .output_nodes
                .iter()
                .position(|&n| n == next_val)
                .expect("state param must have a producer in output_nodes");
            state_out_pos_per_slot.push(pos);
            state_output_positions.insert(pos);
        }

        let mut created = 0usize;
        // Track all NodeIndex slots we newly assign for loop-marker ops.
        // StableGraph reuses freed node indices; removals later in this
        // function might target slots that happen to coincide with a new
        // loop-marker's NodeIndex, so we explicitly exclude those.
        let mut added_loop_ops: FxHashSet<NodeIndex> = FxHashSet::default();

        for (slot_idx, (&p, &out_pos)) in candidate
            .state_param_indices
            .iter()
            .zip(state_out_pos_per_slot.iter())
            .enumerate()
        {
            let initial = candidate.occurrences[0].boundary_inputs[p];
            let body_state_out = candidate.occurrences[0].output_nodes[out_pos];
            let last_state_out = candidate.occurrences[n_iters - 1].output_nodes[out_pos];
            let dtype = self.infer_node_dtype(initial);

            let loop_start = self.graph.add_node(Box::new(LoopStart {
                loop_id,
                slot_idx,
                iters: Expression::from(n_iters as i32),
                dtype,
            }));
            added_loop_ops.insert(loop_start);
            self.graph.add_edge(initial, loop_start, ());

            let edges_out_of_initial: Vec<_> = self
                .graph
                .edges_directed(initial, Direction::Outgoing)
                .filter(|e| body_nodes.contains(&e.target()))
                .map(|e| (e.id(), e.target()))
                .collect();
            for (eid, dst) in edges_out_of_initial {
                self.graph.remove_edge(eid);
                self.graph.add_edge(loop_start, dst, ());
            }

            let loop_end = self.graph.add_node(Box::new(LoopEnd {
                loop_id,
                slot_idx,
                dtype,
            }));
            added_loop_ops.insert(loop_end);
            self.graph.add_edge(body_state_out, loop_end, ());

            let external_edges: Vec<_> = self
                .graph
                .edges_directed(last_state_out, Direction::Outgoing)
                .filter(|e| {
                    let t = e.target();
                    !body_nodes.contains(&t) && !duplicate_body_nodes.contains(&t)
                })
                .map(|e| (e.id(), e.target()))
                .collect();
            for (eid, dst) in external_edges {
                self.graph.remove_edge(eid);
                self.graph.add_edge(loop_end, dst, ());
            }

            created += 2;
        }

        for p in 0..n_boundary {
            if state_set.contains(&p) {
                continue;
            }
            let per_iter_sources: Vec<NodeIndex> = candidate
                .occurrences
                .iter()
                .map(|occ| occ.boundary_inputs[p])
                .collect();
            if per_iter_sources.windows(2).all(|w| w[0] == w[1]) {
                continue;
            }

            let body_input = candidate.occurrences[0].boundary_inputs[p];
            let dtype = self.infer_node_dtype(body_input);
            let loop_input = self.graph.add_node(Box::new(LoopInput {
                loop_id,
                stream_id: p,
                dtype,
            }));
            added_loop_ops.insert(loop_input);
            for &src in &per_iter_sources {
                self.graph.add_edge(src, loop_input, ());
            }

            let body_edges: Vec<_> = self
                .graph
                .edges_directed(body_input, Direction::Outgoing)
                .filter(|e| body_nodes.contains(&e.target()))
                .map(|e| (e.id(), e.target()))
                .collect();
            for (eid, dst) in body_edges {
                self.graph.remove_edge(eid);
                self.graph.add_edge(loop_input, dst, ());
            }

            created += 1;
        }

        let n_outputs = candidate.occurrences[0].output_nodes.len();
        for q in 0..n_outputs {
            if state_output_positions.contains(&q) {
                continue;
            }

            // Per iteration, determine (body_producer, edges_to_rewire):
            //  * If `output_nodes[q]` is an Output HLIR (graph sink): the
            //    body producer is that Output's predecessor, and the edge to
            //    rewire is the predecessor → Output edge itself.
            //  * Otherwise: body producer is `output_nodes[q]`; the edges to
            //    rewire are all of its outgoing edges whose target is OUTSIDE
            //    the rolled region (post-loop consumers — Output HLIR or any
            //    downstream computation, treated identically).
            let mut per_iter_plan: Vec<(NodeIndex, Vec<(petgraph::graph::EdgeIndex, NodeIndex)>)> =
                Vec::with_capacity(n_iters);
            let mut complete = true;
            for occ in &candidate.occurrences {
                let node = occ.output_nodes[q];
                if self.try_get_op::<Output>(node).is_some() {
                    // Output HLIR sink. Its predecessor is the body producer;
                    // the single (pred → Output) edge is what we rewire.
                    let pred_edge = self
                        .graph
                        .edges_directed(node, Direction::Incoming)
                        .next()
                        .map(|e| (e.id(), e.source(), node));
                    match pred_edge {
                        Some((eid, pred, output)) => {
                            per_iter_plan.push((pred, vec![(eid, output)]));
                        }
                        None => {
                            complete = false;
                            break;
                        }
                    }
                } else {
                    // Internal body producer. Cross-region edges = its
                    // outgoing edges whose target is not in any iter's body.
                    let edges: Vec<_> = self
                        .graph
                        .edges_directed(node, Direction::Outgoing)
                        .filter(|e| {
                            let t = e.target();
                            !body_nodes.contains(&t) && !duplicate_body_nodes.contains(&t)
                        })
                        .map(|e| (e.id(), e.target()))
                        .collect();
                    if edges.is_empty() {
                        // Nothing actually crosses the region for this iter.
                        // Skip the whole stream — without a consumer the
                        // Select would dangle.
                        complete = false;
                        break;
                    }
                    per_iter_plan.push((node, edges));
                }
            }
            if !complete {
                continue;
            }

            // Iter-0 body producer feeds the LoopOutput marker.
            let body_output = per_iter_plan[0].0;
            let dtype = self.infer_node_dtype(body_output);

            let loop_output = self.graph.add_node(Box::new(LoopOutput {
                loop_id,
                stream_id: q,
                dtype,
            }));
            self.graph.add_edge(body_output, loop_output, ());
            added_loop_ops.insert(loop_output);

            // For each iter, create a LoopOutputSelect(i) and rewire the
            // cross-region edges to flow through it.
            for (i, (_, edges)) in per_iter_plan.into_iter().enumerate() {
                let select = self.graph.add_node(Box::new(LoopOutputSelect {
                    loop_id,
                    stream_id: q,
                    iter: i,
                    dtype,
                }));
                self.graph.add_edge(loop_output, select, ());
                added_loop_ops.insert(select);

                for (edge_id, consumer) in edges {
                    self.graph.remove_edge(edge_id);
                    self.graph.add_edge(select, consumer, ());
                }
                created += 1;
            }
            created += 1; // for the LoopOutput marker itself
        }

        // Delete duplicate body nodes. Skip any node we just added as a
        // loop-marker op (StableGraph may reuse NodeIndex slots, so an
        // added marker could collide with a previously-freed body node id).
        for &node in &duplicate_body_nodes {
            if added_loop_ops.contains(&node) {
                continue;
            }
            self.graph.remove_node(node);
        }

        if created > 0 {
            let nodes_after = self.graph.node_count();
            // Region partition: body_nodes is the surviving one-iteration body,
            // `created` is the marker scaffold (LoopStart/End/Input/Output),
            // and the rest is graph outside the loop region (embedding,
            // weights, post-loop / lm-head).
            let inside_body = body_nodes.len();
            let inside_markers = created;
            let outside = nodes_after - inside_body - inside_markers;
            println!(
                "   {:>6}  rolled HLIR: {} -> {} nodes ({} loop ops inserted, {} duplicate body nodes deleted)",
                "Rolled".cyan().bold(),
                nodes_before,
                nodes_after,
                created,
                duplicate_body_nodes.len(),
            );
            println!(
                "   {:>6}  region partition: {} inside ({} body + {} markers) / {} outside",
                "Rolled".cyan().bold(),
                inside_body + inside_markers,
                inside_body,
                inside_markers,
                outside,
            );
        }
        created
    }

    /// Best-effort dtype lookup for a NodeIndex in the HLIR graph.
    fn infer_node_dtype(&self, node: NodeIndex) -> DType {
        if let Some((_, dt)) = self.input_meta.get(&node) {
            return *dt;
        }
        if let Some(op) = self.try_get_op::<crate::hlir::Input>(node) {
            return op.dtype;
        }
        if let Some(op) = self.try_get_op::<crate::hlir::Cast>(node) {
            return op.1;
        }
        for pred in self
            .graph
            .neighbors_directed(node, Direction::Incoming)
            .collect::<Vec<_>>()
        {
            let dt = self.infer_node_dtype(pred);
            if dt != DType::F32 || self.input_meta.contains_key(&pred) {
                return dt;
            }
        }
        DType::F32
    }

    /// Set a runtime dimension
    pub fn set_dim(&mut self, dimension: char, val: usize) {
        self.dyn_map.insert(dimension, val);
    }

    /// Define buckets for a dynamic dimension.
    ///
    /// When buckets are set, `search()` compiles a separate optimized LLIR graph
    /// for each bucket combination. At runtime, `execute()` dispatches to the
    /// appropriate compiled graph based on the current `dyn_map` values.
    ///
    /// Buckets must not overlap and must cover all values that will be used at runtime.
    pub fn set_dim_buckets(&mut self, dimension: char, buckets: &[DimBucket]) {
        // Validate no overlapping ranges
        for (i, a) in buckets.iter().enumerate() {
            for b in buckets.iter().skip(i + 1) {
                assert!(
                    a.max < b.min || b.max < a.min,
                    "Overlapping buckets for dim '{}': [{}, {}] and [{}, {}]",
                    dimension,
                    a.min,
                    a.max,
                    b.min,
                    b.max,
                );
            }
        }
        self.dim_buckets.insert(dimension, buckets.to_vec());
    }

    /// Attempt to discover repeated HLIR regions and build explicit region
    /// descriptors for loop-carried state edges.
    /// Returns the number of detected inter-region boundaries.
    ///
    /// This is a conservative prepass:
    /// - only rolls candidates with at least one loop-carried state parameter
    /// - only inserts when the carried edge shapes can be inferred
    pub fn auto_roll_loops_prepass(&mut self) -> usize {
        let max_region_size = self.graph.node_count() / 2;
        if max_region_size < 1 {
            return 0;
        }
        println!(
            "   {:>6}  scanning {} HLIR nodes for loop regions (max body={})",
            "Rolled".cyan().bold(),
            self.graph.node_count(),
            max_region_size,
        );
        let report = self.best_rolling_candidate(max_region_size);
        let Some(candidate) = report.candidate else {
            self.print_rolling_search_diagnostics(&report.diagnostics);
            return 0;
        };
        println!(
            "   {:>6}  candidate: body={} trips={} boundary_inputs={} state_params={:?}",
            "Rolled".yellow().bold(),
            candidate.occurrences[0].nodes.len(),
            candidate.occurrences.len(),
            candidate.occurrences[0].boundary_inputs.len(),
            candidate.state_param_indices,
        );
        if let Some(rejected) = &report.diagnostics.best_rejected {
            println!(
                "   {:>6}  best rejected: body={} trips={} boundary_inputs={} state_params={} savings={}",
                "Rolled".yellow().bold(),
                rejected.window,
                rejected.repetitions,
                rejected.boundary_inputs,
                rejected.state_params,
                rejected.savings,
            );
        }
        for run in report.diagnostics.top_runs.iter().take(5) {
            println!("   {:>6}  run: {}", "Rolled".yellow().bold(), run);
        }
        if candidate.occurrences.len() < 2 {
            return 0;
        }

        // Mutate the HLIR in place — insert LoopStart/LoopEnd/LoopInput/
        // LoopOutput markers, delete N-1 duplicate bodies. The loop structure
        // is encoded in the HLIR graph itself and the downstream single-root
        // egglog path picks it up unchanged.
        self.insert_loop_region_ops(candidate)
    }

    fn print_rolling_search_diagnostics(&self, diagnostics: &RollingSearchDiagnostics) {
        let best_rejected = diagnostics
            .best_rejected
            .as_ref()
            .map(|candidate| {
                format!(
                    "best rejected: body={} trips={} boundary_inputs={} state_params={} savings={}",
                    candidate.window,
                    candidate.repetitions,
                    candidate.boundary_inputs,
                    candidate.state_params,
                    candidate.savings
                )
            })
            .unwrap_or_else(|| "best rejected: none".to_string());
        println!(
            "   {:>6}  diagnostics: windows={} hash_matches={} repeated_runs={} rejected(zero_state={}); {}",
            "Rolled".yellow().bold(),
            diagnostics.windows_probed,
            diagnostics.adjacent_hash_matches,
            diagnostics.repeated_signature_runs,
            diagnostics.rejected_zero_state_params,
            best_rejected,
        );
        for run in diagnostics.top_runs.iter().take(5) {
            println!("   {:>6}  run: {}", "Rolled".yellow().bold(), run);
        }
    }

    fn best_rolling_candidate(&self, max_region_size: usize) -> RollingSearchReport {
        let Some(full_topo) = toposort(&self.graph, None).ok() else {
            return RollingSearchReport {
                candidate: None,
                diagnostics: RollingSearchDiagnostics::default(),
            };
        };
        let topo: Vec<NodeIndex> = full_topo
            .into_iter()
            .filter(|n| self.try_get_op::<crate::hlir::Input>(*n).is_none())
            .collect();
        if topo.len() < 2 {
            return RollingSearchReport {
                candidate: None,
                diagnostics: RollingSearchDiagnostics::default(),
            };
        }
        let uses = build_uses(&self.graph);
        let topo_index: FxHashMap<NodeIndex, usize> =
            topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let max_window = max_region_size.min(topo.len() / 2);
        let probe_windows = rolling_probe_window_sizes(max_window);
        let node_hashes: Vec<u64> = topo
            .iter()
            .map(|&node| cheap_rolling_node_hash(&self.graph, node))
            .collect();
        let rolling_hash = RollingHash64::new(&node_hashes);
        let mut diagnostics = RollingSearchDiagnostics::default();
        let mut best_overall: Option<RollingCandidate> = None;
        let mut discovered_runs: Vec<RollingRun> = Vec::new();

        // Search all window sizes down to 1, using cheap rolling hashes only as a
        // gate for expensive canonicalization. Candidate selection remains purely
        // based on valid HLIR-op reduction.
        for window in probe_windows {
            let mut start = 0usize;
            while start + window * 2 <= topo.len() {
                diagnostics.windows_probed += 1;
                let first_hash = rolling_hash.window_hash(start, window);
                let second_hash = rolling_hash.window_hash(start + window, window);
                if first_hash != second_hash {
                    start += 1;
                    continue;
                }
                diagnostics.adjacent_hash_matches += 1;

                let mut occs = vec![];
                let mut starts = vec![];
                let first_nodes = topo[start..start + window].to_vec();
                let Some((sig, first_boundary, first_outputs)) =
                    canonicalize_occurrence(&self.graph, &first_nodes, &uses, &topo_index)
                else {
                    start += 1;
                    continue;
                };
                starts.push(start);
                occs.push(RollingOccurrence {
                    nodes: first_nodes,
                    boundary_inputs: first_boundary,
                    output_nodes: first_outputs,
                });

                let mut pos = start + window;
                while pos + window <= topo.len() {
                    if rolling_hash.window_hash(pos, window) != first_hash {
                        break;
                    }
                    let nodes = topo[pos..pos + window].to_vec();
                    let Some((next_sig, boundary_inputs, output_nodes)) =
                        canonicalize_occurrence(&self.graph, &nodes, &uses, &topo_index)
                    else {
                        break;
                    };
                    if next_sig != sig {
                        break;
                    }
                    starts.push(pos);
                    occs.push(RollingOccurrence {
                        nodes,
                        boundary_inputs,
                        output_nodes,
                    });
                    pos += window;
                }
                if occs.len() < 2 {
                    start += 1;
                    continue;
                }
                diagnostics.repeated_signature_runs += 1;
                discovered_runs.push(RollingRun {
                    occurrences: occs.clone(),
                    starts: starts.clone(),
                    window,
                });
                let stride = starts
                    .windows(2)
                    .next()
                    .map(|w| w[1].saturating_sub(w[0]))
                    .unwrap_or(0);
                let summary = format!(
                    "body={} trips={} stride={} boundary_inputs={} state_params={} starts={:?}",
                    window,
                    occs.len(),
                    stride,
                    occs[0].boundary_inputs.len(),
                    collect_state_params(&occs, &uses, &self.graph).len(),
                    starts.iter().copied().take(4).collect::<Vec<_>>()
                );
                if occs.len() >= 20 && diagnostics.top_runs.len() < 16 {
                    diagnostics.top_runs.push(summary);
                }

                let state_params = collect_state_params(&occs, &uses, &self.graph);
                if state_params.is_empty() {
                    let rejected = RollingRejectedCandidate {
                        window,
                        repetitions: occs.len(),
                        boundary_inputs: occs[0].boundary_inputs.len(),
                        state_params: state_params.len(),
                        savings: window * (occs.len() - 1),
                    };
                    diagnostics.rejected_zero_state_params += 1;
                    let replace = diagnostics.best_rejected.as_ref().is_none_or(|best| {
                        (rejected.savings, rejected.repetitions, rejected.window)
                            > (best.savings, best.repetitions, best.window)
                    });
                    if replace {
                        diagnostics.best_rejected = Some(rejected);
                    }
                    start = pos.saturating_sub(window).max(start + 1);
                    continue;
                }

                let savings = window * (occs.len() - 1);
                let _ = sig;
                let candidate = RollingCandidate {
                    occurrences: occs,
                    state_param_indices: state_params,
                    savings,
                };
                let replace = best_overall.as_ref().is_none_or(|b| {
                    (candidate.savings, candidate.occurrences.len())
                        > (b.savings, b.occurrences.len())
                });
                if replace {
                    best_overall = Some(candidate);
                }
                start = pos.saturating_sub(window).max(start + 1);
            }
        }
        if let Some(best) = best_overall.take() {
            best_overall = Some(grow_rolling_candidate(
                &self.graph,
                &uses,
                &topo_index,
                best,
                &discovered_runs,
            ));
        }
        RollingSearchReport {
            candidate: best_overall,
            diagnostics,
        }
    }

    /// Create a new tensor with shape S
    pub fn tensor(&mut self, shape: impl ToShape) -> GraphTensor {
        self.named_tensor("", shape)
    }

    /// Create a new tensor with shape S and a name. This name will show up on the graph when displayed
    pub fn named_tensor(&mut self, name: impl ToString, shape: impl ToShape) -> GraphTensor {
        let name = name.to_string();
        let id = self.graph.add_node(Box::new(crate::hlir::Input {
            node: 0,
            label: name.clone(),
            dtype: DType::default(),
        }));
        self.get_op_mut::<crate::hlir::Input>(id).node = id.index();
        self.input_meta.insert(id, (name.clone(), DType::default()));
        GraphTensor {
            id,
            graph_ref: self,
            shape: ShapeTracker::new(shape),
            dtype: DType::default(),
        }
    }

    /// Get the sources of a node given it's id
    pub fn get_sources(&self, node_id: NodeIndex) -> Vec<NodeIndex> {
        self.graph
            .edges_directed(node_id, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect()
    }

    /// Get the dests of a node given it's id
    #[allow(clippy::borrowed_box)]
    pub fn get_dests(&self, node_id: NodeIndex) -> Vec<(NodeIndex, &Box<dyn HLIROp>)> {
        self.graph
            .edges_directed(node_id, Direction::Outgoing)
            .sorted_by_key(|e| e.id())
            .map(|e| (e.target(), &self.graph[e.target()]))
            .collect()
    }

    /// Add an op to the graph with the given input edges. Returns the new node's index.
    ///
    /// ```rust
    /// # use luminal::prelude::*;
    /// # let mut cx = Graph::new();
    /// let a = cx.tensor(3);
    /// let b_id = cx.add_op(
    ///     luminal::hlir::Mul { input_shapes: vec![a.shape, a.shape], ..Default::default() },
    ///     &[a.id],
    /// );
    /// let b = GraphTensor::from_id(b_id, a.shape, a.graph(), a.dtype);
    /// ```
    pub fn add_op<O: HLIROp + 'static>(&mut self, op: O, inputs: &[NodeIndex]) -> NodeIndex {
        let id = self.graph.add_node(Box::new(op));
        for &src in inputs {
            self.graph.add_edge(src, id, ());
        }
        id
    }

    pub fn try_get_op<T: HLIROp + 'static>(&self, node: NodeIndex) -> Option<&T> {
        self.node_weight(node).unwrap().as_any().downcast_ref::<T>()
    }
    pub fn get_op<T: HLIROp + 'static>(&self, node: NodeIndex) -> &T {
        self.try_get_op(node).unwrap()
    }
    pub fn try_get_op_mut<T: HLIROp + 'static>(&mut self, node: NodeIndex) -> Option<&mut T> {
        self.node_weight_mut(node)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<T>()
    }
    pub fn get_op_mut<T: HLIROp + 'static>(&mut self, node: NodeIndex) -> &mut T {
        self.try_get_op_mut(node).unwrap()
    }

    pub fn custom_op(
        &mut self,
        op: impl CustomOp + 'static,
        inputs: impl ToIds,
        shape: impl ToShape,
        dtype: DType,
    ) -> GraphTensor {
        self.custom_ops.push(Box::new(op));
        let input_ids = inputs.to_ids();
        let id = self.add_op(
            CustomOpKind {
                id: self.custom_ops.len() - 1,
                dtype,
            },
            &input_ids,
        );
        GraphTensor::from_id(
            id,
            ShapeTracker::new_with_element_bits(shape, dtype.bits()),
            self,
            dtype,
        )
    }

    #[tracing::instrument(skip_all)]
    pub fn build_search_space<Rt: Runtime + 'static>(&mut self) {
        self.run_auto_loop_rolling_prepass();
        let mut ops = Rt::Ops::into_vec();
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>();

        let (program, root) = hlir_to_egglog(self);
        self.egraphs = vec![run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()];
        self.ops = Some(ops);
    }

    #[tracing::instrument(skip_all)]
    pub fn build_search_space_exclude_ops<Rt: Runtime + 'static, Ex: IntoEgglogOp>(&mut self) {
        self.run_auto_loop_rolling_prepass();
        let exclude_ops = Ex::into_vec()
            .into_iter()
            .map(|e| e.sort().name)
            .collect::<FxHashSet<_>>();
        let mut ops = Rt::Ops::into_vec();
        ops.retain(|o| !exclude_ops.contains(&o.sort().name));
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>();

        let (program, root) = hlir_to_egglog(self);
        self.egraphs = vec![run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()];
        self.ops = Some(ops);
    }

    /// Get a reference to the first e-graph search space (if built)
    pub fn egraph(&self) -> Option<&SerializedEGraph> {
        self.egraphs.first()
    }

    /// Get a reference to the available ops (if search space is built)
    pub fn egglog_ops(&self) -> Option<&Vec<Arc<Box<dyn EgglogOp>>>> {
        self.ops.as_ref()
    }

    #[tracing::instrument(skip_all)]
    pub fn search<R: Runtime + 'static>(&mut self, runtime: R, limit: usize) -> R {
        let mut rng = rand::rng();
        self.search_options(runtime, SearchOptions::new(limit), &mut rng)
    }

    #[tracing::instrument(skip_all)]
    pub fn search_options<R: Runtime + 'static, G: rand::Rng>(
        &mut self,
        mut runtime: R,
        options: SearchOptions,
        rng: &mut G,
    ) -> R {
        runtime.set_profile_timeout(options.profile_timeout);
        if self.dim_buckets.is_empty() {
            // No buckets: existing single-search path
            let stitched =
                self.search_single(&mut runtime, &options, rng, &self.dyn_map.clone(), None);

            runtime.clear_intermediate_buffers();
            runtime.load_llir(&stitched);
            runtime.set_profile_timeout(None);
            runtime
        } else {
            // Bucketed search: compile one LLIR per bucket combination
            let bucket_combos = self.bucket_combinations();
            let n_combos = bucket_combos.len();
            let mut bucket_llirs: Vec<BucketLLIR> = Vec::with_capacity(n_combos);

            for (combo_idx, (bucket_indices, representative_dyn_map)) in
                bucket_combos.into_iter().enumerate()
            {
                let bucket_label = self.format_bucket_label(&bucket_indices);
                println!(
                    "   {:>6}  Group {}/{}: {}",
                    "Search".cyan().bold(),
                    combo_idx + 1,
                    n_combos,
                    bucket_label,
                );

                let stitched = self.search_single(
                    &mut runtime,
                    &options,
                    rng,
                    &representative_dyn_map,
                    Some((combo_idx, n_combos)),
                );
                bucket_llirs.push((bucket_indices, representative_dyn_map, stitched));
            }

            runtime.clear_intermediate_buffers();
            runtime.load_llir_buckets(&self.dim_buckets, &bucket_llirs);
            runtime.set_profile_timeout(None);
            runtime
        }
    }

    /// Compute cartesian product of all bucket combinations.
    /// Returns Vec of (bucket_indices, representative_dyn_map).
    fn bucket_combinations(&self) -> Vec<(FxHashMap<char, usize>, FxHashMap<char, usize>)> {
        let mut dims: Vec<(char, &Vec<DimBucket>)> =
            self.dim_buckets.iter().map(|(c, b)| (*c, b)).collect();
        dims.sort_by_key(|(c, _)| *c);

        let mut combos: Vec<(FxHashMap<char, usize>, FxHashMap<char, usize>)> =
            vec![(FxHashMap::default(), self.dyn_map.clone())];

        for (dim, buckets) in &dims {
            let mut new_combos = Vec::new();
            for (existing_indices, existing_dyn_map) in &combos {
                for (bucket_idx, bucket) in buckets.iter().enumerate() {
                    let mut indices = existing_indices.clone();
                    indices.insert(*dim, bucket_idx);
                    let mut dyn_map = existing_dyn_map.clone();
                    dyn_map.insert(*dim, bucket.representative_value());
                    new_combos.push((indices, dyn_map));
                }
            }
            combos = new_combos;
        }

        combos
    }

    /// Format a human-readable label for a bucket combination.
    fn format_bucket_label(&self, bucket_indices: &FxHashMap<char, usize>) -> String {
        let mut parts: Vec<String> = Vec::new();
        let mut dims: Vec<_> = bucket_indices.iter().collect();
        dims.sort_by_key(|(c, _)| **c);
        for (dim, &idx) in dims {
            let bucket = &self.dim_buckets[dim][idx];
            if bucket.min == bucket.max {
                parts.push(format!("{}={}", dim, bucket.min));
            } else {
                parts.push(format!(
                    "{}=[{},{}]@{}",
                    dim,
                    bucket.min,
                    bucket.max,
                    bucket.representative_value()
                ));
            }
        }
        parts.join(", ")
    }

    /// Run the genetic search and return the unrolled LLIR for the winning
    /// genome. `bucket_progress`: if `Some((current_bucket_idx, total_buckets))`
    /// adds a second "Bucket" progress bar.
    fn search_single<R: Runtime + 'static, G: rand::Rng>(
        &mut self,
        runtime: &mut R,
        options: &SearchOptions,
        rng: &mut G,
        dyn_map: &FxHashMap<char, usize>,
        bucket_progress: Option<(usize, usize)>,
    ) -> LLIRGraph {
        let mut profile_dyn_map = dyn_map.clone();
        for (&dim, &value) in &options.profile_dims {
            profile_dyn_map.insert(dim, value);
        }
        let limit = options.limit;
        let ops = self.ops.as_ref().unwrap();
        let egraph = &self.egraphs[0];
        let start = std::time::Instant::now();

        // Bar layout: one Search bar, plus an optional Bucket bar.
        let n_bar_lines = 1 + if bucket_progress.is_some() { 1 } else { 0 };

        fn make_bar(searched: usize, total: usize) -> String {
            let bar_width = 24;
            let head = ((searched as f32 / total as f32) * bar_width as f32)
                .clamp(0.0, bar_width as f32)
                .floor() as usize;
            if head == 0 {
                format!("[>{}]", " ".repeat(bar_width - 1))
            } else if head >= bar_width {
                format!("[{}>]", "=".repeat(bar_width))
            } else {
                format!(
                    "[{}>{}]",
                    "=".repeat(head),
                    " ".repeat(bar_width - head - 1)
                )
            }
        }

        let render_bars =
            |n_graphs: usize, limit: usize, bucket_progress: Option<(usize, usize)>| {
                print!(
                    "\x1b[2K  {:>6}  {} {n_graphs}/{limit}",
                    "Search".cyan().bold(),
                    make_bar(n_graphs, limit),
                );
                if let Some((bucket_idx, n_buckets)) = bucket_progress {
                    print!(
                        "\n\x1b[2K  {:>6}  {} {}/{n_buckets}",
                        "Bucket".cyan().bold(),
                        make_bar(bucket_idx, n_buckets),
                        bucket_idx,
                    );
                }
            };

        let group_start = std::time::Instant::now();
        let mut prev_selected: FxHashSet<u64> = FxHashSet::default();
        let mut list_cache = FxHashMap::default();
        let mut expr_cache = FxHashMap::default();
        runtime.clear_intermediate_buffers();

        // Find a viable initial genome (may need multiple attempts if some panic)
        let (mut best_genome, mut best_metric, display, mut n_graphs);
        let mut init_attempts = 0;
        loop {
            init_attempts += 1;
            if init_attempts > 100 {
                panic!("Failed to find a viable initial genome after 100 attempts");
            }
            let genome = random_initial_choice(egraph, rng);
            prev_selected.insert(hash_choice_set(&genome));

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut graph = egglog_to_llir(
                    egraph,
                    genome.clone(),
                    ops,
                    &self.custom_ops,
                    &mut list_cache,
                    &mut expr_cache,
                    None,
                );
                // Collapse the rolled body to a single iteration before
                // profiling — one transformer block instead of N×block, so
                // per-candidate profile time scales with body size, not the
                // unrolled graph size.
                collapse_loops_to_first_iter(&mut graph);
                runtime.clear_intermediate_buffers();
                let (rep_metric, rep_display) =
                    runtime.profile(&graph, &profile_dyn_map, options.trials);
                let has_nan = runtime.has_nan_outputs(&graph, &profile_dyn_map);
                (rep_metric, rep_display, has_nan)
            }));

            match result {
                Ok((metric, disp, false)) => {
                    best_genome = genome;
                    best_metric = R::aggregate_profile_metrics(&[metric]);
                    display = disp;
                    n_graphs = 1;
                    break;
                }
                Ok(_) | Err(_) => {
                    if options
                        .group_timeout
                        .is_some_and(|timeout| group_start.elapsed() >= timeout)
                    {
                        panic!("Failed to find a viable initial genome before timeout");
                    }
                    list_cache.clear();
                    expr_cache.clear();
                    continue;
                }
            }
        }

        // Print initial result and progress
        let msg = format!("   {:>6} {}", "Search".cyan().bold(), display);
        println!("{msg}");
        render_bars(n_graphs, limit, bucket_progress);
        std::io::stdout().flush().unwrap();

        // Track top-N parents for offspring generation
        let mut parents: Vec<(R::ProfileMetric, crate::egglog_utils::EGraphChoiceSet<'_>)> =
            vec![(best_metric.clone(), best_genome.clone())];

        while n_graphs < limit {
            if options
                .group_timeout
                .is_some_and(|timeout| group_start.elapsed() >= timeout)
            {
                break;
            }

            // Generate offspring from all parents, dividing budget evenly
            let budget = (limit - n_graphs).min(options.generation_size);
            let per_parent = budget.div_ceil(parents.len());
            let mut all_offspring = Vec::new();
            for (_, parent_genome) in &parents {
                let remaining = budget.saturating_sub(all_offspring.len());
                if remaining == 0 {
                    break;
                }
                all_offspring.extend(extract_generation(
                    egraph,
                    parent_genome,
                    per_parent.min(remaining),
                    options.mutations,
                    &mut prev_selected,
                    rng,
                ));
            }
            if all_offspring.is_empty() {
                break;
            }

            for genome in all_offspring {
                if options
                    .group_timeout
                    .is_some_and(|timeout| group_start.elapsed() >= timeout)
                {
                    break;
                }
                n_graphs += 1;
                list_cache.clear();
                expr_cache.clear();

                let profile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut llir_graph = egglog_to_llir(
                        egraph,
                        genome.clone(),
                        ops,
                        &self.custom_ops,
                        &mut list_cache,
                        &mut expr_cache,
                        None,
                    );
                    // Collapse the rolled body to a single iteration
                    // before profiling — see initial-genome path.
                    collapse_loops_to_first_iter(&mut llir_graph);
                    runtime.clear_intermediate_buffers();
                    let (rep_metric, rep_display) =
                        runtime.profile(&llir_graph, &profile_dyn_map, options.trials);
                    let has_nan = runtime.has_nan_outputs(&llir_graph, &profile_dyn_map);
                    (rep_metric, rep_display, has_nan)
                }));

                let (new_metric, display_metric) = match profile_result {
                    Ok((metric, display, false)) => {
                        (R::aggregate_profile_metrics(&[metric]), display)
                    }
                    Ok((_, _, true)) | Err(_) => {
                        // NaN or panic — redraw bars and skip
                        for _ in 1..n_bar_lines {
                            print!("\x1b[1A");
                        }
                        print!("\r\x1b[2K");
                        render_bars(n_graphs, limit, bucket_progress);
                        std::io::stdout().flush().unwrap();
                        continue;
                    }
                };

                // Update parents list (keep top-N for next generation)
                let dominated_by_all = parents.len() >= options.keep_best
                    && !parents.last().unwrap().0.gt(&new_metric);
                if !dominated_by_all {
                    let pos = parents
                        .iter()
                        .position(|(m, _)| {
                            new_metric
                                .partial_cmp(m)
                                .is_some_and(|o| o == std::cmp::Ordering::Less)
                        })
                        .unwrap_or(parents.len());
                    parents.insert(pos, (new_metric.clone(), genome.clone()));
                    if parents.len() > options.keep_best {
                        parents.truncate(options.keep_best);
                    }
                }

                let new_best = best_metric.gt(&new_metric);
                if new_best {
                    best_metric = new_metric;
                    best_genome = genome.clone();
                }

                if new_best {
                    let msg = format!("   {:>6} {display_metric}", "Searched".green().bold());
                    for _ in 1..n_bar_lines {
                        print!("\x1b[1A");
                    }
                    print!("\r\x1b[2K");
                    println!("{msg}");
                } else {
                    for _ in 1..n_bar_lines {
                        print!("\x1b[1A");
                    }
                    print!("\r\x1b[2K");
                }
                render_bars(n_graphs, limit, bucket_progress);
                std::io::stdout().flush().unwrap();
            }
        }

        // Clear progress bars
        for _ in 1..n_bar_lines {
            print!("\x1b[1A");
        }
        print!("\r");
        for _ in 0..n_bar_lines {
            println!("\x1b[2K");
        }
        for _ in 0..n_bar_lines {
            print!("\x1b[1A");
        }
        print!("\r");
        std::io::stdout().flush().unwrap();

        // Re-extract the winning genome WITHOUT the per-candidate
        // single-iteration collapse, then run the real loop unroll. The
        // resulting LLIR is the full N-iteration graph the runtime executes;
        // the per-candidate collapsed form was used only for ranking.
        let mut stitched = egglog_to_llir(
            egraph,
            best_genome,
            ops,
            &self.custom_ops,
            &mut FxHashMap::default(),
            &mut FxHashMap::default(),
            None,
        );
        unroll_loops_in_llir(&mut stitched);

        println!(
            "   {:>6}  in {}",
            "Searched".green().bold(),
            pretty_duration::pretty_duration(&start.elapsed(), None)
        );

        stitched
    }
}

impl Deref for Graph {
    type Target = HLIRGraph;
    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for Graph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

fn build_uses(graph: &HLIRGraph) -> FxHashMap<NodeIndex, Vec<(NodeIndex, usize)>> {
    let mut uses: FxHashMap<NodeIndex, Vec<(NodeIndex, usize)>> = FxHashMap::default();
    for n in graph.node_indices() {
        uses.entry(n).or_default();
    }
    for dst in graph.node_indices() {
        let sources: Vec<_> = graph
            .edges_directed(dst, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect();
        for (port, src) in sources.into_iter().enumerate() {
            if let Some(v) = uses.get_mut(&src) {
                v.push((dst, port));
            }
        }
    }
    uses
}

struct RollingHash64 {
    prefix: Vec<u64>,
    powers: Vec<u64>,
}

impl RollingHash64 {
    const BASE: u64 = 1_000_000_007;

    fn new(tokens: &[u64]) -> Self {
        let mut prefix = Vec::with_capacity(tokens.len() + 1);
        let mut powers = Vec::with_capacity(tokens.len() + 1);
        prefix.push(0u64);
        powers.push(1u64);
        for &token in tokens {
            let next_prefix = prefix
                .last()
                .copied()
                .unwrap()
                .wrapping_mul(Self::BASE)
                .wrapping_add(token.wrapping_add(1));
            prefix.push(next_prefix);
            let next_power = powers.last().copied().unwrap().wrapping_mul(Self::BASE);
            powers.push(next_power);
        }
        Self { prefix, powers }
    }

    fn window_hash(&self, start: usize, len: usize) -> u64 {
        self.prefix[start + len].wrapping_sub(self.prefix[start].wrapping_mul(self.powers[len]))
    }
}

fn cheap_rolling_node_hash(graph: &HLIRGraph, node: NodeIndex) -> u64 {
    // Use Debug, NOT Display — Display for many HLIR ops drops their
    // shape/stride metadata (e.g. `Display for Mul` emits just "Mul"), so
    // two structurally-different ops with the same kind would hash equal
    // and get falsely grouped as a repeating pattern. Debug captures all
    // op fields, which is the correct notion of op identity for rolling.
    let op = format!("{:?}", graph[node]);
    let mut hash: u64 = 1469598103934665603;
    for byte in op.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1099511628211);
    }
    let in_degree = graph.neighbors_directed(node, Direction::Incoming).count() as u64;
    let out_degree = graph.neighbors_directed(node, Direction::Outgoing).count() as u64;
    hash ^= in_degree.wrapping_mul(0x9e3779b185ebca87);
    hash = hash.rotate_left(13);
    hash ^= out_degree.wrapping_mul(0xc2b2ae3d27d4eb4f);
    hash
}

fn rolling_probe_window_sizes(max_window: usize) -> Vec<usize> {
    if max_window == 0 {
        return vec![];
    }
    (1..=max_window).rev().collect()
}

fn canonicalize_occurrence(
    graph: &HLIRGraph,
    ordered_nodes: &[NodeIndex],
    uses: &FxHashMap<NodeIndex, Vec<(NodeIndex, usize)>>,
    topo_index: &FxHashMap<NodeIndex, usize>,
) -> Option<(String, Vec<NodeIndex>, Vec<NodeIndex>)> {
    let region: FxHashSet<NodeIndex> = ordered_nodes.iter().copied().collect();
    if region.is_empty() {
        return None;
    }
    let internal_index: FxHashMap<NodeIndex, usize> = ordered_nodes
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i))
        .collect();
    let mut param_index: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let mut boundary_inputs = vec![];
    let mut node_parts = vec![];

    for &node in ordered_nodes {
        // Debug, not Display — see `cheap_rolling_node_hash` for why op
        // identity must include all fields (shape/strides), which Display
        // drops for many HLIR ops.
        let op = format!("{:?}", graph[node]);
        let inputs: Vec<NodeIndex> = graph
            .edges_directed(node, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect();
        let mut inp_parts = vec![];
        for src in inputs {
            if let Some(&idx) = internal_index.get(&src) {
                inp_parts.push(format!("n{idx}"));
            } else {
                let p = *param_index.entry(src).or_insert_with(|| {
                    boundary_inputs.push(src);
                    boundary_inputs.len() - 1
                });
                inp_parts.push(format!("p{p}"));
            }
        }
        node_parts.push(format!("{op}({})", inp_parts.join(",")));
    }

    let mut output_nodes: Vec<NodeIndex> = ordered_nodes
        .iter()
        .copied()
        .filter(|n| {
            uses.get(n)
                .is_some_and(|out_uses| out_uses.iter().any(|(user, _)| !region.contains(user)))
                || graph.externals(Direction::Outgoing).any(|root| root == *n)
        })
        .collect();
    output_nodes.sort_by_key(|n| topo_index[n]);
    let outputs: Vec<String> = output_nodes
        .iter()
        .filter_map(|n| internal_index.get(n).copied())
        .map(|idx| format!("o{idx}"))
        .collect();

    let sig = format!("{}|{}", node_parts.join(";"), outputs.join(","));
    Some((sig, boundary_inputs, output_nodes))
}

fn collect_state_params(
    occurrences: &[RollingOccurrence],
    uses: &FxHashMap<NodeIndex, Vec<(NodeIndex, usize)>>,
    graph: &HLIRGraph,
) -> Vec<usize> {
    if occurrences.len() < 2 {
        return vec![];
    }
    let param_count = occurrences[0].boundary_inputs.len();
    let mut state_params = vec![];

    for p in 0..param_count {
        let mut is_state = true;
        for i in 1..occurrences.len() {
            let earlier = &occurrences[i - 1];
            let later = &occurrences[i];
            let val = later.boundary_inputs.get(p).copied();
            let Some(val) = val else {
                is_state = false;
                break;
            };
            if !earlier.output_nodes.contains(&val) {
                is_state = false;
                break;
            }
            let external_uses: Vec<_> = uses
                .get(&val)
                .map(|u| {
                    u.iter()
                        .copied()
                        .filter(|(user, _)| !earlier.nodes.contains(user))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if external_uses.is_empty() {
                is_state = false;
                break;
            }
            if graph.externals(Direction::Outgoing).any(|root| root == val) {
                is_state = false;
                break;
            }
            if external_uses
                .iter()
                .any(|(user, _)| !later.nodes.contains(user))
            {
                is_state = false;
                break;
            }
        }
        if is_state {
            state_params.push(p);
        }
    }
    state_params
}

fn grow_rolling_candidate(
    graph: &HLIRGraph,
    uses: &FxHashMap<NodeIndex, Vec<(NodeIndex, usize)>>,
    topo_index: &FxHashMap<NodeIndex, usize>,
    mut candidate: RollingCandidate,
    discovered_runs: &[RollingRun],
) -> RollingCandidate {
    loop {
        let candidate_starts: Vec<usize> = candidate
            .occurrences
            .iter()
            .map(|occ| {
                occ.nodes
                    .first()
                    .map(|n| topo_index[n])
                    .unwrap_or(usize::MAX)
            })
            .collect();
        let candidate_ends: Vec<usize> = candidate
            .occurrences
            .iter()
            .map(|occ| occ.nodes.last().map(|n| topo_index[n] + 1).unwrap_or(0))
            .collect();

        let mut best_growth: Option<RollingCandidate> = None;
        for run in discovered_runs {
            for shift in 0..=1usize {
                if run.occurrences.len() < candidate.occurrences.len() + shift {
                    continue;
                }
                let aligned = (0..candidate.occurrences.len()).all(|i| {
                    run.starts[i + shift] == candidate_ends[i]
                        || run.starts[i + shift] + run.window == candidate_starts[i]
                });
                if !aligned {
                    continue;
                }

                let mut merged_occs = Vec::with_capacity(candidate.occurrences.len());
                // `i` indexes the candidate side while `i + shift` indexes the
                // run side — explicit range is clearer than zip-with-skip.
                #[allow(clippy::needless_range_loop)]
                for i in 0..candidate.occurrences.len() {
                    let run_occ = &run.occurrences[i + shift];
                    let mut nodes = if run.starts[i + shift] + run.window == candidate_starts[i] {
                        let mut n = run_occ.nodes.clone();
                        n.extend(candidate.occurrences[i].nodes.iter().copied());
                        n
                    } else {
                        let mut n = candidate.occurrences[i].nodes.clone();
                        n.extend(run_occ.nodes.iter().copied());
                        n
                    };
                    nodes.sort_by_key(|n| topo_index[n]);
                    let Some((sig, boundary_inputs, output_nodes)) =
                        canonicalize_occurrence(graph, &nodes, uses, topo_index)
                    else {
                        merged_occs.clear();
                        break;
                    };
                    if i == 0 && sig.is_empty() {
                        merged_occs.clear();
                        break;
                    }
                    merged_occs.push(RollingOccurrence {
                        nodes,
                        boundary_inputs,
                        output_nodes,
                    });
                }
                if merged_occs.len() != candidate.occurrences.len() {
                    continue;
                }
                let first_sig =
                    canonicalize_occurrence(graph, &merged_occs[0].nodes, uses, topo_index)
                        .map(|(sig, _, _)| sig);
                let Some(first_sig) = first_sig else { continue };
                if merged_occs.iter().skip(1).any(|occ| {
                    canonicalize_occurrence(graph, &occ.nodes, uses, topo_index)
                        .map(|(sig, _, _)| sig != first_sig)
                        .unwrap_or(true)
                }) {
                    continue;
                }
                let state_param_indices = collect_state_params(&merged_occs, uses, graph);
                if state_param_indices.is_empty() {
                    continue;
                }
                let savings = merged_occs[0].nodes.len() * (merged_occs.len() - 1);
                let _ = first_sig;
                let grown = RollingCandidate {
                    occurrences: merged_occs,
                    state_param_indices,
                    savings,
                };
                let replace = best_growth.as_ref().is_none_or(|best| {
                    (
                        grown.savings,
                        grown.occurrences[0].nodes.len(),
                        grown.occurrences.len(),
                    ) > (
                        best.savings,
                        best.occurrences[0].nodes.len(),
                        best.occurrences.len(),
                    )
                });
                if replace {
                    best_growth = Some(grown);
                }
            }
        }

        match best_growth {
            Some(grown) if grown.savings > candidate.savings => candidate = grown,
            _ => return candidate,
        }
    }
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
    use crate::hlir::{
        LoopEnd, LoopInput, LoopInputStatic, LoopOutput, LoopOutputSelect, LoopStart, Output,
    };
    use petgraph::visit::EdgeRef;
    use std::collections::BTreeMap;

    let mut starts: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    let mut ends: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    let mut inputs: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    // Iteration-independent inputs. Keyed on LLIR NodeIndex (stream_id is not
    // unique when a single LoopInput splits into multiple LoopInputStatics via
    // egglog rewrites of the same stream).
    let mut static_inputs: FxHashSet<NodeIndex> = FxHashSet::default();
    // LoopOutput stream → its NodeIndex. Each stream has one LoopOutput.
    let mut outputs: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    // (stream_id, iter) → LoopOutputSelect NodeIndex.
    let mut output_selects: FxHashMap<NodeIndex, (usize /*stream*/, usize /*iter*/)> =
        FxHashMap::default();

    let mut iters = 0usize;
    for n in llir.node_indices() {
        let op = &llir[n];
        if let Some(ls) = op.to_op::<LoopStart>() {
            iters = iters.max(ls.iters.to_usize().unwrap_or(1));
            starts.insert(ls.slot_idx, n);
        } else if let Some(le) = op.to_op::<LoopEnd>() {
            ends.insert(le.slot_idx, n);
        } else if op.to_op::<LoopInputStatic>().is_some() {
            // Must be checked before LoopInput because LoopInputStatic is a
            // distinct op with its own sort name, but we want it recognized as
            // a separate category here.
            static_inputs.insert(n);
        } else if let Some(li) = op.to_op::<LoopInput>() {
            inputs.insert(li.stream_id, n);
        } else if let Some(los) = op.to_op::<LoopOutputSelect>() {
            output_selects.insert(n, (los.stream_id, los.iter));
        } else if let Some(lo) = op.to_op::<LoopOutput>() {
            outputs.insert(lo.stream_id, n);
        }
    }
    if iters <= 1 || starts.is_empty() {
        return;
    }

    let loop_markers: FxHashSet<NodeIndex> = starts
        .values()
        .copied()
        .chain(ends.values().copied())
        .chain(inputs.values().copied())
        .chain(static_inputs.iter().copied())
        .chain(outputs.values().copied())
        .chain(output_selects.keys().copied())
        .collect();

    let mut body_nodes: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut worklist: Vec<NodeIndex> = starts
        .values()
        .flat_map(|n| {
            llir.neighbors_directed(*n, Direction::Outgoing)
                .collect::<Vec<_>>()
        })
        .chain(inputs.values().flat_map(|n| {
            llir.neighbors_directed(*n, Direction::Outgoing)
                .collect::<Vec<_>>()
        }))
        .chain(static_inputs.iter().flat_map(|n| {
            llir.neighbors_directed(*n, Direction::Outgoing)
                .collect::<Vec<_>>()
        }))
        .collect();
    while let Some(n) = worklist.pop() {
        if body_nodes.contains(&n) || loop_markers.contains(&n) {
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

    // start_meta[loop_start] = (initial, body_producer):
    //   - `initial` = LoopStart's incoming (state at iter 0).
    //   - `body_producer` = LoopEnd's incoming (state value the body
    //     produces each iter).
    //
    // A state slot is **iteration-invariant** iff `body_producer` is not
    // in `body_nodes` (the forward-walk set from input markers). Such a
    // producer has no input-marker ancestor, so its value can't depend on
    // per-iter state — egglog rewrites prove the body chain reduces to a
    // constant or external value, and extraction picks that directly. A
    // real example is gemma's RoPE frequency factors (`Log2(10000)`,
    // `log2(e)`), which the kernel-rewrite chain folds onto the body slot.
    // For these, every iter sees the same state, so per-iter cloning is
    // skipped — `resolve_src` and `marker_post_sub` use `body_producer`
    // directly, recognised by `clone_map.get(&body_producer).is_none()`.
    let mut start_meta: FxHashMap<NodeIndex, (NodeIndex, NodeIndex)> = FxHashMap::default();
    for (slot_idx, &start_node) in &starts {
        let end_node = *ends
            .get(slot_idx)
            .unwrap_or_else(|| panic!("missing LoopEnd for slot {slot_idx}"));
        let initial = llir
            .neighbors_directed(start_node, Direction::Incoming)
            .next()
            .expect("LoopStart must have an initial-value producer");
        let body_producer = llir
            .neighbors_directed(end_node, Direction::Incoming)
            .next()
            .expect("LoopEnd must have a body producer");
        start_meta.insert(start_node, (initial, body_producer));
    }

    let mut input_per_iter: FxHashMap<NodeIndex, Vec<NodeIndex>> = FxHashMap::default();
    for input_node in inputs.values() {
        let srcs: Vec<NodeIndex> = llir
            .edges_directed(*input_node, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect();
        assert_eq!(
            srcs.len(),
            iters,
            "LoopInput stream must have `iters` sources"
        );
        input_per_iter.insert(*input_node, srcs);
    }
    // LoopInputStatic: single shared source reused across all iterations.
    let mut static_source: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    for &static_node in &static_inputs {
        let srcs: Vec<NodeIndex> = llir
            .edges_directed(static_node, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect();
        assert_eq!(
            srcs.len(),
            1,
            "LoopInputStatic must have exactly 1 source (got {})",
            srcs.len()
        );
        static_source.insert(static_node, srcs[0]);
    }

    let mut clone_map: Vec<FxHashMap<NodeIndex, NodeIndex>> = vec![FxHashMap::default(); iters];
    for &b in &body_nodes {
        clone_map[0].insert(b, b);
    }
    for clone in clone_map.iter_mut().skip(1) {
        for &b in &body_nodes {
            let cloned = llir.add_node(llir[b].clone());
            clone.insert(b, cloned);
        }
    }

    let resolve_src = |src: NodeIndex, i: usize, clone_map: &[FxHashMap<NodeIndex, NodeIndex>]| {
        if let Some(&(initial, body_producer)) = start_meta.get(&src) {
            if i == 0 {
                initial
            } else {
                // Iteration-invariant slot fallback: `body_producer` not in
                // `body_nodes` ⇒ not cloned per iter ⇒ all iters share it.
                clone_map[i - 1]
                    .get(&body_producer)
                    .copied()
                    .unwrap_or(body_producer)
            }
        } else if let Some(sources) = input_per_iter.get(&src) {
            sources[i]
        } else if let Some(&shared) = static_source.get(&src) {
            // LoopInputStatic: same source for every iteration's clone.
            shared
        } else if body_nodes.contains(&src) {
            clone_map[i][&src]
        } else {
            src
        }
    };

    let body_incoming: FxHashMap<NodeIndex, Vec<NodeIndex>> = body_nodes
        .iter()
        .map(|&b| {
            let srcs: Vec<NodeIndex> = llir
                .edges_directed(b, Direction::Incoming)
                .sorted_by_key(|e| e.id())
                .map(|e| e.source())
                .collect();
            (b, srcs)
        })
        .collect();

    // For iter 0, we rebuild each body node's incoming edges in place: we
    // remove each old edge and immediately re-add a new edge with the
    // resolved source. petgraph::stable_graph reuses freed edge indices
    // LIFO, so interleaving remove+add for each edge causes the new edge
    // to reuse exactly the freed slot, preserving edge-id ordering (which
    // the runtime relies on for input positions).
    for &b in &body_nodes {
        let pairs: Vec<(NodeIndex, petgraph::graph::EdgeIndex)> = llir
            .edges_directed(b, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| (e.source(), e.id()))
            .collect();
        for (src, eid) in pairs {
            let new_src = resolve_src(src, 0, &clone_map);
            llir.remove_edge(eid);
            llir.add_edge(new_src, b, ());
        }
    }
    // For iter > 0 clones, there are no existing edges — add fresh ones in
    // body_incoming order so edge-id ordering matches.
    for i in 1..iters {
        for &b in &body_nodes {
            let target = clone_map[i][&b];
            let srcs = &body_incoming[&b];
            for &src in srcs {
                let new_src = resolve_src(src, i, &clone_map);
                llir.add_edge(new_src, target, ());
            }
        }
    }

    let post_loop_consumers: FxHashSet<NodeIndex> = loop_markers
        .iter()
        .flat_map(|n| {
            llir.neighbors_directed(*n, Direction::Outgoing)
                .collect::<Vec<_>>()
        })
        .filter(|n| !loop_markers.contains(n) && !body_nodes.contains(n))
        .collect();

    // Resolve each LoopOutput stream's body producer (its single incoming
    // edge in the LLIR).
    let mut output_body_producer: FxHashMap<usize /*stream_id*/, NodeIndex> = FxHashMap::default();
    for (&stream_id, &output_node) in &outputs {
        let body_producer = llir
            .neighbors_directed(output_node, Direction::Incoming)
            .next()
            .expect("LoopOutput missing body producer during rewire");
        output_body_producer.insert(stream_id, body_producer);
    }

    let mut marker_post_sub: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    for &end_node in ends.values() {
        let body_producer = llir
            .neighbors_directed(end_node, Direction::Incoming)
            .next()
            .expect("LoopEnd missing body producer during rewire");
        // Same iteration-invariant fallback as `resolve_src`.
        let sub = clone_map[iters - 1]
            .get(&body_producer)
            .copied()
            .unwrap_or(body_producer);
        marker_post_sub.insert(end_node, sub);
    }
    // Each LoopOutputSelect(stream, iter) routes to iter's clone of that
    // stream's body producer. Same iteration-invariant fallback as for
    // LoopEnd above: if the body producer isn't in `body_nodes`, it wasn't
    // cloned per iter and every iter shares the single `body_producer`.
    for (&select_node, &(stream_id, iter)) in &output_selects {
        let body_producer = output_body_producer[&stream_id];
        let sub = clone_map[iter]
            .get(&body_producer)
            .copied()
            .unwrap_or(body_producer);
        marker_post_sub.insert(select_node, sub);
    }

    for &consumer in &post_loop_consumers {
        // Per-edge replace to preserve edge-id ordering via LIFO reuse.
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

    for &n in &loop_markers {
        llir.remove_node(n);
    }

    debug_assert_eq!(
        llir.node_indices()
            .filter(|n| {
                let op = &llir[*n];
                op.to_op::<LoopStart>().is_some()
                    || op.to_op::<LoopEnd>().is_some()
                    || op.to_op::<LoopInput>().is_some()
                    || op.to_op::<LoopInputStatic>().is_some()
                    || op.to_op::<LoopOutput>().is_some()
                    || op.to_op::<LoopOutputSelect>().is_some()
            })
            .count(),
        0,
        "unroll left stray loop marker ops in LLIR"
    );

    // Compact the graph into a freshly-allocated StableGraph so all edge
    // IDs are re-assigned sequentially in our chosen insertion order.
    // Without this, later kernel_to_host add_edge calls can reuse edge
    // indices that our unroll freed via remove_node on loop markers,
    // producing sort-by-edge-id orderings where a later-added scheduling
    // edge ends up at a low index — which the runtime interprets as a
    // primary input position and crashes looking up a buffer.
    let compacted = compact_llir_preserving_input_order(llir);
    *llir = compacted;
}

/// Collapse all loop markers in an LLIR graph down to a SINGLE iteration's
/// body, with first-iteration inputs and outputs only. This is the cheap
/// per-candidate form used by the genetic search — profiling one transformer
/// block instead of N×block makes the search ~N× faster, and the relative
/// cost of any extraction choice is preserved on the body shape.
///
/// LoopStart consumers re-route to the initial value, LoopInput consumers
/// re-route to `sources[0]`, LoopInputStatic consumers re-route to the single
/// shared source, LoopEnd's post-loop consumers re-route to the body producer
/// directly, and each `LoopOutput` is replaced with a single `Output { node:
/// targets[0] }`. After collapse the LLIR has no marker ops left and contains
/// exactly the iter-0 body plus the surrounding non-loop graph.
pub fn collapse_loops_to_first_iter(llir: &mut LLIRGraph) {
    use crate::hlir::{
        LoopEnd, LoopInput, LoopInputStatic, LoopOutput, LoopOutputSelect, LoopStart, Output,
    };
    use petgraph::visit::EdgeRef;
    use std::collections::BTreeMap;

    let mut starts: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    let mut ends: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    let mut inputs: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    let mut static_inputs: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut outputs: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    let mut output_selects: FxHashSet<NodeIndex> = FxHashSet::default();

    for n in llir.node_indices() {
        let op = &llir[n];
        if op.to_op::<LoopStart>().is_some() {
            starts.insert(op.to_op::<LoopStart>().unwrap().slot_idx, n);
        } else if let Some(le) = op.to_op::<LoopEnd>() {
            ends.insert(le.slot_idx, n);
        } else if op.to_op::<LoopInputStatic>().is_some() {
            static_inputs.insert(n);
        } else if let Some(li) = op.to_op::<LoopInput>() {
            inputs.insert(li.stream_id, n);
        } else if op.to_op::<LoopOutputSelect>().is_some() {
            output_selects.insert(n);
        } else if let Some(lo) = op.to_op::<LoopOutput>() {
            outputs.insert(lo.stream_id, n);
        }
    }
    if starts.is_empty() {
        return;
    }

    let loop_markers: FxHashSet<NodeIndex> = starts
        .values()
        .copied()
        .chain(ends.values().copied())
        .chain(inputs.values().copied())
        .chain(static_inputs.iter().copied())
        .chain(outputs.values().copied())
        .chain(output_selects.iter().copied())
        .collect();

    // body_nodes = forward-reachable from any marker outgoing, stopping at
    // markers and Output ops. This matches `unroll_loops_in_llir`.
    let mut body_nodes: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut worklist: Vec<NodeIndex> = starts
        .values()
        .flat_map(|n| {
            llir.neighbors_directed(*n, Direction::Outgoing)
                .collect::<Vec<_>>()
        })
        .chain(inputs.values().flat_map(|n| {
            llir.neighbors_directed(*n, Direction::Outgoing)
                .collect::<Vec<_>>()
        }))
        .chain(static_inputs.iter().flat_map(|n| {
            llir.neighbors_directed(*n, Direction::Outgoing)
                .collect::<Vec<_>>()
        }))
        .collect();
    while let Some(n) = worklist.pop() {
        if body_nodes.contains(&n) || loop_markers.contains(&n) {
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
    let mut static_source: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    for &static_node in &static_inputs {
        let src = llir
            .neighbors_directed(static_node, Direction::Incoming)
            .next()
            .expect("LoopInputStatic must have a source");
        static_source.insert(static_node, src);
    }

    // Resolve a source reference to its iter-0 equivalent.
    let resolve_src = |src: NodeIndex| -> NodeIndex {
        if let Some(&initial) = start_initial.get(&src) {
            initial
        } else if let Some(&first) = input_first_source.get(&src) {
            first
        } else if let Some(&shared) = static_source.get(&src) {
            shared
        } else {
            src
        }
    };

    // Rewrite every body node's incoming edges. Per-edge remove+add to keep
    // edge-id ordering via LIFO reuse — runtime reads inputs sorted by edge
    // id so position must be preserved.
    for &b in &body_nodes {
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
    for (&stream_id, &output_node) in &outputs {
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
    for &select_node in &output_selects {
        let stream_id = llir[select_node]
            .to_op::<LoopOutputSelect>()
            .map(|s| s.stream_id)
            .expect("output_selects entries must be LoopOutputSelect");
        if let Some(&body_producer) = output_body_producer.get(&stream_id) {
            marker_post_sub.insert(select_node, body_producer);
        }
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

    for &n in &loop_markers {
        llir.remove_node(n);
    }

    let compacted = compact_llir_preserving_input_order(llir);
    *llir = compacted;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egglog_utils::hash_egglog_normalized;
    use crate::tests::{assert_close, random_vec};

    #[test]
    fn test_hash_egglog_normalized_same_structure() {
        // Two egglog texts differing only in Input node indices and labels
        let text_a = r#"(let t0 (Input 42 "boundary" (F32)))
(let t1 (Input 100 "layers.0.wq.weight" (F32)))
(let t2 (Add (ECons 128 (ECons 4096 (ENil))) t1 (ECons 1 (ECons 128 (ENil))) t0 (ECons 1 (ECons 1 (ENil))) (ECons 1 (ECons 128 (ENil)))))
(let t3 (Output t2 42))
"#;
        let text_b = r#"(let t0 (Input 84 "boundary" (F32)))
(let t1 (Input 200 "layers.1.wq.weight" (F32)))
(let t2 (Add (ECons 128 (ECons 4096 (ENil))) t1 (ECons 1 (ECons 128 (ENil))) t0 (ECons 1 (ECons 1 (ENil))) (ECons 1 (ECons 128 (ENil)))))
(let t3 (Output t2 84))
"#;
        assert_eq!(
            hash_egglog_normalized(text_a),
            hash_egglog_normalized(text_b),
            "Structurally identical chunks should hash the same"
        );
    }

    #[test]
    fn test_hash_egglog_normalized_different_structure() {
        let text_a = r#"(let t0 (Input 42 "boundary" (F32)))
(let t1 (Add (ECons 128 (ENil)) t0 (ECons 1 (ENil)) t0 (ECons 1 (ENil)) (ECons 1 (ENil))))
"#;
        let text_b = r#"(let t0 (Input 42 "boundary" (F32)))
(let t1 (Mul (ECons 128 (ENil)) t0 (ECons 1 (ENil)) t0 (ECons 1 (ENil)) (ECons 1 (ENil))))
"#;
        assert_ne!(
            hash_egglog_normalized(text_a),
            hash_egglog_normalized(text_b),
            "Different op types should produce different hashes"
        );
    }

    #[test]
    fn test_hash_egglog_normalized_different_dtypes() {
        let text_a = "(let t0 (Input 42 \"boundary\" (F32)))\n";
        let text_b = "(let t0 (Input 42 \"boundary\" (F16)))\n";
        assert_ne!(
            hash_egglog_normalized(text_a),
            hash_egglog_normalized(text_b),
            "Different dtypes should produce different hashes"
        );
    }

    #[test]
    fn test_hash_egglog_normalized_output_join_not_normalized() {
        // OutputJoin lines should be hashed verbatim, not treated as Output
        let text_a = "(let t0 (OutputJoin t1 t2))\n";
        let text_b = "(let t0 (OutputJoin t3 t4))\n";
        assert_ne!(
            hash_egglog_normalized(text_a),
            hash_egglog_normalized(text_b),
            "OutputJoin lines should be hashed verbatim"
        );
    }

    #[test]
    fn test_hash_egglog_normalized_custom_op_id() {
        // CustomOpKind lines differ only in the integer ID (layer index)
        let text_a = r#"(let t0 (Input 441 "boundary" (F32)))
(let t1 (Op (CustomOpKind 1 (F32)) (ICons t74 (ICons t120 (ICons t28 (INil))))))
(let t2 (Output t1 585))
"#;
        let text_b = r#"(let t0 (Input 585 "boundary" (F32)))
(let t1 (Op (CustomOpKind 2 (F32)) (ICons t74 (ICons t120 (ICons t28 (INil))))))
(let t2 (Output t1 729))
"#;
        assert_eq!(
            hash_egglog_normalized(text_a),
            hash_egglog_normalized(text_b),
            "CustomOpKind with different IDs should hash the same"
        );
    }

    #[test]
    fn test_hash_egglog_normalized_custom_op_different_structure() {
        // CustomOpKind lines with different input lists should hash differently
        let text_a = "(let t1 (Op (CustomOpKind 1 (F32)) (ICons t74 (ICons t120 (INil)))))\n";
        let text_b = "(let t1 (Op (CustomOpKind 1 (F32)) (ICons t74 (ICons t99 (INil)))))\n";
        assert_ne!(
            hash_egglog_normalized(text_a),
            hash_egglog_normalized(text_b),
            "CustomOpKind with different input lists should hash differently"
        );
    }

    #[test]
    fn test_auto_roll_loops_prepass_creates_regions_for_chain_recurrence() {
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let out = x.exp2().sin().exp2().sin().exp2().sin().output();

        let inserted = cx.auto_roll_loops_prepass();
        assert!(
            inserted >= 2,
            "expected at least two loop boundaries for 3 repeated bodies, got {inserted}"
        );

        let vals = random_vec(8);
        let mut rt = NativeRuntime::default();
        cx.build_search_space::<NativeRuntime>();
        rt = cx.search(rt, 1);
        rt.set_data(x.id, vals.clone());
        rt.execute(&cx.dyn_map);

        let expected = vals
            .into_iter()
            .map(|v| v.exp2().sin().exp2().sin().exp2().sin())
            .collect::<Vec<f32>>();
        assert_close(rt.get_f32(out.id), &expected);
    }

    #[test]
    fn test_auto_roll_loops_prepass_skips_non_recurrent_branches() {
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let y = cx.tensor(8);
        let _out = (x.exp().sin() + y.exp().sin()).output();

        let inserted = cx.auto_roll_loops_prepass();
        assert_eq!(inserted, 0, "branch-only reuse should not roll into loops");
    }
}
