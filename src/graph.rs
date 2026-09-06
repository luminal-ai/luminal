use crate::egglog_utils::{
    hlir_to_egglog, log_channel_enabled, run_egglog_with_late_passes_interval_analysis_and_log,
};
pub use crate::search::unroll::{collapse_loops_to_first_iter, unroll_loops_in_llir};
use crate::search::{BucketSearchSpace, SearchSpace, bucket_index_combinations};
use crate::shape::{DimInterval, DynDimIntervals};
use crate::{
    egglog_utils::SerializedEGraph,
    op::{EgglogOp, IntoEgglogOp, LLIROp},
};
use crate::{hlir::CustomOpKind, op::*, prelude::*};
use colored::Colorize;
use itertools::Itertools;
use petgraph::{Direction, stable_graph::StableGraph, visit::EdgeRef};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
    sync::Arc,
};
use tracing;

mod artifact;

pub use artifact::{ScheduleBucket, SelectedSchedule};

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
pub type BucketLLIR = (DynMap, DynMap, LLIRGraph);

/// Borrowed view of a compiled bucket used for non-committing aggregate
/// candidate filtering.
#[derive(Clone, Copy)]
pub struct BucketLLIRRef<'a> {
    pub bucket_indices: &'a DynMap,
    pub representative_dyn_map: &'a DynMap,
    pub llir: &'a LLIRGraph,
}

impl<'a> From<&'a BucketLLIR> for BucketLLIRRef<'a> {
    fn from((bucket_indices, representative_dyn_map, llir): &'a BucketLLIR) -> Self {
        Self {
            bucket_indices,
            representative_dyn_map,
            llir,
        }
    }
}

/// A bucket for a dynamic dimension, defining a range of valid values.
/// For an exact value, use `min == max` (zero-length range).
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
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
    /// This is not a runtime shape constraint or implicit padding: executions
    /// still use the exact dynamic value anywhere in the bucket's range.
    pub fn representative_value(&self) -> usize {
        self.representative_override
            .unwrap_or((self.min + self.max) / 2)
    }

    /// Check if `val` falls within this bucket's range.
    pub fn contains(&self, val: usize) -> bool {
        val >= self.min && val <= self.max
    }
}

/// Options for building an e-graph search space and searching it.
///
/// Use the builder pattern to configure search parameters:
/// ```
/// use luminal::prelude::CompileOptions;
/// let opts = CompileOptions::default()
///     .search_graph_limit(5)
///     .search_time_limit(std::time::Duration::from_secs(30))
///     .generation_size(50)
///     .mutations(40)
///     .trials(15);
/// ```
#[derive(Debug, Clone)]
pub struct CompileOptions {
    /// Maximum number of graphs to evaluate during search.
    pub limit: usize,
    /// Maximum wall-clock time to spend searching.
    pub search_time_limit: std::time::Duration,
    /// Number of offspring per generation (default: 10)
    pub generation_size: usize,
    /// Number of mutations applied to each offspring (default: 10)
    pub mutations: usize,
    /// Number of profiling trials per candidate (default: 3)
    pub trials: usize,
    /// Number of best genomes to keep as parents per generation (default: 1)
    pub keep_best: usize,
    /// Generations without a new best before exploration escalates:
    /// mutation counts grow per stagnant generation (escaping local minima
    /// needs multi-gene jumps) and every other stagnant generation samples
    /// fresh random genomes. 0 disables (default: 0).
    pub restart_stagnation: usize,
    /// Per-candidate viability budget covering compile (`load_llir`) + run.
    /// Candidates exceeding it are discarded (default: 60 seconds).
    pub candidate_timeout: Option<std::time::Duration>,
    /// Caps how long profiling runs a single trial; not a rejection criterion.
    pub execution_timeout: Option<std::time::Duration>,
    /// Stop profiling a candidate early once its running mean exceeds
    /// `factor ×` the best candidate's metric. The partial mean is still
    /// returned and ranked normally, so this never changes which candidates
    /// are eligible — it only stops spending trials on candidates that have
    /// already lost by at least this margin. `None` disables (default).
    pub early_stop_factor: Option<f64>,
    /// Dynamic dimension values applied after search-space construction and
    /// before search. These values persist in [`Graph::dyn_map`] and provide
    /// the base representative values for unbucketed dimensions. Per-bucket
    /// representatives override them during bucketed search, and
    /// [`CompileOptions::profile_dims`] override them only while profiling.
    pub search_dims: DynMap,
    /// Optional profiling dimension overrides.
    pub profile_dims: DynMap,
    /// Bucket definitions per dynamic dimension. Dimensions without buckets use
    /// a single implicit bucket.
    pub dim_buckets: FxHashMap<Symbol, Vec<DimBucket>>,
    /// Enable egglog progress logging. Quiet by default; overridden by
    /// `EGGLOG_LOG=1` or `LUMINAL_LOG=1`.
    pub egglog_log: bool,
    /// Enable automatic loop rolling and its diagnostics. Disabled by default;
    /// overridden by `ROLLING_LOG=1` or `LUMINAL_LOG=1`.
    pub rolling_log: bool,
    /// Enable search progress logging. Enabled by default; overridden by
    /// `SEARCH_LOG=0`/`1` or `LUMINAL_LOG=1`.
    pub search_log: bool,
}

/// Resolve a caller-supplied dimension name, rejecting the reserved loop index.
///
/// Covers the methods that give a dimension a value, not every way in:
/// `Graph::dyn_map` and `CompileOptions::{search_dims, profile_dims,
/// dim_buckets}` are public fields, and a serialized `ShapeTracker`
/// deserializes without passing through here.
fn checked_dim(dimension: impl Into<Symbol>) -> Symbol {
    let dimension = dimension.into();
    assert!(
        !dimension.is_reserved(),
        "{}",
        crate::shape::InvalidSymbolName::Reserved
    );
    dimension
}

impl CompileOptions {
    /// Set the maximum number of graphs to evaluate during search.
    pub fn search_graph_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set the maximum wall-clock time to spend searching.
    pub fn search_time_limit(mut self, search_time_limit: std::time::Duration) -> Self {
        self.search_time_limit = search_time_limit;
        self
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

    pub fn restart_stagnation(mut self, generations: usize) -> Self {
        self.restart_stagnation = generations;
        self
    }

    /// Set the outer per-candidate timeout (compilation + execution).
    pub fn candidate_timeout(mut self, candidate_timeout: std::time::Duration) -> Self {
        self.candidate_timeout = Some(candidate_timeout);
        self
    }

    /// Set the inner single-execution timeout (execution only, excludes compile).
    pub fn execution_timeout(mut self, execution_timeout: std::time::Duration) -> Self {
        self.execution_timeout = Some(execution_timeout);
        self
    }

    /// Stop profiling a candidate once its running mean exceeds `factor ×`
    /// the current best. See [`CompileOptions::early_stop_factor`].
    pub fn early_stop_factor(mut self, factor: f64) -> Self {
        assert!(
            factor >= 1.0,
            "early_stop_factor below 1.0 would truncate candidates still in contention"
        );
        self.early_stop_factor = Some(factor);
        self
    }

    /// Set a dynamic dimension after search-space construction and before
    /// search. This is equivalent to calling [`Graph::set_dim`] between
    /// [`Graph::build_search_space`] and [`Graph::search`], while still using
    /// the unified [`Graph::compile`] API.
    pub fn search_dim(mut self, dim: impl Into<Symbol>, value: usize) -> Self {
        let dim = checked_dim(dim);
        self.search_dims.insert(dim, value);
        self
    }

    /// Override a dynamic dimension value used during search profiling.
    pub fn profile_dim(mut self, dim: impl Into<Symbol>, value: usize) -> Self {
        let dim = checked_dim(dim);
        self.profile_dims.insert(dim, value);
        self
    }

    /// Define buckets for a dynamic dimension.
    ///
    /// Bucketed compilation builds a separate search space and selected LLIR for
    /// each bucket combination. Buckets must not overlap and must cover all
    /// values that will be used at runtime.
    pub fn dim_buckets(mut self, dimension: impl Into<Symbol>, buckets: &[DimBucket]) -> Self {
        let dimension = checked_dim(dimension);
        validate_dim_buckets(dimension, buckets);
        self.dim_buckets.insert(dimension, buckets.to_vec());
        self
    }

    /// Enable or disable egglog progress logging.
    pub fn egglog_log(mut self, enabled: bool) -> Self {
        self.egglog_log = enabled;
        self
    }

    /// Enable or disable automatic loop rolling and its diagnostics.
    pub fn rolling_log(mut self, enabled: bool) -> Self {
        self.rolling_log = enabled;
        self
    }

    /// Enable or disable search progress logging.
    pub fn search_log(mut self, enabled: bool) -> Self {
        self.search_log = enabled;
        self
    }

    fn egglog_log_enabled(&self) -> bool {
        log_channel_enabled(self.egglog_log, "EGGLOG_LOG")
    }

    fn rolling_log_enabled(&self) -> bool {
        log_channel_enabled(self.rolling_log, "ROLLING_LOG")
    }

    /// Whether search progress logging is on, honoring `SEARCH_LOG` /
    /// `LUMINAL_LOG` overrides.
    pub fn search_log_enabled(&self) -> bool {
        log_channel_enabled(self.search_log, "SEARCH_LOG")
    }
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            limit: 100,
            search_time_limit: std::time::Duration::MAX,
            generation_size: 10,
            mutations: 10,
            trials: 3,
            keep_best: 1,
            restart_stagnation: 0,
            candidate_timeout: Some(std::time::Duration::from_secs(60)),
            execution_timeout: Some(std::time::Duration::from_secs(1)),
            early_stop_factor: None,
            search_dims: FxHashMap::default(),
            profile_dims: FxHashMap::default(),
            dim_buckets: FxHashMap::default(),
            egglog_log: false,
            rolling_log: false,
            search_log: true,
        }
    }
}

fn validate_dim_buckets(dimension: Symbol, buckets: &[DimBucket]) {
    assert!(
        !buckets.is_empty(),
        "Buckets for dim '{dimension}' must not be empty"
    );
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
}

/// A Luminal compute graph.
///
/// All computation is represented as a directed acyclic graph.
#[derive(Debug, Default)]
pub struct Graph {
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: DynMap,
    /// Edge weights: (Input index, Output index, Input shape)
    pub graph: HLIRGraph,
    /// The saturated search space built by [`Graph::build_search_space`]:
    /// one e-graph per bucket combination, handed to the runtime to search.
    search_space: Option<SearchSpace>,
    /// Custom ops
    pub custom_ops: Vec<Box<dyn CustomOp>>,
    /// Optional graph-wide interval assumptions for dynamic dimensions.
    pub dim_intervals: DynDimIntervals,
    /// Metadata for Input nodes: NodeIndex -> (label, dtype).
    /// Stored as plain data so it survives cross-binary type identity mismatches
    /// when external backend plugins are compiled separately.
    pub input_meta: FxHashMap<NodeIndex, (String, DType)>,
    selected_schedule: Option<SelectedSchedule>,
}

impl Graph {
    /// Create a new graph
    pub fn new() -> Graph {
        Graph::default()
    }

    fn run_auto_loop_rolling_prepass(&mut self, options: &CompileOptions) {
        let log = options.rolling_log_enabled();
        let before = self.graph.node_count();
        // Roll to a fixpoint. Each pass rolls the single best repeated
        // region, and rolling one region can expose the next: a periodic
        // layer pattern (e.g. 5 local + 1 global attention layer) first
        // rolls into a multi-layer body, and only then do the identical
        // layers inside that one surviving body form a rollable run of
        // their own. Termination: every roll strictly deletes duplicate
        // body nodes, and marker ops are unique so they never form new
        // repeats.
        let mut rolled = 0usize;
        while self.auto_roll_loops_prepass_with_log(log) > 0 {
            rolled += 1;
        }
        if rolled == 0 {
            println!(
                "   {:>6}  no loop regions found (max body={})",
                "Rolled".cyan().bold(),
                before / 2,
            );
        }
        if log {
            self.debug_validate_rolled_regions();
        }
    }

    /// ROLLING_LOG diagnostic: walk each rolled region's body in the HLIR and
    /// report any path that reaches another region's markers without passing
    /// through this region's own exit markers. Inner regions reaching outer
    /// markers directly means some cross-region edge was not rewired through
    /// a marker at insert time.
    fn debug_validate_rolled_regions(&self) {
        use crate::hlir::{LoopEnd, LoopInput, LoopOutput, LoopOutputSelect, LoopStart, Output};
        let mut markers: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let mut entries: FxHashMap<usize, Vec<NodeIndex>> = FxHashMap::default();
        for n in self.graph.node_indices() {
            if let Some(op) = self.try_get_op::<LoopStart>(n) {
                markers.insert(n, op.loop_id);
                entries.entry(op.loop_id).or_default().push(n);
            } else if let Some(op) = self.try_get_op::<LoopEnd>(n) {
                markers.insert(n, op.loop_id);
            } else if let Some(op) = self.try_get_op::<LoopInput>(n) {
                markers.insert(n, op.loop_id);
                entries.entry(op.loop_id).or_default().push(n);
            } else if let Some(op) = self.try_get_op::<LoopOutput>(n) {
                markers.insert(n, op.loop_id);
            } else if let Some(op) = self.try_get_op::<LoopOutputSelect>(n) {
                markers.insert(n, op.loop_id);
            }
        }
        for (&id, seeds) in entries.iter().sorted_by_key(|(id, _)| **id) {
            let mut seen: FxHashSet<NodeIndex> = FxHashSet::default();
            let mut bridges = 0usize;
            let mut worklist: Vec<(NodeIndex, NodeIndex)> = seeds
                .iter()
                .flat_map(|m| {
                    self.graph
                        .neighbors_directed(*m, Direction::Outgoing)
                        .map(|s| (s, *m))
                        .collect::<Vec<_>>()
                })
                .collect();
            while let Some((n, pred)) = worklist.pop() {
                if seen.contains(&n) {
                    continue;
                }
                if let Some(&owner) = markers.get(&n) {
                    if owner != id {
                        bridges += 1;
                        if bridges <= 3 {
                            println!(
                                "   {:>6}  region {id} bridge: {} -> {} (owner {owner})",
                                "Rolled".red().bold(),
                                self.graph[pred],
                                self.graph[n],
                            );
                        }
                    }
                    continue;
                }
                if self.try_get_op::<Output>(n).is_some() {
                    continue;
                }
                seen.insert(n);
                for succ in self
                    .graph
                    .neighbors_directed(n, Direction::Outgoing)
                    .collect::<Vec<_>>()
                {
                    worklist.push((succ, n));
                }
            }
            println!(
                "   {:>6}  region {id}: body={} foreign-marker bridges={}",
                "Rolled".cyan().bold(),
                seen.len(),
                bridges,
            );
        }
    }

    /// Add edges whose relative edge-id order carries meaning (LoopInput
    /// per-iteration sources) with freshly allocated, ascending edge ids.
    /// StableGraph recycles freed edge indices LIFO, so after any removals a
    /// plain `add_edge` sequence lands on arbitrary ids — and `get_sources`
    /// / serialization order edges by id. Drain the free list with dummy
    /// edges first, add the real edges in fresh index territory, then drop
    /// the dummies.
    fn add_iteration_ordered_edges(&mut self, pending: Vec<(NodeIndex, Vec<NodeIndex>)>) {
        use petgraph::visit::EdgeIndexable;
        let Some(&(anchor, _)) = pending.first() else {
            return;
        };
        let bound = EdgeIndexable::edge_bound(&self.graph);
        let mut dummies = Vec::new();
        loop {
            let e = self.graph.add_edge(anchor, anchor, ());
            let fresh = e.index() >= bound;
            dummies.push(e);
            if fresh {
                break;
            }
        }
        for (marker, sources) in pending {
            for src in sources {
                self.graph.add_edge(src, marker, ());
            }
        }
        for e in dummies {
            self.graph.remove_edge(e);
        }
    }

    /// Mutate the HLIR graph in place to fold N repeated body occurrences into
    /// a single body plus loop-marker ops. See `auto_roll_loops_prepass`.
    fn insert_loop_region_ops(&mut self, candidate: RollingCandidate, log: bool) -> usize {
        use crate::hlir::{LoopEnd, LoopInput, LoopOutput, LoopOutputSelect, LoopStart, Output};
        use petgraph::visit::EdgeRef;

        let nodes_before = self.graph.node_count();
        let n_iters = candidate.occurrences.len();
        // Regions roll one at a time; each gets the next free id so nested
        // and disjoint regions stay distinguishable through egglog and the
        // LLIR unroll/collapse passes.
        let loop_id = self
            .graph
            .node_indices()
            .filter_map(|n| {
                self.try_get_op::<LoopStart>(n)
                    .map(|start| start.loop_id + 1)
            })
            .max()
            .unwrap_or(0);

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
        // Loop markers cross the HLIR -> egglog -> LLIR boundary and therefore
        // carry the same concrete dtype as the tensor they represent. Compute
        // those dtypes from the still-unmodified graph before inserting any
        // markers; an unknown or inconsistent dtype is a compile error, never
        // an F32 default.
        let dtype_map = self.concrete_node_dtypes();
        // Track all NodeIndex slots we newly assign for loop-marker ops.
        // StableGraph reuses freed node indices; removals later in this
        // function might target slots that happen to coincide with a new
        // loop-marker's NodeIndex, so we explicitly exclude those.
        let mut added_loop_ops: FxHashSet<NodeIndex> = FxHashSet::default();
        // LoopInput per-iteration source edges, added together at the end:
        // edge-id order IS logical input order across HLIR (`get_sources`
        // sorts by id), and StableGraph recycles freed edge indices LIFO, so
        // adding these amid the rewiring below — or after a previous pass's
        // duplicate-body deletions — would hand them arbitrary recycled ids
        // and silently permute iteration order at serialization.
        let mut deferred_source_edges: Vec<(NodeIndex, Vec<NodeIndex>)> = Vec::new();

        for (slot_idx, (&p, &out_pos)) in candidate
            .state_param_indices
            .iter()
            .zip(state_out_pos_per_slot.iter())
            .enumerate()
        {
            let initial = candidate.occurrences[0].boundary_inputs[p];
            let body_state_out = candidate.occurrences[0].output_nodes[out_pos];
            let last_state_out = candidate.occurrences[n_iters - 1].output_nodes[out_pos];
            let dtype = dtype_map[&initial];
            for (role, node) in [
                ("loop body state output", body_state_out),
                ("loop final state output", last_state_out),
            ] {
                let actual = dtype_map[&node];
                assert_eq!(
                    actual,
                    dtype,
                    "loop {loop_id} slot {slot_idx} changes dtype: initial node {} is {dtype:?}, {role} node {} is {actual:?}",
                    initial.index(),
                    node.index(),
                );
            }

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
            let dtype = self.uniform_node_dtype(
                &per_iter_sources,
                &dtype_map,
                &format!("loop {loop_id} input stream {p}"),
            );
            assert_eq!(
                dtype, dtype_map[&body_input],
                "loop {loop_id} input stream {p} body input has a different concrete dtype"
            );
            if log {
                println!(
                    "   {:>6}  loop {loop_id} stream {p}: per-iter sources {:?}",
                    "Rolled".cyan().bold(),
                    per_iter_sources
                        .iter()
                        .map(|n| n.index())
                        .collect::<Vec<_>>(),
                );
            }
            let loop_input = self.graph.add_node(Box::new(LoopInput {
                loop_id,
                stream_id: p,
                dtype,
            }));
            added_loop_ops.insert(loop_input);
            // Deferred: added at the end with fresh ascending edge ids —
            // see `add_iteration_ordered_edges`.
            deferred_source_edges.push((loop_input, per_iter_sources.clone()));

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
            let per_iter_outputs: Vec<NodeIndex> = per_iter_plan
                .iter()
                .map(|(producer, _)| *producer)
                .collect();
            let dtype = self.uniform_node_dtype(
                &per_iter_outputs,
                &dtype_map,
                &format!("loop {loop_id} output stream {q}"),
            );

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

        self.add_iteration_ordered_edges(deferred_source_edges);

        if log && created > 0 {
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

    /// Resolve every HLIR node's concrete dtype in topological order.
    ///
    /// Each operation owns its dtype contract through
    /// [`HLIROp::output_dtype`]. There is no graph-level opcode table and no
    /// fallback dtype: malformed graphs fail before a transform can stamp
    /// incorrect metadata onto a structural marker.
    fn concrete_node_dtypes(&self) -> FxHashMap<NodeIndex, DType> {
        let order = petgraph::algo::toposort(&self.graph, None).unwrap_or_else(|cycle| {
            panic!("HLIR contains a cycle at node {}", cycle.node_id().index())
        });
        let mut dtypes = FxHashMap::default();
        for node in order {
            let sources = self.get_sources(node);
            let input_dtypes: Vec<_> = sources
                .iter()
                .map(|source| {
                    dtypes.get(source).copied().unwrap_or_else(|| {
                        panic!(
                            "HLIR node {} ({}) depends on node {} before its concrete dtype is known",
                            node.index(),
                            self.graph[node],
                            source.index(),
                        )
                    })
                })
                .collect();
            let dtype = self.graph[node].output_dtype(&input_dtypes);
            if let Some((_, metadata_dtype)) = self.input_meta.get(&node) {
                assert_eq!(
                    dtype,
                    *metadata_dtype,
                    "HLIR node {} ({}) has conflicting concrete dtypes: op={dtype:?}, metadata={metadata_dtype:?}",
                    node.index(),
                    self.graph[node],
                );
            }
            dtypes.insert(node, dtype);
        }
        dtypes
    }

    fn uniform_node_dtype(
        &self,
        nodes: &[NodeIndex],
        dtypes: &FxHashMap<NodeIndex, DType>,
        context: &str,
    ) -> DType {
        let (&first, rest) = nodes
            .split_first()
            .unwrap_or_else(|| panic!("{context} has no tensor sources"));
        let dtype = dtypes[&first];
        for &node in rest {
            let actual = dtypes[&node];
            assert_eq!(
                actual,
                dtype,
                "{context} mixes concrete dtypes: node {} is {dtype:?}, node {} is {actual:?}",
                first.index(),
                node.index(),
            );
        }
        dtype
    }

    /// Set a runtime dimension
    pub fn set_dim(&mut self, dimension: impl Into<Symbol>, val: usize) {
        let dimension = checked_dim(dimension);
        self.dyn_map.insert(dimension, val);
    }

    pub fn set_dim_interval(&mut self, dimension: impl Into<Symbol>, min: i64, max: i64) {
        let dimension = checked_dim(dimension);
        self.dim_intervals
            .insert(dimension, DimInterval::new(min, max));
    }

    /// Attempt to discover repeated HLIR regions and build explicit region
    /// descriptors for loop-carried state edges.
    /// Returns the number of detected inter-region boundaries.
    ///
    /// This is a conservative prepass:
    /// - only rolls candidates with at least one loop-carried state parameter
    /// - only inserts when the carried edge shapes can be inferred
    pub fn auto_roll_loops_prepass(&mut self) -> usize {
        let log = log_channel_enabled(false, "ROLLING_LOG");
        self.auto_roll_loops_prepass_with_log(log)
    }

    fn auto_roll_loops_prepass_with_log(&mut self, log: bool) -> usize {
        let max_region_size = self.graph.node_count() / 2;
        if max_region_size < 1 {
            return 0;
        }
        if log {
            println!(
                "   {:>6}  scanning {} HLIR nodes for loop regions (max body={})",
                "Rolled".cyan().bold(),
                self.graph.node_count(),
                max_region_size,
            );
        }
        let report = self.best_rolling_candidate(max_region_size);
        let Some(candidate) = report.candidate else {
            if log {
                self.print_rolling_search_diagnostics(&report.diagnostics);
            }
            return 0;
        };
        if log {
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
        }
        if candidate.occurrences.len() < 2 {
            return 0;
        }
        // Reject rolls that grow the graph: a roll that deletes fewer
        // duplicate nodes than the markers it creates is pure overhead.
        // (Termination of the rolling fixpoint doesn't depend on this —
        // every roll irreversibly consumes a repetition run, since markers
        // are unique and can never form new repeats — so break-even rolls
        // are kept for their search-space compression.)
        let net = rolling_net_savings(&candidate);
        if net < 0 {
            if log {
                println!(
                    "   {:>6}  best candidate rejected: net savings {} <= 0 (body={} trips={})",
                    "Rolled".yellow().bold(),
                    net,
                    candidate.occurrences[0].nodes.len(),
                    candidate.occurrences.len(),
                );
            }
            return 0;
        }

        // Mutate the HLIR in place — insert LoopStart/LoopEnd/LoopInput/
        // LoopOutput markers, delete N-1 duplicate bodies. The loop structure
        // is encoded in the HLIR graph itself and the downstream single-root
        // egglog path picks it up unchanged.
        self.insert_loop_region_ops(candidate, log)
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

    /// Innermost enclosing rolled region per HLIR node. Walked outer-to-inner
    /// (ascending loop_id = creation order), so later (inner) regions
    /// overwrite nothing: an outer walk stops at the inner region's markers
    /// and never sees the inner body. Nodes outside every region are absent.
    fn region_scope_map(&self) -> FxHashMap<NodeIndex, usize> {
        use crate::hlir::{LoopInput, LoopStart, Output};
        let mut entries: std::collections::BTreeMap<usize, Vec<NodeIndex>> =
            std::collections::BTreeMap::new();
        for n in self.graph.node_indices() {
            if let Some(op) = self.try_get_op::<LoopStart>(n) {
                entries.entry(op.loop_id).or_default().push(n);
            } else if let Some(op) = self.try_get_op::<LoopInput>(n) {
                entries.entry(op.loop_id).or_default().push(n);
            }
        }
        let mut scope: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        for (&id, seeds) in &entries {
            let mut worklist: Vec<NodeIndex> = seeds
                .iter()
                .flat_map(|m| {
                    self.graph
                        .neighbors_directed(*m, Direction::Outgoing)
                        .collect::<Vec<_>>()
                })
                .collect();
            let mut seen: FxHashSet<NodeIndex> = FxHashSet::default();
            while let Some(n) = worklist.pop() {
                if seen.contains(&n)
                    || self.is_rolled_loop_marker(n)
                    || self.try_get_op::<Output>(n).is_some()
                {
                    continue;
                }
                seen.insert(n);
                scope.insert(n, id);
                for succ in self
                    .graph
                    .neighbors_directed(n, Direction::Outgoing)
                    .collect::<Vec<_>>()
                {
                    worklist.push(succ);
                }
            }
        }
        scope
    }

    fn is_rolled_loop_marker(&self, n: NodeIndex) -> bool {
        use crate::hlir::{
            LoopEnd, LoopInput, LoopInputStatic, LoopOutput, LoopOutputSelect, LoopStart,
        };
        self.try_get_op::<LoopStart>(n).is_some()
            || self.try_get_op::<LoopEnd>(n).is_some()
            || self.try_get_op::<LoopInput>(n).is_some()
            || self.try_get_op::<LoopInputStatic>(n).is_some()
            || self.try_get_op::<LoopOutput>(n).is_some()
            || self.try_get_op::<LoopOutputSelect>(n).is_some()
    }

    fn best_rolling_candidate(&self, max_region_size: usize) -> RollingSearchReport {
        // The signature memo is keyed by NodeIndex; clear it so entries from a
        // prior (now-mutated) graph state can't leak into this read-only search.
        clear_rolling_sig_cache();
        let Some(full_topo) = stable_toposort_by_node_index(&self.graph) else {
            return RollingSearchReport {
                candidate: None,
                diagnostics: RollingSearchDiagnostics::default(),
            };
        };
        // Inputs, Outputs, and loop markers are region boundary, not body:
        // they feed or drain repeated windows without belonging to them.
        // Markers especially must be excluded — after one roll, per-iteration
        // values (e.g. layer weights) arrive through LoopInput markers whose
        // stream ids differ, and letting them into windows breaks the hash
        // match for otherwise-identical bodies nested inside the roll.
        let topo: Vec<NodeIndex> = full_topo
            .into_iter()
            .filter(|n| {
                self.try_get_op::<crate::hlir::Input>(*n).is_none()
                    && self.try_get_op::<crate::hlir::Output>(*n).is_none()
                    && !self.is_rolled_loop_marker(*n)
            })
            .collect();
        if topo.len() < 2 {
            return RollingSearchReport {
                candidate: None,
                diagnostics: RollingSearchDiagnostics::default(),
            };
        }
        let uses = build_uses(&self.graph);
        // A roll must not straddle a rolled-region boundary: occurrences in
        // different scopes would produce overlapping (not nested) regions.
        // Markers are invisible to scan windows, so this scope check is the
        // fence that keeps repetition discovery from crossing into or out of
        // an existing rolled body.
        let node_scope = self.region_scope_map();
        let scope_uniform = |occs: &[RollingOccurrence]| {
            let mut nodes = occs.iter().flat_map(|occ| occ.nodes.iter());
            let first = nodes.next().map(|n| node_scope.get(n).copied());
            match first {
                None => true,
                Some(s0) => nodes.all(|n| node_scope.get(n).copied() == s0),
            }
        };
        let topo_index: FxHashMap<NodeIndex, usize> =
            topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        // Cap the largest probed window. A useful rolling candidate is one
        // repeating unit — a transformer layer (≈3.4k HLIR nodes for a gpt-oss
        // MoE layer; ≈6.7k for a 2-minibatch dual-branch layer). With two
        // structurally-similar branches (default + minibatch prefill share
        // weights) the cheap rolling hash matches MANY large windows spanning
        // both branches; each triggers an O(window) `canonicalize_occurrence`
        // that ultimately fails the signature check. Probing windows all the way
        // to `topo.len()/2` made those dead-end canonicalizes dominate (still
        // minutes even after the externals-scan fix below). Capping the window
        // comfortably above one layer skips them without changing the selected
        // candidate (the per-layer roll has the best savings = window·(reps−1)
        // and lives at a small window). A real body larger than the cap just
        // isn't rolled (correctness preserved). Tunable via LUMINAL_MAX_ROLL_BODY.
        let roll_body_cap = std::env::var("LUMINAL_MAX_ROLL_BODY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(8192);
        let max_window = max_region_size.min(topo.len() / 2).min(roll_body_cap);
        let probe_windows = rolling_probe_window_sizes(max_window);
        let node_hashes: Vec<u64> = topo
            .iter()
            .map(|&node| cheap_rolling_node_hash(&self.graph, node, &self.custom_ops))
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
                let Some((sig, first_boundary, first_outputs)) = canonicalize_occurrence(
                    &self.graph,
                    &first_nodes,
                    &uses,
                    &topo_index,
                    &self.custom_ops,
                ) else {
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
                    let Some((next_sig, boundary_inputs, output_nodes)) = canonicalize_occurrence(
                        &self.graph,
                        &nodes,
                        &uses,
                        &topo_index,
                        &self.custom_ops,
                    ) else {
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
                if state_params.is_empty()
                    || !candidate_is_rollable(&occs, &state_params)
                    || !scope_uniform(&occs)
                {
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
                    (rolling_net_savings(&candidate), candidate.occurrences.len())
                        > (rolling_net_savings(b), b.occurrences.len())
                });
                if replace {
                    best_overall = Some(candidate);
                }
                start = pos.saturating_sub(window).max(start + 1);
            }
        }
        if crate::egglog_utils::log_channel_enabled(false, "ROLLING_LOG")
            && let Some(best) = &best_overall
        {
            // Probe the windows adjacent to the accepted run: if another
            // repetition of the body exists but didn't match, the first
            // divergent node names what broke the periodicity of the linear
            // order (phase misalignment shows up as an immediate mismatch,
            // interleaved shared nodes as a mismatch at their topo slot).
            let window = best.occurrences[0].nodes.len();
            let first_start = topo_index[&best.occurrences[0].nodes[0]];
            let last_start = topo_index[&best.occurrences.last().unwrap().nodes[0]];
            for (label, probe_start) in [
                ("before", first_start.checked_sub(window)),
                ("after", Some(last_start + window)),
            ] {
                let Some(probe_start) = probe_start else {
                    continue;
                };
                if probe_start + window > topo.len() {
                    continue;
                }
                let mismatches = (0..window)
                    .filter(|i| node_hashes[probe_start + i] != node_hashes[first_start + i])
                    .take(4)
                    .map(|i| {
                        format!(
                            "+{i}: {:?} vs body {:?}",
                            self.graph[topo[probe_start + i]],
                            self.graph[topo[first_start + i]]
                        )
                    })
                    .collect::<Vec<_>>();
                let total = (0..window)
                    .filter(|i| node_hashes[probe_start + i] != node_hashes[first_start + i])
                    .count();
                println!(
                    "   Rolled  probe {label} run (start {probe_start}, window {window}): {total} hash mismatches{}{}",
                    if mismatches.is_empty() {
                        ""
                    } else {
                        "; first: "
                    },
                    mismatches.join(" | ")
                );
            }
        }
        let mut grown_best = best_overall.take().map(|best| {
            // `best` is already rollable (gated above). Growing only ever
            // extends occurrences, which could re-introduce a cross-occurrence
            // dependency; if it does, keep the validated (smaller) seed.
            let seed = best.clone();
            let grown = grow_rolling_candidate(
                &self.graph,
                &uses,
                &topo_index,
                best,
                &discovered_runs,
                &self.custom_ops,
            );
            if candidate_is_rollable(&grown.occurrences, &grown.state_param_indices)
                && scope_uniform(&grown.occurrences)
            {
                grown
            } else {
                seed
            }
        });
        for run in &discovered_runs {
            let state_param_indices = collect_state_params(&run.occurrences, &uses, &self.graph);
            let seed = RollingCandidate {
                occurrences: run.occurrences.clone(),
                state_param_indices,
                savings: 0,
            };
            let grown = grow_rolling_candidate(
                &self.graph,
                &uses,
                &topo_index,
                seed,
                &discovered_runs,
                &self.custom_ops,
            );
            if grown.state_param_indices.is_empty()
                || !candidate_is_rollable(&grown.occurrences, &grown.state_param_indices)
                || !scope_uniform(&grown.occurrences)
            {
                continue;
            }
            let replace = grown_best.as_ref().is_none_or(|best| {
                (rolling_net_savings(&grown), grown.occurrences.len())
                    > (rolling_net_savings(best), best.occurrences.len())
            });
            if replace {
                grown_best = Some(grown);
            }
        }
        RollingSearchReport {
            candidate: grown_best,
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
    pub fn build_search_space<Rt: Runtime>(&mut self, options: CompileOptions) {
        let mut ops = Rt::Ops::into_vec();
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        self.build_search_space_with_ops::<Rt>(ops, options);
    }

    #[tracing::instrument(skip_all)]
    pub fn build_search_space_exclude_ops<Rt: Runtime, Ex: IntoEgglogOp>(
        &mut self,
        options: CompileOptions,
    ) {
        let exclude_ops = Ex::into_vec()
            .into_iter()
            .map(|e| e.sort().name)
            .collect::<FxHashSet<_>>();
        let mut ops = Rt::Ops::into_vec();
        ops.retain(|o| !exclude_ops.contains(&o.sort().name));
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        self.build_search_space_with_ops::<Rt>(ops, options);
    }

    /// Roll loops, then saturate one e-graph per bucket combination into a
    /// [`SearchSpace`] the runtime searches in [`Runtime::compile`].
    fn build_search_space_with_ops<Rt: Runtime>(
        &mut self,
        ops: Vec<Arc<Box<dyn EgglogOp>>>,
        options: CompileOptions,
    ) {
        self.run_auto_loop_rolling_prepass(&options);
        let dim_buckets = options.dim_buckets.clone();
        let late_pass_dyn_map = self.late_pass_dyn_map(&dim_buckets);
        let late_passes = Rt::late_egglog_passes(&ops, &options, &late_pass_dyn_map);
        let extra_egglog = Rt::extra_egglog();

        let (program, root) = hlir_to_egglog(self);
        let buckets = bucket_index_combinations(&dim_buckets)
            .into_iter()
            .map(|bucket_indices| {
                let intervals = self.bucket_intervals(&dim_buckets, &bucket_indices);
                let (contextual_program, use_interval_analysis) =
                    self.egglog_program_with_interval_facts(&program, &intervals);
                let egraph = run_egglog_with_late_passes_interval_analysis_and_log(
                    &contextual_program,
                    &root,
                    &ops,
                    Rt::CLEANUP_HLIR,
                    &late_passes,
                    &extra_egglog,
                    use_interval_analysis,
                    options.egglog_log_enabled(),
                )
                .unwrap();
                BucketSearchSpace {
                    egraph,
                    bucket_indices,
                    intervals,
                }
            })
            .collect();
        let custom_ops = self.custom_ops.iter().map(|op| op.to_llir_op()).collect();
        self.search_space = Some(SearchSpace {
            buckets,
            ops,
            custom_ops,
            dim_buckets,
        });
    }

    /// Graph-wide interval assumptions narrowed to one bucket combination.
    fn bucket_intervals(
        &self,
        dim_buckets: &FxHashMap<Symbol, Vec<DimBucket>>,
        bucket_indices: &DynMap,
    ) -> DynDimIntervals {
        let mut intervals = self.dim_intervals.clone();
        for (&dim, &idx) in bucket_indices {
            let bucket = &dim_buckets[&dim][idx];
            let min = i64::try_from(bucket.min)
                .expect("DimBucket min must fit into i64 for interval analysis");
            let max = i64::try_from(bucket.max)
                .expect("DimBucket max must fit into i64 for interval analysis");
            let bucket_interval = DimInterval::new(min, max);
            intervals
                .entry(dim)
                .and_modify(|existing| {
                    existing.min = existing.min.max(bucket_interval.min);
                    existing.max = existing.max.min(bucket_interval.max);
                    assert!(
                        existing.min <= existing.max,
                        "Bucket interval for dim '{dim}' does not overlap graph interval"
                    );
                })
                .or_insert(bucket_interval);
        }
        intervals
    }

    fn egglog_program_with_interval_facts(
        &self,
        program: &str,
        intervals: &DynDimIntervals,
    ) -> (String, bool) {
        let facts = crate::egglog_utils::base::interval_facts_egglog(intervals, []);
        if facts.is_empty() {
            (program.to_string(), false)
        } else {
            (format!("{facts}\n{program}"), true)
        }
    }

    /// Dyn map handed to backend late passes: bucket maxima override the
    /// graph's values so passes plan for the largest shape.
    fn late_pass_dyn_map(&self, dim_buckets: &FxHashMap<Symbol, Vec<DimBucket>>) -> DynMap {
        let mut dyn_map = self.dyn_map.clone();
        for (&dim, buckets) in dim_buckets {
            if let Some(max) = buckets.iter().map(|bucket| bucket.max).max() {
                dyn_map.insert(dim, max);
            }
        }
        dyn_map
    }
    /// The built search space, if [`Graph::build_search_space`] has run.
    pub fn search_space(&self) -> Option<&SearchSpace> {
        self.search_space.as_ref()
    }

    /// Get a reference to the first e-graph search space (if built)
    pub fn egraph(&self) -> Option<&SerializedEGraph> {
        self.search_space
            .as_ref()
            .and_then(|space| space.buckets.first())
            .map(|bucket| &bucket.egraph)
    }

    /// Get a reference to the available ops (if search space is built)
    pub fn egglog_ops(&self) -> Option<&Vec<Arc<Box<dyn EgglogOp>>>> {
        self.search_space.as_ref().map(|space| &space.ops)
    }

    /// Build the search space and search it with one shared set of options.
    ///
    /// This is the usual compile entry point when runtime inputs such as
    /// weights have already been loaded. Use `build_search_space` and `search`
    /// directly when the two phases need to be separated.
    #[tracing::instrument(skip_all)]
    pub fn compile<R: Runtime>(&mut self, runtime: R, options: CompileOptions) -> R {
        let mut rng = rand::rng();
        self.compile_with_rng(runtime, options, &mut rng)
    }

    #[tracing::instrument(skip_all)]
    pub fn compile_with_rng<R: Runtime, G: rand::Rng>(
        &mut self,
        runtime: R,
        options: CompileOptions,
        rng: &mut G,
    ) -> R {
        self.build_search_space::<R>(options.clone());
        let runtime = self.search_with_rng(runtime, options, rng);
        // Legality-by-construction burn-down: any post-extraction mask that
        // fired during this compile is a contract violation to fix, not a
        // normal event — always report it.
        if let Some(report) = crate::mask_events::report() {
            println!("{report}");
        }
        runtime
    }

    #[tracing::instrument(skip_all)]
    pub fn search<R: Runtime>(&mut self, runtime: R, options: CompileOptions) -> R {
        let mut rng = rand::rng();
        self.search_with_rng(runtime, options, &mut rng)
    }

    /// Hand the built search space to the runtime, which searches it by
    /// whatever strategy it implements and loads the programs it selects.
    #[tracing::instrument(skip_all)]
    pub fn search_with_rng<R: Runtime, G: rand::Rng>(
        &mut self,
        mut runtime: R,
        options: CompileOptions,
        rng: &mut G,
    ) -> R {
        for (&dim, &value) in &options.search_dims {
            self.set_dim(dim, value);
        }
        let space = self
            .search_space
            .as_ref()
            .expect("build_search_space must run before search");
        assert!(
            options.dim_buckets.is_empty() || options.dim_buckets == space.dim_buckets,
            "dim buckets must be configured in CompileOptions before build_search_space; search cannot change buckets after build",
        );
        runtime.compile(space, &self.dyn_map, &options, rng);
        self.selected_schedule = runtime.selected_schedule();
        runtime
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

fn stable_toposort_by_node_index(graph: &HLIRGraph) -> Option<Vec<NodeIndex>> {
    let mut indegree: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    for n in graph.node_indices() {
        indegree.insert(n, graph.edges_directed(n, Direction::Incoming).count());
    }

    let mut ready = std::collections::BTreeSet::new();
    for (&node, &degree) in &indegree {
        if degree == 0 {
            ready.insert(node);
        }
    }

    let mut ordered = Vec::with_capacity(graph.node_count());
    while let Some(node) = ready.pop_first() {
        ordered.push(node);
        for edge in graph.edges_directed(node, Direction::Outgoing) {
            let target = edge.target();
            let degree = indegree
                .get_mut(&target)
                .expect("toposort target must exist in indegree map");
            *degree -= 1;
            if *degree == 0 {
                ready.insert(target);
            }
        }
    }

    (ordered.len() == graph.node_count()).then_some(ordered)
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

fn cheap_rolling_node_hash(
    graph: &HLIRGraph,
    node: NodeIndex,
    custom_ops: &[Box<dyn CustomOp>],
) -> u64 {
    let op = rolling_op_signature(graph, node, custom_ops);
    let mut hash: u64 = 1469598103934665603;
    for byte in op.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1099511628211);
    }
    // Only in-degree (op arity) may enter the hash. Out-degree counts external
    // consumers, and boundary nodes legitimately differ in fan-out between
    // occurrences (an interior repetition feeds the next one, the last feeds
    // the epilogue) — the canonical signature models those consumers as loop
    // outputs, so a gate stricter than the matcher would reject valid trips.
    let in_degree = graph.neighbors_directed(node, Direction::Incoming).count() as u64;
    hash ^= in_degree.wrapping_mul(0x9e3779b185ebca87);
    hash
}

thread_local! {
    // Memoized per-node rolling signatures. `rolling_op_signature_uncached`
    // Debug-formats the op (allocating + recursing over its shape/stride
    // metadata), and the rolling search calls it for the SAME node across many
    // overlapping windows — O(window) per `canonicalize_occurrence`, summed over
    // every (window, start) hash-match. On a 36-layer dual-branch graph that
    // re-formatting dominated the prepass (16+ min). The signature is a pure
    // function of (node, custom_ops) while the graph + custom_ops are read-only
    // (the search phase never mutates them), so memoize it keyed by node. Cleared
    // at the start of each `best_rolling_candidate` so stale NodeIndex→signature
    // entries can never leak across a graph mutation.
    static ROLLING_SIG_CACHE: std::cell::RefCell<FxHashMap<NodeIndex, String>> =
        std::cell::RefCell::new(FxHashMap::default());
}

fn clear_rolling_sig_cache() {
    ROLLING_SIG_CACHE.with(|c| c.borrow_mut().clear());
}

fn rolling_op_signature(
    graph: &HLIRGraph,
    node: NodeIndex,
    custom_ops: &[Box<dyn CustomOp>],
) -> String {
    ROLLING_SIG_CACHE.with(|c| {
        if let Some(sig) = c.borrow().get(&node) {
            return sig.clone();
        }
        let sig = rolling_op_signature_uncached(graph, node, custom_ops);
        c.borrow_mut().insert(node, sig.clone());
        sig
    })
}

fn rolling_op_signature_uncached(
    graph: &HLIRGraph,
    node: NodeIndex,
    custom_ops: &[Box<dyn CustomOp>],
) -> String {
    if graph[node].as_any().is::<crate::hlir::Output>() {
        return "Output".to_string();
    }
    if let Some(kind) = graph[node].as_any().downcast_ref::<CustomOpKind>() {
        // The `id` is a global custom_ops index and differs for every call
        // (e.g. one rope per layer), which would make structurally identical
        // layer bodies hash differently and defeat loop rolling. Hash the
        // referenced op's content instead: identical custom ops (same kernel
        // parameters) compare equal across layers, distinct ones stay
        // distinct.
        return format!("CustomOp({:?}, {:?})", custom_ops[kind.id], kind.dtype);
    }

    // Use Debug, NOT Display — Display for many HLIR ops drops their
    // shape/stride metadata (e.g. `Display for Mul` emits just "Mul"), so
    // two structurally-different ops with the same kind would hash equal
    // and get falsely grouped as a repeating pattern. Debug captures those
    // fields. Output is the exception: its `node` field is only the source
    // slot for runtime storage, so it must not participate in rolling
    // identity.
    format!("{:?}", graph[node])
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
    custom_ops: &[Box<dyn CustomOp>],
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
        let op = rolling_op_signature(graph, node, custom_ops);
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
                // A node is an Outgoing-external (graph sink) iff it has no
                // outgoing edges. The previous `graph.externals(Outgoing).any(..)`
                // re-scanned EVERY node in the graph for each node in the window,
                // making this O(window × graph) per canonicalize — the dominant
                // cost that stalled the dual-branch rolling prepass. Check the
                // node's own out-edges instead (O(out-degree)).
                || graph
                    .edges_directed(*n, Direction::Outgoing)
                    .next()
                    .is_none()
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

/// A rolling candidate is only valid to collapse into a single loop body if
/// every boundary-input parameter is either:
///   - a loop-carried state param (its value is produced by the IMMEDIATELY
///     preceding occurrence — already enforced by `collect_state_params`), or
///   - fed from OUTSIDE the candidate's occurrences (an external producer, e.g.
///     a per-iteration weight tensor — these may legitimately differ across
///     occurrences; the loop indexes them by iteration).
///
/// If a NON-state boundary input is produced by ANOTHER occurrence in the
/// candidate, that's a cross-occurrence dependency that is not the adjacent
/// loop-carry (e.g. occurrence `i` consuming occurrence `i-2`'s output, as
/// happens when minibatch-pipelined prefill is rolled at the per-minibatch
/// granularity: stream-0 of layer L+1 depends on stream-0 of layer L, skipping
/// stream-1). Collapsing the duplicate bodies folds that `i ← i-2` edge back
/// onto the single rolled body, creating a directed cycle that makes the egglog
/// Kahn toposort drop nodes and panic. Rejecting such candidates lets the search
/// fall back to a coarser, valid window (e.g. rolling whole layers — both
/// minibatch streams together — where the only cross-occurrence edges are the
/// adjacent layer-to-layer state).
fn candidate_is_rollable(occurrences: &[RollingOccurrence], state_params: &[usize]) -> bool {
    if occurrences.len() < 2 {
        return false;
    }
    let state_set: FxHashSet<usize> = state_params.iter().copied().collect();
    let all_occ_nodes: FxHashSet<NodeIndex> = occurrences
        .iter()
        .flat_map(|o| o.nodes.iter().copied())
        .collect();
    let param_count = occurrences[0].boundary_inputs.len();
    for p in 0..param_count {
        if state_set.contains(&p) {
            continue;
        }
        for occ in occurrences {
            if let Some(&src) = occ.boundary_inputs.get(p) {
                if all_occ_nodes.contains(&src) {
                    return false;
                }
            }
        }
    }
    true
}

/// Net node-count change of rolling a candidate: duplicate body nodes
/// deleted minus marker ops created (mirroring `insert_loop_region_ops`:
/// LoopStart + LoopEnd per state slot, one LoopInput per iteration-varying
/// non-state boundary stream, and one LoopOutput plus a per-iteration Select
/// for each non-state output stream). Rolling is only worth doing — and the
/// rolling fixpoint only terminates — when this is positive.
fn rolling_net_savings(candidate: &RollingCandidate) -> i64 {
    let occs = &candidate.occurrences;
    let n_iters = occs.len();
    let states: FxHashSet<usize> = candidate.state_param_indices.iter().copied().collect();
    let deleted = occs[0].nodes.len() * (n_iters - 1);
    let varying_inputs = (0..occs[0].boundary_inputs.len())
        .filter(|p| !states.contains(p))
        .filter(|&p| {
            !occs
                .windows(2)
                .all(|w| w[0].boundary_inputs[p] == w[1].boundary_inputs[p])
        })
        .count();
    let output_streams = occs[0].output_nodes.len().saturating_sub(states.len());
    let markers = 2 * states.len() + varying_inputs + output_streams * (n_iters + 1);
    deleted as i64 - markers as i64
}

fn grow_rolling_candidate(
    graph: &HLIRGraph,
    uses: &FxHashMap<NodeIndex, Vec<(NodeIndex, usize)>>,
    topo_index: &FxHashMap<NodeIndex, usize>,
    mut candidate: RollingCandidate,
    discovered_runs: &[RollingRun],
    custom_ops: &[Box<dyn CustomOp>],
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
                        canonicalize_occurrence(graph, &nodes, uses, topo_index, custom_ops)
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
                let first_sig = canonicalize_occurrence(
                    graph,
                    &merged_occs[0].nodes,
                    uses,
                    topo_index,
                    custom_ops,
                )
                .map(|(sig, _, _)| sig);
                let Some(first_sig) = first_sig else { continue };
                if merged_occs.iter().skip(1).any(|occ| {
                    canonicalize_occurrence(graph, &occ.nodes, uses, topo_index, custom_ops)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egglog_utils::hash_egglog_normalized;
    use crate::hlir::{Input, LoopEnd, LoopInput, LoopStart, Output, ReferenceOp, Sin};
    use crate::search::unroll::materialize_unrolled_llir;

    // A rolling candidate is only collapsible if every non-state boundary input
    // is fed from OUTSIDE the candidate's occurrences. A non-state input produced
    // by another occurrence (e.g. occ `i` consuming occ `i-2`'s output) is a
    // non-adjacent cross-occurrence dependency that, once the bodies are folded
    // into one, becomes a directed cycle — which makes the egglog Kahn toposort
    // drop nodes and panic. `candidate_is_rollable` must reject those.
    #[test]
    fn candidate_is_rollable_rejects_cross_occurrence_dep() {
        let n = NodeIndex::new;
        let occ = |body: usize, input: usize| RollingOccurrence {
            nodes: vec![n(body)],
            boundary_inputs: vec![n(input)],
            output_nodes: vec![n(body)],
        };

        // Both occurrences' inputs come from outside the candidate (100, 101) →
        // rollable.
        let rollable = vec![occ(0, 100), occ(1, 101)];
        assert!(candidate_is_rollable(&rollable, &[]));

        // Occurrence 1's input is node 0, which lives INSIDE occurrence 0, and
        // param 0 is not a state param → reject.
        let cyclic = vec![occ(0, 100), occ(1, 0)];
        assert!(!candidate_is_rollable(&cyclic, &[]));

        // Same shape, but param 0 is declared a loop-carried state param → the
        // adjacent loop-carry is allowed.
        assert!(candidate_is_rollable(&cyclic, &[0]));

        // Fewer than two occurrences is never rollable.
        assert!(!candidate_is_rollable(&[occ(0, 100)], &[]));
    }
    use crate::tests::{assert_close, random_vec};

    #[test]
    fn materialize_many_disjoint_loops_without_a_global_cartesian_product() {
        const N_LOOPS: usize = 64;
        let mut rolled = LLIRGraph::default();
        for loop_id in 0..N_LOOPS {
            let input = rolled.add_node(LLIROp::new::<Input>(Box::new(Input {
                node: loop_id,
                label: String::new(),
                dtype: DType::F32,
            })));
            let start = rolled.add_node(LLIROp::new::<LoopStart>(Box::new(LoopStart {
                loop_id,
                slot_idx: 0,
                iters: Expression::from(2),
                dtype: DType::F32,
            })));
            let body = rolled.add_node(LLIROp::new::<dyn ReferenceOp>(
                Box::new(Sin::default()) as Box<dyn ReferenceOp>
            ));
            let end = rolled.add_node(LLIROp::new::<LoopEnd>(Box::new(LoopEnd {
                loop_id,
                slot_idx: 0,
                dtype: DType::F32,
            })));
            let output = rolled.add_node(LLIROp::new::<Output>(Box::new(Output {
                node: loop_id,
                persist_only: false,
            })));
            rolled.add_edge(input, start, ());
            rolled.add_edge(start, body, ());
            rolled.add_edge(body, end, ());
            rolled.add_edge(end, output, ());
        }

        let materialized = materialize_unrolled_llir(&rolled)
            .expect("independent loop regions must not multiply one another's contexts");

        // Per region: one input + two body instances + one output, connected
        // as a three-edge chain. A global product would overflow at 2^64.
        assert_eq!(materialized.node_count(), N_LOOPS * 4);
        assert_eq!(materialized.edge_count(), N_LOOPS * 3);
        assert!(
            materialized
                .node_weights()
                .all(|op| { op.to_op::<LoopStart>().is_none() && op.to_op::<LoopEnd>().is_none() })
        );
    }

    #[test]
    fn test_hash_egglog_normalized_same_structure() {
        // Two egglog texts differing only in Input node indices and labels
        let text_a = r#"(let t0 (Input 42 "boundary" (F32)))
(let t1 (Input 100 "layers.0.wq.weight" (F32)))
(let t2 (Add (ECons 128 (ECons 4096 (ENil))) t1 (ECons 1 (ECons 128 (ENil))) t0 (ECons 1 (ECons 1 (ENil))) (ECons 1 (ECons 128 (ENil)))))
(let t3 (Output t2 42 false))
"#;
        let text_b = r#"(let t0 (Input 84 "boundary" (F32)))
(let t1 (Input 200 "layers.1.wq.weight" (F32)))
(let t2 (Add (ECons 128 (ECons 4096 (ENil))) t1 (ECons 1 (ECons 128 (ENil))) t0 (ECons 1 (ECons 1 (ENil))) (ECons 1 (ECons 128 (ENil)))))
(let t3 (Output t2 84 false))
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
    fn test_hash_egglog_normalized_distinguishes_persist_only_output() {
        let observed = "(let t1 (Output t0 42 false))\n";
        let persist_only = "(let t1 (Output t0 42 true))\n";
        assert_ne!(
            hash_egglog_normalized(observed),
            hash_egglog_normalized(persist_only),
            "persistence and observed-output semantics must not share a cached egraph"
        );
    }

    #[test]
    fn test_hash_egglog_normalized_custom_op_id() {
        // CustomOpKind lines differ only in the integer ID (layer index)
        let text_a = r#"(let t0 (Input 441 "boundary" (F32)))
(let t1 (Op (CustomOpKind 1 (F32)) (ICons t74 (ICons t120 (ICons t28 (INil))))))
(let t2 (Output t1 585 false))
"#;
        let text_b = r#"(let t0 (Input 585 "boundary" (F32)))
(let t1 (Op (CustomOpKind 2 (F32)) (ICons t74 (ICons t120 (ICons t28 (INil))))))
(let t2 (Output t1 729 false))
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
    fn test_rolling_op_signature_custom_op_content() {
        #[derive(Debug)]
        struct TestCustomOp {
            #[allow(dead_code)]
            name: &'static str,
        }
        impl CustomOp for TestCustomOp {
            fn to_llir_op(&self) -> LLIROp {
                unimplemented!()
            }
        }

        // The signature cache is keyed by NodeIndex and only cleared by
        // best_rolling_candidate; drop entries another test on this thread
        // may have left behind for the same indices.
        clear_rolling_sig_cache();

        let mut cx = Graph::new();
        cx.custom_ops.push(Box::new(TestCustomOp { name: "rope" }));
        cx.custom_ops.push(Box::new(TestCustomOp { name: "rope" }));
        cx.custom_ops.push(Box::new(TestCustomOp { name: "topk" }));
        let ids: Vec<_> = (0..3)
            .map(|id| {
                cx.add_op(
                    CustomOpKind {
                        id,
                        dtype: DType::F32,
                    },
                    &[],
                )
            })
            .collect();

        assert_eq!(
            rolling_op_signature(&cx.graph, ids[0], &cx.custom_ops),
            rolling_op_signature(&cx.graph, ids[1], &cx.custom_ops),
            "separate instances of the same custom op should sign the same"
        );
        assert_ne!(
            rolling_op_signature(&cx.graph, ids[0], &cx.custom_ops),
            rolling_op_signature(&cx.graph, ids[2], &cx.custom_ops),
            "different custom ops should sign differently"
        );
    }

    #[test]
    fn test_auto_roll_loops_prepass_creates_regions_for_chain_recurrence() {
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let out = x.exp2().sin().exp2().sin().exp2().sin().output();

        let inserted = cx.auto_roll_loops_prepass_with_log(true);
        assert!(
            inserted >= 2,
            "expected at least two loop boundaries for 3 repeated bodies, got {inserted}"
        );

        let vals = random_vec(8);
        let mut rt = ReferenceRuntime::default();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
        rt.set_data(x.id, vals.clone());
        rt.execute(&cx.dyn_map);

        let expected = vals
            .into_iter()
            .map(|v| v.exp2().sin().exp2().sin().exp2().sin())
            .collect::<Vec<f32>>();
        assert_close(rt.get_f32(out.id), &expected);
    }

    #[test]
    fn test_auto_roll_loops_prepass_rolls_recurrence_with_interleaved_outputs() {
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let mut y = x;
        for _ in 0..10 {
            y.exp2().output();
            y = y.sin();
        }
        let y = y.output();

        let before = cx.graph.node_count();
        let inserted = cx.auto_roll_loops_prepass_with_log(true);
        let after = cx.graph.node_count();
        assert!(
            inserted >= 2,
            "expected loop markers for recurrence split by Output nodes, got {inserted}"
        );
        assert!(
            after < before,
            "expected rolling to reduce nodes for recurrence split by Output nodes ({before} -> {after})"
        );

        let vals = random_vec(8);
        let mut rt = ReferenceRuntime::default();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
        rt.set_data(x.id, vals.clone());
        rt.execute(&cx.dyn_map);

        let expected = vals
            .into_iter()
            .map(|mut v| {
                for _ in 0..10 {
                    v = v.sin();
                }
                v
            })
            .collect::<Vec<f32>>();
        assert_close(rt.get_f32(y.id), &expected);
    }

    #[test]
    fn test_auto_roll_loops_prepass_skips_non_recurrent_branches() {
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let y = cx.tensor(8);
        let _out = (x.exp().sin() + y.exp().sin()).output();

        let inserted = cx.auto_roll_loops_prepass_with_log(true);
        assert_eq!(inserted, 0, "branch-only reuse should not roll into loops");
    }

    #[test]
    fn test_auto_roll_loops_prepass_runs_when_logging_is_disabled() {
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let out = x.exp2().sin().exp2().sin().exp2().sin().output();

        let before = cx.graph.node_count();
        let inserted = cx.auto_roll_loops_prepass();
        let after = cx.graph.node_count();

        assert!(
            inserted >= 2,
            "expected loop rolling to run without ROLLING_LOG, got {inserted}"
        );
        assert!(
            after < before,
            "expected loop rolling to reduce nodes ({before} -> {after})"
        );
        assert!(
            cx.graph
                .neighbors_directed(out.id, Direction::Outgoing)
                .next()
                .is_none(),
            "output should remain a graph root"
        );
    }

    #[test]
    fn test_nested_loop_rolling_rolls_periodic_layer_pattern() {
        // Periodic pattern like alternating-attention transformers: blocks
        // of 3 identical "layers" (sin) closed by a distinct one (exp2),
        // repeated 4 times. The first pass rolls the 4 blocks; the second
        // rolls the 3 identical layers inside the surviving body.
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let mut y = x;
        for _ in 0..4 {
            for _ in 0..3 {
                y = y.sin();
            }
            y = y.exp2();
        }
        let out = y.output();

        let first = cx.auto_roll_loops_prepass_with_log(true);
        assert!(first > 0, "expected the block pattern to roll");
        let second = cx.auto_roll_loops_prepass_with_log(true);
        assert!(
            second > 0,
            "expected the repeated layers inside the rolled body to roll"
        );

        let loop_ids: FxHashSet<usize> = cx
            .graph
            .node_indices()
            .filter_map(|n| {
                cx.try_get_op::<crate::hlir::LoopStart>(n)
                    .map(|ls| ls.loop_id)
            })
            .collect();
        assert_eq!(loop_ids.len(), 2, "expected two distinct loop regions");

        let vals = random_vec(8);
        let mut rt = ReferenceRuntime::default();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
        rt.set_data(x.id, vals.clone());
        rt.execute(&cx.dyn_map);

        let expected = vals
            .into_iter()
            .map(|mut v| {
                for _ in 0..4 {
                    for _ in 0..3 {
                        v = v.sin();
                    }
                    v = v.exp2();
                }
                v
            })
            .collect::<Vec<f32>>();
        assert_close(rt.get_f32(out.id), &expected);
    }

    #[test]
    fn test_nested_loop_rolling_chained_sibling_inner_regions() {
        // Mirror of gemma's rolled topology: an outer periodic block whose
        // body contains multiple distinct repeated runs, chained through
        // non-repeating ops. The runs roll into sibling regions nested
        // inside the outer region; unroll must find each sibling innermost.
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let mut y = x;
        for _ in 0..4 {
            for _ in 0..3 {
                y = y.sin();
            }
            y = y.exp2();
            for _ in 0..4 {
                y = y.sin();
            }
            y = y.reciprocal();
        }
        let out = y.output();

        let mut passes = 0;
        while cx.auto_roll_loops_prepass_with_log(true) > 0 {
            passes += 1;
        }
        assert!(
            passes >= 3,
            "expected outer + two sibling inner rolls, got {passes}"
        );

        let vals = random_vec(8);
        let mut rt = ReferenceRuntime::default();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
        rt.set_data(x.id, vals.clone());
        rt.execute(&cx.dyn_map);

        let expected = vals
            .into_iter()
            .map(|mut v| {
                for _ in 0..4 {
                    for _ in 0..3 {
                        v = v.sin();
                    }
                    v = v.exp2();
                    for _ in 0..4 {
                        v = v.sin();
                    }
                    v = v.recip();
                }
                v
            })
            .collect::<Vec<f32>>();
        assert_close(rt.get_f32(out.id), &expected);
    }

    #[test]
    fn test_nested_loop_rolling_with_varying_weights() {
        // Gemma-shaped: an outer periodic block of 3 identical weighted
        // layers plus a distinct closer, repeated 4 times, with a DISTINCT
        // weight tensor per layer. The inner region's per-iteration inputs
        // are then varying streams fed by the outer region's own LoopInput
        // markers — the exact structure of per-layer weights in a rolled
        // transformer.
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let weights: Vec<GraphTensor> = (0..12).map(|_| cx.tensor(8)).collect();
        let mut y = x;
        for block in 0..4 {
            for layer in 0..3 {
                y = (y * weights[block * 3 + layer]).sin();
            }
            y = y.exp2();
        }
        let out = y.output();

        let mut passes = 0;
        while cx.auto_roll_loops_prepass_with_log(true) > 0 {
            passes += 1;
        }
        assert!(passes >= 2, "expected nested rolls, got {passes}");

        let xv = random_vec(8);
        let wvs: Vec<Vec<f32>> = (0..12).map(|_| random_vec(8)).collect();
        let mut rt = ReferenceRuntime::default();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
        rt.set_data(x.id, xv.clone());
        for (w, wv) in weights.iter().zip(&wvs) {
            rt.set_data(w.id, wv.clone());
        }
        rt.execute(&cx.dyn_map);

        let expected: Vec<f32> = (0..8)
            .map(|j| {
                let mut v = xv[j];
                for block in 0..4 {
                    for layer in 0..3 {
                        v = (v * wvs[block * 3 + layer][j]).sin();
                    }
                    v = v.exp2();
                }
                v
            })
            .collect();
        assert_close(rt.get_f32(out.id), &expected);
    }

    #[test]
    fn loop_rolling_stamps_concrete_varying_stream_dtypes() {
        // A transformer-style recurrence carries F32 activations while every
        // repeated layer receives a distinct BF16 weight. The weight casts are
        // part of the repeated body, so the rolled boundary itself must retain
        // BF16 rather than a placeholder chosen independently of its sources.
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let weights: Vec<GraphTensor> =
            (0..4).map(|_| cx.tensor(8).as_dtype(DType::Bf16)).collect();
        let mut y = x;
        for weight in weights {
            y = (y * weight.cast(DType::F32)).sin();
        }
        let _ = y.output();

        assert!(
            cx.auto_roll_loops_prepass_with_log(true) > 0,
            "expected the repeated weighted recurrence to roll"
        );

        let loop_inputs: Vec<_> = cx
            .graph
            .node_indices()
            .filter_map(|node| cx.try_get_op::<LoopInput>(node))
            .collect();
        assert!(!loop_inputs.is_empty(), "expected a varying weight stream");
        assert!(
            loop_inputs.iter().any(|input| input.dtype == DType::Bf16),
            "expected a concrete BF16 LoopInput, got {loop_inputs:?}"
        );
        assert!(
            cx.graph.node_indices().all(|node| {
                cx.try_get_op::<LoopStart>(node)
                    .is_none_or(|start| start.dtype == DType::F32)
            }),
            "the carried F32 activation must remain concretely F32"
        );

        // The concrete field remains the source of truth through egglog
        // construction; this used to turn every marker field into F32.
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    }

    #[test]
    fn loop_rolling_preserves_integer_gather_stream_dtypes() {
        // Regression for mixed-type repeated regions: Gather consumes Int
        // indexes but produces the dtype of its F32 data input. Both facts
        // must survive rolling without one eclass overwriting the other.
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let indexes: Vec<GraphTensor> = (0..4)
            .map(|layer| {
                cx.named_tensor(format!("indexes.{layer}"), 8)
                    .as_dtype(DType::Int)
            })
            .collect();
        let mut y = x;
        for index in indexes {
            y = y.gather(index).sin();
        }
        let _ = y.output();

        assert!(
            cx.auto_roll_loops_prepass_with_log(true) > 0,
            "expected the repeated gather recurrence to roll"
        );

        let loop_inputs: Vec<_> = cx
            .graph
            .node_indices()
            .filter_map(|node| cx.try_get_op::<LoopInput>(node))
            .collect();
        assert!(
            loop_inputs.iter().any(|input| input.dtype == DType::Int),
            "expected a concrete Int LoopInput, got {loop_inputs:?}"
        );
        assert!(
            cx.graph.node_indices().all(|node| {
                cx.try_get_op::<LoopStart>(node)
                    .is_none_or(|start| start.dtype == DType::F32)
            }),
            "Gather must preserve the carried F32 activation dtype"
        );

        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    }

    #[test]
    fn test_nested_loop_rolling_per_layer_outputs_not_permuted() {
        // Cache-analog regression test: every layer persists a side output
        // (like per-layer KV caches). Nested rolling + unroll must route each
        // side output to ITS OWN layer's value — a permutation across layers
        // corrupts state promotion even when the final output is correct.
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let weights: Vec<GraphTensor> = (0..24).map(|_| cx.tensor(8)).collect();
        let mut y = x;
        let mut side = Vec::new();
        for block in 0..4 {
            for layer in 0..5 {
                y = (y * weights[block * 6 + layer]).sin();
                side.push((y * 2.0_f32).output());
            }
            y = (y * weights[block * 6 + 5]).exp2();
            side.push((y * 2.0_f32).output());
        }
        let out = y.output();

        let mut passes = 0;
        while cx.auto_roll_loops_prepass_with_log(true) > 0 {
            passes += 1;
        }
        assert!(passes >= 2, "expected nested rolls, got {passes}");

        let xv = random_vec(8);
        let wvs: Vec<Vec<f32>> = (0..24).map(|_| random_vec(8)).collect();
        let mut rt = ReferenceRuntime::default();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
        rt.set_data(x.id, xv.clone());
        for (w, wv) in weights.iter().zip(&wvs) {
            rt.set_data(w.id, wv.clone());
        }
        rt.execute(&cx.dyn_map);

        let mut refs: Vec<Vec<f32>> = Vec::new();
        let mut v: Vec<f32> = xv.clone();
        for block in 0..4 {
            for layer in 0..5 {
                v = v
                    .iter()
                    .zip(&wvs[block * 6 + layer])
                    .map(|(a, b)| (a * b).sin())
                    .collect();
                refs.push(v.iter().map(|a| a * 2.0).collect());
            }
            v = v
                .iter()
                .zip(&wvs[block * 6 + 5])
                .map(|(a, b)| (a * b).exp2())
                .collect();
            refs.push(v.iter().map(|a| a * 2.0).collect());
        }
        let mut permutation = Vec::new();
        for (idx, t) in side.iter().enumerate() {
            let got = rt.get_f32(t.id);
            let matches: Vec<usize> = refs
                .iter()
                .enumerate()
                .filter(|(_, r)| got.iter().zip(*r).all(|(a, b)| (a - b).abs() < 1e-5))
                .map(|(j, _)| j)
                .collect();
            permutation.push((idx, matches));
        }
        let bad: Vec<_> = permutation.iter().filter(|(i, m)| !m.contains(i)).collect();
        assert!(
            bad.is_empty(),
            "side outputs carry other layers' values: {permutation:?}"
        );
        let final_expected: Vec<f32> = refs.last().unwrap().iter().map(|a| a / 2.0).collect();
        assert!(
            rt.get_f32(out.id)
                .iter()
                .zip(&final_expected)
                .all(|(a, b)| (a - b).abs() < 1e-5),
            "final output mismatch"
        );
    }

    #[test]
    fn test_nested_loop_rolling_handles_disjoint_regions() {
        // Two independent recurrence chains roll into two disjoint regions
        // across successive passes; unroll must handle both.
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let y = cx.tensor(8);
        let a = x.sin().sin().sin().sin();
        let b = y.exp2().exp2().exp2().exp2();
        let out = (a + b).output();

        let mut passes = 0;
        while cx.auto_roll_loops_prepass_with_log(true) > 0 {
            passes += 1;
        }
        assert!(
            passes >= 2,
            "expected both chains to roll, got {passes} passes"
        );

        let xv = random_vec(8);
        let yv = random_vec(8);
        let mut rt = ReferenceRuntime::default();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
        rt.set_data(x.id, xv.clone());
        rt.set_data(y.id, yv.clone());
        rt.execute(&cx.dyn_map);

        let expected = xv
            .into_iter()
            .zip(yv)
            .map(|(mut a, mut b)| {
                for _ in 0..4 {
                    a = a.sin();
                    b = b.exp2();
                }
                a + b
            })
            .collect::<Vec<f32>>();
        assert_close(rt.get_f32(out.id), &expected);
    }
}
