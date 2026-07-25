use crate::egglog_utils::{
    EGraphChoiceSet, count_choice_sets_up_to, egglog_to_llir, extract_reachable_generation,
    hash_choice_set, hlir_to_egglog, log_channel_enabled, random_initial_choice,
    run_egglog_with_late_passes_interval_analysis_and_log,
};
use crate::shape::{DimInterval, DynDimIntervals};
use crate::{
    egglog_utils::SerializedEGraph,
    op::{EgglogOp, IntoEgglogOp, LLIROp},
};
use crate::{hlir::CustomOpKind, op::*, prelude::*};
use colored::Colorize;
use itertools::Itertools;
use petgraph::{
    Direction,
    dot::{Config, Dot},
    stable_graph::StableGraph,
    visit::EdgeRef,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    any::TypeId,
    fmt::{Debug, Write as FmtWrite},
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

#[derive(Debug, Clone, Default)]
struct SearchSpaceContext {
    bucket_indices: FxHashMap<char, usize>,
    representative_dyn_map: FxHashMap<char, usize>,
    intervals: DynDimIntervals,
}

#[derive(Debug, Clone)]
struct SearchProfileBucketContext {
    dim_buckets: FxHashMap<char, Vec<DimBucket>>,
    bucket_indices: FxHashMap<char, usize>,
    representative_dyn_map: FxHashMap<char, usize>,
}

struct Finalist<M> {
    metric: M,
    pre_unroll: Option<LLIRGraph>,
    llir: LLIRGraph,
}

struct LazyFinalists<'a, M> {
    ranked: Vec<(M, EGraphChoiceSet<'a>)>,
    next_ranked: usize,
    finalists: Vec<Finalist<M>>,
    rejections: usize,
    last_rejection: Option<String>,
    stopped_reason: Option<String>,
}

impl<'a, M> LazyFinalists<'a, M> {
    fn new(ranked: Vec<(M, EGraphChoiceSet<'a>)>) -> Self {
        Self {
            ranked,
            next_ranked: 0,
            finalists: Vec::new(),
            rejections: 0,
            last_rejection: None,
            stopped_reason: None,
        }
    }
}

struct BucketFinalistSearch<'a, M> {
    context: SearchProfileBucketContext,
    egraph_index: usize,
    candidates: LazyFinalists<'a, M>,
}

/// A compiled bucket: (bucket_indices, representative_dyn_map, stitched_llir).
pub type BucketLLIR = (FxHashMap<char, usize>, FxHashMap<char, usize>, LLIRGraph);

/// Borrowed view of a compiled bucket used for non-committing aggregate
/// candidate filtering.
#[derive(Clone, Copy)]
pub struct BucketLLIRRef<'a> {
    pub bucket_indices: &'a FxHashMap<char, usize>,
    pub representative_dyn_map: &'a FxHashMap<char, usize>,
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Candidates exceeding it are discarded.
    pub candidate_timeout: Option<std::time::Duration>,
    /// Caps how long profiling runs a single trial; not a rejection criterion.
    pub execution_timeout: Option<std::time::Duration>,
    /// Dynamic dimension values applied after search-space construction and
    /// before search. These values persist in [`Graph::dyn_map`] and provide
    /// the base representative values for unbucketed dimensions. Per-bucket
    /// representatives override them during bucketed search, and
    /// [`CompileOptions::profile_dims`] override them only while profiling.
    pub search_dims: FxHashMap<char, usize>,
    /// Optional profiling dimension overrides.
    pub profile_dims: FxHashMap<char, usize>,
    /// Bucket definitions per dynamic dimension. Dimensions without buckets use
    /// a single implicit bucket.
    pub dim_buckets: FxHashMap<char, Vec<DimBucket>>,
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

    /// Set a dynamic dimension after search-space construction and before
    /// search. This is equivalent to calling [`Graph::set_dim`] between
    /// [`Graph::build_search_space`] and [`Graph::search`], while still using
    /// the unified [`Graph::compile`] API.
    pub fn search_dim(mut self, dim: char, value: usize) -> Self {
        self.search_dims.insert(dim, value);
        self
    }

    /// Override a dynamic dimension value used during search profiling.
    pub fn profile_dim(mut self, dim: char, value: usize) -> Self {
        self.profile_dims.insert(dim, value);
        self
    }

    /// Define buckets for a dynamic dimension.
    ///
    /// Bucketed compilation builds a separate search space and selected LLIR for
    /// each bucket combination. Buckets must not overlap and must cover all
    /// values that will be used at runtime.
    pub fn dim_buckets(mut self, dimension: char, buckets: &[DimBucket]) -> Self {
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

    fn search_log_enabled(&self) -> bool {
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
            // 3 trials proved too noisy to discriminate fast-vs-slow backends
            // now that search keeps equivalent alternatives alive; llama has
            // long used 10 and shows the most stable selections.
            trials: 5,
            keep_best: 1,
            restart_stagnation: 0,
            candidate_timeout: Some(std::time::Duration::from_secs(5)),
            execution_timeout: Some(std::time::Duration::from_secs(1)),
            search_dims: FxHashMap::default(),
            profile_dims: FxHashMap::default(),
            dim_buckets: FxHashMap::default(),
            egglog_log: false,
            rolling_log: false,
            search_log: true,
        }
    }
}

fn validate_dim_buckets(dimension: char, buckets: &[DimBucket]) {
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

fn maybe_dump_selected_llir(label: &str, dyn_map: &FxHashMap<char, usize>, llir: &LLIRGraph) {
    let Ok(dir) = std::env::var("LLIR_DUMP_DIR") else {
        return;
    };

    if let Err(err) = std::fs::create_dir_all(&dir) {
        eprintln!("failed to create LLIR_DUMP_DIR={dir}: {err}");
        return;
    }

    let dims = dyn_map
        .iter()
        .sorted_by_key(|(dim, _)| **dim)
        .map(|(dim, value)| format!("{dim}{value}"))
        .join("_");
    let stem = if dims.is_empty() {
        format!("selected-llir-{label}")
    } else {
        format!("selected-llir-{label}-{dims}")
    };
    let dot_path = format!("{dir}/{stem}.dot");
    let summary_path = format!("{dir}/{stem}.txt");

    let dot = format!("{:?}", Dot::with_config(llir, &[Config::EdgeNoLabel]));
    if let Err(err) = std::fs::write(&dot_path, dot) {
        eprintln!("failed to write {dot_path}: {err}");
    }

    let mut op_counts = std::collections::BTreeMap::<String, usize>::new();
    for node in llir.node_indices() {
        *op_counts.entry(format!("{}", llir[node])).or_default() += 1;
    }

    let mut summary = String::new();
    let _ = writeln!(
        summary,
        "selected LLIR {label}: {} nodes, {} edges",
        llir.node_count(),
        llir.edge_count()
    );
    let _ = writeln!(summary, "dyn_map: {dyn_map:?}");
    let _ = writeln!(summary, "\nop counts:");
    for (op, count) in op_counts {
        let _ = writeln!(summary, "  {count:5} {op}");
    }
    let _ = writeln!(summary, "\nnodes:");
    for node in llir.node_indices().sorted_by_key(|n| n.index()) {
        let inputs = llir
            .edges_directed(node, Direction::Incoming)
            .sorted_by_key(|edge| edge.id())
            .map(|edge| edge.source().index().to_string())
            .join(", ");
        let _ = writeln!(
            summary,
            "  n{} <- [{}] {}",
            node.index(),
            inputs,
            llir[node]
        );
    }

    if let Err(err) = std::fs::write(&summary_path, summary) {
        eprintln!("failed to write {summary_path}: {err}");
    } else {
        println!("   LLIR dump {summary_path}");
    }
}

fn random_choice_generation<'a, G: rand::Rng>(
    egraph: &'a SerializedEGraph,
    generation_size: usize,
    prev_selected: &mut FxHashSet<u64>,
    rng: &mut G,
) -> Vec<crate::egglog_utils::EGraphChoiceSet<'a>> {
    let mut generation = Vec::with_capacity(generation_size);
    let max_attempts = generation_size.saturating_mul(100);
    let mut attempts = 0;

    while generation.len() < generation_size && attempts < max_attempts {
        attempts += 1;
        let genome = random_initial_choice(egraph, rng);
        if prev_selected.insert(hash_choice_set(&genome)) {
            generation.push(genome);
        }
    }

    generation
}

fn panic_initial_filter_limit(filter_fails: usize, last_rejection: Option<&str>) -> ! {
    if let Some(last_rejection) = last_rejection {
        panic!(
            "Failed to find a viable initial genome after {filter_fails} runtime filter failures; last rejection: {last_rejection}"
        );
    }
    panic!("Failed to find a viable initial genome after {filter_fails} runtime filter failures");
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
    /// E-Graph search spaces. Bucketed compilation stores one egraph per
    /// bucket combination; unbucketed compilation stores one egraph.
    egraphs: Vec<SerializedEGraph>,
    egraph_contexts: Vec<SearchSpaceContext>,
    /// Available ops
    pub ops: Option<Vec<Arc<Box<dyn EgglogOp>>>>,
    /// Custom ops
    pub custom_ops: Vec<Box<dyn CustomOp>>,
    /// Bucket definitions used by the currently built search space.
    search_space_dim_buckets: FxHashMap<char, Vec<DimBucket>>,
    /// Optional graph-wide interval assumptions for dynamic dimensions.
    pub dim_intervals: DynDimIntervals,
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

            // Marker dtype fields are placeholders: the e-graph derives each
            // marker's dtype fact from its input via dtype_prop (see hlir.rs).
            // Marker dtype fields are placeholders, not sources of truth:
            // each marker's dtype fact is derived inside the e-graph from
            // its input by the generic dtype_prop propagation rule (see the
            // marker EgglogOp impls in hlir.rs). Nothing downstream reads
            // the field semantically.
            let loop_start = self.graph.add_node(Box::new(LoopStart {
                loop_id,
                slot_idx,
                iters: Expression::from(n_iters as i32),
                dtype: DType::F32,
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
                dtype: DType::F32,
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
                dtype: DType::F32,
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

            let loop_output = self.graph.add_node(Box::new(LoopOutput {
                loop_id,
                stream_id: q,
                dtype: DType::F32,
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
                    dtype: DType::F32,
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

    /// Set a runtime dimension
    pub fn set_dim(&mut self, dimension: char, val: usize) {
        self.dyn_map.insert(dimension, val);
    }

    pub fn set_dim_interval(&mut self, dimension: char, min: i64, max: i64) {
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
    pub fn build_search_space<Rt: Runtime + 'static>(&mut self, options: CompileOptions) {
        self.run_auto_loop_rolling_prepass(&options);
        let mut ops = Rt::Ops::into_vec();
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<ReferenceRuntime>();
        let dim_buckets = options.dim_buckets.clone();
        let late_pass_dyn_map = self.late_pass_dyn_map(&dim_buckets);
        let late_passes = Rt::late_egglog_passes(&ops, &options, &late_pass_dyn_map);
        let extra_egglog = Rt::extra_egglog();

        let (program, root) = hlir_to_egglog(self);
        let contexts = self.search_space_contexts(&dim_buckets);
        self.egraphs = contexts
            .iter()
            .map(|context| {
                let (contextual_program, use_interval_analysis) =
                    self.egglog_program_with_interval_facts(&program, &context.intervals);
                run_egglog_with_late_passes_interval_analysis_and_log(
                    &contextual_program,
                    &root,
                    &ops,
                    cleanup_hlir,
                    &late_passes,
                    &extra_egglog,
                    use_interval_analysis,
                    options.egglog_log_enabled(),
                )
                .unwrap()
            })
            .collect();
        self.egraph_contexts = contexts;
        self.ops = Some(ops);
        self.search_space_dim_buckets = dim_buckets;
    }

    fn search_space_contexts(
        &self,
        dim_buckets: &FxHashMap<char, Vec<DimBucket>>,
    ) -> Vec<SearchSpaceContext> {
        if dim_buckets.is_empty() {
            return vec![SearchSpaceContext {
                bucket_indices: FxHashMap::default(),
                representative_dyn_map: self.dyn_map.clone(),
                intervals: self.dim_intervals.clone(),
            }];
        }

        self.bucket_combinations(dim_buckets)
            .into_iter()
            .map(|(bucket_indices, representative_dyn_map)| {
                let mut intervals = self.dim_intervals.clone();
                for (&dim, &idx) in &bucket_indices {
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
                SearchSpaceContext {
                    bucket_indices,
                    representative_dyn_map,
                    intervals,
                }
            })
            .collect()
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

    #[tracing::instrument(skip_all)]
    pub fn build_search_space_exclude_ops<Rt: Runtime + 'static, Ex: IntoEgglogOp>(
        &mut self,
        options: CompileOptions,
    ) {
        self.run_auto_loop_rolling_prepass(&options);
        let exclude_ops = Ex::into_vec()
            .into_iter()
            .map(|e| e.sort().name)
            .collect::<FxHashSet<_>>();
        let mut ops = Rt::Ops::into_vec();
        ops.retain(|o| !exclude_ops.contains(&o.sort().name));
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<ReferenceRuntime>();
        let dim_buckets = options.dim_buckets.clone();
        let late_pass_dyn_map = self.late_pass_dyn_map(&dim_buckets);
        let late_passes = Rt::late_egglog_passes(&ops, &options, &late_pass_dyn_map);
        let extra_egglog = Rt::extra_egglog();

        let (program, root) = hlir_to_egglog(self);
        let contexts = self.search_space_contexts(&dim_buckets);
        self.egraphs = contexts
            .iter()
            .map(|context| {
                let (contextual_program, use_interval_analysis) =
                    self.egglog_program_with_interval_facts(&program, &context.intervals);
                run_egglog_with_late_passes_interval_analysis_and_log(
                    &contextual_program,
                    &root,
                    &ops,
                    cleanup_hlir,
                    &late_passes,
                    &extra_egglog,
                    use_interval_analysis,
                    options.egglog_log_enabled(),
                )
                .unwrap()
            })
            .collect();
        self.egraph_contexts = contexts;
        self.ops = Some(ops);
        self.search_space_dim_buckets = dim_buckets;
    }

    /// Get a reference to the first e-graph search space (if built)
    pub fn egraph(&self) -> Option<&SerializedEGraph> {
        self.egraphs.first()
    }

    /// Get a reference to the available ops (if search space is built)
    pub fn egglog_ops(&self) -> Option<&Vec<Arc<Box<dyn EgglogOp>>>> {
        self.ops.as_ref()
    }

    /// Build the search space and search it with one shared set of options.
    ///
    /// This is the usual compile entry point when runtime inputs such as
    /// weights have already been loaded. Use `build_search_space` and `search`
    /// directly when the two phases need to be separated.
    #[tracing::instrument(skip_all)]
    pub fn compile<R: Runtime + 'static>(&mut self, runtime: R, options: CompileOptions) -> R {
        let mut rng = rand::rng();
        self.compile_with_rng(runtime, options, &mut rng)
    }

    #[tracing::instrument(skip_all)]
    pub fn compile_with_rng<R: Runtime + 'static, G: rand::Rng>(
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
    pub fn search<R: Runtime + 'static>(&mut self, runtime: R, options: CompileOptions) -> R {
        let mut rng = rand::rng();
        self.search_with_rng(runtime, options, &mut rng)
    }

    #[tracing::instrument(skip_all)]
    pub fn search_with_rng<R: Runtime + 'static, G: rand::Rng>(
        &mut self,
        mut runtime: R,
        options: CompileOptions,
        rng: &mut G,
    ) -> R {
        for (&dim, &value) in &options.search_dims {
            self.set_dim(dim, value);
        }

        assert!(
            options.dim_buckets.is_empty() || options.dim_buckets == self.search_space_dim_buckets,
            "dim buckets must be configured in CompileOptions before build_search_space; search cannot change buckets after build",
        );

        let search_started_at = std::time::Instant::now();
        let search_log = options.search_log_enabled();
        if self.search_space_dim_buckets.is_empty() {
            // No buckets: existing single-search path
            let ranked = self.search_single(
                &mut runtime,
                &options,
                rng,
                &self.dyn_map.clone(),
                None,
                None,
                0,
                search_started_at,
            );
            let mut candidates = LazyFinalists::new(ranked);
            if !self.ensure_finalist(
                &mut runtime,
                &mut candidates,
                0,
                &options,
                &self.dyn_map,
                None,
                0,
                search_started_at,
            ) {
                panic!(
                    "Failed to find a viable final graph: {}",
                    Self::no_finalist_message(&candidates)
                );
            }
            let finalist = candidates.finalists.remove(0);
            Self::dump_selected_finalist(&finalist, &self.dyn_map, None);

            runtime.clear_intermediate_buffers();
            runtime.load_llir(&finalist.llir);
            runtime
        } else {
            // Bucketed search: retain ranked genomes for every bucket, then
            // choose a collectively viable finalist set below.
            let bucket_contexts = self.search_space_contexts(&self.search_space_dim_buckets);
            let n_combos = bucket_contexts.len();
            let mut bucket_searches = Vec::with_capacity(n_combos);
            assert!(
                self.egraphs.len() == n_combos,
                "dim buckets must be configured before build_search_space; search space has {} egraphs but current bucket configuration has {n_combos} combinations",
                self.egraphs.len(),
            );

            for (combo_idx, context) in bucket_contexts.into_iter().enumerate() {
                let bucket_label = self
                    .format_bucket_label(&self.search_space_dim_buckets, &context.bucket_indices);
                if search_log {
                    println!(
                        "   {:>6}  Group {}/{}: {}",
                        "Search".cyan().bold(),
                        combo_idx + 1,
                        n_combos,
                        bucket_label,
                    );
                }

                let profile_context = SearchProfileBucketContext {
                    dim_buckets: self.search_space_dim_buckets.clone(),
                    bucket_indices: context.bucket_indices.clone(),
                    representative_dyn_map: context.representative_dyn_map.clone(),
                };
                let ranked = self.search_single(
                    &mut runtime,
                    &options,
                    rng,
                    &context.representative_dyn_map,
                    Some((combo_idx, n_combos)),
                    Some(profile_context.clone()),
                    combo_idx,
                    search_started_at,
                );
                bucket_searches.push(BucketFinalistSearch {
                    context: profile_context,
                    egraph_index: combo_idx,
                    candidates: LazyFinalists::new(ranked),
                });
            }

            // Materialize only the fastest individually viable graph for each
            // bucket to seed a best-first walk over the Cartesian finalist
            // lattice. Slower full LLIRs are extracted only when an aggregate
            // rejection makes their coordinate reachable.
            for (bucket_idx, bucket) in bucket_searches.iter_mut().enumerate() {
                if !self.ensure_finalist(
                    &mut runtime,
                    &mut bucket.candidates,
                    0,
                    &options,
                    &bucket.context.representative_dyn_map,
                    Some(&bucket.context),
                    bucket.egraph_index,
                    search_started_at,
                ) {
                    let label = self.format_bucket_label(
                        &self.search_space_dim_buckets,
                        &bucket.context.bucket_indices,
                    );
                    panic!(
                        "Failed to find a viable final graph for bucket {bucket_idx} ({label}): {}",
                        Self::no_finalist_message(&bucket.candidates)
                    );
                }
            }

            let initial_indices = vec![0usize; bucket_searches.len()];
            let initial_metrics = bucket_searches
                .iter()
                .map(|bucket| bucket.candidates.finalists[0].metric.clone())
                .collect_vec();
            let mut frontier = vec![(
                R::aggregate_profile_metrics(&initial_metrics),
                initial_indices.clone(),
            )];
            let mut visited = FxHashSet::default();
            visited.insert(initial_indices);
            let mut selected_indices = None;
            let mut aggregate_attempts = 0usize;
            let mut aggregate_rejections = 0usize;
            let mut last_aggregate_rejection = None;
            let mut aggregate_stopped_reason = None;

            while !frontier.is_empty() {
                if aggregate_attempts > 0
                    && search_started_at.elapsed() >= options.search_time_limit
                {
                    aggregate_stopped_reason = Some(
                        "search time limit expired during aggregate bucket finalization"
                            .to_string(),
                    );
                    break;
                }
                let best_pos = (1..frontier.len()).fold(0, |best, candidate| {
                    if frontier[candidate]
                        .0
                        .partial_cmp(&frontier[best].0)
                        .is_some_and(|ordering| ordering == std::cmp::Ordering::Less)
                    {
                        candidate
                    } else {
                        best
                    }
                });
                let (_, indices) = frontier.swap_remove(best_pos);
                aggregate_attempts += 1;

                let bucket_refs = bucket_searches
                    .iter()
                    .zip(&indices)
                    .map(|(bucket, &candidate_idx)| BucketLLIRRef {
                        bucket_indices: &bucket.context.bucket_indices,
                        representative_dyn_map: &bucket.context.representative_dyn_map,
                        llir: &bucket.candidates.finalists[candidate_idx].llir,
                    })
                    .collect_vec();
                runtime.clear_intermediate_buffers();
                let aggregate_started_at = std::time::Instant::now();
                let aggregate_result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        runtime.filter_llir_bucket_set(
                            &self.search_space_dim_buckets,
                            &bucket_refs,
                            &options,
                        )
                    }))
                    .unwrap_or_else(|_| {
                        CandidateFilterResult::reject_with_display(
                            "aggregate bucket candidate filter panicked",
                        )
                    });
                let aggregate_timed_out = options
                    .candidate_timeout
                    .is_some_and(|timeout| aggregate_started_at.elapsed() >= timeout);
                if aggregate_result.accepted && !aggregate_timed_out {
                    if search_log && aggregate_rejections > 0 {
                        println!(
                            "   {:>6}  aggregate fallback: selected per-bucket finalist ranks {:?} after {} rejection(s)",
                            "Search".yellow().bold(),
                            indices,
                            aggregate_rejections,
                        );
                    }
                    selected_indices = Some(indices);
                    break;
                }
                aggregate_rejections += 1;
                crate::mask_events::AGGREGATE_REJECT.record();
                last_aggregate_rejection = if aggregate_timed_out {
                    Some(format!(
                        "candidate timeout expired while filtering aggregate bucket set {aggregate_attempts}"
                    ))
                } else {
                    aggregate_result.display
                };
                if search_log {
                    println!(
                        "   {:>6}  aggregate reject per-bucket finalist ranks {:?}: {}",
                        "Search".yellow().bold(),
                        indices,
                        last_aggregate_rejection.as_deref().unwrap_or("(no reason)"),
                    );
                }

                // Any one-coordinate successor is the next possible slower
                // combination. A visited set prevents duplicate lattice paths.
                for bucket_idx in 0..bucket_searches.len() {
                    let mut successor = indices.clone();
                    successor[bucket_idx] += 1;
                    if !visited.insert(successor.clone()) {
                        continue;
                    }
                    let bucket = &mut bucket_searches[bucket_idx];
                    if !self.ensure_finalist(
                        &mut runtime,
                        &mut bucket.candidates,
                        successor[bucket_idx],
                        &options,
                        &bucket.context.representative_dyn_map,
                        Some(&bucket.context),
                        bucket.egraph_index,
                        search_started_at,
                    ) {
                        if bucket.candidates.stopped_reason.is_some()
                            || bucket
                                .candidates
                                .last_rejection
                                .as_deref()
                                .is_some_and(|reason| reason.contains("timeout"))
                        {
                            let label = self.format_bucket_label(
                                &self.search_space_dim_buckets,
                                &bucket.context.bucket_indices,
                            );
                            aggregate_stopped_reason.get_or_insert_with(|| {
                                format!(
                                    "bucket {bucket_idx} ({label}) fallback stopped: {}",
                                    Self::no_finalist_message(&bucket.candidates)
                                )
                            });
                        }
                        continue;
                    }
                    let metrics = bucket_searches
                        .iter()
                        .zip(&successor)
                        .map(|(bucket, &candidate_idx)| {
                            bucket.candidates.finalists[candidate_idx].metric.clone()
                        })
                        .collect_vec();
                    frontier.push((R::aggregate_profile_metrics(&metrics), successor));
                }
            }

            let Some(selected_indices) = selected_indices else {
                let reason = aggregate_stopped_reason
                    .or(last_aggregate_rejection)
                    .unwrap_or_else(|| "no aggregate candidate combinations remain".to_string());
                panic!(
                    "Failed to find a viable aggregate bucket set after {aggregate_rejections} rejections: {reason}"
                );
            };

            let mut bucket_llirs = Vec::with_capacity(n_combos);
            for (bucket_idx, (mut bucket, candidate_idx)) in bucket_searches
                .into_iter()
                .zip(selected_indices)
                .enumerate()
            {
                let finalist = bucket.candidates.finalists.swap_remove(candidate_idx);
                Self::dump_selected_finalist(
                    &finalist,
                    &bucket.context.representative_dyn_map,
                    Some((bucket_idx, n_combos)),
                );
                bucket_llirs.push((
                    bucket.context.bucket_indices,
                    bucket.context.representative_dyn_map,
                    finalist.llir,
                ));
            }

            runtime.clear_intermediate_buffers();
            runtime.load_llir_buckets(&self.search_space_dim_buckets, &bucket_llirs);
            runtime
        }
    }

    /// Compute cartesian product of all bucket combinations.
    /// Returns Vec of (bucket_indices, representative_dyn_map).
    fn bucket_combinations(
        &self,
        dim_buckets: &FxHashMap<char, Vec<DimBucket>>,
    ) -> Vec<(FxHashMap<char, usize>, FxHashMap<char, usize>)> {
        let mut dims: Vec<(char, &Vec<DimBucket>)> =
            dim_buckets.iter().map(|(c, b)| (*c, b)).collect();
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

    fn late_pass_dyn_map(
        &self,
        dim_buckets: &FxHashMap<char, Vec<DimBucket>>,
    ) -> FxHashMap<char, usize> {
        let mut dyn_map = self.dyn_map.clone();
        for (&dim, buckets) in dim_buckets {
            if let Some(max) = buckets.iter().map(|bucket| bucket.max).max() {
                dyn_map.insert(dim, max);
            }
        }
        dyn_map
    }

    /// Format a human-readable label for a bucket combination.
    fn format_bucket_label(
        &self,
        dim_buckets: &FxHashMap<char, Vec<DimBucket>>,
        bucket_indices: &FxHashMap<char, usize>,
    ) -> String {
        let mut parts: Vec<String> = Vec::new();
        let mut dims: Vec<_> = bucket_indices.iter().collect();
        dims.sort_by_key(|(c, _)| **c);
        for (dim, &idx) in dims {
            let bucket = &dim_buckets[dim][idx];
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

    /// Run the genetic search and return every successfully profiled genome in
    /// metric order. Final extraction and hard filtering are deliberately
    /// deferred so bucketed compilation can choose a viable retained set.
    /// `bucket_progress`: if `Some((current_bucket_idx, total_buckets))` adds a
    /// second "Bucket" progress bar.
    #[allow(clippy::too_many_arguments)]
    fn search_single<'a, R: Runtime + 'static, G: rand::Rng>(
        &'a self,
        runtime: &mut R,
        options: &CompileOptions,
        rng: &mut G,
        dyn_map: &FxHashMap<char, usize>,
        bucket_progress: Option<(usize, usize)>,
        bucket_profile_context: Option<SearchProfileBucketContext>,
        egraph_index: usize,
        search_started_at: std::time::Instant,
    ) -> Vec<(R::ProfileMetric, EGraphChoiceSet<'a>)> {
        let mut profile_dyn_map = dyn_map.clone();
        for (&dim, &value) in &options.profile_dims {
            profile_dyn_map.insert(dim, value);
        }
        let limit = options.limit;
        let ops = self.ops.as_ref().unwrap();
        let egraph = &self.egraphs[egraph_index];
        let search_limit = count_choice_sets_up_to(egraph, limit);
        let start = std::time::Instant::now();
        let search_log = options.search_log_enabled();

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

        let mut prev_selected: FxHashSet<u64> = FxHashSet::default();
        let mut list_cache = FxHashMap::default();
        let mut expr_cache = FxHashMap::default();
        runtime.clear_intermediate_buffers();
        let search_time_limit_reached = || search_started_at.elapsed() >= options.search_time_limit;
        // `profile_start.elapsed()` wraps the whole compile+run, so this is the
        // candidate-timeout (viability) check, not the per-execution one.
        let candidate_timed_out = |elapsed: std::time::Duration| {
            options
                .candidate_timeout
                .is_some_and(|timeout| elapsed >= timeout)
        };

        // Find a viable initial genome. Runtime-filtered candidates are dry
        // failures, not searched graphs: they are never profiled and do not
        // count toward the graph search limit.
        let (initial_genome, mut best_metric, display, mut n_graphs) = {
            let mut invalid_attempts = 0usize;
            let mut filter_fails = 0usize;
            let mut last_filter_rejection: Option<String> = None;
            // Breakdown of why profiled candidates were invalid, for the give-up message.
            let (mut n_timed_out, mut n_nan, mut n_invalid_profile, mut n_panicked) = (0, 0, 0, 0);
            let max_invalid_attempts = 100usize;
            let max_filter_fails = options
                .limit
                .max(1)
                .saturating_mul(options.generation_size.max(1))
                .saturating_mul(100)
                .max(10_000);

            loop {
                let mut generation = random_choice_generation(egraph, 1, &mut prev_selected, rng);
                let Some(genome) = generation.pop() else {
                    panic_initial_filter_limit(filter_fails, last_filter_rejection.as_deref());
                };

                list_cache.clear();
                expr_cache.clear();
                let graph_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut graph = egglog_to_llir(
                        egraph,
                        genome.clone(),
                        ops,
                        &self.custom_ops,
                        &mut list_cache,
                        &mut expr_cache,
                        None,
                    );
                    // Profile the deployment graph itself: fully unrolled.
                    // Every scaled-down proxy (collapsed bodies, trip-count
                    // differencing) leaked family-dependent costs and
                    // inverted rankings; measuring the real graph is slower
                    // per candidate but cannot misorder families.
                    unroll_loops_in_llir(&mut graph);
                    graph
                }));
                let Ok(graph) = graph_result else {
                    invalid_attempts += 1;
                    if invalid_attempts > max_invalid_attempts {
                        panic!(
                            "Failed to find a viable initial genome after {max_invalid_attempts} invalid attempts"
                        );
                    }
                    continue;
                };

                // A candidate whose LLIR fails to compile (e.g. an egglog
                // rule that mis-fires and produces an inconsistent kernel op)
                // must be rejected like any other, not abort the search.
                let filter_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.candidate_filter_result(
                        runtime,
                        &graph,
                        &profile_dyn_map,
                        options,
                        bucket_profile_context.as_ref(),
                    )
                }))
                .unwrap_or_else(|_| {
                    CandidateFilterResult::reject_with_display("candidate compile panicked")
                });
                if !filter_result.accepted {
                    filter_fails += 1;
                    last_filter_rejection = filter_result.display;
                    // Rejections are otherwise silent until the 10k-fail
                    // panic; surface them early — a structural rejection
                    // (e.g. every candidate over the memory cap) loops
                    // here for hours looking like a hang.
                    if filter_fails <= 5 || filter_fails % 100 == 0 {
                        eprintln!(
                            "   Search  initial-genome filter reject #{filter_fails}: {}",
                            last_filter_rejection.as_deref().unwrap_or("(no reason)")
                        );
                    }
                    if filter_fails >= max_filter_fails {
                        panic_initial_filter_limit(filter_fails, last_filter_rejection.as_deref());
                    }
                    continue;
                }
                let filter_display = filter_result.display;

                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    runtime.clear_intermediate_buffers();
                    let profile_start = std::time::Instant::now();
                    let (rep_metric, rep_display) =
                        if let Some(bucket_context) = &bucket_profile_context {
                            runtime.profile_with_bucket_context(
                                &graph,
                                &profile_dyn_map,
                                options.trials,
                                options.execution_timeout,
                                ProfileBucketContext {
                                    dim_buckets: &bucket_context.dim_buckets,
                                    bucket_indices: &bucket_context.bucket_indices,
                                    representative_dyn_map: &bucket_context.representative_dyn_map,
                                },
                            )
                        } else {
                            runtime.profile(
                                &graph,
                                &profile_dyn_map,
                                options.trials,
                                options.execution_timeout,
                            )
                        };
                    let timed_out = candidate_timed_out(profile_start.elapsed());
                    let has_nan = !timed_out && runtime.has_nan_outputs(&graph, &profile_dyn_map);
                    if has_nan {
                        crate::mask_events::NAN_OUTPUT_REJECT.record();
                    }
                    let invalid_profile = rep_display.starts_with("invalid ");
                    if !has_nan && !timed_out && !invalid_profile {
                        log_best_llir(&graph, &format!("candidate=0 {rep_display}"));
                    }
                    (
                        rep_metric,
                        append_filter_display(rep_display, filter_display.as_deref()),
                        has_nan,
                        timed_out,
                        invalid_profile,
                    )
                }));

                match result {
                    Ok((metric, disp, false, false, false)) => {
                        break (genome, R::aggregate_profile_metrics(&[metric]), disp, 1);
                    }
                    Ok((_, _, has_nan, timed_out, invalid_profile)) => {
                        if timed_out {
                            n_timed_out += 1;
                        } else if has_nan {
                            n_nan += 1;
                        } else if invalid_profile {
                            n_invalid_profile += 1;
                        }
                        invalid_attempts += 1;
                    }
                    Err(_) => {
                        n_panicked += 1;
                        invalid_attempts += 1;
                    }
                }
                if invalid_attempts > max_invalid_attempts {
                    panic!(
                        "Failed to find a viable initial genome after {max_invalid_attempts} invalid attempts \
                         (candidate_timed_out={n_timed_out} nan={n_nan} invalid_profile={n_invalid_profile} panicked={n_panicked})"
                    );
                }
            }
        };

        // Print initial result and progress
        if search_log {
            let msg = format!("   {:>6} {}", "Search".cyan().bold(), display);
            println!("{msg}");
            render_bars(n_graphs, search_limit, bucket_progress);
            std::io::stdout().flush().unwrap();
        }

        // Retain every successfully profiled genome in metric order. Profiling
        // uses a collapsed loop body, so the fastest collapsed candidate is not
        // necessarily viable after the final loop unroll. Final selection below
        // re-extracts and hard-filters these candidates fastest-first.
        let mut ranked_candidates = vec![(best_metric.clone(), initial_genome.clone())];

        // Track top-N parents for offspring generation
        let mut parents: Vec<(R::ProfileMetric, crate::egglog_utils::EGraphChoiceSet<'_>)> =
            vec![(best_metric.clone(), initial_genome)];
        let mut resample_generation = false;
        let mut stagnant_generations = 0usize;

        while n_graphs < search_limit {
            if search_time_limit_reached() {
                break;
            }

            // Generate offspring from all parents, dividing budget evenly
            let budget = (search_limit - n_graphs).min(options.generation_size);
            let all_offspring = if resample_generation {
                random_choice_generation(egraph, budget, &mut prev_selected, rng)
            } else {
                let per_parent = budget.div_ceil(parents.len());
                let mut offspring = Vec::new();
                for (_, parent_genome) in &parents {
                    let remaining = budget.saturating_sub(offspring.len());
                    if remaining == 0 {
                        break;
                    }
                    // Stagnation kick: escaping a family basin needs
                    // multi-gene jumps, so mutation counts escalate with
                    // consecutive stagnant generations (capped 16x).
                    let kick = if options.restart_stagnation > 0
                        && stagnant_generations >= options.restart_stagnation
                    {
                        (1 + stagnant_generations - options.restart_stagnation).min(16)
                    } else {
                        1
                    };
                    offspring.extend(extract_reachable_generation(
                        egraph,
                        parent_genome,
                        per_parent.min(remaining),
                        options.mutations * kick,
                        &mut prev_selected,
                        rng,
                    ));
                }
                offspring
            };
            if all_offspring.is_empty() {
                break;
            }

            let mut generation_found_non_timeout = false;
            let mut generation_found_new_best = false;

            for genome in all_offspring {
                if search_time_limit_reached() {
                    break;
                }
                list_cache.clear();
                expr_cache.clear();

                let graph_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut llir_graph = egglog_to_llir(
                        egraph,
                        genome.clone(),
                        ops,
                        &self.custom_ops,
                        &mut list_cache,
                        &mut expr_cache,
                        None,
                    );
                    let pre_collapse = std::env::var_os("LLIR_DUMP_DIR")
                        .is_some()
                        .then(|| llir_graph.clone());
                    // Profile fully unrolled — see initial-genome path.
                    unroll_loops_in_llir(&mut llir_graph);
                    (pre_collapse, llir_graph)
                }));
                if let Err(payload) = &graph_result {
                    crate::mask_events::CANDIDATE_PANIC
                        .record_with(|| crate::mask_events::panic_payload(payload.as_ref()));
                }
                let Ok((pre_collapse, llir_graph)) = graph_result else {
                    if search_log {
                        for _ in 1..n_bar_lines {
                            print!("\x1b[1A");
                        }
                        print!("\r\x1b[2K");
                        render_bars(n_graphs, search_limit, bucket_progress);
                        std::io::stdout().flush().unwrap();
                    }
                    continue;
                };

                let filter_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.candidate_filter_result(
                        runtime,
                        &llir_graph,
                        &profile_dyn_map,
                        options,
                        bucket_profile_context.as_ref(),
                    )
                }))
                .unwrap_or_else(|payload| {
                    crate::mask_events::CANDIDATE_PANIC
                        .record_with(|| crate::mask_events::panic_payload(payload.as_ref()));
                    maybe_dump_selected_llir("failed-filter-candidate", dyn_map, &llir_graph);
                    if let Some(pre) = &pre_collapse {
                        maybe_dump_selected_llir("failed-filter-precollapse", dyn_map, pre);
                    }
                    CandidateFilterResult::reject_with_display("candidate compile panicked")
                });
                if !filter_result.accepted {
                    continue;
                }
                let filter_display = filter_result.display;

                n_graphs += 1;
                let profile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    runtime.clear_intermediate_buffers();
                    let profile_start = std::time::Instant::now();
                    let (rep_metric, rep_display) =
                        if let Some(bucket_context) = &bucket_profile_context {
                            runtime.profile_with_bucket_context(
                                &llir_graph,
                                &profile_dyn_map,
                                options.trials,
                                options.execution_timeout,
                                ProfileBucketContext {
                                    dim_buckets: &bucket_context.dim_buckets,
                                    bucket_indices: &bucket_context.bucket_indices,
                                    representative_dyn_map: &bucket_context.representative_dyn_map,
                                },
                            )
                        } else {
                            runtime.profile(
                                &llir_graph,
                                &profile_dyn_map,
                                options.trials,
                                options.execution_timeout,
                            )
                        };
                    let timed_out = candidate_timed_out(profile_start.elapsed());
                    let has_nan =
                        !timed_out && runtime.has_nan_outputs(&llir_graph, &profile_dyn_map);
                    if has_nan {
                        crate::mask_events::NAN_OUTPUT_REJECT.record();
                    }
                    let invalid_profile = rep_display.starts_with("invalid ");
                    (
                        rep_metric,
                        append_filter_display(rep_display, filter_display.as_deref()),
                        has_nan,
                        timed_out,
                        invalid_profile,
                    )
                }));

                let (new_metric, display_metric) = match profile_result {
                    Ok((metric, display, false, false, false)) => {
                        generation_found_non_timeout = true;
                        (R::aggregate_profile_metrics(&[metric]), display)
                    }
                    Ok((_, _, _, true, _)) | Err(_) => {
                        // Timed out or panicked — redraw bars and skip.
                        if search_log {
                            for _ in 1..n_bar_lines {
                                print!("\x1b[1A");
                            }
                            print!("\r\x1b[2K");
                            render_bars(n_graphs, search_limit, bucket_progress);
                            std::io::stdout().flush().unwrap();
                        }
                        continue;
                    }
                    Ok((_, _, true, false, _)) => {
                        generation_found_non_timeout = true;
                        // Completed profiling but produced NaNs — redraw bars and skip.
                        if search_log {
                            for _ in 1..n_bar_lines {
                                print!("\x1b[1A");
                            }
                            print!("\r\x1b[2K");
                            render_bars(n_graphs, search_limit, bucket_progress);
                            std::io::stdout().flush().unwrap();
                        }
                        continue;
                    }
                    Ok((_, _, false, false, true)) => {
                        // Backend rejected this candidate during load/profile.
                        if search_log {
                            for _ in 1..n_bar_lines {
                                print!("\x1b[1A");
                            }
                            print!("\r\x1b[2K");
                            render_bars(n_graphs, search_limit, bucket_progress);
                            std::io::stdout().flush().unwrap();
                        }
                        continue;
                    }
                };

                let rank = ranked_candidates
                    .iter()
                    .position(|(metric, _)| {
                        new_metric
                            .partial_cmp(metric)
                            .is_some_and(|ordering| ordering == std::cmp::Ordering::Less)
                    })
                    .unwrap_or(ranked_candidates.len());
                ranked_candidates.insert(rank, (new_metric.clone(), genome.clone()));

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

                log_candidate_ops(&llir_graph, &format!("cand={n_graphs} {display_metric}"));
                let new_best = best_metric.gt(&new_metric);
                if new_best {
                    generation_found_new_best = true;
                    best_metric = new_metric;
                    log_best_llir(
                        &llir_graph,
                        &format!("candidate={n_graphs} {display_metric}"),
                    );
                }

                if new_best {
                    if search_log {
                        let msg = format!("   {:>6} {display_metric}", "Searched".green().bold());
                        for _ in 1..n_bar_lines {
                            print!("\x1b[1A");
                        }
                        print!("\r\x1b[2K");
                        println!("{msg}");
                    }
                } else if search_log {
                    for _ in 1..n_bar_lines {
                        print!("\x1b[1A");
                    }
                    print!("\r\x1b[2K");
                }
                if search_log {
                    render_bars(n_graphs, search_limit, bucket_progress);
                    std::io::stdout().flush().unwrap();
                }
            }

            if generation_found_new_best {
                stagnant_generations = 0;
            } else {
                stagnant_generations += 1;
            }
            // Every other stagnant generation past the threshold explores
            // from fresh random genomes instead of the converged parents.
            let stagnation_resample = options.restart_stagnation > 0
                && stagnant_generations >= options.restart_stagnation
                && stagnant_generations % 2 == 0;
            resample_generation = !generation_found_non_timeout || stagnation_resample;
        }

        // Clear progress bars
        if search_log {
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
        }

        if search_log {
            println!(
                "   {:>6}  in {}",
                "Searched".green().bold(),
                pretty_duration::pretty_duration(&start.elapsed(), None)
            );
        }

        ranked_candidates
    }

    /// Lazily materialize individually viable final LLIRs until `target` is
    /// available. Ranked genomes remain compact e-graph choices; full graphs
    /// are only retained when aggregate bucket backtracking actually reaches
    /// them.
    #[allow(clippy::too_many_arguments)]
    fn ensure_finalist<'a, R: Runtime + 'static>(
        &'a self,
        runtime: &mut R,
        candidates: &mut LazyFinalists<'a, R::ProfileMetric>,
        target: usize,
        options: &CompileOptions,
        dyn_map: &FxHashMap<char, usize>,
        bucket_profile_context: Option<&SearchProfileBucketContext>,
        egraph_index: usize,
        search_started_at: std::time::Instant,
    ) -> bool {
        let egraph = &self.egraphs[egraph_index];
        let ops = self.ops.as_ref().unwrap();
        let search_log = options.search_log_enabled();
        let dump_pre_unroll = std::env::var_os("LLIR_DUMP_PRE_UNROLL").is_some();
        let final_filter_dyn_map = if bucket_profile_context.is_some() {
            dyn_map.clone()
        } else {
            let mut profile_dyn_map = dyn_map.clone();
            for (&dim, &value) in &options.profile_dims {
                profile_dyn_map.insert(dim, value);
            }
            profile_dyn_map
        };

        while candidates.finalists.len() <= target {
            if candidates.next_ranked >= candidates.ranked.len() {
                return false;
            }
            // Always give the fastest profiled genome one finalization attempt.
            // Later fallbacks respect the overall search budget.
            if candidates.next_ranked > 0
                && search_started_at.elapsed() >= options.search_time_limit
            {
                candidates.stopped_reason = Some(format!(
                    "search time limit expired before finalizing ranked candidate {}",
                    candidates.next_ranked + 1
                ));
                return false;
            }

            let finalist_started_at = std::time::Instant::now();
            let (metric, genome) = &candidates.ranked[candidates.next_ranked];
            let metric = metric.clone();
            let genome = genome.clone();
            candidates.next_ranked += 1;

            let final_graph_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut stitched = egglog_to_llir(
                    egraph,
                    genome,
                    ops,
                    &self.custom_ops,
                    &mut FxHashMap::default(),
                    &mut FxHashMap::default(),
                    None,
                );
                let pre_unroll = dump_pre_unroll.then(|| stitched.clone());
                unroll_loops_in_llir(&mut stitched);
                (pre_unroll, stitched)
            }));
            if let Err(payload) = &final_graph_result {
                crate::mask_events::CANDIDATE_PANIC
                    .record_with(|| crate::mask_events::panic_payload(payload.as_ref()));
            }
            let Ok((pre_unroll, stitched)) = final_graph_result else {
                candidates.rejections += 1;
                candidates.last_rejection =
                    Some("final extraction or loop unroll panicked".to_string());
                if search_log {
                    println!(
                        "   {:>6}  finalist reject ranked #{}: final extraction or loop unroll panicked",
                        "Search".yellow().bold(),
                        candidates.next_ranked,
                    );
                }
                continue;
            };

            runtime.clear_intermediate_buffers();
            let filter_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.candidate_filter_result(
                    runtime,
                    &stitched,
                    &final_filter_dyn_map,
                    options,
                    bucket_profile_context,
                )
            }))
            .unwrap_or_else(|_| {
                CandidateFilterResult::reject_with_display("final candidate filter panicked")
            });

            if options
                .candidate_timeout
                .is_some_and(|timeout| finalist_started_at.elapsed() >= timeout)
            {
                candidates.rejections += 1;
                candidates.last_rejection = Some(format!(
                    "candidate timeout expired while finalizing ranked candidate {}",
                    candidates.next_ranked
                ));
                if search_log {
                    println!(
                        "   {:>6}  finalist reject ranked #{}: finalization timeout after {:?}",
                        "Search".yellow().bold(),
                        candidates.next_ranked,
                        finalist_started_at.elapsed(),
                    );
                }
                continue;
            }
            if !filter_result.accepted {
                candidates.rejections += 1;
                candidates.last_rejection = filter_result.display;
                crate::mask_events::FINALIST_REJECT.record_with(|| {
                    candidates
                        .last_rejection
                        .clone()
                        .unwrap_or_else(|| "(no reason)".to_string())
                });
                if search_log {
                    println!(
                        "   {:>6}  finalist reject ranked #{}: {}",
                        "Search".yellow().bold(),
                        candidates.next_ranked,
                        candidates
                            .last_rejection
                            .as_deref()
                            .unwrap_or("(no reason)"),
                    );
                }
                continue;
            }

            // Falling past rank #1 silently substitutes a slower-profiled
            // graph; make the substitution visible.
            if search_log && candidates.next_ranked > 1 {
                println!(
                    "   {:>6}  finalist fallback: loading ranked #{} after {} rejection(s)",
                    "Search".yellow().bold(),
                    candidates.next_ranked,
                    candidates.rejections,
                );
            }
            candidates.finalists.push(Finalist {
                metric,
                pre_unroll,
                llir: stitched,
            });
        }
        true
    }

    fn no_finalist_message<M>(candidates: &LazyFinalists<'_, M>) -> String {
        if let Some(stopped_reason) = &candidates.stopped_reason {
            return format!(
                "no viable final graph after {} hard-filter rejections: {stopped_reason}",
                candidates.rejections
            );
        }
        format!(
            "no viable final graph after hard-filtering {} profiled candidates: {}",
            candidates.rejections,
            candidates
                .last_rejection
                .as_deref()
                .unwrap_or("no rejection reason")
        )
    }

    fn dump_selected_finalist<M>(
        finalist: &Finalist<M>,
        dyn_map: &FxHashMap<char, usize>,
        bucket_progress: Option<(usize, usize)>,
    ) {
        if let Some(pre_unroll) = &finalist.pre_unroll {
            let dump_label = bucket_progress
                .map(|(bucket_idx, n_buckets)| {
                    format!("pre-unroll-bucket-{:02}-of-{n_buckets:02}", bucket_idx + 1)
                })
                .unwrap_or_else(|| "pre-unroll-single".to_string());
            maybe_dump_selected_llir(&dump_label, dyn_map, pre_unroll);
        }
        let dump_label = bucket_progress
            .map(|(bucket_idx, n_buckets)| {
                format!("bucket-{:02}-of-{n_buckets:02}", bucket_idx + 1)
            })
            .unwrap_or_else(|| "single".to_string());
        maybe_dump_selected_llir(&dump_label, dyn_map, &finalist.llir);
    }

    fn candidate_filter_result<R: Runtime + 'static>(
        &self,
        runtime: &mut R,
        llir_graph: &LLIRGraph,
        dyn_map: &FxHashMap<char, usize>,
        search_options: &CompileOptions,
        bucket_profile_context: Option<&SearchProfileBucketContext>,
    ) -> CandidateFilterResult {
        runtime.filter_llir_candidate(
            llir_graph,
            CandidateFilterContext {
                search_options,
                dyn_map,
                bucket_context: bucket_profile_context.map(|bucket_context| ProfileBucketContext {
                    dim_buckets: &bucket_context.dim_buckets,
                    bucket_indices: &bucket_context.bucket_indices,
                    representative_dyn_map: &bucket_context.representative_dyn_map,
                }),
            },
        )
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

fn append_filter_display(display: String, filter_display: Option<&str>) -> String {
    if let Some(filter_display) = filter_display.filter(|s| !s.is_empty()) {
        format!("{display} | {filter_display}")
    } else {
        display
    }
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

/// Append one line per profiled candidate (op-type histogram + metric) to
/// the file named by `LUMINAL_CANDIDATE_OPS` — search-trajectory forensics
/// for "was family X ever generated, and what did it measure".
fn log_candidate_ops(llir: &LLIRGraph, tag: &str) {
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let Some(path) = PATH.get_or_init(|| std::env::var("LUMINAL_CANDIDATE_OPS").ok()) else {
        return;
    };
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for op in llir.node_weights() {
        let debug = format!("{op:?}");
        let name = debug
            .split(['{', '(', ' ', ')'])
            .find(|s| !s.is_empty() && *s != "LLIROp" && *s != "DialectOp")
            .unwrap_or("?")
            .to_string();
        *counts.entry(name).or_default() += 1;
    }
    let line = format!(
        "{tag} | {}\n",
        counts
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = f.write_all(line.as_bytes());
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
/// When `LUMINAL_LOG_LLIR=1`, print a canonical, diffable dump of a
/// candidate LLIR each time the search finds a new fastest graph. Nodes are
/// numbered canonically (Kahn topological order with a deterministic
/// tie-break on op text and canonical input ids), so two runs of an
/// identical graph produce byte-identical output regardless of NodeIndex
/// assignment — best-so-far graphs from different runs can be compared with
/// plain `diff`.
pub fn log_best_llir(llir: &LLIRGraph, context: &str) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("LUMINAL_LOG_LLIR").is_some()) {
        return;
    }
    use petgraph::visit::EdgeRef;
    use std::collections::BTreeMap;

    let mut indegree: FxHashMap<NodeIndex, usize> = llir
        .node_indices()
        .map(|n| (n, llir.edges_directed(n, Direction::Incoming).count()))
        .collect();
    // Ready nodes keyed by (op text, canonical input ids) for deterministic
    // pops; true duplicates tie and are interchangeable.
    let mut ready: BTreeMap<(String, Vec<usize>), Vec<NodeIndex>> = BTreeMap::new();
    let mut canonical: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let inputs_of = |n: NodeIndex, canonical: &FxHashMap<NodeIndex, usize>| -> Vec<usize> {
        llir.edges_directed(n, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| canonical.get(&e.source()).copied().unwrap_or(usize::MAX))
            .collect()
    };
    for (&n, &d) in &indegree {
        if d == 0 {
            ready
                .entry((format!("{:?}", llir[n]), Vec::new()))
                .or_default()
                .push(n);
        }
    }
    let mut lines: Vec<String> = Vec::with_capacity(indegree.len());
    while let Some((key, nodes)) = ready.pop_first() {
        for n in nodes {
            let id = lines.len();
            canonical.insert(n, id);
            let inputs = key
                .1
                .iter()
                .map(|i| format!("n{i}"))
                .collect::<Vec<_>>()
                .join(",");
            lines.push(format!("n{id}: {} <- [{inputs}]", key.0));
            for succ in llir
                .neighbors_directed(n, Direction::Outgoing)
                .collect::<Vec<_>>()
            {
                let d = indegree.get_mut(&succ).unwrap();
                *d -= 1;
                if *d == 0 {
                    ready
                        .entry((format!("{:?}", llir[succ]), inputs_of(succ, &canonical)))
                        .or_default()
                        .push(succ);
                }
            }
        }
    }
    println!("LLIR_BEST {context} nodes={}", lines.len());
    for line in &lines {
        println!("{line}");
    }
    println!("LLIR_BEST_END");
}

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

pub fn unroll_loops_in_llir(llir: &mut LLIRGraph) {
    inline_static_loop_inputs(llir);
    // Innermost first: unrolling an inner region turns it back into plain
    // body nodes, which is exactly what the enclosing region's clone step
    // needs to replicate.
    while let Some((region, body)) = take_innermost_region(llir) {
        if region.iters <= 1 || region.starts.is_empty() {
            eprintln!(
                "[loop-debug] unroll abandoned on degenerate region: iters={} starts={} ends={} inputs={} outputs={} selects={}",
                region.iters,
                region.starts.len(),
                region.ends.len(),
                region.inputs.len(),
                region.outputs.len(),
                region.output_selects.len(),
            );
            return;
        }
        unroll_loop_region(llir, &region, &body);
        // Compact into a freshly-allocated StableGraph so all edge IDs are
        // re-assigned sequentially in our chosen insertion order. Without
        // this, later add_edge calls (the next region's rewiring, or
        // kernel_to_host scheduling edges) can reuse edge indices freed by
        // remove_node on loop markers, producing sort-by-edge-id orderings
        // where a later-added edge lands at a low index — which the runtime
        // interprets as a primary input position.
        let compacted = compact_llir_preserving_input_order(llir);
        *llir = compacted;
    }
    debug_assert!(
        collect_loop_regions(llir).is_empty(),
        "unroll left stray loop marker ops in LLIR"
    );
}

fn unroll_loop_region(
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
        iters,
        markers: loop_markers,
    } = region;
    let iters = *iters;

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
    for (slot_idx, &start_node) in starts {
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

    let mut clone_map: Vec<FxHashMap<NodeIndex, NodeIndex>> = vec![FxHashMap::default(); iters];
    for &b in body_nodes {
        clone_map[0].insert(b, b);
    }
    for clone in clone_map.iter_mut().skip(1) {
        for &b in body_nodes {
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
    for &b in body_nodes {
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
        for &b in body_nodes {
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
    for (&stream_id, &output_node) in outputs {
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
        let sub = resolve_src(body_producer, iters - 1, &clone_map);
        marker_post_sub.insert(end_node, sub);
    }
    // Each LoopOutputSelect(stream, iter) routes to iter's clone of that
    // stream's body producer. Same iteration-invariant fallback as for
    // LoopEnd above: if the body producer isn't in `body_nodes`, it wasn't
    // cloned per iter and every iter shares the single `body_producer`.
    // Entry markers can also have consumers outside the walked body:
    // extraction may elect a marker as the representative of a value class
    // read anywhere in the graph. Such aliasing only arises for
    // iteration-invariant values (that is what egglog unions), so route
    // stray consumers to the iter-0 value — never leave a consumer pointing
    // at a marker about to be removed.
    for (&input_node, sources) in &input_per_iter {
        marker_post_sub.insert(input_node, sources[0]);
    }
    for (&start_node, &(initial, _)) in &start_meta {
        marker_post_sub.insert(start_node, initial);
    }
    for (&select_node, &(stream_id, iter)) in output_selects {
        let body_producer = output_body_producer[&stream_id];
        let sub = resolve_src(body_producer, iter, &clone_map);
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

    for &n in loop_markers {
        llir.remove_node(n);
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
    // Innermost first, same as `unroll_loops_in_llir`: collapsing an inner
    // region leaves its iter-0 body as plain nodes inside the enclosing
    // region, which then collapses over them in turn.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egglog_utils::hash_egglog_normalized;
    use crate::egglog_utils::{
        api::{Rule, SortDef, sort},
        base::OP_KIND,
    };
    use crate::hlir::{ReferenceData, ReferenceOp};
    use rand::SeedableRng;

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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    static SEARCH_DIM_LATE_PASS_CALLED: AtomicBool = AtomicBool::new(false);
    static SEARCH_DIM_LATE_PASS_SAW_C: AtomicBool = AtomicBool::new(false);

    #[derive(Default)]
    struct SearchDimRecordingRuntime {
        profile_dyn_maps: Vec<FxHashMap<char, usize>>,
        bucket_representative_dyn_maps: Vec<FxHashMap<char, usize>>,
    }

    impl Runtime for SearchDimRecordingRuntime {
        type Ops = ();
        type CompileArg = ();
        type ExecReturn = ();
        type ProfileMetric = usize;

        fn late_egglog_passes(
            _: &[Arc<Box<dyn EgglogOp>>],
            _: &CompileOptions,
            dyn_map: &FxHashMap<char, usize>,
        ) -> Vec<crate::egglog_utils::LateEgglogPass> {
            SEARCH_DIM_LATE_PASS_CALLED.store(true, Ordering::SeqCst);
            SEARCH_DIM_LATE_PASS_SAW_C.store(dyn_map.contains_key(&'c'), Ordering::SeqCst);
            vec![]
        }

        fn initialize(_: Self::CompileArg) -> Self {
            Self::default()
        }

        fn load_llir(&mut self, _: &LLIRGraph) {}

        fn load_llir_buckets(
            &mut self,
            _: &FxHashMap<char, Vec<DimBucket>>,
            bucket_llirs: &[BucketLLIR],
        ) {
            self.bucket_representative_dyn_maps = bucket_llirs
                .iter()
                .map(|(_, representative_dyn_map, _)| representative_dyn_map.clone())
                .collect();
        }

        fn execute(&mut self, _: &FxHashMap<char, usize>) -> Self::ExecReturn {}

        fn profile(
            &mut self,
            _: &LLIRGraph,
            dyn_map: &FxHashMap<char, usize>,
            _: usize,
            _: Option<std::time::Duration>,
        ) -> (Self::ProfileMetric, String) {
            self.profile_dyn_maps.push(dyn_map.clone());
            (0, "0 ms".to_string())
        }
    }

    #[derive(Default)]
    struct TestFilterRuntime {
        reject_candidates: bool,
    }

    impl Runtime for TestFilterRuntime {
        type Ops = ();
        type CompileArg = ();
        type ExecReturn = ();
        type ProfileMetric = usize;

        fn initialize(_: Self::CompileArg) -> Self {
            Self::default()
        }

        fn load_llir(&mut self, _: &LLIRGraph) {}

        fn execute(&mut self, _: &FxHashMap<char, usize>) -> Self::ExecReturn {}

        fn profile(
            &mut self,
            _: &LLIRGraph,
            _: &FxHashMap<char, usize>,
            _: usize,
            _: Option<std::time::Duration>,
        ) -> (Self::ProfileMetric, String) {
            (0, "0 ms".to_string())
        }

        fn filter_llir_candidate(
            &mut self,
            _: &LLIRGraph,
            _: CandidateFilterContext<'_>,
        ) -> CandidateFilterResult {
            if self.reject_candidates {
                CandidateFilterResult::reject_with_display("test filter rejected candidate")
            } else {
                CandidateFilterResult::accept()
            }
        }
    }

    static PROFILE_CALLS: AtomicUsize = AtomicUsize::new(0);

    #[derive(Default)]
    struct CountingRuntime {
        reject_candidates: bool,
    }

    impl Runtime for CountingRuntime {
        type Ops = ();
        type CompileArg = ();
        type ExecReturn = ();
        type ProfileMetric = usize;

        fn initialize(_: Self::CompileArg) -> Self {
            Self::default()
        }

        fn load_llir(&mut self, _: &LLIRGraph) {}

        fn execute(&mut self, _: &FxHashMap<char, usize>) -> Self::ExecReturn {}

        fn profile(
            &mut self,
            _: &LLIRGraph,
            _: &FxHashMap<char, usize>,
            _: usize,
            _: Option<std::time::Duration>,
        ) -> (Self::ProfileMetric, String) {
            let count = PROFILE_CALLS.fetch_add(1, Ordering::SeqCst);
            (count, format!("{count} ms"))
        }

        fn filter_llir_candidate(
            &mut self,
            _: &LLIRGraph,
            _: CandidateFilterContext<'_>,
        ) -> CandidateFilterResult {
            if self.reject_candidates {
                CandidateFilterResult::reject_with_display("test filter rejected candidate")
            } else {
                CandidateFilterResult::accept()
            }
        }
    }

    macro_rules! final_filter_test_op {
        ($name:ident, $sort_name:literal, $rule_name:literal) => {
            #[derive(Debug, Default)]
            struct $name;

            impl EgglogOp for $name {
                fn sort(&self) -> SortDef {
                    sort(OP_KIND, $sort_name, &[])
                }

                fn rewrites(&self) -> Vec<Rule> {
                    vec![Rule::raw(format!(
                        "(rule
                            (
                                (= ?sin (Op (Sin ?shape ?strides ?out_strides) ?inputs))
                                (= (F32) (dtype ?sin))
                            )
                            (
                                (let ?candidate (Op ({}) ?inputs))
                                (union ?sin ?candidate)
                                (set (dtype ?candidate) (F32))
                            )
                            :ruleset kernel_lower
                            :name \"{}\"
                        )",
                        $sort_name, $rule_name
                    ))]
                }

                fn cleanup(&self) -> bool {
                    false
                }

                fn n_inputs(&self) -> usize {
                    1
                }

                fn extract<'a>(
                    &'a self,
                    _: &'a SerializedEGraph,
                    _: &[&'a ENodeId],
                    input_enodes: Vec<&'a ENodeId>,
                    _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
                    _: &mut FxHashMap<&'a ENodeId, Expression>,
                ) -> (LLIROp, Vec<&'a ENodeId>) {
                    (
                        LLIROp::new::<dyn ReferenceOp>(Box::new(Self) as Box<dyn ReferenceOp>),
                        input_enodes,
                    )
                }
            }

            impl ReferenceOp for $name {
                fn execute(
                    &self,
                    inputs: Vec<&ReferenceData>,
                    _: &FxHashMap<char, usize>,
                ) -> ReferenceData {
                    let ReferenceData::F32(input) = inputs[0] else {
                        panic!("final-filter test Sin candidates only support F32")
                    };
                    ReferenceData::F32(input.iter().map(|value| value.sin()).collect())
                }
            }
        };
    }

    final_filter_test_op!(FastButInvalidAfterUnroll, "TestFastSin", "test fast sin");
    final_filter_test_op!(SlowerValidAfterUnroll, "TestSafeSin", "test safe sin");

    #[derive(Default)]
    struct FinalFilterRuntime {
        accepted_signatures: FxHashSet<(usize, usize, usize)>,
        loaded_signatures: Vec<(usize, usize, usize)>,
        profiled_fast: usize,
        profiled_safe: usize,
        rejected_unrolled_fast: usize,
    }

    impl FinalFilterRuntime {
        fn signature(llir: &LLIRGraph) -> (usize, usize, usize) {
            let mut fast = 0;
            let mut safe = 0;
            for op in llir.node_weights() {
                let Some(reference_op) = op.to_dialect::<dyn ReferenceOp>() else {
                    continue;
                };
                let reference_op = reference_op.as_ref().as_ref();
                fast += usize::from(reference_op.as_any().is::<FastButInvalidAfterUnroll>());
                safe += usize::from(reference_op.as_any().is::<SlowerValidAfterUnroll>());
            }
            (fast, safe, llir.node_count())
        }
    }

    impl Runtime for FinalFilterRuntime {
        type Ops = (FastButInvalidAfterUnroll, SlowerValidAfterUnroll);
        type CompileArg = ();
        type ExecReturn = ();
        type ProfileMetric = usize;

        fn initialize(_: Self::CompileArg) -> Self {
            Self::default()
        }

        fn load_llir(&mut self, llir: &LLIRGraph) {
            let signature = Self::signature(llir);
            assert!(
                self.accepted_signatures.contains(&signature),
                "loaded final LLIR {signature:?} did not pass the hard candidate filter"
            );
            self.loaded_signatures.push(signature);
        }

        fn execute(&mut self, _: &FxHashMap<char, usize>) -> Self::ExecReturn {}

        fn profile(
            &mut self,
            llir: &LLIRGraph,
            _: &FxHashMap<char, usize>,
            _: usize,
            _: Option<std::time::Duration>,
        ) -> (Self::ProfileMetric, String) {
            let (fast, safe, _) = Self::signature(llir);
            assert_eq!(
                fast + safe,
                3,
                "profiling should see the fully unrolled candidate"
            );
            if fast == 1 {
                self.profiled_fast += 1;
                (0, "fast".to_string())
            } else {
                self.profiled_safe += 1;
                (1, "safe".to_string())
            }
        }

        fn filter_llir_candidate(
            &mut self,
            llir: &LLIRGraph,
            _: CandidateFilterContext<'_>,
        ) -> CandidateFilterResult {
            let signature = Self::signature(llir);
            if signature.0 > 1 {
                self.rejected_unrolled_fast += 1;
                CandidateFilterResult::reject_with_display(
                    "fast candidate exceeds final unrolled resource limit",
                )
            } else {
                self.accepted_signatures.insert(signature);
                CandidateFilterResult::accept()
            }
        }
    }

    #[derive(Default)]
    struct ProfileDimFinalFilterRuntime {
        last_profile_dim: Option<usize>,
        loaded_signatures: Vec<(usize, usize, usize)>,
        rejected_unrolled_fast: usize,
    }

    impl Runtime for ProfileDimFinalFilterRuntime {
        type Ops = (FastButInvalidAfterUnroll, SlowerValidAfterUnroll);
        type CompileArg = ();
        type ExecReturn = ();
        type ProfileMetric = usize;

        fn initialize(_: Self::CompileArg) -> Self {
            Self::default()
        }

        fn load_llir(&mut self, llir: &LLIRGraph) {
            let signature = FinalFilterRuntime::signature(llir);
            let profile_dim = self
                .last_profile_dim
                .expect("a selected LLIR must be loaded after profiling");
            assert!(
                signature.0 == 0 || signature.0 + signature.1 == 1 || profile_dim <= 1,
                "load rejected unrolled fast LLIR {signature:?} at profile dim {profile_dim}"
            );
            self.loaded_signatures.push(signature);
        }

        fn execute(&mut self, _: &FxHashMap<char, usize>) -> Self::ExecReturn {}

        fn profile(
            &mut self,
            llir: &LLIRGraph,
            dyn_map: &FxHashMap<char, usize>,
            _: usize,
            _: Option<std::time::Duration>,
        ) -> (Self::ProfileMetric, String) {
            let (fast, safe, _) = FinalFilterRuntime::signature(llir);
            assert_eq!(
                fast + safe,
                3,
                "profiling should see the fully unrolled candidate"
            );
            self.last_profile_dim = dyn_map.get(&'s').copied();
            if fast == 1 {
                (0, "fast".to_string())
            } else {
                (1, "safe".to_string())
            }
        }

        fn filter_llir_candidate(
            &mut self,
            llir: &LLIRGraph,
            context: CandidateFilterContext<'_>,
        ) -> CandidateFilterResult {
            let (fast, safe, _) = FinalFilterRuntime::signature(llir);
            let is_unrolled = fast + safe > 1;
            let filter_dim = context.dyn_map.get(&'s').copied().unwrap_or(1);
            if is_unrolled && fast > 0 && filter_dim > 1 {
                self.rejected_unrolled_fast += 1;
                CandidateFilterResult::reject_with_display(format!(
                    "fast candidate exceeds final resource limit at s={filter_dim}"
                ))
            } else {
                CandidateFilterResult::accept()
            }
        }
    }

    #[derive(Default)]
    struct AggregateBucketFilterRuntime {
        individually_accepted: FxHashSet<(usize, usize, usize)>,
        aggregate_attempts: Vec<Vec<(usize, usize, usize)>>,
        aggregate_accepted: FxHashSet<Vec<(usize, usize, usize)>>,
        loaded_sets: Vec<Vec<(usize, usize, usize)>>,
    }

    impl Runtime for AggregateBucketFilterRuntime {
        type Ops = (FastButInvalidAfterUnroll, SlowerValidAfterUnroll);
        type CompileArg = ();
        type ExecReturn = ();
        type ProfileMetric = usize;

        fn initialize(_: Self::CompileArg) -> Self {
            Self::default()
        }

        fn load_llir(&mut self, _: &LLIRGraph) {
            panic!("aggregate bucket regression must load a retained bucket set")
        }

        fn load_llir_buckets(
            &mut self,
            _: &FxHashMap<char, Vec<DimBucket>>,
            bucket_llirs: &[BucketLLIR],
        ) {
            let signatures = bucket_llirs
                .iter()
                .map(|(_, _, llir)| FinalFilterRuntime::signature(llir))
                .collect_vec();
            assert!(
                signatures
                    .iter()
                    .all(|signature| self.individually_accepted.contains(signature)),
                "load received a bucket LLIR that did not pass individual final filtering"
            );
            assert!(
                self.aggregate_accepted.contains(&signatures),
                "load received bucket set {signatures:?} that did not pass aggregate filtering"
            );
            self.loaded_sets.push(signatures);
        }

        fn execute(&mut self, _: &FxHashMap<char, usize>) -> Self::ExecReturn {}

        fn profile(
            &mut self,
            llir: &LLIRGraph,
            _: &FxHashMap<char, usize>,
            _: usize,
            _: Option<std::time::Duration>,
        ) -> (Self::ProfileMetric, String) {
            let (fast, safe, _) = FinalFilterRuntime::signature(llir);
            if fast > 0 {
                (0, "fast".to_string())
            } else if safe > 0 {
                (1, "safe".to_string())
            } else {
                (2, "original".to_string())
            }
        }

        fn aggregate_profile_metrics(metrics: &[Self::ProfileMetric]) -> Self::ProfileMetric {
            metrics.iter().sum()
        }

        fn filter_llir_candidate(
            &mut self,
            llir: &LLIRGraph,
            _: CandidateFilterContext<'_>,
        ) -> CandidateFilterResult {
            self.individually_accepted
                .insert(FinalFilterRuntime::signature(llir));
            CandidateFilterResult::accept()
        }

        fn filter_llir_bucket_set(
            &mut self,
            _: &FxHashMap<char, Vec<DimBucket>>,
            bucket_llirs: &[BucketLLIRRef<'_>],
            _: &CompileOptions,
        ) -> CandidateFilterResult {
            let signatures = bucket_llirs
                .iter()
                .map(|bucket| FinalFilterRuntime::signature(bucket.llir))
                .collect_vec();
            assert!(
                signatures
                    .iter()
                    .all(|signature| self.individually_accepted.contains(signature)),
                "aggregate filter received a bucket that skipped individual filtering"
            );
            self.aggregate_attempts.push(signatures.clone());
            let all_fast = signatures
                .iter()
                .all(|(fast, safe, _)| *fast > 0 && *safe == 0);
            if all_fast {
                CandidateFilterResult::reject_with_display(
                    "fast bucket finalists conflict in retained resources",
                )
            } else {
                self.aggregate_accepted.insert(signatures);
                CandidateFilterResult::accept()
            }
        }
    }

    static SEARCH_BUDGET_PROFILE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static SEARCH_BUDGET_FINAL_FILTER_CALLS: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug, Default)]
    struct SearchBudgetFinalFilterRuntime;

    impl Runtime for SearchBudgetFinalFilterRuntime {
        type Ops = (FastButInvalidAfterUnroll, SlowerValidAfterUnroll);
        type CompileArg = ();
        type ExecReturn = ();
        type ProfileMetric = usize;

        fn initialize(_: Self::CompileArg) -> Self {
            Self
        }

        fn load_llir(&mut self, _: &LLIRGraph) {
            panic!("a runtime with no viable final candidate must not be loaded")
        }

        fn execute(&mut self, _: &FxHashMap<char, usize>) -> Self::ExecReturn {}

        fn profile(
            &mut self,
            llir: &LLIRGraph,
            _: &FxHashMap<char, usize>,
            _: usize,
            _: Option<std::time::Duration>,
        ) -> (Self::ProfileMetric, String) {
            let profile_call = SEARCH_BUDGET_PROFILE_CALLS.fetch_add(1, Ordering::SeqCst);
            if profile_call == 1 {
                // Cross the global budget only after a second genome has
                // entered profiling, so it is retained as a possible fallback.
                std::thread::sleep(std::time::Duration::from_millis(150));
            }
            let (fast, _, _) = FinalFilterRuntime::signature(llir);
            if fast == 1 {
                (0, "fast".to_string())
            } else {
                (1, "safe".to_string())
            }
        }

        fn filter_llir_candidate(
            &mut self,
            llir: &LLIRGraph,
            _: CandidateFilterContext<'_>,
        ) -> CandidateFilterResult {
            let (fast, safe, _) = FinalFilterRuntime::signature(llir);
            if fast + safe > 1 {
                // Candidates are filtered unrolled at search time AND at
                // finalization now. Accept the first two calls (the two
                // search candidates), then reject the finalization
                // re-checks, burning the search time limit on the first.
                let call = SEARCH_BUDGET_FINAL_FILTER_CALLS.fetch_add(1, Ordering::SeqCst);
                if call < 2 {
                    return CandidateFilterResult::accept();
                }
                std::thread::sleep(std::time::Duration::from_millis(120));
                CandidateFilterResult::reject_with_display("forced final rejection")
            } else {
                CandidateFilterResult::accept()
            }
        }
    }

    static CANDIDATE_BUDGET_FINAL_FILTER_CALLS: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug, Default)]
    struct CandidateBudgetFinalFilterRuntime;

    impl Runtime for CandidateBudgetFinalFilterRuntime {
        type Ops = (FastButInvalidAfterUnroll, SlowerValidAfterUnroll);
        type CompileArg = ();
        type ExecReturn = ();
        type ProfileMetric = usize;

        fn initialize(_: Self::CompileArg) -> Self {
            Self
        }

        fn load_llir(&mut self, _: &LLIRGraph) {
            panic!("a timed-out final candidate must not be loaded")
        }

        fn execute(&mut self, _: &FxHashMap<char, usize>) -> Self::ExecReturn {}

        fn profile(
            &mut self,
            llir: &LLIRGraph,
            _: &FxHashMap<char, usize>,
            _: usize,
            _: Option<std::time::Duration>,
        ) -> (Self::ProfileMetric, String) {
            let (fast, _, _) = FinalFilterRuntime::signature(llir);
            if fast == 1 {
                (0, "fast".to_string())
            } else {
                (1, "safe".to_string())
            }
        }

        fn filter_llir_candidate(
            &mut self,
            llir: &LLIRGraph,
            _: CandidateFilterContext<'_>,
        ) -> CandidateFilterResult {
            let (fast, safe, _) = FinalFilterRuntime::signature(llir);
            if fast + safe > 1 {
                CANDIDATE_BUDGET_FINAL_FILTER_CALLS.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
            CandidateFilterResult::accept()
        }
    }

    #[test]
    fn compile_options_defaults_and_search_time_limit_builder() {
        let opts = CompileOptions::default();
        assert_eq!(opts.limit, 100);
        assert_eq!(opts.search_time_limit, std::time::Duration::MAX);
        assert!(!opts.egglog_log);
        assert!(!opts.rolling_log);
        assert!(opts.search_log);
        assert!(opts.search_dims.is_empty());

        let time_limit = std::time::Duration::from_millis(25);
        let opts = CompileOptions::default()
            .search_graph_limit(7)
            .search_time_limit(time_limit)
            .search_dim('c', 16)
            .egglog_log(true)
            .rolling_log(true)
            .search_log(false);
        assert_eq!(opts.limit, 7);
        assert_eq!(opts.search_time_limit, time_limit);
        assert_eq!(opts.search_dims[&'c'], 16);
        assert!(opts.egglog_log);
        assert!(opts.rolling_log);
        assert!(!opts.search_log);
    }

    #[test]
    fn compile_applies_search_dims_after_build_with_documented_precedence() {
        SEARCH_DIM_LATE_PASS_CALLED.store(false, Ordering::SeqCst);
        SEARCH_DIM_LATE_PASS_SAW_C.store(false, Ordering::SeqCst);

        let mut cx = Graph::new();
        let _ = cx.tensor(('s', 'c')).output();
        let options = CompileOptions::default()
            .dim_buckets('s', &[DimBucket::new(1, 4).representative(3)])
            .search_dim('s', 4)
            .search_dim('c', 16)
            .profile_dim('s', 2)
            .profile_dim('c', 7)
            .search_graph_limit(1)
            .search_log(false);

        let runtime = cx.compile(SearchDimRecordingRuntime::default(), options);

        assert!(SEARCH_DIM_LATE_PASS_CALLED.load(Ordering::SeqCst));
        assert!(
            !SEARCH_DIM_LATE_PASS_SAW_C.load(Ordering::SeqCst),
            "search-only dimensions must not leak into build-time late passes"
        );
        assert_eq!(runtime.profile_dyn_maps.len(), 1);
        assert_eq!(runtime.profile_dyn_maps[0][&'s'], 2);
        assert_eq!(runtime.profile_dyn_maps[0][&'c'], 7);
        assert_eq!(runtime.bucket_representative_dyn_maps.len(), 1);
        assert_eq!(runtime.bucket_representative_dyn_maps[0][&'s'], 3);
        assert_eq!(runtime.bucket_representative_dyn_maps[0][&'c'], 16);
        assert_eq!(cx.dyn_map[&'s'], 4);
        assert_eq!(cx.dyn_map[&'c'], 16);
    }

    #[test]
    fn search_time_limit_stops_after_initial_viable_candidate() {
        let mut cx = Graph::new();
        let a = cx.tensor((4, 8));
        let b = cx.tensor((8, 4));
        let c = cx.tensor((4, 4));
        let _ = (a.matmul(b) + c).relu().softmax(1).output();

        cx.build_search_space::<CountingRuntime>(CompileOptions::default());

        PROFILE_CALLS.store(0, Ordering::SeqCst);
        let _ = cx.search(
            CountingRuntime::default(),
            CompileOptions::default().search_time_limit(std::time::Duration::ZERO),
        );
        assert_eq!(PROFILE_CALLS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn build_search_space_does_not_configure_runtime_filter() {
        let mut cx = Graph::new();
        let _ = cx.tensor(1).output();

        cx.build_search_space::<TestFilterRuntime>(CompileOptions::default());
        assert_eq!(cx.egraphs.len(), 1);
    }

    #[test]
    #[should_panic(expected = "runtime filter failures")]
    fn runtime_filter_rejects_candidates() {
        let mut cx = Graph::new();
        let _ = cx.tensor(1).output();
        cx.build_search_space::<TestFilterRuntime>(CompileOptions::default());

        let _ = cx.search(
            TestFilterRuntime {
                reject_candidates: true,
            },
            CompileOptions::default().search_graph_limit(1),
        );
    }

    #[test]
    fn runtime_filter_rejects_candidate_before_profile() {
        let mut cx = Graph::new();
        let _ = cx.tensor(1).output();
        cx.build_search_space::<CountingRuntime>(CompileOptions::default());

        PROFILE_CALLS.store(0, Ordering::SeqCst);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = cx.search(
                CountingRuntime {
                    reject_candidates: true,
                },
                CompileOptions::default().search_graph_limit(1),
            );
        }));

        assert!(result.is_err());
        assert_eq!(PROFILE_CALLS.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn final_filter_skips_fast_invalid_unroll_and_loads_filtered_fallback() {
        let mut cx = Graph::new();
        let input = cx.tensor(8);
        let _ = input.sin().sin().sin().output();

        let options = CompileOptions::default()
            .search_graph_limit(32)
            .generation_size(8)
            .mutations(2)
            .search_log(false);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xF1A1_F11E);
        let runtime = cx.compile_with_rng(FinalFilterRuntime::default(), options, &mut rng);

        assert_eq!(
            runtime.profiled_fast, 0,
            "invalid-unrolled fast candidates must be filtered before profiling"
        );
        assert!(runtime.profiled_safe > 0, "safe candidate was not profiled");
        assert!(
            runtime.rejected_unrolled_fast > 0,
            "the fast candidate should be rejected by the unrolled candidate filter"
        );
        assert_eq!(
            runtime.loaded_signatures.len(),
            1,
            "search should load exactly one final LLIR"
        );
        let (fast, safe, _) = runtime.loaded_signatures[0];
        assert_eq!(fast, 0, "the rejected fast candidate was loaded");
        assert!(
            safe > 1,
            "the loaded fallback should be the fully unrolled safe graph"
        );
    }

    #[test]
    fn unbucketed_final_filter_uses_profile_dims_seen_by_load() {
        let mut cx = Graph::new();
        cx.set_dim('s', 1);
        let input = cx.tensor('s');
        let _ = input.sin().sin().sin().output();

        let options = CompileOptions::default()
            .profile_dim('s', 8)
            .search_graph_limit(32)
            .generation_size(8)
            .mutations(2)
            .search_log(false);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xF1A1_F11E);
        let runtime =
            cx.compile_with_rng(ProfileDimFinalFilterRuntime::default(), options, &mut rng);

        assert_eq!(runtime.last_profile_dim, Some(8));
        assert!(
            runtime.rejected_unrolled_fast > 0,
            "the final hard filter did not receive the profiling dimension"
        );
        assert_eq!(runtime.loaded_signatures.len(), 1);
        let (fast, safe, _) = runtime.loaded_signatures[0];
        assert_eq!(fast, 0, "load received a candidate invalid at profile_dim");
        assert!(safe > 1, "load should receive a fully unrolled fallback");
    }

    #[test]
    fn final_fallback_validates_one_candidate_then_stops_at_search_time_limit() {
        SEARCH_BUDGET_PROFILE_CALLS.store(0, Ordering::SeqCst);
        SEARCH_BUDGET_FINAL_FILTER_CALLS.store(0, Ordering::SeqCst);

        let mut cx = Graph::new();
        let input = cx.tensor(8);
        let _ = input.sin().sin().sin().output();
        let options = CompileOptions::default()
            .search_graph_limit(2)
            .generation_size(1)
            .mutations(2)
            .search_time_limit(std::time::Duration::from_millis(100))
            .search_log(false);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xF1A1_F11E);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cx.compile_with_rng(SearchBudgetFinalFilterRuntime, options, &mut rng)
        }));

        let panic = result.expect_err("every finalized candidate should be rejected");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic");
        assert!(
            message.contains("search time limit expired before finalizing ranked candidate 2"),
            "unexpected finalization failure: {message}"
        );
        assert_eq!(
            SEARCH_BUDGET_PROFILE_CALLS.load(Ordering::SeqCst),
            2,
            "the fallback genome must be retained before the budget expires"
        );
        assert_eq!(
            SEARCH_BUDGET_FINAL_FILTER_CALLS.load(Ordering::SeqCst),
            3,
            "two search-time filter calls plus one finalization attempt; no fallback after expiry"
        );
    }

    #[test]
    fn final_fallback_rejects_candidates_that_exceed_candidate_timeout() {
        CANDIDATE_BUDGET_FINAL_FILTER_CALLS.store(0, Ordering::SeqCst);

        let mut cx = Graph::new();
        let input = cx.tensor(8);
        let _ = input.sin().sin().sin().output();
        let options = CompileOptions::default()
            .search_graph_limit(2)
            .generation_size(1)
            .mutations(2)
            .candidate_timeout(std::time::Duration::from_millis(20))
            .search_log(false);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xF1A1_F11E);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cx.compile_with_rng(CandidateBudgetFinalFilterRuntime, options, &mut rng)
        }));

        let panic = result.expect_err("timed-out finalists must not be loaded");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic");
        assert!(
            message.contains("candidate timeout expired while finalizing ranked candidate 2"),
            "unexpected finalization failure: {message}"
        );
        assert_eq!(
            CANDIDATE_BUDGET_FINAL_FILTER_CALLS.load(Ordering::SeqCst),
            4,
            "both candidates are filtered unrolled at search time and again at finalization"
        );
    }

    #[test]
    fn aggregate_bucket_filter_backtracks_to_fastest_viable_retained_set() {
        let mut cx = Graph::new();
        let _ = cx.tensor('s').sin().output();

        let options = CompileOptions::default()
            .dim_buckets('s', &[DimBucket::new(1, 1), DimBucket::new(2, 2)])
            .search_graph_limit(32)
            .generation_size(8)
            .mutations(2)
            .search_log(false);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xA66E_6A7E);
        let runtime =
            cx.compile_with_rng(AggregateBucketFilterRuntime::default(), options, &mut rng);

        assert!(
            runtime.aggregate_attempts.len() >= 2,
            "aggregate rejection should trigger a slower finalist combination"
        );
        assert!(
            runtime.aggregate_attempts[0]
                .iter()
                .all(|(fast, safe, _)| *fast > 0 && *safe == 0),
            "the independently fastest all-fast set should be tried first"
        );
        assert_eq!(runtime.loaded_sets.len(), 1);
        let loaded = &runtime.loaded_sets[0];
        assert_eq!(
            loaded
                .iter()
                .filter(|(fast, safe, _)| *fast > 0 && *safe == 0)
                .count(),
            1,
            "best-first fallback should keep one fast bucket"
        );
        assert_eq!(
            loaded.iter().filter(|(_, safe, _)| *safe > 0).count(),
            1,
            "best-first fallback should replace only one conflicting finalist"
        );
    }

    #[test]
    fn compile_builds_search_space_and_searches_it() {
        let mut cx = Graph::new();
        let _ = cx.tensor(1).output();

        let _ = cx.compile(
            TestFilterRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );

        assert_eq!(cx.egraphs.len(), 1);
    }

    #[test]
    fn bucketed_build_search_space_builds_one_egraph_per_bucket() {
        let mut cx = Graph::new();
        let _ = cx.tensor('s').output();

        cx.build_search_space::<TestFilterRuntime>(
            CompileOptions::default()
                .dim_buckets('s', &[DimBucket::new(1, 1), DimBucket::new(2, 4)]),
        );

        assert_eq!(cx.egraphs.len(), 2);
        assert_eq!(cx.egraph_contexts.len(), 2);
        assert_eq!(
            cx.egraph_contexts[0].intervals[&'s'],
            DimInterval::new(1, 1)
        );
        assert_eq!(
            cx.egraph_contexts[1].intervals[&'s'],
            DimInterval::new(2, 4)
        );
    }

    #[test]
    #[should_panic(expected = "search cannot change buckets after build")]
    fn search_with_rng_dim_buckets_after_build_search_space_panics() {
        let mut cx = Graph::new();
        let _ = cx.tensor('s').output();

        cx.build_search_space::<TestFilterRuntime>(CompileOptions::default());
        let mut rng = rand::rng();
        let _ = cx.search_with_rng(
            TestFilterRuntime::default(),
            CompileOptions::default().search_graph_limit(1).dim_buckets(
                's',
                &[DimBucket::new(1, 1), DimBucket::new(2, 4).representative(4)],
            ),
            &mut rng,
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
