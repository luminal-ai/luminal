use crate::egglog_utils::{
    egglog_to_llir, extract_generation, hash_choice_set, hash_egglog_normalized,
    hlir_subgraph_to_egglog, hlir_to_egglog, random_initial_choice, run_egglog, stitch_llir_graphs,
};
use crate::{
    egglog_utils::SerializedEGraph,
    op::{EgglogOp, IntoEgglogOp, LLIROp},
};
use crate::{hlir::CustomOpHLIR, op::*, prelude::*};
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
pub type HLIRGraph = StableGraph<Box<dyn HLIROp>, ShapeTracker>;

/// A group of structurally identical chunks that share the same e-graph.
#[derive(Debug, Clone)]
struct ChunkGroup {
    /// The representative chunk index (used for e-graph building and search)
    representative: usize,
    /// All chunk indices in this group (including the representative)
    members: Vec<usize>,
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
    /// E-Graph search spaces (one per unique group when graph breaks are used)
    egraphs: Vec<SerializedEGraph>,
    /// Subgraph descriptors (one per chunk when graph breaks are used)
    subgraph_descriptors: Vec<SubgraphDescriptor>,
    /// Groups of structurally identical chunks
    chunk_groups: Vec<ChunkGroup>,
    /// Available ops
    pub ops: Option<Vec<Arc<Box<dyn EgglogOp>>>>,
    /// Custom ops
    pub custom_ops: Vec<Box<dyn CustomOp>>,
}

impl Graph {
    /// Create a new graph
    pub fn new() -> Graph {
        Graph::default()
    }

    /// Set a runtime dimension
    pub fn set_dim(&mut self, dimension: char, val: usize) {
        self.dyn_map.insert(dimension, val);
    }

    /// Create a new tensor with shape S
    pub fn tensor(&mut self, shape: impl ToShape) -> GraphTensor {
        self.named_tensor("Tensor", shape)
    }

    /// Create a new tensor with shape S and a name. This name will show up on the graph when displayed
    pub fn named_tensor(&mut self, name: impl ToString, shape: impl ToShape) -> GraphTensor {
        let id = self.graph.add_node(Box::new(crate::hlir::Input {
            node: 0,
            label: name.to_string(),
            dtype: DType::default(),
        }));
        self.get_op_mut::<crate::hlir::Input>(id).node = id.index();
        GraphTensor {
            id,
            graph_ref: self,
            shape: ShapeTracker::new(shape),
            dtype: DType::default(),
        }
    }

    /// Get the sources of a node given it's id
    pub fn get_sources(&self, node_id: NodeIndex) -> Vec<(NodeIndex, ShapeTracker)> {
        self.graph
            .edges_directed(node_id, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| (e.source(), *e.weight()))
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

    /// Add op on the graph, and get back a NewOp
    ///
    /// ```rust
    /// # use luminal::prelude::*;
    /// # let mut cx = Graph::new();
    /// let a = cx.tensor(3);
    /// let b_id = cx
    ///     .add_op(luminal::hlir::Mul::default())
    ///     .input(a.id, a.shape)
    ///     .finish();
    /// let b = GraphTensor::from_id(b_id, a.shape, a.graph(), a.dtype);
    /// ```
    pub fn add_op<O: HLIROp + 'static>(&mut self, op: O) -> NewOp<'_> {
        NewOp {
            new_op_id: self.graph.add_node(Box::new(op)),
            graph_ref: self,
            num_srcs: 0,
        }
    }
    /// Add op on the graph, and get back a NewOp. Just like add_op, except a boxed op is expected.
    pub fn add_boxed_op(&mut self, op: Box<dyn HLIROp + 'static>) -> NewOp<'_> {
        NewOp {
            new_op_id: self.graph.add_node(op),
            graph_ref: self,
            num_srcs: 0,
        }
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
        let mut add = self.add_op(CustomOpHLIR {
            id: self.custom_ops.len() - 1,
            dtype,
        });
        for input in inputs.to_ids() {
            add = add.input(input, ShapeTracker::new(()));
        }
        GraphTensor::from_id(add.finish(), ShapeTracker::new(shape), self, dtype)
    }

    #[tracing::instrument(skip_all)]
    pub fn build_search_space<Rt: Runtime + 'static>(&mut self) {
        let mut ops = Rt::Ops::into_vec();
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>();

        let subgraphs = split_at_graph_breaks(self);

        if subgraphs.len() <= 1 {
            let (program, root) = hlir_to_egglog(self);
            self.egraphs = vec![run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()];
            self.chunk_groups = vec![ChunkGroup {
                representative: 0,
                members: vec![0],
            }];
        } else {
            println!(
                "   {:>6}  {} chunks from graph breaks",
                "Split".cyan().bold(),
                subgraphs.len()
            );
            self.build_grouped_egraphs(&subgraphs, &ops, cleanup_hlir);
        }
        self.subgraph_descriptors = subgraphs;
        self.ops = Some(ops);
    }

    #[tracing::instrument(skip_all)]
    pub fn build_search_space_exclude_ops<Rt: Runtime + 'static, Ex: IntoEgglogOp>(&mut self) {
        let exclude_ops = Ex::into_vec()
            .into_iter()
            .map(|e| e.term().0)
            .collect::<FxHashSet<_>>();
        let mut ops = Rt::Ops::into_vec();
        ops.retain(|o| !exclude_ops.contains(&o.term().0));
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>();

        let subgraphs = split_at_graph_breaks(self);
        if subgraphs.len() <= 1 {
            let (program, root) = hlir_to_egglog(self);
            self.egraphs = vec![run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()];
            self.chunk_groups = vec![ChunkGroup {
                representative: 0,
                members: vec![0],
            }];
        } else {
            self.build_grouped_egraphs(&subgraphs, &ops, cleanup_hlir);
        }
        self.subgraph_descriptors = subgraphs;
        self.ops = Some(ops);
    }

    /// Build e-graphs for multi-chunk subgraphs, grouping structurally identical
    /// chunks and only building one e-graph per unique group.
    fn build_grouped_egraphs(
        &mut self,
        subgraphs: &[SubgraphDescriptor],
        ops: &[Arc<Box<dyn EgglogOp>>],
        cleanup_hlir: bool,
    ) {
        // Get egglog text for each subgraph
        let egglog_texts: Vec<(String, String)> = subgraphs
            .iter()
            .map(|sg| hlir_subgraph_to_egglog(self, sg))
            .collect();

        // Group by normalized egglog hash
        let mut hash_to_chunks: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
        for (i, (text, _)) in egglog_texts.iter().enumerate() {
            let h = hash_egglog_normalized(text);
            hash_to_chunks.entry(h).or_default().push(i);
        }
        let mut groups: Vec<ChunkGroup> = hash_to_chunks
            .into_values()
            .map(|members| ChunkGroup {
                representative: members[0],
                members,
            })
            .collect();
        groups.sort_by_key(|g| g.representative);

        println!(
            "   {:>6}  {} unique groups from {} chunks",
            "Groups".cyan().bold(),
            groups.len(),
            subgraphs.len()
        );

        // Build e-graphs only for representative chunks
        self.egraphs = groups
            .iter()
            .map(|g| {
                let (ref program, ref root) = egglog_texts[g.representative];
                run_egglog(program, root, ops, cleanup_hlir).unwrap()
            })
            .collect();

        self.chunk_groups = groups;
    }

    /// Get a reference to the first e-graph search space (if built)
    pub fn egraph(&self) -> Option<&SerializedEGraph> {
        self.egraphs.first()
    }

    /// Get a reference to the available ops (if search space is built)
    pub fn egglog_ops(&self) -> Option<&Vec<Arc<Box<dyn EgglogOp>>>> {
        self.ops.as_ref()
    }

    const DEFAULT_GENERATION_SIZE: usize = 30;
    const MUTATIONS_PER_OFFSPRING: usize = 30;
    const TRIALS_PER_PROFILE: usize = 10;

    #[tracing::instrument(skip_all)]
    pub fn search<R: Runtime>(&mut self, mut runtime: R, limit: usize) -> R {
        let n_chunks = self.subgraph_descriptors.len();
        let n_groups = self.chunk_groups.len();
        let multi_chunk = n_chunks > 1;
        let ops = self.ops.as_ref().unwrap();
        let start = std::time::Instant::now();

        // Allocate dummy buffers for boundary inputs so groups can be profiled
        for desc in &self.subgraph_descriptors {
            for bi in &desc.boundary_inputs {
                let n_elements = bi
                    .shape
                    .n_elements()
                    .exec(&self.dyn_map)
                    .expect("Failed to resolve boundary input shape");
                runtime.allocate_dummy_input(bi.break_node.index(), n_elements);
            }
        }

        // Search each group's representative
        let mut rng = rand::rng();
        let mut group_best_llirs: Vec<Option<LLIRGraph>> = (0..n_groups).map(|_| None).collect();
        let mut group_best_genomes: Vec<Option<crate::egglog_utils::EGraphChoiceSet>> =
            (0..n_groups).map(|_| None).collect();
        let mut bars_drawn = false;

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

        for (group_idx, group) in self.chunk_groups.iter().enumerate() {
            let egraph = &self.egraphs[group_idx];

            let mut prev_selected: FxHashSet<u64> = FxHashSet::default();
            let mut list_cache = FxHashMap::default();
            let mut expr_cache = FxHashMap::default();

            // Clear intermediate buffers from previous group's profiling
            runtime.clear_intermediate_buffers();

            let mut best_genome = random_initial_choice(egraph, &mut rng);
            prev_selected.insert(hash_choice_set(&best_genome));

            let mut best_graph = egglog_to_llir(
                egraph,
                best_genome.clone(),
                ops,
                &self.custom_ops,
                &mut list_cache,
                &mut expr_cache,
                None,
            );
            let (mut best_metric, display) =
                runtime.profile(&best_graph, &self.dyn_map, Self::TRIALS_PER_PROFILE);

            let mut n_graphs = 1;

            // Print initial result and progress
            {
                let multiplier = if group.members.len() > 1 {
                    format!(" ({}x)", group.members.len())
                } else {
                    String::new()
                };
                let msg = format!(
                    "   {:>8} {}{multiplier}",
                    format!("Group {group_idx}").cyan().bold(),
                    display,
                );
                if bars_drawn {
                    print!("\x1b[1A\r\x1b[2K");
                }
                println!("{msg}");
                if multi_chunk {
                    print!(
                        "\x1b[2K  {:>6}  {} {n_graphs}/{limit}\n\x1b[2K  {:>6}  {} {group_idx}/{n_groups}",
                        "Group".cyan().bold(),
                        make_bar(n_graphs, limit),
                        "Total".cyan().bold(),
                        make_bar(group_idx, n_groups)
                    );
                } else {
                    print!(
                        "\x1b[2K  {:>6}  {} {n_graphs}/{limit}",
                        "Search".cyan().bold(),
                        make_bar(n_graphs, limit),
                    );
                }
                std::io::stdout().flush().unwrap();
                bars_drawn = true;
            }

            while n_graphs < limit {
                let offspring = extract_generation(
                    egraph,
                    &best_genome,
                    (limit - n_graphs).min(Self::DEFAULT_GENERATION_SIZE),
                    Self::MUTATIONS_PER_OFFSPRING,
                    &mut prev_selected,
                    &mut rng,
                );
                if offspring.is_empty() {
                    break;
                }

                for genome in offspring {
                    n_graphs += 1;
                    list_cache.clear();
                    expr_cache.clear();

                    let llir_graph = egglog_to_llir(
                        egraph,
                        genome.clone(),
                        ops,
                        &self.custom_ops,
                        &mut list_cache,
                        &mut expr_cache,
                        None,
                    );

                    // Use catch_unwind to handle CUDA errors from invalid LLIR
                    runtime.clear_intermediate_buffers();
                    let profile_result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            runtime.profile(&llir_graph, &self.dyn_map, Self::TRIALS_PER_PROFILE)
                        }));

                    let (new_metric, display_metric) = match profile_result {
                        Ok(result) => result,
                        Err(_) => {
                            if multi_chunk {
                                print!(
                                    "\x1b[1A\r\x1b[2K  {:>6}  {} {n_graphs}/{limit}\n\x1b[2K  {:>6}  {} {group_idx}/{n_groups}",
                                    "Group".cyan().bold(),
                                    make_bar(n_graphs, limit),
                                    "Total".cyan().bold(),
                                    make_bar(group_idx, n_groups)
                                );
                            } else {
                                print!(
                                    "\r\x1b[2K  {:>6}  {} {n_graphs}/{limit}",
                                    "Search".cyan().bold(),
                                    make_bar(n_graphs, limit),
                                );
                            }
                            std::io::stdout().flush().unwrap();
                            continue;
                        }
                    };

                    let new_best = best_metric.gt(&new_metric);
                    if new_best {
                        best_metric = new_metric;
                        best_graph = llir_graph;
                        best_genome = genome;
                    }

                    if new_best {
                        let msg = format!("   {:>6} {display_metric}", "Searched".green().bold());
                        if multi_chunk {
                            print!("\x1b[1A\r\x1b[2K");
                            println!("{msg}");
                            print!(
                                "\x1b[2K  {:>6}  {} {n_graphs}/{limit}\n\x1b[2K  {:>6}  {} {group_idx}/{n_groups}",
                                "Group".cyan().bold(),
                                make_bar(n_graphs, limit),
                                "Total".cyan().bold(),
                                make_bar(group_idx, n_groups)
                            );
                        } else {
                            print!("\r\x1b[2K");
                            println!("{msg}");
                            print!(
                                "\x1b[2K  {:>6}  {} {n_graphs}/{limit}",
                                "Search".cyan().bold(),
                                make_bar(n_graphs, limit),
                            );
                        }
                    } else if multi_chunk {
                        print!(
                            "\x1b[1A\r\x1b[2K  {:>6}  {} {n_graphs}/{limit}\n\x1b[2K  {:>6}  {} {group_idx}/{n_groups}",
                            "Group".cyan().bold(),
                            make_bar(n_graphs, limit),
                            "Total".cyan().bold(),
                            make_bar(group_idx, n_groups)
                        );
                    } else {
                        print!(
                            "\r\x1b[2K  {:>6}  {} {n_graphs}/{limit}",
                            "Search".cyan().bold(),
                            make_bar(n_graphs, limit),
                        );
                    }
                    std::io::stdout().flush().unwrap();
                }
            }

            group_best_llirs[group_idx] = Some(best_graph);
            group_best_genomes[group_idx] = Some(best_genome);
        }

        // Clear progress bars
        if bars_drawn {
            if multi_chunk {
                print!("\x1b[1A\r\x1b[2K\n\x1b[2K\x1b[1A\r");
            } else {
                print!("\r\x1b[2K");
            }
            std::io::stdout().flush().unwrap();
        }

        // Build per-chunk LLIRs: representative uses searched LLIR,
        // others re-extract from same e-graph/genome with remapped custom op IDs + IO nodes
        let mut chunk_best_llirs: Vec<Option<LLIRGraph>> = (0..n_chunks).map(|_| None).collect();

        for (group_idx, group) in self.chunk_groups.iter().enumerate() {
            let egraph = &self.egraphs[group_idx];
            let genome = group_best_genomes[group_idx].as_ref().unwrap();

            for &chunk_idx in &group.members {
                if chunk_idx == group.representative {
                    continue;
                }
                let (node_remap, custom_op_id_remap) = build_chunk_remaps(
                    &self.subgraph_descriptors[group.representative],
                    &self.subgraph_descriptors[chunk_idx],
                    &self.graph,
                );
                let mut list_cache = FxHashMap::default();
                let mut expr_cache = FxHashMap::default();
                let custom_remap = if custom_op_id_remap.is_empty() {
                    None
                } else {
                    Some(&custom_op_id_remap)
                };
                let mut llir = egglog_to_llir(
                    egraph,
                    genome.clone(),
                    ops,
                    &self.custom_ops,
                    &mut list_cache,
                    &mut expr_cache,
                    custom_remap,
                );
                remap_llir_io_nodes(&mut llir, &node_remap);
                chunk_best_llirs[chunk_idx] = Some(llir);
            }

            // Move the representative's LLIR (avoids clone)
            chunk_best_llirs[group.representative] = group_best_llirs[group_idx].take();
        }

        let chunk_best_llirs: Vec<LLIRGraph> = chunk_best_llirs
            .into_iter()
            .enumerate()
            .map(|(i, opt)| opt.unwrap_or_else(|| panic!("Missing LLIR for chunk {i}")))
            .collect();

        // Stitch chunk LLIRs into a single graph (no-op for single chunk)
        let stitched = stitch_llir_graphs(&chunk_best_llirs, &self.subgraph_descriptors);

        println!(
            "   {:>6}  {} groups ({} chunks) in {}",
            "Searched".green().bold(),
            n_groups,
            n_chunks,
            pretty_duration::pretty_duration(&start.elapsed(), None)
        );

        // Clear stale buffers from chunk profiling before loading the final graph
        runtime.clear_intermediate_buffers();
        runtime.load_llir(&stitched);
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

pub struct NewOp<'a> {
    new_op_id: NodeIndex,
    graph_ref: &'a mut Graph,
    num_srcs: u8,
}

impl NewOp<'_> {
    pub fn finish(self) -> NodeIndex {
        self.new_op_id
    }

    pub fn input(mut self, id: NodeIndex, shape: ShapeTracker) -> Self {
        self.graph_ref.graph.add_edge(id, self.new_op_id, shape);
        self.num_srcs += 1;
        self
    }
}

/// Describes a tensor value crossing a graph break boundary into a chunk.
#[derive(Debug, Clone)]
pub struct BoundaryInput {
    /// The HLIR NodeIndex of the GraphBreak node (unique ID for matching)
    pub break_node: NodeIndex,
    /// Shape of the tensor at the boundary
    pub shape: ShapeTracker,
    /// DType of the tensor at the boundary
    pub dtype: DType,
}

/// Describes a subgraph (chunk) of the HLIR graph between graph breaks.
#[derive(Debug, Clone)]
pub struct SubgraphDescriptor {
    /// HLIR nodes in this chunk (excludes GraphBreak nodes themselves)
    pub nodes: FxHashSet<NodeIndex>,
    /// Boundary inputs entering from prior chunks
    pub boundary_inputs: Vec<BoundaryInput>,
    /// GraphBreak node indices whose predecessor is in this chunk
    pub boundary_outputs: Vec<NodeIndex>,
}

/// Split the HLIR graph at GraphBreak nodes into independent subgraphs.
///
/// Each non-GraphBreak node is assigned to a chunk based on the latest
/// GraphBreak in its transitive dependency chain. Real Input nodes (weights,
/// data) are included in every chunk that uses them.
pub fn split_at_graph_breaks(graph: &Graph) -> Vec<SubgraphDescriptor> {
    use crate::hlir::GraphBreak;

    // Find all GraphBreak nodes
    let break_nodes: FxHashSet<NodeIndex> = graph
        .graph
        .node_indices()
        .filter(|n| graph.try_get_op::<GraphBreak>(*n).is_some())
        .collect();

    if break_nodes.is_empty() {
        // No breaks: single chunk with all nodes
        return vec![SubgraphDescriptor {
            nodes: graph.graph.node_indices().collect(),
            boundary_inputs: vec![],
            boundary_outputs: vec![],
        }];
    }

    // Topological sort of the full graph
    let topo = toposort(&graph.graph, None).expect("HLIR graph has a cycle");

    // Assign each GraphBreak a sequential break index (0, 1, 2, ...)
    // so chunk 0 is before break 0, chunk 1 is between break 0 and break 1, etc.
    let mut break_index: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let mut next_break_idx = 0;
    for &n in &topo {
        if break_nodes.contains(&n) {
            break_index.insert(n, next_break_idx);
            next_break_idx += 1;
        }
    }
    let n_chunks = next_break_idx + 1;

    // Assign each non-break node to a chunk.
    // A node's chunk = max over all predecessors' chunks.
    // A GraphBreak's "output chunk" = break_index + 1 (it pushes successors to the next chunk).
    let mut node_chunk: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    for &n in &topo {
        if break_nodes.contains(&n) {
            continue; // GraphBreak nodes aren't in any chunk
        }
        let mut chunk = 0usize;
        for pred in graph.graph.neighbors_directed(n, Direction::Incoming) {
            if let Some(&bi) = break_index.get(&pred) {
                // Predecessor is a GraphBreak: this node is in chunk bi+1
                chunk = chunk.max(bi + 1);
            } else if let Some(&pred_chunk) = node_chunk.get(&pred) {
                chunk = chunk.max(pred_chunk);
            }
        }
        node_chunk.insert(n, chunk);
    }

    // Build per-chunk node sets.
    // For non-Input nodes: they go to their assigned chunk.
    // For Input nodes (no predecessors): they go to EVERY chunk where they have
    // at least one direct consumer. This avoids polluting chunk 0 with hundreds
    // of weight tensors that are only used by later layers.
    let mut chunk_nodes: Vec<FxHashSet<NodeIndex>> = vec![FxHashSet::default(); n_chunks];

    let input_nodes: FxHashSet<NodeIndex> = graph
        .graph
        .node_indices()
        .filter(|n| graph.try_get_op::<crate::hlir::Input>(*n).is_some())
        .collect();

    for (&node, &chunk) in &node_chunk {
        if input_nodes.contains(&node) {
            continue; // Handle Input nodes separately below
        }
        chunk_nodes[chunk].insert(node);
    }

    // Place each Input node in every chunk that has a consumer of it
    for &inp in &input_nodes {
        let mut target_chunks = FxHashSet::default();
        for consumer in graph.graph.neighbors_directed(inp, Direction::Outgoing) {
            if break_nodes.contains(&consumer) {
                // Consumer is a GraphBreak — the chunk after it needs this input
                let bi = break_index[&consumer];
                let target_chunk = bi + 1;
                if target_chunk < n_chunks {
                    target_chunks.insert(target_chunk);
                }
            } else if let Some(&consumer_chunk) = node_chunk.get(&consumer) {
                target_chunks.insert(consumer_chunk);
            }
        }
        // If no consumers found (orphan input), put it in chunk 0
        if target_chunks.is_empty() {
            target_chunks.insert(0);
        }
        for chunk in target_chunks {
            chunk_nodes[chunk].insert(inp);
        }
    }

    // Closure: for each chunk, ensure all transitive predecessors of its nodes
    // are included (except GraphBreak boundary nodes). This handles shared
    // computation nodes (e.g. RoPE frequencies, sqrt(d_k)) that are assigned to
    // an earlier chunk but used by later chunks.
    for chunk_node in chunk_nodes.iter_mut().take(n_chunks) {
        let mut to_visit: Vec<NodeIndex> = chunk_node.iter().copied().collect();
        while let Some(node) = to_visit.pop() {
            for pred in graph.graph.neighbors_directed(node, Direction::Incoming) {
                if break_nodes.contains(&pred) {
                    continue; // Boundary handled separately
                }
                if chunk_node.insert(pred) {
                    // Newly added — also visit its predecessors
                    to_visit.push(pred);
                }
            }
        }
    }

    // Build SubgraphDescriptors
    let mut descriptors: Vec<SubgraphDescriptor> = Vec::with_capacity(n_chunks);
    for (chunk_idx, chunk_node) in chunk_nodes.iter().enumerate().take(n_chunks) {
        let nodes = chunk_node.clone();

        // Boundary inputs: GraphBreak nodes whose successors are in this chunk
        let mut boundary_inputs = vec![];
        for &brk in &break_nodes {
            let bi = break_index[&brk];
            if bi + 1 == chunk_idx {
                // This break feeds into this chunk
                // Get shape/dtype from the incoming edge of the GraphBreak
                let edge = graph
                    .graph
                    .edges_directed(brk, Direction::Incoming)
                    .next()
                    .expect("GraphBreak must have exactly one input");
                let shape = *edge.weight();
                // Get dtype from the predecessor
                let pred = edge.source();
                let dtype = if let Some(inp) = graph.try_get_op::<crate::hlir::Input>(pred) {
                    inp.dtype
                } else {
                    DType::F32 // Default; most intermediate values are F32
                };
                boundary_inputs.push(BoundaryInput {
                    break_node: brk,
                    shape,
                    dtype,
                });
            }
        }

        // Boundary outputs: GraphBreak nodes whose predecessor is in this chunk
        let mut boundary_outputs = vec![];
        for &brk in &break_nodes {
            let bi = break_index[&brk];
            if bi + 1 == chunk_idx + 1 {
                // This break's predecessor should be in this chunk
                let pred = graph
                    .graph
                    .neighbors_directed(brk, Direction::Incoming)
                    .next()
                    .expect("GraphBreak must have exactly one input");
                if nodes.contains(&pred) || node_chunk.get(&pred) == Some(&chunk_idx) {
                    boundary_outputs.push(brk);
                }
            }
        }

        descriptors.push(SubgraphDescriptor {
            nodes,
            boundary_inputs,
            boundary_outputs,
        });
    }

    descriptors
}

/// Clone a representative chunk's LLIR graph and remap Input/Output node indices
/// to match a different (structurally identical) target chunk.
///
/// The remapping handles three categories of nodes:
/// 1. **Boundary inputs**: Matched positionally via SubgraphDescriptor.boundary_inputs
/// 2. **Boundary outputs**: Matched positionally via SubgraphDescriptor.boundary_outputs
/// 3. **Weight/data inputs**: Chunk-specific Input nodes (in one subgraph but not the other) are sorted by index and matched positionally.
///
/// Shared Input nodes (same NodeIndex in both subgraphs) need no remapping.
/// Build remapping tables for cloning a representative chunk's LLIR to a target chunk.
///
/// Returns:
/// - `node_remap`: maps rep HLIR node indices → target HLIR node indices (for Input/Output nodes)
/// - `custom_op_id_remap`: maps rep CustomOpHLIR IDs → target CustomOpHLIR IDs
fn build_chunk_remaps(
    rep_desc: &SubgraphDescriptor,
    target_desc: &SubgraphDescriptor,
    hlir_graph: &HLIRGraph,
) -> (FxHashMap<usize, usize>, FxHashMap<usize, usize>) {
    let mut node_remap: FxHashMap<usize, usize> = FxHashMap::default();

    // 1. Boundary inputs: positional match
    for (r, t) in rep_desc
        .boundary_inputs
        .iter()
        .zip(&target_desc.boundary_inputs)
    {
        node_remap.insert(r.break_node.index(), t.break_node.index());
    }

    // 2. Boundary outputs: positional match
    for (r, t) in rep_desc
        .boundary_outputs
        .iter()
        .zip(&target_desc.boundary_outputs)
    {
        node_remap.insert(r.index(), t.index());
    }

    // 3. Weight/data inputs: chunk-specific Input HLIR nodes
    let rep_input_nodes: FxHashSet<NodeIndex> = rep_desc
        .nodes
        .iter()
        .filter(|n| {
            hlir_graph
                .node_weight(**n)
                .and_then(|w| w.as_any().downcast_ref::<crate::hlir::Input>())
                .is_some()
        })
        .copied()
        .collect();
    let target_input_nodes: FxHashSet<NodeIndex> = target_desc
        .nodes
        .iter()
        .filter(|n| {
            hlir_graph
                .node_weight(**n)
                .and_then(|w| w.as_any().downcast_ref::<crate::hlir::Input>())
                .is_some()
        })
        .copied()
        .collect();

    // Chunk-specific = in one subgraph but not the other
    let rep_specific: Vec<usize> = rep_input_nodes
        .difference(&target_input_nodes)
        .map(|n| n.index())
        .sorted()
        .collect();
    let target_specific: Vec<usize> = target_input_nodes
        .difference(&rep_input_nodes)
        .map(|n| n.index())
        .sorted()
        .collect();

    assert_eq!(
        rep_specific.len(),
        target_specific.len(),
        "Chunk-specific input count mismatch: rep has {}, target has {}",
        rep_specific.len(),
        target_specific.len()
    );
    for (r, t) in rep_specific.iter().zip(&target_specific) {
        node_remap.insert(*r, *t);
    }

    // 4. CustomOpHLIR ID remapping: match positionally by sorted HLIR node index
    let mut custom_op_id_remap: FxHashMap<usize, usize> = FxHashMap::default();
    let rep_custom_ops: Vec<usize> = rep_desc
        .nodes
        .iter()
        .filter_map(|n| {
            hlir_graph
                .node_weight(*n)
                .and_then(|w| w.as_any().downcast_ref::<crate::hlir::CustomOpHLIR>())
                .map(|op| op.id)
        })
        .sorted()
        .collect();
    let target_custom_ops: Vec<usize> = target_desc
        .nodes
        .iter()
        .filter_map(|n| {
            hlir_graph
                .node_weight(*n)
                .and_then(|w| w.as_any().downcast_ref::<crate::hlir::CustomOpHLIR>())
                .map(|op| op.id)
        })
        .sorted()
        .collect();
    assert_eq!(
        rep_custom_ops.len(),
        target_custom_ops.len(),
        "CustomOpHLIR count mismatch: rep has {}, target has {}",
        rep_custom_ops.len(),
        target_custom_ops.len()
    );
    for (r, t) in rep_custom_ops.iter().zip(&target_custom_ops) {
        if r != t {
            custom_op_id_remap.insert(*r, *t);
        }
    }

    (node_remap, custom_op_id_remap)
}

/// Apply Input/Output node index remapping to an LLIR graph (in-place modification).
fn remap_llir_io_nodes(llir: &mut LLIRGraph, node_remap: &FxHashMap<usize, usize>) {
    // We need to replace nodes in-place. Collect node indices first.
    let node_indices: Vec<NodeIndex> = llir.node_indices().collect();
    for node_idx in node_indices {
        let op = &llir[node_idx];
        let new_op = if let Some(input_op) = op.to_op::<crate::hlir::Input>() {
            if let Some(&new_node) = node_remap.get(&input_op.node) {
                Some(LLIROp::new::<crate::hlir::Input>(Box::new(
                    crate::hlir::Input {
                        node: new_node,
                        label: input_op.label.clone(),
                        dtype: input_op.dtype,
                    },
                )))
            } else {
                None
            }
        } else if let Some(output_op) = op.to_op::<crate::hlir::Output>() {
            if let Some(&new_node) = node_remap.get(&output_op.node) {
                Some(LLIROp::new::<crate::hlir::Output>(Box::new(
                    crate::hlir::Output { node: new_node },
                )))
            } else {
                None
            }
        } else {
            None
        };
        if let Some(new_op) = new_op {
            llir[node_idx] = new_op;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egglog_utils::hash_egglog_normalized;

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
    fn test_build_chunk_remaps_and_remap_io() {
        use petgraph::stable_graph::NodeIndex as NI;

        // Build a simple LLIR: Input(node=10) -> Output(node=20), Input(node=100) -> Output(node=20)
        let mut llir = LLIRGraph::default();
        let input_node = llir.add_node(LLIROp::new::<crate::hlir::Input>(Box::new(
            crate::hlir::Input {
                node: 10,
                label: "boundary".to_string(),
                dtype: DType::F32,
            },
        )));
        let weight_node = llir.add_node(LLIROp::new::<crate::hlir::Input>(Box::new(
            crate::hlir::Input {
                node: 100,
                label: "weight_a".to_string(),
                dtype: DType::F32,
            },
        )));
        let output_node = llir.add_node(LLIROp::new::<crate::hlir::Output>(Box::new(
            crate::hlir::Output { node: 20 },
        )));
        llir.add_edge(input_node, output_node, ());
        llir.add_edge(weight_node, output_node, ());

        // Build minimal HLIR graph with Input nodes for both rep and target
        let mut hlir_graph = HLIRGraph::default();
        while hlir_graph.node_count() < 10 {
            hlir_graph.add_node(Box::new(crate::hlir::Input {
                node: 0,
                label: "pad".to_string(),
                dtype: DType::F32,
            }));
        }
        hlir_graph.add_node(Box::new(crate::hlir::Input {
            node: 10,
            label: "boundary".to_string(),
            dtype: DType::F32,
        }));
        while hlir_graph.node_count() < 20 {
            hlir_graph.add_node(Box::new(crate::hlir::Input {
                node: 0,
                label: "pad".to_string(),
                dtype: DType::F32,
            }));
        }
        hlir_graph.add_node(Box::new(crate::hlir::Input {
            node: 20,
            label: "brk".to_string(),
            dtype: DType::F32,
        }));
        while hlir_graph.node_count() < 50 {
            hlir_graph.add_node(Box::new(crate::hlir::Input {
                node: 0,
                label: "pad".to_string(),
                dtype: DType::F32,
            }));
        }
        hlir_graph.add_node(Box::new(crate::hlir::Input {
            node: 50,
            label: "boundary".to_string(),
            dtype: DType::F32,
        }));
        while hlir_graph.node_count() < 60 {
            hlir_graph.add_node(Box::new(crate::hlir::Input {
                node: 0,
                label: "pad".to_string(),
                dtype: DType::F32,
            }));
        }
        hlir_graph.add_node(Box::new(crate::hlir::Input {
            node: 60,
            label: "brk".to_string(),
            dtype: DType::F32,
        }));
        while hlir_graph.node_count() < 100 {
            hlir_graph.add_node(Box::new(crate::hlir::Input {
                node: 0,
                label: "pad".to_string(),
                dtype: DType::F32,
            }));
        }
        hlir_graph.add_node(Box::new(crate::hlir::Input {
            node: 100,
            label: "weight_a".to_string(),
            dtype: DType::F32,
        }));
        while hlir_graph.node_count() < 200 {
            hlir_graph.add_node(Box::new(crate::hlir::Input {
                node: 0,
                label: "pad".to_string(),
                dtype: DType::F32,
            }));
        }
        hlir_graph.add_node(Box::new(crate::hlir::Input {
            node: 200,
            label: "weight_b".to_string(),
            dtype: DType::F32,
        }));

        let rep_desc = SubgraphDescriptor {
            nodes: [NI::new(10), NI::new(100)].into_iter().collect(),
            boundary_inputs: vec![BoundaryInput {
                break_node: NI::new(10),
                shape: ShapeTracker::new(()),
                dtype: DType::F32,
            }],
            boundary_outputs: vec![NI::new(20)],
        };
        let target_desc = SubgraphDescriptor {
            nodes: [NI::new(50), NI::new(200)].into_iter().collect(),
            boundary_inputs: vec![BoundaryInput {
                break_node: NI::new(50),
                shape: ShapeTracker::new(()),
                dtype: DType::F32,
            }],
            boundary_outputs: vec![NI::new(60)],
        };

        let (node_remap, custom_op_remap) =
            build_chunk_remaps(&rep_desc, &target_desc, &hlir_graph);

        // No custom ops in this test
        assert!(custom_op_remap.is_empty());

        // Apply IO remap
        remap_llir_io_nodes(&mut llir, &node_remap);

        // Verify remapped nodes
        let mut input_nodes: Vec<(usize, String)> = vec![];
        let mut output_nodes: Vec<usize> = vec![];
        for node in llir.node_indices() {
            let op = &llir[node];
            if let Some(inp) = op.to_op::<crate::hlir::Input>() {
                input_nodes.push((inp.node, inp.label.clone()));
            }
            if let Some(out) = op.to_op::<crate::hlir::Output>() {
                output_nodes.push(out.node);
            }
        }
        input_nodes.sort_by_key(|(n, _)| *n);
        assert_eq!(input_nodes.len(), 2);
        assert_eq!(
            input_nodes[0].0, 50,
            "Boundary input should be remapped to 50"
        );
        assert_eq!(
            input_nodes[1].0, 200,
            "Weight input should be remapped to 200"
        );
        assert_eq!(output_nodes, vec![60], "Output should be remapped to 60");
        assert_eq!(llir.edge_count(), 2, "Should have 2 edges");
    }

    #[test]
    fn test_hash_egglog_normalized_custom_op_id() {
        // CustomOpHLIR lines differ only in the integer ID (layer index)
        let text_a = r#"(let t0 (Input 441 "boundary" (F32)))
(let t1 (CustomOpHLIR (ICons t74 (ICons t120 (ICons t28 (INil)))) 1 (F32)))
(let t2 (Output t1 585))
"#;
        let text_b = r#"(let t0 (Input 585 "boundary" (F32)))
(let t1 (CustomOpHLIR (ICons t74 (ICons t120 (ICons t28 (INil)))) 2 (F32)))
(let t2 (Output t1 729))
"#;
        assert_eq!(
            hash_egglog_normalized(text_a),
            hash_egglog_normalized(text_b),
            "CustomOpHLIR with different IDs should hash the same"
        );
    }

    #[test]
    fn test_hash_egglog_normalized_custom_op_different_structure() {
        // CustomOpHLIR lines with different input lists should hash differently
        let text_a = "(let t1 (CustomOpHLIR (ICons t74 (ICons t120 (INil))) 1 (F32)))\n";
        let text_b = "(let t1 (CustomOpHLIR (ICons t74 (ICons t99 (INil))) 1 (F32)))\n";
        assert_ne!(
            hash_egglog_normalized(text_a),
            hash_egglog_normalized(text_b),
            "CustomOpHLIR with different input lists should hash differently"
        );
    }
}
