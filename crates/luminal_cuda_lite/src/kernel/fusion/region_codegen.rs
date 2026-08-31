// =========================================================================
// Region codegen for FusionStart / FusionEnd-bracketed fused regions.
//
// Older fusion lowering left elementwise / FusionStart / FusionEnd nodes in the post-extraction
// LLIR, each compiling to its own standalone CUDA kernel. PR2 collapses
// every FusionEnd-rooted region into ONE fused CUDA kernel at codegen
// time — without rewriting the LLIR.
//
// Pipeline:
//   `kernel_to_host` builds a Vec<CompileUnit> from the topo order:
//     - CompileUnit::Single(node)  — unfused non-region kernels, compiled as before.
//     - CompileUnit::Region(rgn)   — one FE + its interior elementwise DAG +
//                                    its FS leaves. Compiled here as a
//                                    single CUDA kernel that reads from
//                                    the region's external inputs once,
//                                    chains all elementwise bodies through
//                                    register-resident locals, and writes
//                                    the FE's output.
//
// The CompiledKernel for a Region is keyed on the FE node and stores
// `inputs = external producer NodeIndices` (one per interior FusionStart),
// so the existing buffer-pointer wiring in to_host.rs picks up the right
// device pointers at execute time. Interior Cuda*Elementwise / FusionStart nodes
// never enter the kernels Vec — they have no buffers, no launches.
// =========================================================================

use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    graph::LLIRGraph,
    prelude::{
        petgraph::{Direction, algo::toposort, visit::EdgeRef},
        *,
    },
};

use as_any::Downcast;
use rustc_hash::FxHasher;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::KernelOp,
    kernel::fusion::elementwise::{CudaBinaryElementwise, CudaUnaryElementwise},
    kernel::fusion::markers::{FusionEnd, FusionStart},
    kernel::hlir::{dtype_includes, generate_dyn_dims_defines},
};

// =========================================================================
// Compile units — what `kernel_to_host` iterates over instead of nodes.
// =========================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RegionUnit {
    /// The FusionEnd node that anchors this region.
    pub fe_node: NodeIndex,
    /// Interior Cuda*Elementwise nodes, in topological order (predecessors before
    /// consumers). Used to emit register-binding statements in dependency
    /// order in the fused CUDA kernel body.
    pub elementwise_topo: Vec<NodeIndex>,
    /// FusionStart nodes that bound the region's leaves. One per external
    /// read site — duplicates (different FS LLIR nodes wrapping the same
    /// upstream tensor) are kept separate so each read uses its own
    /// strides; the host launch passes the same device pointer twice.
    pub fs_nodes: Vec<NodeIndex>,
    /// External producer NodeIndices, one per `fs_nodes` entry in the same
    /// order. Becomes the `inputs` field of the FE's `CompiledKernel`, and
    /// the kernel function's `in0`, `in1`, ... parameters in that order.
    pub external_inputs: Vec<NodeIndex>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CompileUnit {
    Single(NodeIndex),
    Region(RegionUnit),
}

impl CompileUnit {
    pub(crate) fn root_node(&self) -> NodeIndex {
        match self {
            Self::Single(node) => *node,
            Self::Region(region) => region.fe_node,
        }
    }
}

#[derive(Debug)]
pub(crate) struct PreparedRegionKernel {
    pub source: Arc<str>,
    pub output_size: Expression,
}

#[derive(Debug, Default)]
pub(crate) struct RegionSourceCache {
    sources: FxHashMap<u64, Vec<CachedRegionSource>>,
    hits: usize,
    misses: usize,
}

impl RegionSourceCache {
    pub(crate) fn counters(&self) -> (usize, usize) {
        (self.hits, self.misses)
    }
}

#[derive(Debug)]
struct CachedRegionSource {
    key: SingletonRegionProgramKey,
    source: Arc<str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FusionStartProgramKey {
    strides: Vec<Expression>,
    dtype: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum SingletonElementProgramKey {
    Unary { opcode: u8, dtype: u8 },
    Binary { opcode: u8, dtype: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SingletonRegionProgramKey {
    global_dyn_dims: Vec<Symbol>,
    output_shape: Vec<Expression>,
    output_strides: Vec<Expression>,
    output_dtype: u8,
    inputs: Vec<FusionStartProgramKey>,
    element: SingletonElementProgramKey,
    operand_slots: Vec<u16>,
}

// The singleton egglog rules copy the operation's shape/output strides into
// FusionEnd, and copy each input stride into its FusionStart. Consequently the
// fields above are a complete source descriptor: repeating the elementwise
// shape/stride metadata would only make every cache hit hash it twice.

/// Candidate-local fusion work shared by static resource validation and the
/// subsequent compilation of that exact LLIR. Node indices remain valid when
/// `compile_bucket` clones the graph because `StableGraph::clone` preserves
/// them.
#[derive(Debug)]
pub(crate) struct PreparedFusionPlan {
    compile_units: Vec<CompileUnit>,
    region_kernels: FxHashMap<NodeIndex, PreparedRegionKernel>,
    absorbed_markers: FxHashSet<NodeIndex>,
}

impl PreparedFusionPlan {
    pub(crate) fn discover(kernel_topo: &[NodeIndex], llir: &LLIRGraph) -> Self {
        let (compile_units, absorbed) = build_compile_units(kernel_topo, llir);
        Self {
            compile_units,
            region_kernels: FxHashMap::default(),
            absorbed_markers: absorbed,
        }
    }

    pub(crate) fn prepare_region_kernels_for(
        &mut self,
        nodes: &FxHashSet<NodeIndex>,
        llir: &LLIRGraph,
        source_cache: &mut RegionSourceCache,
        global_dyn_dims: &[Symbol],
    ) {
        for unit in &self.compile_units {
            let CompileUnit::Region(region) = unit else {
                continue;
            };
            if !nodes.contains(&region.fe_node) || self.region_kernels.contains_key(&region.fe_node)
            {
                continue;
            }
            let output_size = region_output_size(region, llir);
            let source = if let Some(fingerprint) =
                singleton_program_fingerprint(region, llir, global_dyn_dims)
            {
                let cached = source_cache.sources.get(&fingerprint).and_then(|entries| {
                    entries
                        .iter()
                        .find(|entry| {
                            singleton_program_matches(&entry.key, region, llir, global_dyn_dims)
                        })
                        .map(|entry| Arc::clone(&entry.source))
                });
                if let Some(source) = cached {
                    source_cache.hits += 1;
                    source
                } else {
                    let (source, rendered_output_size) = region_kernel_source(region, llir);
                    debug_assert_eq!(rendered_output_size, output_size);
                    let source: Arc<str> = Arc::from(source);
                    let key = singleton_program_key(region, llir, global_dyn_dims)
                        .expect("fingerprinted singleton region must have an owned program key");
                    source_cache
                        .sources
                        .entry(fingerprint)
                        .or_default()
                        .push(CachedRegionSource {
                            key,
                            source: Arc::clone(&source),
                        });
                    source_cache.misses += 1;
                    source
                }
            } else {
                Arc::from(region_kernel_source(region, llir).0)
            };
            self.region_kernels.insert(
                region.fe_node,
                PreparedRegionKernel {
                    source,
                    output_size,
                },
            );
        }
    }

    pub(crate) fn compile_units(&self) -> &[CompileUnit] {
        &self.compile_units
    }

    pub(crate) fn compile_units_for<'a>(
        &'a self,
        nodes: &'a FxHashSet<NodeIndex>,
    ) -> impl Iterator<Item = &'a CompileUnit> + 'a {
        self.compile_units
            .iter()
            .filter(|unit| nodes.contains(&unit.root_node()))
    }

    pub(crate) fn region_kernel(&self, fe_node: NodeIndex) -> Option<&PreparedRegionKernel> {
        self.region_kernels.get(&fe_node)
    }

    pub(crate) fn absorbed_markers(&self) -> &FxHashSet<NodeIndex> {
        &self.absorbed_markers
    }
}

// =========================================================================
// Region detection.
// =========================================================================

/// Group a sub-DAG's topo order into compile units. Each FusionEnd node
/// becomes the root of a `CompileUnit::Region`; the region's interior
/// Cuda*Elementwise and FusionStart nodes are absorbed into that region and removed
/// from the per-node iteration. Anything else is wrapped in
/// `CompileUnit::Single`.
pub(crate) fn build_compile_units(
    topo_order: &[NodeIndex],
    llir_graph: &LLIRGraph,
) -> (Vec<CompileUnit>, FxHashSet<NodeIndex>) {
    let name_of = |idx: NodeIndex| -> Option<&'static str> {
        llir_graph
            .node_weight(idx)
            .and_then(|op| op.to_dialect::<dyn KernelOp>().map(|k| k.kernel_name()))
    };

    // First pass: every FusionEnd in the subgraph anchors a region; gather
    // the region's interior + FS leaves by walking incoming edges
    // backward, stopping at FusionStart (a leaf — its predecessor is the
    // external producer, outside the region).
    let mut absorbed: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut regions: FxHashMap<NodeIndex, RegionUnit> = FxHashMap::default();

    for &node in topo_order {
        if name_of(node) != Some("FusionEnd") {
            continue;
        }

        if let Some(region) = singleton_region(llir_graph, node) {
            absorbed.extend(region.elementwise_topo.iter().copied());
            absorbed.extend(region.fs_nodes.iter().copied());
            regions.insert(node, region);
            continue;
        }

        let mut interior: Vec<NodeIndex> = Vec::new();
        let mut fs_nodes: Vec<NodeIndex> = Vec::new();
        let mut visited: FxHashSet<NodeIndex> = FxHashSet::default();
        let mut stack: Vec<NodeIndex> = Vec::new();
        stack.push(node);
        visited.insert(node);

        while let Some(cur) = stack.pop() {
            for pred in llir_graph.neighbors_directed(cur, Direction::Incoming) {
                if !visited.insert(pred) {
                    continue;
                }
                match name_of(pred) {
                    Some("FusionStart") => {
                        fs_nodes.push(pred);
                        // Don't recurse past FS — its predecessor is
                        // external (outside the region).
                    }
                    Some("FusionEnd") => {
                        // Fusion rewrites do not create direct nested ends.
                        // Keep walking defensively so this discovery helper
                        // remains total for hand-built test graphs.
                        absorbed.insert(pred);
                        stack.push(pred);
                    }
                    Some(_) if is_region_elementwise(llir_graph, pred) => {
                        interior.push(pred);
                        stack.push(pred);
                    }
                    _ => {
                        // Non-marker, non-elementwise predecessor inside what
                        // we thought was a region. Shouldn't happen with
                        // the current rules; treat conservatively: do
                        // not absorb it. This means the region is
                        // malformed and we likely should not have a
                        // region at all; caller will see incomplete
                        // interior.
                    }
                }
            }
        }

        // Canonical orders for interior + FS nodes. `egglog_to_llir`
        // reissues NodeIndexes for every search candidate, so any
        // NodeIndex-driven order (like the previous global-toposort
        // filter) renumbers the kernel's inputs and locals across
        // candidates, defeating the source-keyed compile cache for
        // regions that are structurally identical. Order by content
        // instead — see `canonicalize_region`.
        let (interior_topo, fs_topo) = canonicalize_region(llir_graph, node, &interior, &fs_nodes);

        // External producer for each FS leaf, in the same order.
        let external_inputs: Vec<NodeIndex> = fs_topo
            .iter()
            .map(|&fs| {
                llir_graph
                    .neighbors_directed(fs, Direction::Incoming)
                    .next()
                    .unwrap_or_else(|| {
                        // Dump the malformed structure: which FE
                        // triggered the walk. A malformed region (an FS leaf
                        // with no external producer) should never reach here.
                        panic!("FusionStart with no predecessor")
                    })
            })
            .collect();

        absorbed.extend(interior_topo.iter().copied());
        absorbed.extend(fs_topo.iter().copied());

        regions.insert(
            node,
            RegionUnit {
                fe_node: node,
                elementwise_topo: interior_topo,
                fs_nodes: fs_topo,
                external_inputs,
            },
        );
    }

    // Second pass: emit compile units in original topo order, replacing
    // FE nodes with their RegionUnit and skipping anything absorbed —
    // by any region in the LLIR. Discovery receives the global kernel topo
    // order, so shared FS markers are present in this one absorbed set even
    // when convex packaging later places their consumers in another subgraph.
    let mut units: Vec<CompileUnit> = Vec::new();
    for &node in topo_order {
        if let Some(region) = regions.remove(&node) {
            units.push(CompileUnit::Region(region));
        } else if absorbed.contains(&node) {
            continue;
        } else {
            units.push(CompileUnit::Single(node));
        }
    }
    (units, absorbed)
}

/// Fast path for the only fusion shape currently constructed by the rewrites:
/// one unary/binary elementwise op bracketed by FusionStart leaves and a
/// FusionEnd. Avoid the generic region DAG's maps, heap, structural-hash pass,
/// and duplicate indexing-expression rendering for these tiny regions.
fn singleton_region(llir_graph: &LLIRGraph, fe_node: NodeIndex) -> Option<RegionUnit> {
    let mut fe_inputs = llir_graph.neighbors_directed(fe_node, Direction::Incoming);
    let elementwise = fe_inputs.next()?;
    if fe_inputs.next().is_some() || !is_region_elementwise(llir_graph, elementwise) {
        return None;
    }

    let elem_op = llir_graph[elementwise].to_dialect::<dyn KernelOp>()?;
    let expected_inputs = if (***elem_op)
        .downcast_ref::<CudaUnaryElementwise>()
        .is_some()
    {
        1
    } else if (***elem_op)
        .downcast_ref::<CudaBinaryElementwise>()
        .is_some()
    {
        2
    } else {
        return None;
    };

    let mut fs_nodes: Vec<NodeIndex> = llir_graph
        .neighbors_directed(elementwise, Direction::Incoming)
        .collect();
    if fs_nodes.len() != expected_inputs
        || fs_nodes.iter().any(|&node| {
            llir_graph[node]
                .to_dialect::<dyn KernelOp>()
                .and_then(|op| (***op).downcast_ref::<FusionStart>())
                .is_none()
        })
    {
        return None;
    }

    // All currently constructed binary region ops are commutative. Order
    // leaves by their complete structural metadata so source and launch ABI
    // stay NodeIndex-independent across candidate extraction.
    fs_nodes.sort_by(|&a, &b| compare_fusion_starts(llir_graph, a, b));
    // Congruence can make both input edges of `x op x` reference one FS node.
    // A region has one load/local/launch argument per distinct FS, while both
    // elementwise edges consume that same local.
    fs_nodes.dedup();
    let external_inputs = fs_nodes
        .iter()
        .map(|&fs| {
            llir_graph
                .neighbors_directed(fs, Direction::Incoming)
                .next()
                .expect("FusionStart with no predecessor")
        })
        .collect();

    Some(RegionUnit {
        fe_node,
        elementwise_topo: vec![elementwise],
        fs_nodes,
        external_inputs,
    })
}

fn compare_fusion_starts(llir_graph: &LLIRGraph, a: NodeIndex, b: NodeIndex) -> std::cmp::Ordering {
    let fusion_start = |node: NodeIndex| {
        let op = llir_graph[node].to_dialect::<dyn KernelOp>().unwrap();
        (***op).downcast_ref::<FusionStart>().unwrap()
    };
    let a_start = fusion_start(a);
    let b_start = fusion_start(b);
    a_start
        .strides
        .iter()
        .map(|expression| expression.to_kernel())
        .cmp(
            b_start
                .strides
                .iter()
                .map(|expression| expression.to_kernel()),
        )
        .then_with(|| cuda_dtype(a_start.dtype).cmp(cuda_dtype(b_start.dtype)))
        .then_with(|| a.index().cmp(&b.index()))
}

// =========================================================================
// Region canonicalization.
//
// The emitted kernel string must be a function of the region's *structure*
// only, never of NodeIndexes: every search candidate reissues NodeIndexes,
// and structurally identical regions recur constantly across candidates
// (one gemma search was measured compiling 200k+ kernels where ~20% were
// the same program with inputs/locals renumbered by NodeIndex churn).
// =========================================================================

/// Structural hash per region node. Captures exactly the text-relevant
/// content: FS leaves hash (read index expression, dtype); interior
/// elementwise nodes hash (op name, dtype, child hashes). Child hashes are
/// sorted — the only binary region ops, Add and Mul, are commutative, so
/// operand order is presentation, not structure. NodeIndexes never enter a
/// hash.
fn region_structural_hashes(
    llir_graph: &LLIRGraph,
    fe_node: NodeIndex,
    interior: &[NodeIndex],
    fs_nodes: &[NodeIndex],
) -> FxHashMap<NodeIndex, u64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let fe_op = llir_graph[fe_node].to_dialect::<dyn KernelOp>().unwrap();
    let fe_struct: &FusionEnd = (***fe_op)
        .downcast_ref::<FusionEnd>()
        .expect("region root must be FusionEnd");
    let out_shape: &[Expression] = &fe_struct.shape;

    let mut hashes: FxHashMap<NodeIndex, u64> = FxHashMap::default();
    for &fs in fs_nodes {
        let fs_op = llir_graph[fs].to_dialect::<dyn KernelOp>().unwrap();
        let fs_struct: &FusionStart = (***fs_op).downcast_ref::<FusionStart>().unwrap();
        let read_idx = flatten_strides(out_shape, &fs_struct.strides).to_kernel();
        let mut h = DefaultHasher::new();
        ("FS", read_idx.as_str(), cuda_dtype(fs_struct.dtype)).hash(&mut h);
        hashes.insert(fs, h.finish());
    }

    // Interior nodes bottom-up in one Kahn pass over the region-induced
    // subgraph (in-degree counts only in-region predecessors, so FS
    // leaves and external/malformed predecessors never gate readiness —
    // the latter hash as a constant tag). O(V + E); rolled prefill
    // regions have thousands of interior nodes in long chains, so
    // anything multi-pass is quadratic and stalls the search.
    let interior_set: FxHashSet<NodeIndex> = interior.iter().copied().collect();
    let mut indeg: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    for &n in interior {
        let d = llir_graph
            .neighbors_directed(n, Direction::Incoming)
            .filter(|p| interior_set.contains(p))
            .count();
        indeg.insert(n, d);
    }
    let mut queue: std::collections::VecDeque<NodeIndex> =
        interior.iter().copied().filter(|n| indeg[n] == 0).collect();
    while let Some(n) = queue.pop_front() {
        let mut child_hashes: Vec<u64> = llir_graph
            .neighbors_directed(n, Direction::Incoming)
            .map(|src| hashes.get(&src).copied().unwrap_or(0x4558_5445_524e_414c)) // "EXTERNAL"
            .collect();
        child_hashes.sort_unstable();
        let op_ref = llir_graph[n].to_dialect::<dyn KernelOp>().unwrap();
        let (op_name, dt) = if let Some(e) = (***op_ref).downcast_ref::<CudaUnaryElementwise>() {
            (e.op.as_str(), e.dtype)
        } else if let Some(e) = (***op_ref).downcast_ref::<CudaBinaryElementwise>() {
            (e.op.as_str(), e.dtype)
        } else {
            (op_ref.kernel_name(), op_ref.output_dtype())
        };
        let mut h = DefaultHasher::new();
        (op_name, cuda_dtype(dt), &child_hashes).hash(&mut h);
        hashes.insert(n, h.finish());
        for succ in llir_graph.neighbors_directed(n, Direction::Outgoing) {
            if let Some(d) = indeg.get_mut(&succ) {
                *d -= 1;
                if *d == 0 {
                    queue.push_back(succ);
                }
            }
        }
    }
    hashes
}

/// Canonical orders for a region's interior and FS nodes:
/// - interior: topological (Kahn over the region-induced subgraph), ties
///   broken by structural hash;
/// - FS leaves: sorted by (read index expression, dtype), ties broken by
///   first use in the canonical body. Two FS leaves tied on all keys are
///   textually interchangeable loads feeding commutative ops, so their
///   relative order cannot change the emitted kernel.
fn canonicalize_region(
    llir_graph: &LLIRGraph,
    fe_node: NodeIndex,
    interior: &[NodeIndex],
    fs_nodes: &[NodeIndex],
) -> (Vec<NodeIndex>, Vec<NodeIndex>) {
    let hashes = region_structural_hashes(llir_graph, fe_node, interior, fs_nodes);
    let interior_set: FxHashSet<NodeIndex> = interior.iter().copied().collect();

    let mut indeg: FxHashMap<NodeIndex, usize> = interior
        .iter()
        .map(|&n| {
            let d = llir_graph
                .neighbors_directed(n, Direction::Incoming)
                .filter(|p| interior_set.contains(p))
                .count();
            (n, d)
        })
        .collect();
    // Min-heap keyed by (structural hash, NodeIndex): O(V log V) — regions
    // from rolled prefill graphs have thousands of interior nodes.
    let mut ready: std::collections::BinaryHeap<std::cmp::Reverse<(u64, usize, NodeIndex)>> =
        interior
            .iter()
            .copied()
            .filter(|n| indeg[n] == 0)
            .map(|n| std::cmp::Reverse((hashes.get(&n).copied().unwrap_or(0), n.index(), n)))
            .collect();
    let mut interior_topo: Vec<NodeIndex> = Vec::with_capacity(interior.len());
    while let Some(std::cmp::Reverse((_, _, n))) = ready.pop() {
        interior_topo.push(n);
        for succ in llir_graph.neighbors_directed(n, Direction::Outgoing) {
            if let Some(d) = indeg.get_mut(&succ) {
                *d -= 1;
                if *d == 0 {
                    ready.push(std::cmp::Reverse((
                        hashes.get(&succ).copied().unwrap_or(0),
                        succ.index(),
                        succ,
                    )));
                }
            }
        }
    }
    debug_assert_eq!(interior_topo.len(), interior.len());

    // First use of each FS leaf, walking consumers in canonical body order
    // with operands in hash order (matching emission).
    let mut first_use: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    for &n in interior_topo.iter().chain(std::iter::once(&fe_node)) {
        let mut srcs: Vec<NodeIndex> = llir_graph
            .neighbors_directed(n, Direction::Incoming)
            .collect();
        srcs.sort_by_key(|s| (hashes.get(s).copied().unwrap_or(0), s.index()));
        for s in srcs {
            if !interior_set.contains(&s) && hashes.contains_key(&s) {
                let next = first_use.len();
                first_use.entry(s).or_insert(next);
            }
        }
    }

    let fe_op = llir_graph[fe_node].to_dialect::<dyn KernelOp>().unwrap();
    let fe_struct: &FusionEnd = (***fe_op).downcast_ref::<FusionEnd>().unwrap();
    let fs_keys: FxHashMap<NodeIndex, (String, &'static str)> = fs_nodes
        .iter()
        .map(|&fs| {
            let fs_op = llir_graph[fs].to_dialect::<dyn KernelOp>().unwrap();
            let fs_struct: &FusionStart = (***fs_op).downcast_ref::<FusionStart>().unwrap();
            let read_idx = flatten_strides(&fe_struct.shape, &fs_struct.strides).to_kernel();
            (fs, (read_idx, cuda_dtype(fs_struct.dtype)))
        })
        .collect();
    let mut fs_topo = fs_nodes.to_vec();
    fs_topo.sort_by(|a, b| {
        fs_keys[a]
            .cmp(&fs_keys[b])
            .then_with(|| {
                first_use
                    .get(a)
                    .unwrap_or(&usize::MAX)
                    .cmp(first_use.get(b).unwrap_or(&usize::MAX))
            })
            .then_with(|| a.index().cmp(&b.index()))
    });
    (interior_topo, fs_topo)
}

// =========================================================================
// Per-elementwise body templates.
//
// Each entry takes the names of the local variables holding the op's
// inputs and returns a CUDA expression evaluating to the op's output
// (a register-resident value, no buffer involved).
// =========================================================================

fn is_region_elementwise(llir_graph: &LLIRGraph, node: NodeIndex) -> bool {
    llir_graph
        .node_weight(node)
        .and_then(|op| op.to_dialect::<dyn KernelOp>())
        .is_some_and(|op| {
            (***op).downcast_ref::<CudaUnaryElementwise>().is_some()
                || (***op).downcast_ref::<CudaBinaryElementwise>().is_some()
        })
}

/// Convert a local to its in-register compute form. 16-bit and FP8 locals
/// are widened to float for compute; each node's local then rounds back to
/// the node's own dtype on store (see `elementwise_init_expr`). Per-op this
/// is numerically identical to native 16-bit arithmetic (exact widening,
/// one rounding per node) and avoids relying on device operator overloads.
/// `dtype` is the dtype of the local's *producer* node, not the consumer.
fn elementwise_value(local: &str, dtype: DType) -> String {
    if matches!(
        dtype,
        DType::F16 | DType::Bf16 | DType::F8E4M3 | DType::F8E5M2 | DType::F8UE8M0
    ) {
        format!("static_cast<float>({local})")
    } else {
        local.to_string()
    }
}

fn elementwise_init_expr(expr: &str, dtype: DType, cuda_ty: &str) -> String {
    match dtype {
        DType::F8E4M3 | DType::F8E5M2 | DType::F8UE8M0 => format!("{cuda_ty}({expr})"),
        DType::F16 | DType::Bf16 => format!("({cuda_ty})({expr})"),
        _ => expr.to_string(),
    }
}

/// `locals` are already widened to compute form by `elementwise_value`.
fn elementwise_body(op: &str, locals: &[&str]) -> String {
    let a = || locals[0].to_string();
    let b = || locals[1].to_string();
    match op {
        "Sin" => format!("sinf({})", a()),
        "Sqrt" => format!("sqrtf({})", a()),
        "Rsqrt" => format!("rsqrtf({})", a()),
        "Exp" => format!("expf({})", a()),
        "Exp2" => format!("exp2f({})", a()),
        "Log2" => format!("log2f({})", a()),
        // Operands are widened to `float` by `elementwise_value` (16-bit/fp8
        // → static_cast<float>), so a float reciprocal is unambiguous and the
        // result rounds back to the node dtype at store (elementwise_init_expr).
        // A `static_cast<dtype>(1.0f)` numerator would make this `bf16 / float`
        // — ambiguous in NVRTC against cuda_bf16.h's operator/ overloads.
        "Recip" => format!("1.0f / {}", a()),
        "Sigmoid" => format!("1.0f / (1.0f + expf(-{}))", a()),
        // Dtype conversion happens in the widen (input) / round (store)
        // helpers, so the cast body is the identity.
        "Cast" => a(),
        "Add" => format!("{} + {}", a(), b()),
        "Mul" => format!("{} * {}", a(), b()),
        other => panic!("region_codegen: unknown elementwise op {other}"),
    }
}

fn dtype_program_tag(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F64 => 1,
        DType::F16 => 2,
        DType::Bf16 => 3,
        DType::TF32 => 4,
        DType::Int => 5,
        DType::I64 => 6,
        DType::I4 => 7,
        DType::U4 => 8,
        DType::I8 => 9,
        DType::U8 => 10,
        DType::I16 => 11,
        DType::U16 => 12,
        DType::Bool => 13,
        DType::F8UE8M0 => 14,
        DType::F8E4M3 => 15,
        DType::F8E5M2 => 16,
        DType::F6E2M3 => 17,
        DType::F6E3M2 => 18,
        DType::F4E2M1 => 19,
    }
}

fn elementwise_program_tag(op: &str) -> u8 {
    match op {
        "Sin" => 0,
        "Sqrt" => 1,
        "Rsqrt" => 2,
        "Exp" => 3,
        "Exp2" => 4,
        "Log2" => 5,
        "Recip" => 6,
        "Sigmoid" => 7,
        "Cast" => 8,
        "Add" => 9,
        "Mul" => 10,
        other => panic!("region_codegen: unknown elementwise op {other}"),
    }
}

fn region_output_size(region: &RegionUnit, llir_graph: &LLIRGraph) -> Expression {
    let fe_op = llir_graph[region.fe_node]
        .to_dialect::<dyn KernelOp>()
        .expect("FE node must be a KernelOp");
    let fe: &FusionEnd = (***fe_op)
        .downcast_ref::<FusionEnd>()
        .expect("region root must be FusionEnd");
    fe.shape.iter().copied().product()
}

fn singleton_program_key(
    region: &RegionUnit,
    llir_graph: &LLIRGraph,
    global_dyn_dims: &[Symbol],
) -> Option<SingletonRegionProgramKey> {
    let &[elementwise] = region.elementwise_topo.as_slice() else {
        return None;
    };
    let fe_op = llir_graph[region.fe_node].to_dialect::<dyn KernelOp>()?;
    let fe = (***fe_op).downcast_ref::<FusionEnd>()?;
    let inputs = region
        .fs_nodes
        .iter()
        .map(|&node| {
            let op = llir_graph[node].to_dialect::<dyn KernelOp>().unwrap();
            let start = (***op).downcast_ref::<FusionStart>().unwrap();
            FusionStartProgramKey {
                strides: start.strides.clone(),
                dtype: dtype_program_tag(start.dtype),
            }
        })
        .collect();
    let elementwise_op = llir_graph[elementwise].to_dialect::<dyn KernelOp>()?;
    let element = if let Some(unary) = (***elementwise_op).downcast_ref::<CudaUnaryElementwise>() {
        SingletonElementProgramKey::Unary {
            opcode: elementwise_program_tag(&unary.op),
            dtype: dtype_program_tag(unary.dtype),
        }
    } else {
        let binary = (***elementwise_op).downcast_ref::<CudaBinaryElementwise>()?;
        SingletonElementProgramKey::Binary {
            opcode: elementwise_program_tag(&binary.op),
            dtype: dtype_program_tag(binary.dtype),
        }
    };
    let (operand_slots, operand_count) = singleton_operand_slots(region, llir_graph)?;

    Some(SingletonRegionProgramKey {
        global_dyn_dims: global_dyn_dims.to_vec(),
        output_shape: fe.shape.clone(),
        output_strides: fe.strides.clone(),
        output_dtype: dtype_program_tag(fe.dtype),
        inputs,
        element,
        operand_slots: operand_slots[..operand_count].to_vec(),
    })
}

/// Return the canonical input-local slots for the singleton operation without
/// allocating. The only currently constructible region operation has at most
/// two inputs, and Add/Mul source emission orders them by local slot.
fn singleton_operand_slots(
    region: &RegionUnit,
    llir_graph: &LLIRGraph,
) -> Option<([u16; 2], usize)> {
    let &[elementwise] = region.elementwise_topo.as_slice() else {
        return None;
    };
    let mut slots = [0; 2];
    let mut count = 0;
    for source in llir_graph.neighbors_directed(elementwise, Direction::Incoming) {
        if count == slots.len() {
            return None;
        }
        slots[count] = region
            .fs_nodes
            .iter()
            .position(|&node| node == source)
            .and_then(|slot| u16::try_from(slot).ok())?;
        count += 1;
    }
    slots[..count].sort_unstable();
    Some((slots, count))
}

/// Allocation-free hash of all CUDA-source-relevant singleton metadata. A
/// hash hit is always checked by `singleton_program_matches`, so collisions
/// can only add a comparison and can never return the wrong program.
fn singleton_program_fingerprint(
    region: &RegionUnit,
    llir_graph: &LLIRGraph,
    global_dyn_dims: &[Symbol],
) -> Option<u64> {
    let &[elementwise] = region.elementwise_topo.as_slice() else {
        return None;
    };
    let fe_op = llir_graph[region.fe_node].to_dialect::<dyn KernelOp>()?;
    let fe = (***fe_op).downcast_ref::<FusionEnd>()?;

    let mut hasher = FxHasher::default();
    0_u8.hash(&mut hasher); // Cache-key format version.
    global_dyn_dims.hash(&mut hasher);
    hash_expression_slice(&fe.shape, &mut hasher);
    hash_expression_slice(&fe.strides, &mut hasher);
    dtype_program_tag(fe.dtype).hash(&mut hasher);
    region.fs_nodes.len().hash(&mut hasher);
    for &node in &region.fs_nodes {
        let op = llir_graph[node].to_dialect::<dyn KernelOp>()?;
        let start = (***op).downcast_ref::<FusionStart>()?;
        hash_expression_slice(&start.strides, &mut hasher);
        dtype_program_tag(start.dtype).hash(&mut hasher);
    }

    let elementwise_op = llir_graph[elementwise].to_dialect::<dyn KernelOp>()?;
    if let Some(unary) = (***elementwise_op).downcast_ref::<CudaUnaryElementwise>() {
        0_u8.hash(&mut hasher);
        elementwise_program_tag(&unary.op).hash(&mut hasher);
        dtype_program_tag(unary.dtype).hash(&mut hasher);
    } else {
        let binary = (***elementwise_op).downcast_ref::<CudaBinaryElementwise>()?;
        1_u8.hash(&mut hasher);
        elementwise_program_tag(&binary.op).hash(&mut hasher);
        dtype_program_tag(binary.dtype).hash(&mut hasher);
    }
    let (operand_slots, operand_count) = singleton_operand_slots(region, llir_graph)?;
    operand_slots[..operand_count].hash(&mut hasher);
    Some(hasher.finish())
}

fn hash_expression_slice(expressions: &[Expression], hasher: &mut FxHasher) {
    expressions.len().hash(hasher);
    for expression in expressions {
        expression.hash_intern_id(hasher);
    }
}

fn same_expression_slice(left: &[Expression], right: &[Expression]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.has_same_intern_id(right))
}

/// Collision-safe comparison against the owned cache entry, performed
/// directly against LLIR metadata without cloning any vectors.
fn singleton_program_matches(
    key: &SingletonRegionProgramKey,
    region: &RegionUnit,
    llir_graph: &LLIRGraph,
    global_dyn_dims: &[Symbol],
) -> bool {
    let &[elementwise] = region.elementwise_topo.as_slice() else {
        return false;
    };
    let Some(fe_op) = llir_graph[region.fe_node].to_dialect::<dyn KernelOp>() else {
        return false;
    };
    let Some(fe) = (***fe_op).downcast_ref::<FusionEnd>() else {
        return false;
    };
    if key.global_dyn_dims != global_dyn_dims
        || !same_expression_slice(&key.output_shape, &fe.shape)
        || !same_expression_slice(&key.output_strides, &fe.strides)
        || key.output_dtype != dtype_program_tag(fe.dtype)
        || key.inputs.len() != region.fs_nodes.len()
    {
        return false;
    }
    for (input_key, &node) in key.inputs.iter().zip(&region.fs_nodes) {
        let Some(op) = llir_graph[node].to_dialect::<dyn KernelOp>() else {
            return false;
        };
        let Some(start) = (***op).downcast_ref::<FusionStart>() else {
            return false;
        };
        if !same_expression_slice(&input_key.strides, &start.strides)
            || input_key.dtype != dtype_program_tag(start.dtype)
        {
            return false;
        }
    }

    let Some(elementwise_op) = llir_graph[elementwise].to_dialect::<dyn KernelOp>() else {
        return false;
    };
    let element_matches = match &key.element {
        SingletonElementProgramKey::Unary { opcode, dtype } => (***elementwise_op)
            .downcast_ref::<CudaUnaryElementwise>()
            .is_some_and(|unary| {
                *opcode == elementwise_program_tag(&unary.op)
                    && *dtype == dtype_program_tag(unary.dtype)
            }),
        SingletonElementProgramKey::Binary { opcode, dtype } => (***elementwise_op)
            .downcast_ref::<CudaBinaryElementwise>()
            .is_some_and(|binary| {
                *opcode == elementwise_program_tag(&binary.op)
                    && *dtype == dtype_program_tag(binary.dtype)
            }),
    };
    if !element_matches {
        return false;
    }
    let Some((operand_slots, operand_count)) = singleton_operand_slots(region, llir_graph) else {
        return false;
    };
    key.operand_slots.as_slice() == &operand_slots[..operand_count]
}

// =========================================================================
// Region compilation — emit one CUDA kernel for the whole region.
// =========================================================================

#[allow(clippy::type_complexity)]
pub(crate) struct CompiledRegion {
    pub function: CudaFunction,
    pub module: Arc<CudaModule>,
    pub source_bytes: usize,
    pub has_dyn_dims_param: bool,
    pub grid: (Expression, Expression, Expression),
    pub block: (Expression, Expression, Expression),
    pub shared_mem: Expression,
    pub constants: FxHashMap<Symbol, CudaSlice<u8>>,
}

/// Generate the fused kernel source plus launch geometry for a region.
/// Pure — no CUDA calls — so canonicalization invariants are testable
/// without a device. The string this returns is the compile-cache key:
/// it must depend only on region structure, never on NodeIndexes.
pub(crate) fn region_kernel_source(
    region: &RegionUnit,
    llir_graph: &LLIRGraph,
) -> (String, Expression) {
    // Resolve FE: shape, strides (for the write), dtype.
    let fe_op = llir_graph[region.fe_node]
        .to_dialect::<dyn KernelOp>()
        .expect("FE node must be a KernelOp");
    let fe_struct: &FusionEnd = (***fe_op)
        .downcast_ref::<FusionEnd>()
        .expect("region root must be FusionEnd");
    let out_shape: &[Expression] = &fe_struct.shape;
    let out_strides: &[Expression] = &fe_struct.strides;
    let dtype: DType = fe_struct.dtype;

    // Aggregate all dynamic vars used anywhere in the region (FS strides,
    // FE strides and elementwise shapes.
    // own strides are likewise relevant for any future stride-affine ops).
    let mut all_vars: FxHashSet<Symbol> = FxHashSet::default();
    all_vars.extend(out_shape.iter().flat_map(|e| e.dyn_vars()));
    all_vars.extend(out_strides.iter().flat_map(|e| e.dyn_vars()));
    for &fs_idx in &region.fs_nodes {
        let fs_op = llir_graph[fs_idx].to_dialect::<dyn KernelOp>().unwrap();
        let fs_struct: &FusionStart = (***fs_op).downcast_ref::<FusionStart>().unwrap();
        all_vars.extend(fs_struct.strides.iter().flat_map(|e| e.dyn_vars()));
    }
    for &elem_idx in &region.elementwise_topo {
        let elem_op = llir_graph[elem_idx].to_dialect::<dyn KernelOp>().unwrap();
        if let Some(elem) = (***elem_op).downcast_ref::<CudaUnaryElementwise>() {
            all_vars.extend(elem.shape.iter().flat_map(|e| e.dyn_vars()));
            all_vars.extend(elem.in_strides.iter().flat_map(|e| e.dyn_vars()));
            all_vars.extend(elem.out_strides.iter().flat_map(|e| e.dyn_vars()));
        } else if let Some(elem) = (***elem_op).downcast_ref::<CudaBinaryElementwise>() {
            all_vars.extend(elem.out_shape.iter().flat_map(|e| e.dyn_vars()));
            all_vars.extend(elem.a_stride.iter().flat_map(|e| e.dyn_vars()));
            all_vars.extend(elem.b_stride.iter().flat_map(|e| e.dyn_vars()));
            all_vars.extend(elem.out_stride.iter().flat_map(|e| e.dyn_vars()));
        }
    }

    // Per-node dtypes: regions are dtype-uniform except at explicit Cast
    // nodes, so every FS leaf, interior node, and the FE carry their own
    // dtype. Locals and kernel parameters are typed per node.
    let node_dtype = |idx: NodeIndex| -> DType {
        let op = llir_graph[idx].to_dialect::<dyn KernelOp>().unwrap();
        if let Some(fs) = (***op).downcast_ref::<FusionStart>() {
            fs.dtype
        } else if let Some(elem) = (***op).downcast_ref::<CudaUnaryElementwise>() {
            elem.dtype
        } else if let Some(elem) = (***op).downcast_ref::<CudaBinaryElementwise>() {
            elem.dtype
        } else {
            op.output_dtype()
        }
    };

    let cuda_ty = cuda_dtype(dtype);
    let mut region_dtypes: Vec<DType> = vec![dtype];
    region_dtypes.extend(region.fs_nodes.iter().map(|&n| node_dtype(n)));
    region_dtypes.extend(region.elementwise_topo.iter().map(|&n| node_dtype(n)));
    let includes = dtype_includes(&region_dtypes);
    let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&all_vars);
    let dyn_dims_param = if all_vars.is_empty() {
        ""
    } else {
        ", const int* dyn_dims"
    };

    let n_elements = out_shape
        .iter()
        .copied()
        .product::<Expression>()
        .to_kernel();

    // Build kernel signature: out, then one input per FS leaf in
    // `region.fs_nodes` order. The `external_inputs` list (parallel to
    // `fs_nodes`) is what the host wires into the launch params.
    let mut signature_params: Vec<String> = vec![format!("{cuda_ty} *out")];
    for (i, &fs_idx) in region.fs_nodes.iter().enumerate() {
        let fs_ty = cuda_dtype(node_dtype(fs_idx));
        signature_params.push(format!("const {fs_ty} *in{i}"));
    }
    let signature = signature_params.join(", ");

    // Body: read FS leaves, then walk elementwise nodes in topo order emitting a
    // local per op, then write FE output. Every node gets a local keyed
    // by a position-in-region index so the kernel string is invariant
    // under NodeIndex churn (each `egglog_to_llir` reissues NodeIndexes,
    // so naming locals by `n.index()` would invalidate the kernel
    // string cache on every search candidate). Indices: FS leaves get
    // 0..fs_nodes.len(), elementwise nodes get fs_nodes.len()..(+ elementwise_topo.len()).
    let mut local_idx_map: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    for (i, &fs_idx) in region.fs_nodes.iter().enumerate() {
        local_idx_map.insert(fs_idx, i);
    }
    let fs_count = region.fs_nodes.len();
    for (i, &op_idx) in region.elementwise_topo.iter().enumerate() {
        local_idx_map.insert(op_idx, fs_count + i);
    }
    let local_name = |n: NodeIndex| format!("v_{}", local_idx_map[&n]);

    let mut body = String::new();
    body.push_str(&format!(
        "        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20       if (const_z >= {n_elements}) return;\n"
    ));

    // FS leaves: each reads from its corresponding `in_i` parameter using
    // its own strides.
    for (i, &fs_idx) in region.fs_nodes.iter().enumerate() {
        let fs_op = llir_graph[fs_idx].to_dialect::<dyn KernelOp>().unwrap();
        let fs_struct: &FusionStart = (***fs_op).downcast_ref::<FusionStart>().unwrap();
        let fs_ty = cuda_dtype(fs_struct.dtype);
        let read_idx = flatten_strides(out_shape, &fs_struct.strides).to_kernel();
        body.push_str(&format!(
            "        {fs_ty} {name} = in{i}[{read_idx}];\n",
            name = local_name(fs_idx),
        ));
    }

    // Elementwise ops in topo order. Each looks up its predecessor locals
    // (in incoming-edge id order to match the original op's input
    // arity / position).
    for &op_idx in &region.elementwise_topo {
        let op_ref = llir_graph[op_idx].to_dialect::<dyn KernelOp>().unwrap();
        let (elem_name, elem_dtype) =
            if let Some(elem) = (***op_ref).downcast_ref::<CudaUnaryElementwise>() {
                (elem.op.as_str(), elem.dtype)
            } else if let Some(elem) = (***op_ref).downcast_ref::<CudaBinaryElementwise>() {
                (elem.op.as_str(), elem.dtype)
            } else {
                panic!(
                    "region_codegen: expected Cuda*Elementwise op, got {}",
                    op_ref.kernel_name()
                );
            };

        // Operand order must be canonical, not edge-id order: edge ids
        // track LLIR construction order, which varies across search
        // candidates. All binary region ops (Add / Mul) are commutative,
        // so ordering operands by their producer's local position is both
        // safe and NodeIndex-invariant given canonical region orders.
        let mut edges: Vec<(_, NodeIndex)> = llir_graph
            .edges_directed(op_idx, Direction::Incoming)
            .map(|e| (e.id(), e.source()))
            .collect();
        edges.sort_by_key(|&(eid, src)| (local_idx_map.get(&src).copied(), eid));
        let input_locals: Vec<String> = edges
            .into_iter()
            .map(|(_, src)| elementwise_value(&local_name(src), node_dtype(src)))
            .collect();
        let inputs_ref: Vec<&str> = input_locals.iter().map(|s| s.as_str()).collect();

        let elem_ty = cuda_dtype(elem_dtype);
        let expr = elementwise_body(elem_name, &inputs_ref);
        let expr = elementwise_init_expr(&expr, elem_dtype, elem_ty);
        body.push_str(&format!(
            "        {elem_ty} {name} = {expr};\n",
            name = local_name(op_idx),
        ));
    }

    // FE write: pick the elementwise node feeding FE (its single incoming edge in
    // the region — an elementwise node or, in degenerate single-FS regions which
    // shouldn't arise, an FS).
    let fe_input: NodeIndex = llir_graph
        .neighbors_directed(region.fe_node, Direction::Incoming)
        .next()
        .expect("FusionEnd with no predecessor");
    let fe_input_local = local_name(fe_input);
    let write_idx = flatten_strides(out_shape, out_strides).to_kernel();
    body.push_str(&format!("        out[{write_idx}] = {fe_input_local};\n"));

    let kernel = format!(
        "{includes}\n\
         {dyn_defines}\n\
         extern \"C\" {{\n\
         \x20   __global__ void fused_region_k({signature}{dyn_dims_param}) {{\n\
         {body}\
         \x20   }}\n\
         }}"
    );

    let out_size = out_shape.iter().copied().product::<Expression>();
    (kernel, out_size)
}

#[allow(clippy::type_complexity)]
pub(crate) fn compile_region(
    region: &RegionUnit,
    llir_graph: &LLIRGraph,
    stream: &Arc<CudaStream>,
    compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
) -> CompiledRegion {
    let (kernel, out_size) = region_kernel_source(region, llir_graph);
    compile_prepared_region(&kernel, out_size, stream, compile_cache)
}

pub(crate) fn compile_prepared_region(
    kernel: &str,
    out_size: Expression,
    stream: &Arc<CudaStream>,
    compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
) -> CompiledRegion {
    let source_bytes = kernel.len();
    let has_dyn_dims_param = kernel.contains("dyn_dims");

    let (module, function) = if let Some((m, f)) = compile_cache.get(kernel) {
        (m.clone(), f.clone())
    } else {
        let ptx = compile_module_image_for_current_device(stream.context(), kernel)
            .expect("region kernel PTX compile failed");
        let module = stream
            .context()
            .load_module(ptx)
            .expect("module load failed");
        let function = module
            .load_function("fused_region_k")
            .expect("region kernel function not found");
        compile_cache.insert(kernel.to_owned(), (module.clone(), function.clone()));
        (module, function)
    };

    CompiledRegion {
        function,
        module,
        source_bytes,
        has_dyn_dims_param,
        grid: (out_size.ceil_div(256), 1.into(), 1.into()),
        block: (out_size.min(256), 1.into(), 1.into()),
        shared_mem: 0.into(),
        constants: FxHashMap::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::fusion::elementwise::{CudaBinaryElementwise, CudaUnaryElementwise};
    use luminal::op::LLIROp;
    use luminal::prelude::{DType, petgraph::algo::toposort};

    /// Helper: wrap a `KernelOp` in an `LLIROp` of the kernel dialect.
    fn llir_of(op: impl KernelOp + 'static) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(op) as Box<dyn KernelOp>)
    }

    /// Build the test region used by the canonicalization tests:
    ///
    ///   P_sqrt → FS_a (f32) ──┐
    ///   P_sin  → FS_b (bf16) ─┤→ Mul → Add → FE (f32, shape [8])
    ///   P_exp  → FS_c (f32) ──┘         ↑
    ///            (FS_c feeds Add's second operand)
    ///
    /// `order` permutes node insertion and `flip_edges` reverses the
    /// operand-edge insertion order, so the two graphs differ in every
    /// NodeIndex and edge id while being structurally identical.
    fn build_test_region(reversed: bool) -> (LLIRGraph, Vec<NodeIndex>) {
        let shape: Vec<Expression> = vec![8.into()];
        let z: Vec<Expression> = vec![Expression::from('z')];
        let fs = |dt: DType| FusionStart {
            shape: shape.clone(),
            strides: z.clone(),
            dtype: dt,
        };
        let bin = |op: &str, dt: DType| CudaBinaryElementwise {
            op: op.to_string(),
            out_shape: shape.clone(),
            a_stride: z.clone(),
            b_stride: z.clone(),
            out_stride: z.clone(),
            dtype: dt,
        };
        let unary = |op: &str| CudaUnaryElementwise {
            op: op.to_string(),
            shape: shape.clone(),
            in_strides: z.clone(),
            out_strides: z.clone(),
            dtype: DType::F32,
        };
        let fe = FusionEnd {
            shape: shape.clone(),
            strides: z.clone(),
            dtype: DType::F32,
        };

        let mut g: LLIRGraph = LLIRGraph::default();
        let mut add_nodes = |g: &mut LLIRGraph| {
            let p_sqrt = g.add_node(llir_of(unary("Sqrt")));
            let p_sin = g.add_node(llir_of(unary("Sin")));
            let p_exp = g.add_node(llir_of(unary("Exp")));
            let fs_a = g.add_node(llir_of(fs(DType::F32)));
            let fs_b = g.add_node(llir_of(fs(DType::Bf16)));
            let fs_c = g.add_node(llir_of(fs(DType::F32)));
            let mul = g.add_node(llir_of(bin("Mul", DType::F32)));
            let add = g.add_node(llir_of(bin("Add", DType::F32)));
            let fe_n = g.add_node(llir_of(fe.clone()));
            vec![p_sqrt, p_sin, p_exp, fs_a, fs_b, fs_c, mul, add, fe_n]
        };
        // Insert nodes in reverse for the permuted graph so every
        // NodeIndex differs. (StableGraph indices follow insertion order.)
        let nodes = if reversed {
            let p_exp = g.add_node(llir_of(unary("Exp")));
            let fe_n = g.add_node(llir_of(fe.clone()));
            let add = g.add_node(llir_of(bin("Add", DType::F32)));
            let fs_c = g.add_node(llir_of(fs(DType::F32)));
            let mul = g.add_node(llir_of(bin("Mul", DType::F32)));
            let fs_b = g.add_node(llir_of(fs(DType::Bf16)));
            let fs_a = g.add_node(llir_of(fs(DType::F32)));
            let p_sin = g.add_node(llir_of(unary("Sin")));
            let p_sqrt = g.add_node(llir_of(unary("Sqrt")));
            vec![p_sqrt, p_sin, p_exp, fs_a, fs_b, fs_c, mul, add, fe_n]
        } else {
            add_nodes(&mut g)
        };
        let [p_sqrt, p_sin, p_exp, fs_a, fs_b, fs_c, mul, add, fe_n]: [NodeIndex; 9] =
            nodes.clone().try_into().unwrap();

        let mut edges: Vec<(NodeIndex, NodeIndex)> = vec![
            (p_sqrt, fs_a),
            (p_sin, fs_b),
            (p_exp, fs_c),
            (fs_a, mul),
            (fs_b, mul),
            (mul, add),
            (fs_c, add),
            (add, fe_n),
        ];
        if reversed {
            edges.reverse();
        }
        for (a, b) in edges {
            g.add_edge(a, b, ());
        }
        (g, nodes)
    }

    fn region_source_and_producers(g: &LLIRGraph) -> (String, Vec<String>) {
        let topo = toposort(g, None).unwrap();
        let (units, _) = build_compile_units(&topo, g);
        let region = units
            .iter()
            .find_map(|u| match u {
                CompileUnit::Region(r) => Some(r),
                _ => None,
            })
            .expect("no region built");
        let (kernel, _) = region_kernel_source(region, g);
        // Producer identity per input slot, via the producer's unary op
        // name (Sqrt / Sin / Exp).
        let producers = region
            .external_inputs
            .iter()
            .map(|&p| {
                (***g[p].to_dialect::<dyn KernelOp>().unwrap())
                    .downcast_ref::<CudaUnaryElementwise>()
                    .unwrap()
                    .op
                    .clone()
            })
            .collect();
        (kernel, producers)
    }

    #[test]
    fn prepared_fusion_plan_preserves_compile_units_and_sources() {
        let (graph, _) = build_test_region(false);
        let topo = toposort(&graph, None).unwrap();
        let (expected_units, absorbed) = build_compile_units(&topo, &graph);
        let all_nodes = topo.iter().copied().collect();
        let mut prepared = PreparedFusionPlan::discover(&topo, &graph);
        let mut source_cache = RegionSourceCache::default();
        prepared.prepare_region_kernels_for(&all_nodes, &graph, &mut source_cache, &[]);

        assert_eq!(prepared.compile_units(), expected_units);
        assert_eq!(prepared.absorbed_markers(), &absorbed);
        assert_eq!(
            prepared
                .compile_units_for(&all_nodes)
                .cloned()
                .collect::<Vec<_>>(),
            expected_units,
        );
        for unit in prepared.compile_units() {
            let CompileUnit::Region(region) = unit else {
                continue;
            };
            let (source, output_size) = region_kernel_source(region, &graph);
            let prepared_kernel = prepared.region_kernel(region.fe_node).unwrap();
            assert_eq!(prepared_kernel.source.as_ref(), source);
            assert_eq!(prepared_kernel.output_size, output_size);
        }
    }

    #[test]
    fn singleton_region_deduplicates_repeated_fusion_start_inputs() {
        let shape = vec![8.into()];
        let strides = vec![Expression::from('z')];
        let mut graph = LLIRGraph::default();
        let producer = graph.add_node(llir_of(CudaUnaryElementwise {
            op: "Sqrt".to_string(),
            shape: shape.clone(),
            in_strides: strides.clone(),
            out_strides: strides.clone(),
            dtype: DType::F32,
        }));
        let start = graph.add_node(llir_of(FusionStart {
            shape: shape.clone(),
            strides: strides.clone(),
            dtype: DType::F32,
        }));
        let add = graph.add_node(llir_of(CudaBinaryElementwise {
            op: "Add".to_string(),
            out_shape: shape.clone(),
            a_stride: strides.clone(),
            b_stride: strides.clone(),
            out_stride: strides.clone(),
            dtype: DType::F32,
        }));
        let end = graph.add_node(llir_of(FusionEnd {
            shape,
            strides,
            dtype: DType::F32,
        }));
        graph.add_edge(producer, start, ());
        graph.add_edge(start, add, ());
        graph.add_edge(start, add, ());
        graph.add_edge(add, end, ());

        let topo = toposort(&graph, None).unwrap();
        let (units, _) = build_compile_units(&topo, &graph);
        let region = units
            .iter()
            .find_map(|unit| match unit {
                CompileUnit::Region(region) => Some(region),
                CompileUnit::Single(_) => None,
            })
            .unwrap();
        assert_eq!(region.fs_nodes, vec![start]);
        assert_eq!(region.external_inputs, vec![producer]);

        let (source, _) = region_kernel_source(region, &graph);
        assert!(source.contains("const float *in0"));
        assert!(!source.contains("in1"));
        assert!(source.contains("v_0 + v_0"));
    }

    #[test]
    fn singleton_region_orders_inputs_by_structure() {
        let shape = vec![8.into()];
        let mut graph = LLIRGraph::default();
        let indexed = graph.add_node(llir_of(FusionStart {
            shape: shape.clone(),
            strides: vec![Expression::from('z')],
            dtype: DType::F32,
        }));
        let broadcast = graph.add_node(llir_of(FusionStart {
            shape,
            strides: vec![0.into()],
            dtype: DType::F32,
        }));

        assert_eq!(
            compare_fusion_starts(&graph, broadcast, indexed),
            std::cmp::Ordering::Less,
        );
        assert_eq!(
            compare_fusion_starts(&graph, indexed, broadcast),
            std::cmp::Ordering::Greater,
        );
    }

    #[test]
    fn singleton_source_cache_separates_dynamic_dimension_abis() {
        let shape = vec![Expression::from('b')];
        let strides = vec![Expression::from('z')];
        let mut graph = LLIRGraph::default();
        let producer = graph.add_node(llir_of(CudaUnaryElementwise {
            op: "Sqrt".to_string(),
            shape: shape.clone(),
            in_strides: strides.clone(),
            out_strides: strides.clone(),
            dtype: DType::F32,
        }));
        let start = graph.add_node(llir_of(FusionStart {
            shape: shape.clone(),
            strides: strides.clone(),
            dtype: DType::F32,
        }));
        let unary = graph.add_node(llir_of(CudaUnaryElementwise {
            op: "Sin".to_string(),
            shape: shape.clone(),
            in_strides: strides.clone(),
            out_strides: strides.clone(),
            dtype: DType::F32,
        }));
        let end = graph.add_node(llir_of(FusionEnd {
            shape,
            strides,
            dtype: DType::F32,
        }));
        graph.add_edge(producer, start, ());
        graph.add_edge(start, unary, ());
        graph.add_edge(unary, end, ());

        let topo = toposort(&graph, None).unwrap();
        let all_nodes = topo.iter().copied().collect();
        let mut cache = RegionSourceCache::default();
        let ab = [Symbol::from('a'), Symbol::from('b')];
        let ba = [Symbol::from('b'), Symbol::from('a')];
        let prepare = |dims: &[Symbol], cache: &mut RegionSourceCache| {
            crate::kernel::hlir::set_global_dyn_dims(dims.to_vec());
            let mut plan = PreparedFusionPlan::discover(&topo, &graph);
            plan.prepare_region_kernels_for(&all_nodes, &graph, cache, dims);
            plan.region_kernel(end).unwrap().source.clone()
        };
        let source_ab = prepare(&ab, &mut cache);
        let source_ba = prepare(&ba, &mut cache);
        let source_ab_again = prepare(&ab, &mut cache);
        crate::kernel::hlir::clear_global_dyn_dims();

        assert!(source_ab.contains("dyn_dims[1]"));
        assert!(source_ba.contains("dyn_dims[0]"));
        assert_eq!(source_ab, source_ab_again);
        assert_eq!(cache.counters(), (1, 2));
    }

    /// Structurally identical regions must emit byte-identical kernel
    /// sources (the compile-cache key) and bind the same producers to the
    /// same input slots, regardless of NodeIndex / edge-id churn.
    #[test]
    fn region_kernel_source_is_nodeindex_invariant() {
        let (g1, _) = build_test_region(false);
        let (g2, _) = build_test_region(true);
        let (k1, p1) = region_source_and_producers(&g1);
        let (k2, p2) = region_source_and_producers(&g2);
        assert_eq!(k1, k2, "kernel source must not depend on NodeIndexes");
        assert_eq!(p1, p2, "input-slot → producer binding must match");
    }
}
