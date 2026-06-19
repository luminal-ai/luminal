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

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    graph::LLIRGraph,
    prelude::{
        petgraph::{Direction, algo::toposort, visit::EdgeRef},
        *,
    },
};

use as_any::Downcast;

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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub(crate) enum CompileUnit {
    Single(NodeIndex),
    Region(RegionUnit),
}

// =========================================================================
// Region detection.
// =========================================================================

/// Group a sub-DAG's topo order into compile units. Each FusionEnd node
/// becomes the root of a `CompileUnit::Region`; the region's interior
/// Cuda*Elementwise and FusionStart nodes are absorbed into that region and removed
/// from the per-node iteration. Anything else is wrapped in
/// `CompileUnit::Single`.
/// Globally-absorbed FS / FE markers — the set of marker nodes that any
/// `FusionEnd` in the LLIR walks back to during region detection. A
/// marker is "absorbed" iff some FE in the LLIR can reach it by walking
/// incoming edges through `FusionEnd` / Cuda*Elementwise nodes, stopping at
/// `FusionStart` leaves.
///
/// This is computed once over the full LLIR rather than per-convex-
/// subgraph, because `partition_marked_convex` may put a shared FS leaf
/// (one whose e-graph congruence-deduplicated it across multiple
/// regions) into a different subgraph than the FE that absorbs it.
/// Without this global view, `build_compile_units` running on the FS's
/// subgraph would not see any FE walking back to the FS and would emit the
/// FS as `CompileUnit::Single`; marker standalone compilation is not supported.
pub(crate) fn globally_absorbed_markers(llir_graph: &LLIRGraph) -> FxHashSet<NodeIndex> {
    let name_of = |idx: NodeIndex| -> Option<&'static str> {
        llir_graph
            .node_weight(idx)
            .and_then(|op| op.to_dialect::<dyn KernelOp>().map(|k| k.kernel_name()))
    };

    let mut absorbed: FxHashSet<NodeIndex> = FxHashSet::default();
    for fe in llir_graph.node_indices() {
        if name_of(fe) != Some("FusionEnd") {
            continue;
        }
        let mut visited: FxHashSet<NodeIndex> = FxHashSet::default();
        let mut stack: Vec<NodeIndex> = vec![fe];
        visited.insert(fe);
        while let Some(cur) = stack.pop() {
            for pred in llir_graph.neighbors_directed(cur, Direction::Incoming) {
                if !visited.insert(pred) {
                    continue;
                }
                match name_of(pred) {
                    Some("FusionStart") => {
                        absorbed.insert(pred);
                    }
                    Some("FusionEnd") => {
                        absorbed.insert(pred);
                        stack.push(pred);
                    }
                    Some(_) if is_region_elementwise(llir_graph, pred) => {
                        absorbed.insert(pred);
                        stack.push(pred);
                    }
                    _ => {}
                }
            }
        }
    }
    absorbed
}

pub(crate) fn build_compile_units(
    topo_order: &[NodeIndex],
    llir_graph: &LLIRGraph,
    globally_absorbed: &FxHashSet<NodeIndex>,
) -> Vec<CompileUnit> {
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
                        // A nested FE inside a region. Under the current
                        // rule design these are cascade artifacts — treat
                        // them as transparent (walk through) rather than
                        // as a separate region. The outer region absorbs
                        // them. They do not become CompileUnit::Region
                        // anchors because their eclass is already the
                        // outer region's.
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
    // either by a region in *this* subgraph (`absorbed`) or by any
    // region anywhere in the LLIR (`globally_absorbed`). Skipping the
    // latter prevents shared FS markers whose consumers live in other
    // convex subgraphs from being emitted as standalone compile units:
    // those FSes are absorbed by some other region, and the consuming
    // region reads from FS's external producer.
    let mut units: Vec<CompileUnit> = Vec::new();
    for &node in topo_order {
        if let Some(region) = regions.remove(&node) {
            units.push(CompileUnit::Region(region));
        } else if absorbed.contains(&node) || globally_absorbed.contains(&node) {
            continue;
        } else {
            units.push(CompileUnit::Single(node));
        }
    }
    units
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

// =========================================================================
// Region compilation — emit one CUDA kernel for the whole region.
// =========================================================================

#[allow(clippy::type_complexity)]
pub(crate) struct CompiledRegion {
    pub function: CudaFunction,
    pub module: Arc<CudaModule>,
    pub kernel_str: String,
    pub grid: (Expression, Expression, Expression),
    pub block: (Expression, Expression, Expression),
    pub shared_mem: Expression,
    pub constants: FxHashMap<char, CudaSlice<u8>>,
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
    let mut all_vars: FxHashSet<char> = FxHashSet::default();
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

    let (module, function) = if let Some((m, f)) = compile_cache.get(&kernel) {
        (m.clone(), f.clone())
    } else {
        let ptx = compile_module_image_for_current_device(stream.context(), &kernel)
            .expect("region kernel PTX compile failed");
        let module = stream
            .context()
            .load_module(ptx)
            .expect("module load failed");
        let function = module
            .load_function("fused_region_k")
            .expect("region kernel function not found");
        compile_cache.insert(kernel.clone(), (module.clone(), function.clone()));
        (module, function)
    };

    CompiledRegion {
        function,
        module,
        kernel_str: kernel,
        grid: (out_size.ceil_div(256), 1.into(), 1.into()),
        block: (out_size.min(256), 1.into(), 1.into()),
        shared_mem: 0.into(),
        constants: FxHashMap::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::fusion::elementwise::CudaBinaryElementwise;
    use luminal::op::LLIROp;
    use luminal::prelude::petgraph::algo::toposort;

    /// Helper: wrap a `KernelOp` in an `LLIROp` of the kernel dialect.
    fn llir_of(op: impl KernelOp + 'static) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(op) as Box<dyn KernelOp>)
    }

    /// Reproducer for the `FusionStart with no predecessor` panic at
    /// `region_codegen.rs:232`. The egglog rolling pass + iterated mode
    /// (`LUMINAL_LOOP_ROLL_ITERATE=1`) has been observed to produce LLIR
    /// graphs where a `FusionStart` marker is reached as a region leaf
    /// during the FE→FS walk but has no incoming edge — meaning the
    /// region has nothing to read from. `build_compile_units` then
    /// panics when constructing `external_inputs` because every FS leaf
    /// is required to have exactly one external producer.
    ///
    /// Until that path is fixed, this test pins the failure mode so a
    /// regression doesn't silently change the panic message or location.
    /// `should_panic` rather than `ignore` so it stays runnable in CI
    /// and surfaces if the panic ever moves.
    #[test]
    #[should_panic(expected = "FusionStart with no predecessor")]
    fn fusion_start_with_no_predecessor_panics() {
        // Minimal reproducer:
        //
        //   (no input) ──▶ FusionStart ──▶ CudaBinaryElementwise ──▶ FusionEnd
        //
        // CudaBinaryElementwise is a binary op (n_inputs = 2) so a real region would
        // have two FS leaves. For this panic-shape test only the *first*
        // FS leaf needs a missing predecessor — `build_compile_units`
        // panics in `expect("FusionStart with no predecessor")` as soon
        // as any FS in `fs_topo` lacks one. We add only one FS edge so
        // CudaBinaryElementwise has a dangling second input slot, but that's fine:
        // we're testing the specific panic path inside `build_compile_units`,
        // not full kernel codegen.
        let mut llir: LLIRGraph = LLIRGraph::default();

        let fs_node = llir.add_node(llir_of(FusionStart::default()));
        let fadd_node = llir.add_node(llir_of(CudaBinaryElementwise::default()));
        let fe_node = llir.add_node(llir_of(FusionEnd::default()));

        // FusionStart → CudaBinaryElementwise → FusionEnd.
        llir.add_edge(fs_node, fadd_node, ());
        llir.add_edge(fadd_node, fe_node, ());

        let topo = toposort(&llir, None).expect("LLIR cycle in test setup");
        let absorbed = globally_absorbed_markers(&llir);

        // This is the call that panics with `FusionStart with no
        // predecessor` because `fs_node`'s incoming-edges iterator is
        // empty.
        let _ = build_compile_units(&topo, &llir, &absorbed);
    }

    use crate::kernel::fusion::elementwise::CudaUnaryElementwise;
    use luminal::prelude::DType;

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
        let absorbed = globally_absorbed_markers(g);
        let units = build_compile_units(&topo, g, &absorbed);
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
