//! Device-resource prepass over a serialized e-graph.
//!
//! Some e-nodes describe a value that the target device can never hold or
//! dispatch, no matter what the rest of the graph does. The canonical case is
//! an unfused matmul: `Sum(Mul(...))` materializes the whole `[M, N, K]`
//! product tensor before reducing it, while the fused alternative in the same
//! e-class only ever holds the `[M, N]` answer. Both are correct, so the
//! search picks between them at random — and on a large convolution the
//! unfused draw needs orders of magnitude more memory than the card has.
//!
//! Rejecting those candidates after extraction does not work: the choice that
//! blows up is made in the matmul's e-class, but the cost lands two nodes
//! below on the `Mul`, whose own e-class has no cheaper alternative. Every
//! draw that touches it is rejected, so the search never finds a viable
//! initial genome and never starts.
//!
//! This pass removes such e-nodes from the search space before the search
//! begins, so the initial genome is drawn from a space where every choice is
//! viable. Removing an OpKind e-node cascades: the `Op` e-nodes that reference
//! an emptied kind e-class go too, and so on up — which is exactly what makes
//! the fused matmul the only remaining alternative.
//!
//! Two guarantees keep that cascade safe:
//!
//! * A single tensor larger than the device limit can never execute, so
//!   dropping it removes nothing that was reachable. This is a necessary, not
//!   a sufficient, condition for the graph to fit — peak memory is about
//!   simultaneously-live buffers, and this pass does not model that.
//! * If the cascade runs all the way to a root, the whole prune is reverted.
//!   An e-graph with an empty root e-class has no valid graphs at all, which
//!   is a worse failure than the one being fixed.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::dtype::DType;
use crate::op::EgglogOp;
use crate::shape::{Expression, Symbol};

use super::{
    ClassId, NodeId, SerializedEGraph, api::SortDef, extract_expr_list, try_extract_dtype,
};

/// Field names, in priority order, that hold an OpKind's *output* shape.
/// Anything else (iteration shapes, input shapes, index shapes) is not an
/// allocation this pass can reason about.
const OUTPUT_SHAPE_FIELDS: &[&str] = &["shape", "out_shape", "dest_shape"];

/// Bytes assumed per element when the e-graph carries no dtype for a value.
const DEFAULT_DTYPE: DType = DType::F32;

/// Per-e-node ceilings for one device.
///
/// Both are limits on a *single* value's output tensor, not on the graph as a
/// whole. A backend reports these from [`crate::op::Runtime::enode_resource_limits`];
/// leaving both `None` disables the pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EnodeResourceLimits {
    /// Largest single buffer the device can allocate, in bytes.
    pub max_output_bytes: Option<usize>,
    /// Largest number of output elements one kernel launch can cover. Backends
    /// whose dispatch packet counts work-items in a fixed-width field hit this
    /// before they hit the memory ceiling.
    pub max_output_elements: Option<usize>,
}

impl EnodeResourceLimits {
    /// No limits configured: the pass is a no-op.
    pub fn is_unbounded(&self) -> bool {
        self.max_output_bytes.is_none() && self.max_output_elements.is_none()
    }

    fn violation(&self, elements: usize, bytes: usize) -> Option<LimitViolation> {
        if self.max_output_elements.is_some_and(|max| elements > max) {
            return Some(LimitViolation::Elements);
        }
        if self.max_output_bytes.is_some_and(|max| bytes > max) {
            return Some(LimitViolation::Bytes);
        }
        None
    }
}

/// Which ceiling an e-node broke. The same oversized tensor usually breaks
/// both; this records the first one checked, so a graph that is only over the
/// dispatch limit is not reported as an out-of-memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitViolation {
    /// Too large to allocate.
    Bytes,
    /// Too many work-items to dispatch.
    Elements,
}

impl std::fmt::Display for LimitViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LimitViolation::Bytes => write!(f, "memory"),
            LimitViolation::Elements => write!(f, "dispatch"),
        }
    }
}

/// One OpKind e-node that exceeds a device limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OversizedEnode {
    /// The OpKind label, e.g. `"Mul"` or `"FusionEnd"`.
    pub label: String,
    /// Output elements the e-node produces.
    pub elements: usize,
    /// Output bytes the e-node produces.
    pub bytes: usize,
    /// The ceiling it broke.
    pub limit: LimitViolation,
}

impl std::fmt::Display for OversizedEnode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} over {} limit ({} elements, {:.2} GiB)",
            self.label,
            self.limit,
            self.elements,
            self.bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        )
    }
}

/// What [`prune_oversized_enodes`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EnodePruneReport {
    /// Distinct oversized OpKind e-nodes found, largest first. Retained even
    /// when the prune was reverted — this is the diagnostic for a graph that
    /// cannot run on the device at all.
    pub oversized: Vec<OversizedEnode>,
    /// E-nodes actually removed, including everything the cascade took.
    pub removed_enodes: usize,
    /// E-classes emptied by the cascade and dropped.
    pub removed_eclasses: usize,
    /// The prune would have emptied a root e-class, so the e-graph was left
    /// untouched. The search will run against the original space and most
    /// likely fail, but with its own error rather than "No valid graphs
    /// present in the e-graph!".
    pub reverted: bool,
}

impl EnodePruneReport {
    /// Whether the pass changed the e-graph.
    pub fn pruned_anything(&self) -> bool {
        self.removed_enodes > 0
    }

    /// One-line summary for search logging.
    pub fn summary(&self) -> String {
        if self.reverted {
            return format!(
                "reverted: pruning {} oversized e-node(s) would leave no valid graph (largest: {})",
                self.oversized.len(),
                self.oversized
                    .first()
                    .map(|node| node.to_string())
                    .unwrap_or_else(|| "none".to_string()),
            );
        }
        if !self.pruned_anything() {
            return "no oversized e-nodes".to_string();
        }
        format!(
            "dropped {} e-node(s) and {} e-class(es) over device limits (largest: {})",
            self.removed_enodes,
            self.removed_eclasses,
            self.oversized
                .first()
                .map(|node| node.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
        )
    }
}

/// Remove every e-node whose own output tensor exceeds `limits`, then cascade
/// out the e-nodes that depended on them.
///
/// Reverts and reports rather than emptying a root e-class. See the module
/// docs for why this is sound and what it does not catch.
pub fn prune_oversized_enodes(
    egraph: &mut SerializedEGraph,
    ops: &[Arc<Box<dyn EgglogOp>>],
    limits: EnodeResourceLimits,
    dyn_map: &crate::shape::DynMap,
) -> EnodePruneReport {
    if limits.is_unbounded() {
        return EnodePruneReport::default();
    }
    let sorts: FxHashMap<String, SortDef> = ops
        .iter()
        .map(|op| op.sort())
        .filter(|sort| sort.class == "OpKind")
        .map(|sort| (sort.name.clone(), sort))
        .collect();
    if sorts.is_empty() {
        return EnodePruneReport::default();
    }

    // Expression extraction panics on malformed terms. A prepass that cannot
    // measure the graph must degrade to doing nothing, never abort the
    // compile, so the whole measurement phase is caught as one unit.
    let Ok((oversized_nodes, mut oversized)) = std::panic::catch_unwind(AssertUnwindSafe(|| {
        find_oversized_enodes(egraph, &sorts, limits, dyn_map)
    })) else {
        return EnodePruneReport::default();
    };
    if oversized_nodes.is_empty() {
        return EnodePruneReport::default();
    }
    oversized.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.label.cmp(&b.label)));
    oversized.dedup();

    let before_enodes = egraph.enodes.len();
    let before_eclasses = egraph.eclasses.len();

    let mut pruned = egraph.clone();
    for node in &oversized_nodes {
        pruned.enodes.remove(node);
    }
    cascade_empty_eclasses(&mut pruned);

    // An empty root e-class means every way of computing the output was
    // pruned. Better to hand the untouched space to the search and let the
    // candidate filter report the real limit than to panic during extraction.
    if pruned.roots.iter().any(|root| {
        pruned
            .eclasses
            .get(root)
            .is_none_or(|(_, nodes)| nodes.is_empty())
    }) {
        return EnodePruneReport {
            oversized,
            reverted: true,
            ..Default::default()
        };
    }

    let report = EnodePruneReport {
        oversized,
        removed_enodes: before_enodes.saturating_sub(pruned.enodes.len()),
        removed_eclasses: before_eclasses.saturating_sub(pruned.eclasses.len()),
        reverted: false,
    };
    *egraph = pruned;
    report
}

/// Measure every OpKind e-node and collect the ones over the limits.
fn find_oversized_enodes(
    egraph: &SerializedEGraph,
    sorts: &FxHashMap<String, SortDef>,
    limits: EnodeResourceLimits,
    dyn_map: &crate::shape::DynMap,
) -> (Vec<NodeId>, Vec<OversizedEnode>) {
    let dtypes = kind_class_dtypes(egraph);
    let mut list_cache = FxHashMap::default();
    let mut expr_cache = FxHashMap::default();
    let mut nodes = Vec::new();
    let mut oversized = Vec::new();

    // Sorted so the report and the pruned set are identical run to run.
    for node in egraph
        .enodes
        .keys()
        .sorted_by(|a, b| a.as_ref().cmp(b.as_ref()))
    {
        let (label, children) = &egraph.enodes[node];
        let Some(sort) = sorts.get(label.as_str()) else {
            continue;
        };
        let Some(elements) = output_elements(
            egraph,
            sort,
            children,
            &mut list_cache,
            &mut expr_cache,
            dyn_map,
        ) else {
            continue;
        };
        let dtype = kind_dtype(egraph, sort, children)
            .or_else(|| dtypes.get(&egraph.node_to_class[node]).copied())
            .unwrap_or(DEFAULT_DTYPE);
        let bytes = elements.saturating_mul(dtype.bits()).div_ceil(8);
        if let Some(limit) = limits.violation(elements, bytes) {
            nodes.push(node.clone());
            oversized.push(OversizedEnode {
                label: label.clone(),
                elements,
                bytes,
                limit,
            });
        }
    }
    (nodes, oversized)
}

/// Number of elements in an OpKind e-node's output tensor, or `None` when the
/// sort has no output-shape field or the shape is not fully resolved by
/// `dyn_map`.
fn output_elements<'a>(
    egraph: &'a SerializedEGraph,
    sort: &SortDef,
    kind_children: &'a [ClassId],
    list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
    expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
    dyn_map: &crate::shape::DynMap,
) -> Option<usize> {
    let field = OUTPUT_SHAPE_FIELDS
        .iter()
        .find(|name| field_index(sort, name).is_some_and(|i| sort.fields[i].sort == "EList"))?;
    let class = kind_children.get(field_index(sort, field)?)?;
    // `extract_expr_list` only understands a spine of `ECons`/`ENil`. An
    // e-class whose first e-node is an unapplied list rewrite (`RowMajor`,
    // `RemoveNthFromEnd`, …) is left unmeasured rather than walked into.
    let head = first_node_labeled(egraph, class, &["ECons", "ENil"])?;
    let elements: Expression = extract_expr_list(egraph, head, list_cache, expr_cache)?
        .into_iter()
        .product::<Expression>()
        .max(1);
    // Loop bodies are measured one iteration at a time; the reserved loop
    // index is a placeholder, not a dimension.
    let mut dyn_map = dyn_map.clone();
    dyn_map.entry(Symbol::reserved_index()).or_insert(1);
    elements.simplify().exec(&dyn_map)
}

/// The dtype an OpKind declares directly, for sorts that carry one.
fn kind_dtype(
    egraph: &SerializedEGraph,
    sort: &SortDef,
    kind_children: &[ClassId],
) -> Option<DType> {
    let index =
        field_index(sort, "dtype").filter(|&index| sort.fields[index].sort == "DType")?;
    dtype_of_class(egraph, kind_children.get(index)?)
}

/// Read the dtype a DType e-class stands for.
///
/// A DType e-class holds more than its constructor: every `(dtype ?ir)`
/// application that resolved to this dtype is an e-node in the same class. So
/// this looks for the e-node that actually names a dtype rather than taking
/// whichever one happens to be stored first.
fn dtype_of_class(egraph: &SerializedEGraph, class: &ClassId) -> Option<DType> {
    egraph
        .eclasses
        .get(class)?
        .1
        .iter()
        .filter(|node| egraph.enodes.contains_key(*node))
        .find_map(|node| try_extract_dtype(egraph, node))
}

/// Dtypes reachable through the egglog `dtype` function, keyed by the OpKind
/// e-class of the `Op` e-nodes that carry them. Sorts without a dtype field
/// (the HLIR elementwise ops) get their element size from here.
///
/// When one kind e-class is shared by `Op` e-nodes of differing dtype, the
/// narrowest wins: this pass must never overstate a tensor's size.
fn kind_class_dtypes(egraph: &SerializedEGraph) -> FxHashMap<ClassId, DType> {
    let mut ir_dtypes: FxHashMap<&ClassId, DType> = FxHashMap::default();
    for (node, (label, children)) in &egraph.enodes {
        if label != "dtype" {
            continue;
        }
        let (Some(ir_class), Some(dtype_class)) = (children.first(), egraph.node_to_class.get(node))
        else {
            continue;
        };
        // `dtype` is a `:merge new` function, so the e-class settles on one
        // dtype; an e-class that names none is skipped rather than fatal.
        let Some(dtype) = dtype_of_class(egraph, dtype_class) else {
            continue;
        };
        ir_dtypes.insert(ir_class, dtype);
    }

    let mut kind_dtypes: FxHashMap<ClassId, DType> = FxHashMap::default();
    for (node, (label, children)) in &egraph.enodes {
        if label != "Op" {
            continue;
        }
        let (Some(kind_class), Some(ir_class)) = (children.first(), egraph.node_to_class.get(node))
        else {
            continue;
        };
        let Some(&dtype) = ir_dtypes.get(ir_class) else {
            continue;
        };
        kind_dtypes
            .entry(kind_class.clone())
            .and_modify(|existing| {
                if dtype.bits() < existing.bits() {
                    *existing = dtype;
                }
            })
            .or_insert(dtype);
    }
    kind_dtypes
}

fn field_index(sort: &SortDef, name: &str) -> Option<usize> {
    sort.fields.iter().position(|field| field.name == name)
}

/// The first surviving e-node in `class` carrying one of `labels`.
fn first_node_labeled<'a>(
    egraph: &'a SerializedEGraph,
    class: &ClassId,
    labels: &[&str],
) -> Option<&'a NodeId> {
    egraph.eclasses.get(class)?.1.iter().find(|node| {
        egraph
            .enodes
            .get(*node)
            .is_some_and(|(label, _)| labels.contains(&label.as_str()))
    })
}

/// Drop e-nodes that reference an emptied e-class until nothing changes, then
/// drop the e-classes that are left empty. Mirrors the cascade the serializer
/// already runs after egglog's own cleanup.
fn cascade_empty_eclasses(egraph: &mut SerializedEGraph) {
    loop {
        let mut to_remove = Vec::new();
        for (node, (_, children)) in &egraph.enodes {
            if children.iter().any(|class| {
                egraph.eclasses.get(class).is_none_or(|(_, nodes)| {
                    !nodes.iter().any(|node| egraph.enodes.contains_key(node))
                })
            }) {
                to_remove.push(node.clone());
            }
        }
        if to_remove.is_empty() {
            break;
        }
        for node in to_remove {
            egraph.enodes.remove(&node);
        }
    }

    for (_, nodes) in egraph.eclasses.values_mut() {
        nodes.retain(|node| egraph.enodes.contains_key(node));
    }
    egraph.eclasses.retain(|_, (_, nodes)| !nodes.is_empty());
    egraph
        .node_to_class
        .retain(|node, _| egraph.enodes.contains_key(node));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::IntoEgglogOp;

    /// Hand-built e-graph shaped like the failing matmul: an output e-class
    /// with two alternatives, one of which routes through a huge intermediate.
    ///
    /// ```text
    /// out ─┬─ Op(Sin, [mid])   mid = Op(Log2, [in])  shape [big]
    ///      └─ Op(Exp2, [in])                         shape [4]
    /// ```
    ///
    /// `direct_alternative` controls whether the second, cheap arm exists.
    struct Builder {
        egraph: SerializedEGraph,
        next: usize,
    }

    impl Builder {
        fn new() -> Self {
            Self {
                egraph: SerializedEGraph {
                    enodes: FxHashMap::default(),
                    eclasses: FxHashMap::default(),
                    node_to_class: FxHashMap::default(),
                    roots: Vec::new(),
                },
                next: 0,
            }
        }

        /// Add a one-e-node e-class and return its id.
        fn class(&mut self, sort: &str, label: &str, children: Vec<ClassId>) -> ClassId {
            let class = ClassId::from(format!("{sort}-{}", self.next));
            let node = NodeId::from(format!("node-{}", self.next));
            self.next += 1;
            self.egraph
                .enodes
                .insert(node.clone(), (label.to_string(), children));
            self.egraph
                .eclasses
                .insert(class.clone(), (sort.to_string(), vec![node.clone()]));
            self.egraph.node_to_class.insert(node, class.clone());
            class
        }

        /// Add another e-node to an existing e-class: a search choice.
        fn alternative(&mut self, class: &ClassId, label: &str, children: Vec<ClassId>) {
            let node = NodeId::from(format!("node-{}", self.next));
            self.next += 1;
            self.egraph
                .enodes
                .insert(node.clone(), (label.to_string(), children));
            self.egraph
                .eclasses
                .get_mut(class)
                .unwrap()
                .1
                .push(node.clone());
            self.egraph.node_to_class.insert(node, class.clone());
        }

        /// A single-dimension EList holding `dim`.
        fn shape(&mut self, dim: usize) -> ClassId {
            let literal = self.class("i64", &dim.to_string(), vec![]);
            let expr = self.class("Expression", "MNum", vec![literal]);
            let nil = self.class("EList", "ENil", vec![]);
            self.class("EList", "ECons", vec![expr, nil])
        }

        /// An OpKind e-class for a unary sort: (shape, strides, out_strides).
        fn unary_kind(&mut self, label: &str, dim: usize) -> ClassId {
            let shape = self.shape(dim);
            self.class(
                "OpKind",
                label,
                vec![shape.clone(), shape.clone(), shape],
            )
        }

        /// An IR e-class holding `Op(kind, [input])`.
        fn op(&mut self, kind: ClassId, input: ClassId) -> ClassId {
            let nil = self.class("IList", "INil", vec![]);
            let list = self.class("IList", "ICons", vec![input, nil]);
            self.class("IR", "Op", vec![kind, list])
        }
    }

    fn build(direct_alternative: bool) -> SerializedEGraph {
        let mut b = Builder::new();
        let node_id = b.class("i64", "0", vec![]);
        let label = b.class("String", "\"\"", vec![]);
        let dtype = b.class("DType", "F32", vec![]);
        let input = b.class("IR", "Input", vec![node_id, label, dtype]);

        let big_kind = b.unary_kind("Log2", 1 << 20);
        let mid = b.op(big_kind, input.clone());

        let via_mid_kind = b.unary_kind("Sin", 4);
        let out = b.op(via_mid_kind, mid);

        if direct_alternative {
            let direct_kind = b.unary_kind("Exp2", 4);
            let nil = b.class("IList", "INil", vec![]);
            let list = b.class("IList", "ICons", vec![input, nil]);
            b.alternative(&out, "Op", vec![direct_kind, list]);
        }

        b.egraph.roots = vec![out];
        b.egraph
    }

    fn hlir_ops() -> Vec<Arc<Box<dyn EgglogOp>>> {
        <crate::hlir::HLIROps as IntoEgglogOp>::into_vec()
    }

    /// 1 Mi F32 elements = 4 MiB; the cheap arms are 4 elements = 16 B.
    fn four_mib_limit() -> EnodeResourceLimits {
        EnodeResourceLimits {
            max_output_bytes: Some(1 << 20),
            max_output_elements: None,
        }
    }

    #[test]
    fn prunes_oversized_alternative_and_cascades_to_its_consumers() {
        let mut egraph = build(true);
        let report = prune_oversized_enodes(
            &mut egraph,
            &hlir_ops(),
            four_mib_limit(),
            &Default::default(),
        );

        assert!(!report.reverted);
        assert_eq!(report.oversized.len(), 1);
        assert_eq!(report.oversized[0].label, "Log2");
        assert_eq!(report.oversized[0].elements, 1 << 20);
        assert_eq!(report.oversized[0].bytes, 4 << 20);

        // The oversized kind is gone, the `Op` that consumed it cascaded out
        // with it, and so did the arm of the root that routed through that
        // `Op`. The cheap arm is what the search is now forced to pick.
        assert!(
            !egraph
                .enodes
                .values()
                .any(|(label, _)| label == "Log2"),
            "the oversized kind survived the prune"
        );
        let root = &egraph.roots[0];
        let (_, root_nodes) = &egraph.eclasses[root];
        assert_eq!(root_nodes.len(), 1, "both arms of the root should not remain");
        let kind_class = &egraph.enodes[&root_nodes[0]].1[0];
        assert!(
            egraph.eclasses[kind_class]
                .1
                .iter()
                .any(|node| egraph.enodes[node].0 == "Exp2"),
            "the surviving arm should be the one that never built the big tensor"
        );
    }

    #[test]
    fn reverts_rather_than_emptying_a_root_eclass() {
        let mut egraph = build(false);
        let before = egraph.clone();
        let report = prune_oversized_enodes(
            &mut egraph,
            &hlir_ops(),
            four_mib_limit(),
            &Default::default(),
        );

        assert!(report.reverted);
        assert_eq!(report.removed_enodes, 0);
        assert_eq!(report.oversized.len(), 1);
        assert_eq!(egraph.enodes.len(), before.enodes.len());
        assert_eq!(egraph.eclasses.len(), before.eclasses.len());
    }

    #[test]
    fn element_count_limit_fires_independently_of_the_byte_limit() {
        let mut egraph = build(true);
        let report = prune_oversized_enodes(
            &mut egraph,
            &hlir_ops(),
            EnodeResourceLimits {
                max_output_bytes: None,
                max_output_elements: Some(1024),
            },
            &Default::default(),
        );
        assert!(!report.reverted);
        assert_eq!(report.oversized[0].label, "Log2");
    }

    /// End-to-end over a real egglog-built search space: the HLIR matmul is
    /// `Sum(Mul(...))`, so the `Mul` holds the whole `[M, N, K]` product. Its
    /// dtype comes from the egglog `dtype` function, not from a field on the
    /// kind, which is the path the elementwise ops take.
    #[test]
    fn measures_the_product_tensor_of_a_real_matmul() {
        let mut cx = crate::prelude::Graph::new();
        let a = cx.tensor((8, 8));
        let b = cx.tensor((8, 8));
        let _out = a.matmul(b).output();
        cx.build_search_space::<crate::hlir::ReferenceRuntime>(
            crate::graph::CompileOptions::default(),
        );

        let ops = cx.egglog_ops().unwrap().clone();
        let mut egraph = cx.egraph().unwrap().clone();
        // Between the 8·8·8 product (2 KiB of f32) and the 8·8 answer.
        let report = prune_oversized_enodes(
            &mut egraph,
            &ops,
            EnodeResourceLimits {
                max_output_bytes: Some(1024),
                max_output_elements: None,
            },
            &cx.dyn_map,
        );

        let product = report
            .oversized
            .iter()
            .find(|node| node.label == "Mul")
            .expect("the [M, N, K] product tensor should be over the limit");
        assert_eq!(product.elements, 8 * 8 * 8);
        assert_eq!(product.bytes, 8 * 8 * 8 * 4);
        // The reference backend has no fused matmul, so the product is the
        // only way to compute the output and the prune must back off.
        assert!(report.reverted);
    }

    #[test]
    fn unbounded_limits_leave_the_egraph_untouched() {
        let mut egraph = build(true);
        let before = egraph.clone();
        let report = prune_oversized_enodes(
            &mut egraph,
            &hlir_ops(),
            EnodeResourceLimits::default(),
            &Default::default(),
        );
        assert!(!report.pruned_anything());
        assert!(!report.reverted);
        assert_eq!(egraph.enodes.len(), before.enodes.len());
    }

    #[test]
    fn a_graph_within_limits_is_untouched() {
        let mut egraph = build(true);
        let before = egraph.clone();
        let report = prune_oversized_enodes(
            &mut egraph,
            &hlir_ops(),
            EnodeResourceLimits {
                max_output_bytes: Some(1 << 30),
                max_output_elements: Some(1 << 30),
            },
            &Default::default(),
        );
        assert!(!report.pruned_anything());
        assert_eq!(egraph.enodes.len(), before.enodes.len());
    }
}
