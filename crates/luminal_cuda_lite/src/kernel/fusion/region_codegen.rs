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
    hlir::Input,
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
// Candidate-local region validation.
// =========================================================================

/// A selected fusion graph violated a contract required by region codegen.
///
/// Fusion alternatives remain in the egraph. This error rejects only the
/// selected LLIR candidate; it never deletes an alternative globally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FusionValidationError {
    pub node: NodeIndex,
    pub reason: String,
}

impl std::fmt::Display for FusionValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fusion node {} is invalid: {}",
            self.node.index(),
            self.reason
        )
    }
}

impl std::error::Error for FusionValidationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SemanticRelation {
    Equal,
    Different,
    Unknown,
}

impl SemanticRelation {
    fn conjunction(self, other: Self) -> Self {
        match (self, other) {
            (Self::Different, _) | (_, Self::Different) => Self::Different,
            (Self::Equal, Self::Equal) => Self::Equal,
            _ => Self::Unknown,
        }
    }

    fn either(self, other: Self) -> Self {
        match (self, other) {
            (Self::Equal, _) | (_, Self::Equal) => Self::Equal,
            (Self::Different, Self::Different) => Self::Different,
            _ => Self::Unknown,
        }
    }
}

#[derive(Clone, Copy)]
struct FusionLayout<'a> {
    shape: &'a [Expression],
    strides: &'a [Expression],
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OwnedFusionLayout {
    shape: Vec<Expression>,
    strides: Vec<Expression>,
}

impl From<FusionLayout<'_>> for OwnedFusionLayout {
    fn from(layout: FusionLayout<'_>) -> Self {
        Self {
            shape: layout.shape.to_vec(),
            strides: layout.strides.to_vec(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ExpressionProofKey {
    lhs: Expression,
    rhs: Expression,
    z_witnesses: Vec<usize>,
    exhaustive: bool,
}

#[derive(Default)]
struct FusionValidationCache {
    expressions: FxHashMap<ExpressionProofKey, SemanticRelation>,
    symbolic_expressions: FxHashMap<(Expression, Expression), bool>,
    layouts: FxHashMap<(OwnedFusionLayout, OwnedFusionLayout), SemanticRelation>,
}

/// Deterministic associative/commutative normalization for the expression
/// operators where operand order and grouping have no semantic meaning. This
/// proves common layout equalities without saturating an egraph with every
/// permutation of a long shape expression.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum CanonicalSemanticExpression {
    Number(i64),
    Variable(char),
    Binary(u8, Box<Self>, Box<Self>),
    AssociativeCommutative(u8, Vec<Self>),
}

fn expression_operator_tag(term: Term) -> Option<(u8, bool)> {
    Some(match term {
        Term::Add => (0, true),
        Term::Sub => (1, false),
        Term::Mul => (2, true),
        Term::Div => (3, false),
        Term::CeilDiv => (4, false),
        Term::Mod => (5, false),
        Term::Min => (6, true),
        Term::Max => (7, true),
        Term::And => (8, true),
        Term::Or => (9, true),
        Term::Gte => (10, false),
        Term::Lt => (11, false),
        Term::Num(_) | Term::Var(_) => return None,
    })
}

fn append_associative_operand(
    tag: u8,
    operand: CanonicalSemanticExpression,
    operands: &mut Vec<CanonicalSemanticExpression>,
) {
    match operand {
        CanonicalSemanticExpression::AssociativeCommutative(operand_tag, nested)
            if operand_tag == tag =>
        {
            operands.extend(nested);
        }
        operand => operands.push(operand),
    }
}

fn canonical_semantic_expression(expression: Expression) -> Option<CanonicalSemanticExpression> {
    let terms = expression.terms.read();
    let mut stack = Vec::with_capacity(terms.len());
    for &term in terms.iter() {
        match term {
            Term::Num(number) => stack.push(CanonicalSemanticExpression::Number(number)),
            Term::Var(variable) => stack.push(CanonicalSemanticExpression::Variable(variable)),
            operator => {
                // Expression RPN stores rhs before lhs, so the first pop is
                // the source-level left operand.
                let lhs = stack.pop()?;
                let rhs = stack.pop()?;
                let (tag, associative_commutative) = expression_operator_tag(operator)?;
                if associative_commutative {
                    let mut operands = Vec::new();
                    append_associative_operand(tag, lhs, &mut operands);
                    append_associative_operand(tag, rhs, &mut operands);
                    operands.sort_unstable();
                    stack.push(CanonicalSemanticExpression::AssociativeCommutative(
                        tag, operands,
                    ));
                } else {
                    stack.push(CanonicalSemanticExpression::Binary(
                        tag,
                        Box::new(lhs),
                        Box::new(rhs),
                    ));
                }
            }
        }
    }
    (stack.len() == 1).then(|| stack.pop().unwrap())
}

fn invalid(node: NodeIndex, reason: impl Into<String>) -> FusionValidationError {
    FusionValidationError {
        node,
        reason: reason.into(),
    }
}

fn fusion_start(llir: &LLIRGraph, node: NodeIndex) -> Option<&FusionStart> {
    let op = llir[node].to_dialect::<dyn KernelOp>()?;
    (***op).downcast_ref::<FusionStart>()
}

fn fusion_end(llir: &LLIRGraph, node: NodeIndex) -> Option<&FusionEnd> {
    let op = llir[node].to_dialect::<dyn KernelOp>()?;
    (***op).downcast_ref::<FusionEnd>()
}

fn fusion_unary(llir: &LLIRGraph, node: NodeIndex) -> Option<&CudaUnaryElementwise> {
    let op = llir[node].to_dialect::<dyn KernelOp>()?;
    (***op).downcast_ref::<CudaUnaryElementwise>()
}

fn fusion_binary(llir: &LLIRGraph, node: NodeIndex) -> Option<&CudaBinaryElementwise> {
    let op = llir[node].to_dialect::<dyn KernelOp>()?;
    (***op).downcast_ref::<CudaBinaryElementwise>()
}

fn is_fusion_internal(llir: &LLIRGraph, node: NodeIndex) -> bool {
    fusion_start(llir, node).is_some()
        || fusion_unary(llir, node).is_some()
        || fusion_binary(llir, node).is_some()
}

fn fusion_output_layout(llir: &LLIRGraph, node: NodeIndex) -> Option<FusionLayout<'_>> {
    if let Some(start) = fusion_start(llir, node) {
        Some(FusionLayout {
            shape: &start.shape,
            strides: &start.strides,
        })
    } else if let Some(unary) = fusion_unary(llir, node) {
        Some(FusionLayout {
            shape: &unary.shape,
            strides: &unary.out_strides,
        })
    } else if let Some(binary) = fusion_binary(llir, node) {
        Some(FusionLayout {
            shape: &binary.out_shape,
            strides: &binary.out_stride,
        })
    } else {
        fusion_end(llir, node).map(|end| FusionLayout {
            shape: &end.shape,
            strides: &end.strides,
        })
    }
}

fn fusion_output_dtype(llir: &LLIRGraph, node: NodeIndex) -> Option<DType> {
    if let Some(start) = fusion_start(llir, node) {
        Some(start.dtype)
    } else if let Some(unary) = fusion_unary(llir, node) {
        Some(unary.dtype)
    } else if let Some(binary) = fusion_binary(llir, node) {
        Some(binary.dtype)
    } else {
        fusion_end(llir, node).map(|end| end.dtype)
    }
}

fn known_buffer_dtype(llir: &LLIRGraph, node: NodeIndex) -> Option<DType> {
    if let Some(dtype) = fusion_output_dtype(llir, node) {
        return Some(dtype);
    }
    if let Some(kernel) = llir[node].to_dialect::<dyn KernelOp>() {
        return Some(kernel.output_dtype());
    }
    llir[node].to_op::<Input>().map(|input| input.dtype)
}

fn fusion_storage_dtype_supported(dtype: DType) -> bool {
    // These formats are packed (or, for TF32, have a storage width different
    // from `DType::bits()`). Region codegen currently indexes one CUDA scalar
    // per logical element, so treating them as ordinary pointer element types
    // would address and allocate the buffer inconsistently.
    !matches!(
        dtype,
        DType::TF32 | DType::F6E2M3 | DType::F6E3M2 | DType::F4E2M1 | DType::I4 | DType::U4
    )
}

fn fusion_unary_dtype_supported(op: &str, dtype: DType) -> bool {
    if op == "Cast" {
        return fusion_storage_dtype_supported(dtype);
    }
    matches!(
        dtype,
        DType::F32 | DType::F16 | DType::Bf16 | DType::F8E4M3 | DType::F8E5M2 | DType::F8UE8M0
    )
}

fn expression_relation(
    lhs: Expression,
    rhs: Expression,
    dyn_map: Option<&FxHashMap<char, usize>>,
    z_witnesses: &[usize],
    exhaustive: bool,
    cache: &mut FusionValidationCache,
) -> SemanticRelation {
    if lhs == rhs {
        return SemanticRelation::Equal;
    }

    // Keep the expressions symbolic for every positive proof. Resolving a
    // representative dyn_map first would be unsound for dimension buckets:
    // two expressions can coincide at the representative size but disagree at
    // another size covered by the same selected candidate.
    let (lhs_simplified, rhs_simplified) = (lhs.simplify(), rhs.simplify());
    if lhs_simplified == rhs_simplified {
        return SemanticRelation::Equal;
    }

    let key = ExpressionProofKey {
        lhs: lhs_simplified,
        rhs: rhs_simplified,
        z_witnesses: z_witnesses.to_vec(),
        exhaustive,
    };
    if let Some(relation) = cache.expressions.get(&key).copied() {
        return relation;
    }

    let symbolic_key = (lhs_simplified, rhs_simplified);
    let semantically_equal = if let Some(equal) = cache.symbolic_expressions.get(&symbolic_key) {
        *equal
    } else {
        let equal = canonical_semantic_expression(lhs_simplified)
            .zip(canonical_semantic_expression(rhs_simplified))
            .is_some_and(|(lhs, rhs)| lhs == rhs)
            || std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                lhs_simplified.egglog_equal(rhs_simplified)
            }))
            .unwrap_or(false);
        cache.symbolic_expressions.insert(symbolic_key, equal);
        cache
            .symbolic_expressions
            .insert((symbolic_key.1, symbolic_key.0), equal);
        equal
    };
    if semantically_equal {
        cache
            .expressions
            .insert(key.clone(), SemanticRelation::Equal);
        return SemanticRelation::Equal;
    }

    let empty = FxHashMap::default();
    let concrete_map = dyn_map.unwrap_or(&empty);
    let relation = if let (Some(lhs), Some(rhs)) = (
        eval_expression(lhs_simplified, concrete_map),
        eval_expression(rhs_simplified, concrete_map),
    ) {
        if lhs != rhs {
            // One concrete counterexample is sufficient to refute symbolic
            // equality.
            SemanticRelation::Different
        } else if lhs_simplified.to_symbols().is_empty() && rhs_simplified.to_symbols().is_empty() {
            // Constant expressions have no other assignment on which they can
            // diverge. A representative assignment is never used this way.
            SemanticRelation::Equal
        } else {
            SemanticRelation::Unknown
        }
    } else {
        // A single concrete counterexample proves inequality for this
        // candidate. An exhaustive finite-domain match is also a positive
        // proof only when every non-z symbol has already disappeared;
        // non-exhaustive or representative-only matching proves nothing.
        let mut all_evaluated_equal = true;
        let mut different = false;
        for &z in z_witnesses {
            let mut values = concrete_map.clone();
            values.insert('z', z);
            match (
                eval_expression(lhs_simplified, &values),
                eval_expression(rhs_simplified, &values),
            ) {
                (Some(lhs), Some(rhs)) if lhs == rhs => {}
                (Some(_), Some(_)) => {
                    different = true;
                    break;
                }
                _ => all_evaluated_equal = false,
            }
        }
        if different {
            SemanticRelation::Different
        } else if exhaustive
            && all_evaluated_equal
            && lhs_simplified.dyn_vars().is_empty()
            && rhs_simplified.dyn_vars().is_empty()
        {
            SemanticRelation::Equal
        } else {
            SemanticRelation::Unknown
        }
    };

    cache.expressions.insert(key.clone(), relation);
    cache.expressions.insert(
        ExpressionProofKey {
            lhs: key.rhs,
            rhs: key.lhs,
            z_witnesses: key.z_witnesses,
            exhaustive: key.exhaustive,
        },
        relation,
    );
    relation
}

fn eval_expression(expression: Expression, values: &FxHashMap<char, usize>) -> Option<usize> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| expression.exec(values)))
        .ok()
        .flatten()
}

fn layout_relation(
    lhs: FusionLayout<'_>,
    rhs: FusionLayout<'_>,
    dyn_map: Option<&FxHashMap<char, usize>>,
    cache: &mut FusionValidationCache,
) -> SemanticRelation {
    debug_assert_eq!(lhs.shape.len(), lhs.strides.len());
    debug_assert_eq!(rhs.shape.len(), rhs.strides.len());

    let cache_key = (OwnedFusionLayout::from(lhs), OwnedFusionLayout::from(rhs));
    if let Some(relation) = cache.layouts.get(&cache_key).copied() {
        return relation;
    }

    let lhs_elements = lhs.shape.iter().copied().product::<Expression>();
    let rhs_elements = rhs.shape.iter().copied().product::<Expression>();
    let element_relation =
        expression_relation(lhs_elements, rhs_elements, dyn_map, &[], false, cache);
    if element_relation == SemanticRelation::Different {
        cache
            .layouts
            .insert(cache_key.clone(), SemanticRelation::Different);
        cache
            .layouts
            .insert((cache_key.1, cache_key.0), SemanticRelation::Different);
        return SemanticRelation::Different;
    }

    let concrete_elements = dyn_map
        .and_then(|values| {
            eval_expression(lhs_elements, values).zip(eval_expression(rhs_elements, values))
        })
        .or_else(|| {
            let values = FxHashMap::default();
            eval_expression(lhs_elements, &values).zip(eval_expression(rhs_elements, &values))
        })
        .and_then(|(lhs, rhs)| (lhs == rhs).then_some(lhs));
    let symbolically_fixed_domain = element_relation == SemanticRelation::Equal
        && lhs_elements.to_symbols().is_empty()
        && rhs_elements.to_symbols().is_empty();
    let mut z_witnesses = Vec::new();
    let mut exhaustive = false;
    if let Some(elements) = concrete_elements {
        if symbolically_fixed_domain && elements <= 4_096 {
            z_witnesses.extend(0..elements);
            exhaustive = true;
        } else if elements > 0 {
            z_witnesses.extend([0, 1, elements / 2, elements - 1]);
            z_witnesses.retain(|z| *z < elements);
            z_witnesses.sort_unstable();
            z_witnesses.dedup();
        }
    }

    let lhs_index = flatten_strides(lhs.shape, lhs.strides);
    let rhs_index = flatten_strides(rhs.shape, rhs.strides);
    let index_relation = expression_relation(
        lhs_index,
        rhs_index,
        dyn_map,
        &z_witnesses,
        exhaustive,
        cache,
    );
    let relation = element_relation.conjunction(index_relation);
    cache.layouts.insert(cache_key.clone(), relation);
    cache.layouts.insert((cache_key.1, cache_key.0), relation);
    relation
}

fn require_layout_compatible(
    producer: NodeIndex,
    consumer: NodeIndex,
    input_slot: usize,
    produced: FusionLayout<'_>,
    expected: FusionLayout<'_>,
    dyn_map: Option<&FxHashMap<char, usize>>,
    cache: &mut FusionValidationCache,
) -> Result<(), FusionValidationError> {
    match layout_relation(produced, expected, dyn_map, cache) {
        SemanticRelation::Equal => {}
        SemanticRelation::Different => {
            return Err(invalid(
                consumer,
                format!(
                    "input {input_slot} layout is provably incompatible with producer {}",
                    producer.index()
                ),
            ));
        }
        SemanticRelation::Unknown => {
            return Err(invalid(
                consumer,
                format!(
                    "input {input_slot} layout equivalence with producer {} could not be proved",
                    producer.index()
                ),
            ));
        }
    }
    Ok(())
}

fn incoming_nodes(llir: &LLIRGraph, node: NodeIndex) -> Vec<NodeIndex> {
    let mut edges = llir
        .edges_directed(node, Direction::Incoming)
        .map(|edge| (edge.id(), edge.source()))
        .collect::<Vec<_>>();
    edges.sort_by_key(|(edge, _)| *edge);
    edges.into_iter().map(|(_, source)| source).collect()
}

/// Validate only contracts that region codegen actually relies on. Expression
/// relations are tri-state: structural/concrete/semantic equality is accepted,
/// concrete inequality is rejected as malformed, and an unproved relation is
/// rejected as an unsupported layout because codegen erases that metadata.
pub(crate) fn validate_fusion_regions(
    llir: &LLIRGraph,
    dyn_map: Option<&FxHashMap<char, usize>>,
) -> Result<(), FusionValidationError> {
    let mut validation_cache = FusionValidationCache::default();
    for node in llir.node_indices() {
        let inputs = incoming_nodes(llir, node);

        if let Some(start) = fusion_start(llir, node) {
            if inputs.len() != 1 {
                return Err(invalid(
                    node,
                    format!("FusionStart requires 1 input, found {}", inputs.len()),
                ));
            }
            if start.shape.len() != start.strides.len() {
                return Err(invalid(
                    node,
                    format!(
                        "FusionStart shape rank {} differs from stride rank {}",
                        start.shape.len(),
                        start.strides.len()
                    ),
                ));
            }
            if !fusion_storage_dtype_supported(start.dtype) {
                return Err(invalid(
                    node,
                    format!("FusionStart does not support packed dtype {}", start.dtype),
                ));
            }
            if is_fusion_internal(llir, inputs[0]) {
                return Err(invalid(
                    node,
                    "FusionStart must read a materialized external value, not a region-internal node",
                ));
            }
            if let Some(producer_dtype) = known_buffer_dtype(llir, inputs[0])
                && producer_dtype != start.dtype
            {
                let consumer = llir
                    .neighbors_directed(node, Direction::Outgoing)
                    .next()
                    .map(|n| format!("{:?}", llir[n]))
                    .unwrap_or_default();
                return Err(invalid(
                    node,
                    format!(
                        "FusionStart dtype {} disagrees with external producer dtype {} \
                         (producer: {:.120}; consumer: {consumer:.120})",
                        start.dtype,
                        producer_dtype,
                        format!("{:?}", llir[inputs[0]])
                    ),
                ));
            }
        } else if let Some(end) = fusion_end(llir, node) {
            if inputs.len() != 1 {
                return Err(invalid(
                    node,
                    format!("FusionEnd requires 1 input, found {}", inputs.len()),
                ));
            }
            if end.shape.len() != end.strides.len() {
                return Err(invalid(
                    node,
                    format!(
                        "FusionEnd shape rank {} differs from stride rank {}",
                        end.shape.len(),
                        end.strides.len()
                    ),
                ));
            }
            if !fusion_storage_dtype_supported(end.dtype) {
                return Err(invalid(
                    node,
                    format!("FusionEnd does not support packed dtype {}", end.dtype),
                ));
            }
            if fusion_end(llir, inputs[0]).is_some() || !is_fusion_internal(llir, inputs[0]) {
                return Err(invalid(
                    node,
                    "FusionEnd input must be a FusionStart or Cuda*Elementwise node",
                ));
            }
        } else if let Some(unary) = fusion_unary(llir, node) {
            if inputs.len() != 1 {
                return Err(invalid(
                    node,
                    format!(
                        "CudaUnaryElementwise requires 1 input, found {}",
                        inputs.len()
                    ),
                ));
            }
            if unary.shape.len() != unary.in_strides.len()
                || unary.shape.len() != unary.out_strides.len()
            {
                return Err(invalid(
                    node,
                    format!(
                        "CudaUnaryElementwise metadata ranks differ: shape {}, input {}, output {}",
                        unary.shape.len(),
                        unary.in_strides.len(),
                        unary.out_strides.len()
                    ),
                ));
            }
            if !matches!(
                unary.op.as_str(),
                "Sin" | "Sqrt" | "Rsqrt" | "Exp" | "Exp2" | "Log2" | "Recip" | "Sigmoid" | "Cast"
            ) {
                return Err(invalid(
                    node,
                    format!("unsupported unary region opcode {:?}", unary.op),
                ));
            }
            if !fusion_unary_dtype_supported(&unary.op, unary.dtype) {
                return Err(invalid(
                    node,
                    format!(
                        "unary opcode {} does not support dtype {}",
                        unary.op, unary.dtype
                    ),
                ));
            }
            if fusion_end(llir, inputs[0]).is_some() || !is_fusion_internal(llir, inputs[0]) {
                return Err(invalid(
                    node,
                    "CudaUnaryElementwise input must be a FusionStart or Cuda*Elementwise node",
                ));
            }
        } else if let Some(binary) = fusion_binary(llir, node) {
            if inputs.len() != 2 {
                return Err(invalid(
                    node,
                    format!(
                        "CudaBinaryElementwise requires 2 inputs, found {}",
                        inputs.len()
                    ),
                ));
            }
            if binary.out_shape.len() != binary.a_stride.len()
                || binary.out_shape.len() != binary.b_stride.len()
                || binary.out_shape.len() != binary.out_stride.len()
            {
                return Err(invalid(
                    node,
                    format!(
                        "CudaBinaryElementwise metadata ranks differ: shape {}, lhs {}, rhs {}, output {}",
                        binary.out_shape.len(),
                        binary.a_stride.len(),
                        binary.b_stride.len(),
                        binary.out_stride.len()
                    ),
                ));
            }
            if !matches!(binary.op.as_str(), "Add" | "Mul") {
                return Err(invalid(
                    node,
                    format!("unsupported binary region opcode {:?}", binary.op),
                ));
            }
            if !fusion_storage_dtype_supported(binary.dtype) || binary.dtype == DType::Bool {
                return Err(invalid(
                    node,
                    format!(
                        "binary opcode {} does not support dtype {}",
                        binary.op, binary.dtype
                    ),
                ));
            }
            if inputs.iter().any(|input| {
                fusion_end(llir, *input).is_some() || !is_fusion_internal(llir, *input)
            }) {
                return Err(invalid(
                    node,
                    "CudaBinaryElementwise inputs must be FusionStart or Cuda*Elementwise nodes",
                ));
            }
        } else {
            continue;
        }

        // Region internals have no standalone compile implementation. Every
        // outgoing edge must remain in a region; an acyclic chain therefore
        // terminates at a FusionEnd.
        if is_fusion_internal(llir, node) {
            let consumers = llir
                .neighbors_directed(node, Direction::Outgoing)
                .collect::<Vec<_>>();
            if consumers.is_empty() {
                return Err(invalid(
                    node,
                    "region-internal node has no FusionEnd consumer",
                ));
            }
            if consumers.iter().any(|consumer| {
                fusion_end(llir, *consumer).is_none()
                    && fusion_unary(llir, *consumer).is_none()
                    && fusion_binary(llir, *consumer).is_none()
            }) {
                return Err(invalid(
                    node,
                    "region-internal output escapes without a FusionEnd materialization",
                ));
            }
        }

        // A materialized FE may feed arbitrary external work or a downstream
        // FS, but may not be treated as an in-register local directly.
        if fusion_end(llir, node).is_some()
            && llir
                .neighbors_directed(node, Direction::Outgoing)
                .any(|consumer| {
                    fusion_end(llir, consumer).is_some()
                        || fusion_unary(llir, consumer).is_some()
                        || fusion_binary(llir, consumer).is_some()
                })
        {
            return Err(invalid(
                node,
                "FusionEnd must cross a FusionStart before entering another region",
            ));
        }
    }

    // Validate per-edge dtype and layout contracts. Unknown symbolic layout
    // equality fails closed because interior layout metadata is erased.
    for consumer in llir.node_indices() {
        let inputs = incoming_nodes(llir, consumer);
        if let Some(end) = fusion_end(llir, consumer) {
            let producer = inputs[0];
            let producer_dtype = fusion_output_dtype(llir, producer).unwrap();
            if producer_dtype != end.dtype {
                return Err(invalid(
                    consumer,
                    format!(
                        "FusionEnd dtype {} disagrees with producer dtype {}",
                        end.dtype, producer_dtype
                    ),
                ));
            }
            require_layout_compatible(
                producer,
                consumer,
                0,
                fusion_output_layout(llir, producer).unwrap(),
                FusionLayout {
                    shape: &end.shape,
                    strides: &end.strides,
                },
                dyn_map,
                &mut validation_cache,
            )?;
        } else if let Some(unary) = fusion_unary(llir, consumer) {
            let producer = inputs[0];
            if unary.op != "Cast" {
                let producer_dtype = fusion_output_dtype(llir, producer).unwrap();
                if producer_dtype != unary.dtype {
                    return Err(invalid(
                        consumer,
                        format!(
                            "unary dtype {} disagrees with producer dtype {}",
                            unary.dtype, producer_dtype
                        ),
                    ));
                }
            }
            require_layout_compatible(
                producer,
                consumer,
                0,
                fusion_output_layout(llir, producer).unwrap(),
                FusionLayout {
                    shape: &unary.shape,
                    strides: &unary.in_strides,
                },
                dyn_map,
                &mut validation_cache,
            )?;
        } else if let Some(binary) = fusion_binary(llir, consumer) {
            for &producer in &inputs {
                let producer_dtype = fusion_output_dtype(llir, producer).unwrap();
                if producer_dtype != binary.dtype {
                    return Err(invalid(
                        consumer,
                        format!(
                            "binary dtype {} disagrees with producer {} dtype {}",
                            binary.dtype,
                            producer.index(),
                            producer_dtype
                        ),
                    ));
                }
            }

            let a = FusionLayout {
                shape: &binary.out_shape,
                strides: &binary.a_stride,
            };
            let b = FusionLayout {
                shape: &binary.out_shape,
                strides: &binary.b_stride,
            };
            let p0 = fusion_output_layout(llir, inputs[0]).unwrap();
            let p1 = fusion_output_layout(llir, inputs[1]).unwrap();
            let direct = layout_relation(p0, a, dyn_map, &mut validation_cache)
                .conjunction(layout_relation(p1, b, dyn_map, &mut validation_cache));
            let swapped = layout_relation(p0, b, dyn_map, &mut validation_cache)
                .conjunction(layout_relation(p1, a, dyn_map, &mut validation_cache));
            match direct.either(swapped) {
                SemanticRelation::Equal => {}
                SemanticRelation::Different => {
                    return Err(invalid(
                        consumer,
                        "both binary operand layout assignments are provably incompatible",
                    ));
                }
                SemanticRelation::Unknown => {
                    return Err(invalid(
                        consumer,
                        "binary operand layout equivalence could not be proved",
                    ));
                }
            }
        }
    }

    // Region codegen indexes every FS with the root FE's iteration shape. This
    // check catches a rank panic immediately and requires a positive proof that
    // the emitted and declared mappings agree.
    for fe_node in llir
        .node_indices()
        .filter(|node| fusion_end(llir, *node).is_some())
    {
        let end = fusion_end(llir, fe_node).unwrap();
        let mut visited = FxHashSet::default();
        let mut stack = incoming_nodes(llir, fe_node);
        while let Some(node) = stack.pop() {
            if !visited.insert(node) {
                continue;
            }
            if let Some(start) = fusion_start(llir, node) {
                if end.shape.len() != start.strides.len() {
                    return Err(invalid(
                        node,
                        format!(
                            "FusionStart stride rank {} cannot be indexed by FusionEnd rank {}",
                            start.strides.len(),
                            end.shape.len()
                        ),
                    ));
                }
                let emitted = FusionLayout {
                    shape: &end.shape,
                    strides: &start.strides,
                };
                let declared = FusionLayout {
                    shape: &start.shape,
                    strides: &start.strides,
                };
                match layout_relation(emitted, declared, dyn_map, &mut validation_cache) {
                    SemanticRelation::Equal => {}
                    SemanticRelation::Different => {
                        return Err(invalid(
                            node,
                            format!(
                                "FusionEnd {} iteration shape provably changes this FusionStart read mapping",
                                fe_node.index()
                            ),
                        ));
                    }
                    SemanticRelation::Unknown => {
                        return Err(invalid(
                            node,
                            format!(
                                "FusionEnd {} iteration mapping equivalence could not be proved",
                                fe_node.index()
                            ),
                        ));
                    }
                }
            } else {
                stack.extend(incoming_nodes(llir, node));
            }
        }
    }

    Ok(())
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
                        // Defensive traversal only: selected candidates with
                        // a direct nested FE are rejected by
                        // `validate_fusion_regions` before compile-unit
                        // construction. Keep walking here so this structural
                        // helper remains total for diagnostic/test callers.
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

    #[test]
    fn fusion_start_with_no_predecessor_is_rejected_before_codegen() {
        let mut llir: LLIRGraph = LLIRGraph::default();
        let fs_node = llir.add_node(llir_of(FusionStart::default()));
        let fadd_node = llir.add_node(llir_of(CudaBinaryElementwise::default()));
        let fe_node = llir.add_node(llir_of(FusionEnd::default()));
        llir.add_edge(fs_node, fadd_node, ());
        llir.add_edge(fadd_node, fe_node, ());
        let error = validate_fusion_regions(&llir, None).unwrap_err();
        assert_eq!(error.node, fs_node);
        assert!(error.reason.contains("requires 1 input"));
    }

    use crate::kernel::fusion::elementwise::CudaUnaryElementwise;
    use luminal::prelude::DType;

    fn input_op(dtype: DType) -> LLIROp {
        LLIROp::new::<Input>(Box::new(Input {
            node: 0,
            label: String::new(),
            dtype,
        }))
    }

    fn start(shape: Vec<Expression>, strides: Vec<Expression>, dtype: DType) -> LLIROp {
        llir_of(FusionStart {
            shape,
            strides,
            dtype,
        })
    }

    fn unary(
        op: &str,
        shape: Vec<Expression>,
        in_strides: Vec<Expression>,
        out_strides: Vec<Expression>,
        dtype: DType,
    ) -> LLIROp {
        llir_of(CudaUnaryElementwise {
            op: op.to_string(),
            shape,
            in_strides,
            out_strides,
            dtype,
        })
    }

    fn end(shape: Vec<Expression>, strides: Vec<Expression>, dtype: DType) -> LLIROp {
        llir_of(FusionEnd {
            shape,
            strides,
            dtype,
        })
    }

    fn valid_unary_region(op: &str, input_dtype: DType, output_dtype: DType) -> LLIRGraph {
        let shape = vec![16.into()];
        let strides = vec![Expression::from('z')];
        let mut llir = LLIRGraph::default();
        let input = llir.add_node(input_op(input_dtype));
        let fs = llir.add_node(start(shape.clone(), strides.clone(), input_dtype));
        let unary = llir.add_node(unary(
            op,
            shape.clone(),
            strides.clone(),
            strides.clone(),
            output_dtype,
        ));
        let fe = llir.add_node(end(shape, strides, output_dtype));
        llir.add_edge(input, fs, ());
        llir.add_edge(fs, unary, ());
        llir.add_edge(unary, fe, ());
        llir
    }

    #[test]
    fn fusion_validation_accepts_scalar_and_materialized_split_regions() {
        let mut scalar = LLIRGraph::default();
        let input = scalar.add_node(input_op(DType::F32));
        let fs = scalar.add_node(start(vec![], vec![], DType::F32));
        let cast = scalar.add_node(unary("Cast", vec![], vec![], vec![], DType::F32));
        let fe = scalar.add_node(end(vec![], vec![], DType::F32));
        scalar.add_edge(input, fs, ());
        scalar.add_edge(fs, cast, ());
        scalar.add_edge(cast, fe, ());
        validate_fusion_regions(&scalar, Some(&FxHashMap::default())).unwrap();

        let shape = vec![16.into()];
        let strides = vec![Expression::from('z')];
        let mut split = LLIRGraph::default();
        let input = split.add_node(input_op(DType::F32));
        let fs0 = split.add_node(start(shape.clone(), strides.clone(), DType::F32));
        let sin = split.add_node(unary(
            "Sin",
            shape.clone(),
            strides.clone(),
            strides.clone(),
            DType::F32,
        ));
        let fe0 = split.add_node(end(shape.clone(), strides.clone(), DType::F32));
        let fs1 = split.add_node(start(shape.clone(), strides.clone(), DType::F32));
        let sqrt = split.add_node(unary(
            "Sqrt",
            shape.clone(),
            strides.clone(),
            strides.clone(),
            DType::F32,
        ));
        let fe1 = split.add_node(end(shape, strides, DType::F32));
        split.add_edge(input, fs0, ());
        split.add_edge(fs0, sin, ());
        split.add_edge(sin, fe0, ());
        split.add_edge(fe0, fs1, ());
        split.add_edge(fs1, sqrt, ());
        split.add_edge(sqrt, fe1, ());
        validate_fusion_regions(&split, Some(&FxHashMap::default())).unwrap();
    }

    #[test]
    fn fusion_validation_accepts_semantically_equal_unified_layouts() {
        let a = Expression::from('a');
        let b = Expression::from('b');
        let c = Expression::from('c');
        let assoc_l = (a + b) + c;
        let assoc_r = a + (b + c);
        assert_ne!(assoc_l, assoc_r, "test requires distinct expression ASTs");

        let stride = vec![Expression::from('z')];
        let mut llir = LLIRGraph::default();
        let input = llir.add_node(input_op(DType::F32));
        let fs = llir.add_node(start(vec![assoc_l], stride.clone(), DType::F32));
        let sin = llir.add_node(unary(
            "Sin",
            vec![assoc_l],
            stride.clone(),
            stride.clone(),
            DType::F32,
        ));
        let fe = llir.add_node(end(vec![assoc_r], stride, DType::F32));
        llir.add_edge(input, fs, ());
        llir.add_edge(fs, sin, ());
        llir.add_edge(sin, fe, ());

        let dims = [('a', 2), ('b', 3), ('c', 4)].into_iter().collect();
        validate_fusion_regions(&llir, Some(&dims)).unwrap();
    }

    #[test]
    fn fusion_validation_accepts_exact_fixed_domain_layout_equivalence() {
        let shape = vec![4.into()];
        let z = Expression::from('z');
        let wrapped_z = z % 4;
        assert!(!wrapped_z.egglog_equal(z));

        let mut llir = LLIRGraph::default();
        let input = llir.add_node(input_op(DType::F32));
        let fs = llir.add_node(start(shape.clone(), vec![wrapped_z], DType::F32));
        let sin = llir.add_node(unary("Sin", shape.clone(), vec![z], vec![z], DType::F32));
        let fe = llir.add_node(end(shape, vec![z], DType::F32));
        llir.add_edge(input, fs, ());
        llir.add_edge(fs, sin, ());
        llir.add_edge(sin, fe, ());

        validate_fusion_regions(&llir, None).unwrap();
    }

    #[test]
    fn fusion_validation_does_not_promote_representative_equality_to_symbolic_equality() {
        let a = Expression::from('a');
        let b = Expression::from('b');
        let z = Expression::from('z');
        let mut llir = LLIRGraph::default();
        let input = llir.add_node(input_op(DType::F32));
        let fs = llir.add_node(start(vec![a], vec![z], DType::F32));
        let sin = llir.add_node(unary("Sin", vec![a], vec![z], vec![z], DType::F32));
        let fe = llir.add_node(end(vec![b], vec![z], DType::F32));
        llir.add_edge(input, fs, ());
        llir.add_edge(fs, sin, ());
        llir.add_edge(sin, fe, ());

        let representative = [('a', 16), ('b', 16)].into_iter().collect();
        let error = validate_fusion_regions(&llir, Some(&representative)).unwrap_err();
        assert!(error.reason.contains("could not be proved"));
    }

    #[test]
    fn fusion_validation_rejects_proven_layout_and_rank_mismatches() {
        let mut wrong_layout = valid_unary_region("Sin", DType::F32, DType::F32);
        let fe = wrong_layout
            .node_indices()
            .find(|node| fusion_end(&wrong_layout, *node).is_some())
            .unwrap();
        assert!(wrong_layout.remove_node(fe).is_some());
        let wrong_fe = wrong_layout.add_node(end(
            vec![17.into()],
            vec![Expression::from('z')],
            DType::F32,
        ));
        let producer = wrong_layout
            .node_indices()
            .find(|node| fusion_unary(&wrong_layout, *node).is_some())
            .unwrap();
        wrong_layout.add_edge(producer, wrong_fe, ());
        let error =
            validate_fusion_regions(&wrong_layout, Some(&FxHashMap::default())).unwrap_err();
        assert!(error.reason.contains("layout") || error.reason.contains("mapping"));

        let shape = vec![2.into(), 3.into()];
        let mut wrong_rank = LLIRGraph::default();
        let input = wrong_rank.add_node(input_op(DType::F32));
        let fs = wrong_rank.add_node(start(
            shape.clone(),
            vec![Expression::from('z'), Expression::from('z')],
            DType::F32,
        ));
        let binary = wrong_rank.add_node(llir_of(CudaBinaryElementwise {
            op: "Add".to_string(),
            out_shape: shape.clone(),
            a_stride: vec![Expression::from('z')],
            b_stride: vec![Expression::from('z'), Expression::from('z')],
            out_stride: vec![Expression::from('z'), Expression::from('z')],
            dtype: DType::F32,
        }));
        let fe = wrong_rank.add_node(end(
            shape,
            vec![Expression::from('z'), Expression::from('z')],
            DType::F32,
        ));
        wrong_rank.add_edge(input, fs, ());
        wrong_rank.add_edge(fs, binary, ());
        wrong_rank.add_edge(fs, binary, ());
        wrong_rank.add_edge(binary, fe, ());
        let error = validate_fusion_regions(&wrong_rank, None).unwrap_err();
        assert!(error.reason.contains("metadata ranks differ"));
    }

    #[test]
    fn fusion_validation_rejects_nested_end_opcode_and_dtype_contradictions() {
        let mut nested = valid_unary_region("Sin", DType::F32, DType::F32);
        let inner = nested
            .node_indices()
            .find(|node| fusion_end(&nested, *node).is_some())
            .unwrap();
        let outer = nested.add_node(end(
            vec![16.into()],
            vec![Expression::from('z')],
            DType::F32,
        ));
        nested.add_edge(inner, outer, ());
        assert!(
            validate_fusion_regions(&nested, None)
                .unwrap_err()
                .reason
                .contains("must cross a FusionStart")
        );

        let bad_opcode = valid_unary_region("Add", DType::F32, DType::F32);
        assert!(
            validate_fusion_regions(&bad_opcode, None)
                .unwrap_err()
                .reason
                .contains("unsupported unary")
        );

        let bad_dtype = valid_unary_region("Sin", DType::F16, DType::F32);
        assert!(
            validate_fusion_regions(&bad_dtype, None)
                .unwrap_err()
                .reason
                .contains("unary dtype")
        );

        let cast = valid_unary_region("Cast", DType::F16, DType::F32);
        validate_fusion_regions(&cast, None).unwrap();
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
