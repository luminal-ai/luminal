//! The graph a model is authored into: dynamic-dim assumptions plus the
//! LOGICAL structure the recorder captures. The old layout-bearing HLIR
//! graph and compile ladder are gone; this module owns the petgraph-backed
//! logical SSA that feeds the e-graph, while runtimes own
//! load/bind/with_ops/search.

use petgraph::{
    Direction,
    stable_graph::{NodeIndex, StableDiGraph},
    visit::EdgeRef,
};
use rustc_hash::FxHashSet;

use crate::dtype::DType;
use crate::frontend::GraphTensor;
use crate::shape::ToShape;

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

#[derive(Default)]
pub struct Graph {
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: crate::shape::DynMap,
    /// The logical-model recorder — GraphTensor methods emit their
    /// logical ops here; it IS the graph (absorbed into this struct at
    /// M3 Step 4e).
    pub logical: crate::graph::LogicalGraph,
}

impl Graph {
    /// Create a new graph
    pub fn new() -> Graph {
        Graph::default()
    }

    pub fn set_dim(&mut self, dimension: impl Into<crate::shape::Symbol>, val: usize) {
        self.dyn_map.insert(dimension.into(), val);
    }

    /// Create a new tensor with shape S and this dtype. Dtype is DECLARED
    /// at creation (purity ruling 2026-07-30: as_dtype is gone — a
    /// different dtype downstream is a logical cast, never a mutation of
    /// the declaration).
    pub fn tensor(&mut self, shape: impl ToShape, dtype: DType) -> GraphTensor {
        self.named_tensor("", shape, dtype)
    }

    /// Create a new tensor with a name, shape, and dtype. This name will show up on the graph when displayed.
    pub fn named_tensor(
        &mut self,
        name: impl ToString,
        shape: impl ToShape,
        dtype: DType,
    ) -> GraphTensor {
        let name = name.to_string();
        let dims = shape.to_shape();
        let id = self
            .logical
            .input(&name, &dims, dtype)
            .unwrap_or_else(crate::graph::unrecorded_value);
        GraphTensor::from_id(id, dims, self, dtype)
    }
}

// ---------------------------------------------------------------------
// THE LOGICAL GRAPH (M3 Step 4e: absorbed into this module — the
// recorder IS the graph; LogicalGraph keeps its name, Austin's ruling
// 2026-08-01). Formerly src/logical_graph.rs:
// ---------------------------------------------------------------------
// The LOGICAL GRAPH — the model the frontend actually builds (renamed
// from logical_recorder, ruling 2026-07-31: this IS the durable thing;
// absorbed into this module at Step 4e; the layout-bearing HLIR graph it
// once stood beside remains deleted).
//
// GraphTensor methods insert typed logical nodes and numbered operand
// edges here as the graph is built. There is no parallel tensor-id or
// HLIR-node keyspace. Movement methods emit `IndexMapApply` views DIRECTLY
// from their own parameters, at the source of truth — replacing
// tracker-lift reconstruction entirely.
//
// Model/binding split (M3 Step 1): the recorder emits MODEL text only —
// input declarations, ops, output naming, signature lists. Boundary
// vocabulary (layouts, buffers, access, freed-by, Bool8 casts) is the
// runtime binding generator's business (`runtime_binding`), never the
// model's.
//
// Coverage is honest: any construct the recorder does not understand
// POISONS it with a reason — the first reason wins, the native path
// refuses loudly at load, and their pipeline is untouched. Handle dims
// are DERIVED from the recorded value after every record call (R-D
// ruling 2026-08-26), so a tracker/recorder dims divergence is
// unrepresentable — the old resolve() cross-check tripwire is deleted.

use crate::shape::{IntExpr, Term};
use anyhow::{Result as AnyResult, bail};

/// One index-map entry, in-memory. Movement composition happens on this
/// tree — substituting our OWN just-emitted terms, never reconstructing
/// from strides.
#[derive(Debug, Clone, PartialEq)]
pub enum MapEntry {
    /// The consuming view's coordinate, zero-based FROM THE END (the
    /// de Bruijn house convention), with its extent.
    Coord {
        from_end: usize,
        extent: IntExpr,
    },
    /// A dim-expression literal (a number or a symbolic dim var).
    Lit(IntExpr),
    Add(Box<MapEntry>, Box<MapEntry>),
    Mul(Box<MapEntry>, IntExpr),
    Div(Box<MapEntry>, IntExpr),
    Rem(Box<MapEntry>, IntExpr),
    Min(Box<MapEntry>, Box<MapEntry>),
    Max(Box<MapEntry>, Box<MapEntry>),
}

// MapEntry::substitute as a RECORDER fold (composing maps across
// already-recorded values) is DELETED with fold 1 (Austin's ruling
// 2026-08-26): composition of recorded index-map applies is egglog's
// job. What remains below is CONSTRUCTION-TIME composition for ONE
// macro call — the same ruling's flip side: "macro interiors mint ONE
// apply per logical construct", with the map built at construction
// inside the one call. `ViewChain` (frontend/movement.rs) is the only
// user.
impl MapEntry {
    /// Replace each Coord leaf (an axis of the intermediate space,
    /// whose rank is `rank`) with that axis's entry over the next
    /// space. Purely construction-time; nothing recorded is rewritten.
    pub(crate) fn substitute(&self, replacements: &[MapEntry], rank: usize) -> MapEntry {
        match self {
            MapEntry::Coord {
                from_end,
                extent: _,
            } => replacements[rank - 1 - from_end].clone(),
            MapEntry::Lit(value) => MapEntry::Lit(*value),
            MapEntry::Add(a, b) => MapEntry::Add(
                Box::new(a.substitute(replacements, rank)),
                Box::new(b.substitute(replacements, rank)),
            ),
            MapEntry::Mul(a, e) => MapEntry::Mul(Box::new(a.substitute(replacements, rank)), *e),
            MapEntry::Div(a, e) => MapEntry::Div(Box::new(a.substitute(replacements, rank)), *e),
            MapEntry::Rem(a, e) => MapEntry::Rem(Box::new(a.substitute(replacements, rank)), *e),
            MapEntry::Min(a, b) => MapEntry::Min(
                Box::new(a.substitute(replacements, rank)),
                Box::new(b.substitute(replacements, rank)),
            ),
            MapEntry::Max(a, b) => MapEntry::Max(
                Box::new(a.substitute(replacements, rank)),
                Box::new(b.substitute(replacements, rank)),
            ),
        }
    }
}

/// A movement transform, as the frontend method states it — its own
/// parameters, not a tracker diff.
#[derive(Debug, Clone)]
pub enum Movement {
    /// out dim i = in dim axes[i] (front-based, their convention).
    Permute(Vec<usize>),
    /// New broadcast dim inserted at front position `axis`.
    ExpandDim { axis: usize, size: IntExpr },
    /// Size-1 front dim at `axis` removed (the squeeze).
    RemoveDim { axis: usize },
    /// dims[axis] = old/inner (outer), inner inserted after (their
    /// split_dims): parent coord = outer·inner_size + inner.
    SplitDims { axis: usize, inner: IntExpr },
    /// axis2 moved adjacent then merged into axis1 (their merge_dims):
    /// axis1 reads merged/inner, axis2 reads merged%inner.
    MergeDims { axis1: usize, axis2: usize },
    /// Per-axis tile (their repeat): dim → dim·r, coord reads % old dim.
    Repeat(Vec<IntExpr>),
    /// Zero-start slice: same coords, smaller extents (in-bounds shrink).
    Shrink { new_dims: Vec<IntExpr> },
}

/// The logical SSA identity. A tensor names the node that produces its
/// value; there is no parallel frontend-id keyspace.
pub type ValueId = NodeIndex;

/// The sentinel id of an UNRECORDED handle (a source op recorded onto an
/// already-poisoned graph): never present in the graph, so any use of the
/// handle resolves to a loud poison instead of a panic. Handles are TOTAL
/// (the poison-door discipline: the frontend never panics; the graph
/// refuses at load with the reason).
pub(crate) fn unrecorded_value() -> ValueId {
    ValueId::end()
}

/// An operand as a record call sees it: the handle's value plus its
/// dims. The dims payload is VESTIGIAL (R-D ruling 2026-08-26,
/// reasserted 2026-09-01: handle dims are derived from the recorded
/// value after every record call, so the divergence tripwire it fed is
/// deleted; the tuple shape is kept only to avoid churn at ~40 call
/// sites).
pub type Operand = (ValueId, Vec<IntExpr>);

/// Logical operation carried by an SSA node. Rendering details remain at
/// the Egglog boundary; the graph itself stores a closed operation
/// vocabulary rather than free-form constructor strings.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOp {
    Input { label: String },
    Constant(f64),
    Iota { value_expr: String },
    Cast(DType),
    Sqrt,
    Exp,
    Exp2,
    Log2,
    Sin,
    Recip,
    Add,
    Mul,
    Div,
    Mod,
    LessThan,
    TruncDiv,
    TruncRem,
    ReduceSum { axis_from_end: usize },
    ReduceMax { axis_from_end: usize },
    Gather,
    Scatter,
    IndexMapApply { entries: Vec<MapEntry> },
}

impl LogicalOp {
    pub fn constructor(&self) -> &'static str {
        match self {
            Self::Input { .. } => "LogicalTensorInputLit",
            Self::Constant(_) => "LogicalConstant",
            Self::Iota { .. } => "LogicalIota",
            Self::Cast(_) => "LogicalCast",
            Self::Sqrt => "LogicalSqrt",
            Self::Exp => "LogicalExp",
            Self::Exp2 => "LogicalExp2",
            Self::Log2 => "LogicalLog2",
            Self::Sin => "LogicalSin",
            Self::Recip => "LogicalRecip",
            Self::Add => "LogicalAdd",
            Self::Mul => "LogicalMul",
            Self::Div => "LogicalDiv",
            Self::Mod => "LogicalMod",
            Self::LessThan => "LogicalLessThan",
            Self::TruncDiv => "LogicalTruncDiv",
            Self::TruncRem => "LogicalTruncRem",
            Self::ReduceSum { .. } => "LogicalReduceSum",
            Self::ReduceMax { .. } => "LogicalReduceMax",
            Self::Gather => "LogicalGather",
            Self::Scatter => "LogicalScatter",
            Self::IndexMapApply { .. } => "LogicalIndexMapApply",
        }
    }

    fn render_form(&self) -> RenderForm {
        match self {
            Self::Gather => RenderForm::GatherList,
            Self::Scatter => RenderForm::ScatterList,
            _ => RenderForm::Plain,
        }
    }

    fn fixed_arity(&self) -> Option<usize> {
        Some(match self {
            Self::Input { .. } | Self::Constant(_) | Self::Iota { .. } => 0,
            Self::Cast(_)
            | Self::Sqrt
            | Self::Exp
            | Self::Exp2
            | Self::Log2
            | Self::Sin
            | Self::Recip
            | Self::ReduceSum { .. }
            | Self::ReduceMax { .. }
            | Self::IndexMapApply { .. } => 1,
            Self::Add
            | Self::Mul
            | Self::Div
            | Self::Mod
            | Self::LessThan
            | Self::TruncDiv
            | Self::TruncRem => 2,
            Self::Gather | Self::Scatter => return None,
        })
    }
}

/// Operand position is graph structure, not edge insertion order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InputPort(pub usize);

/// Egglog rendering form. Gather and scatter wrap coordinate operands in
/// a `LogicalTensorList`; every other typed node uses the plain form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderForm {
    /// ({constructor} {operands...} {aux})
    Plain,
    /// (LogicalGather data (Cons c1 (Cons c2 ...)))
    GatherList,
    /// (LogicalScatter init (Cons ...) src) — src is the LAST operand.
    ScatterList,
}

/// One SSA value. Operands live exclusively on incoming graph edges;
/// views keep their map entries structured in the operation payload so
/// movement composition works on trees, never on rendered text.
#[derive(Debug, Clone)]
pub struct LogicalNode {
    pub op: LogicalOp,
    pub dims: Vec<IntExpr>,
    pub dtype: DType,
}

/// One `.output()` designation: the value and optional authored name.
#[derive(Debug)]
struct OutputRecord {
    value: ValueId,
    label: Option<String>,
}

/// One bound input of the recorded model: the pristine label, the
/// staging id (what set_data/search key on), and the declared
/// geometry. `dtype` is the AUTHORED dtype — Bool inputs stage as
/// Bool8 buffers per the Bool8 boundary contract.
pub struct InputSpec {
    pub label: String,
    pub id: petgraph::graph::NodeIndex,
    pub dims: Vec<IntExpr>,
    pub dtype: DType,
}

/// One output designation: the label (authored via `output_named`,
/// else the synthesized "out_{key}") and the readback id (the
/// get_f32-family key).
pub struct OutputSpec {
    pub label: String,
    pub id: petgraph::graph::NodeIndex,
}

#[derive(Debug, Default)]
pub struct LogicalGraph {
    graph: StableDiGraph<LogicalNode, InputPort>,
    /// Output designations in .output() order.
    outputs: Vec<OutputRecord>,
    /// Anonymous-input counter — mints "arg.{k}" labels (Stage 3).
    anon_inputs: usize,
    post_checks: String,
    /// The same checks as LABELED units (label carries the named door —
    /// what failed and how to unblock it): on a saturation CheckError
    /// the runtime re-runs these one by one to name the culprit
    /// (ruling 2026-08-13: contract failures never surface as a bare
    /// "native saturation failed").
    labeled_checks: Vec<(String, String)>,
    poisoned: Option<String>,
}

impl LogicalGraph {
    /// First poison reason wins; everything after is a no-op.
    pub fn poison(&mut self, reason: impl Into<String>) {
        if self.poisoned.is_none() {
            self.poisoned = Some(reason.into());
        }
    }

    pub fn poisoned(&self) -> Option<&str> {
        self.poisoned.as_deref()
    }

    /// The backing petgraph. Nodes are logical SSA values and incoming
    /// edges are explicitly numbered operand ports.
    pub fn petgraph(&self) -> &StableDiGraph<LogicalNode, InputPort> {
        &self.graph
    }

    /// The recorded dims of a value — THE dims (R-D ruling 2026-08-26:
    /// `GraphTensor.dims` is derived from this after every record call
    /// by `with_logical`).
    pub fn value_dims(&self, id: ValueId) -> &[IntExpr] {
        &self.graph[id].dims
    }

    pub(crate) fn viz_nodes(&self) -> impl Iterator<Item = (ValueId, &LogicalNode)> {
        self.graph.node_indices().map(|id| (id, &self.graph[id]))
    }

    pub(crate) fn viz_operands(&self, id: ValueId) -> Vec<(usize, ValueId)> {
        self.operand_edges(id)
    }

    /// The recorded output designations, in .output() order.
    pub(crate) fn viz_outputs(&self) -> impl Iterator<Item = (ValueId, usize)> + '_ {
        self.outputs
            .iter()
            .map(|record| (record.value, record.value.index()))
    }

    fn dim_term(expr: &IntExpr) -> Result<String, String> {
        let terms = expr.terms.read();
        match &terms[..] {
            [Term::Num(n)] => Ok(format!("(IntLit {n})")),
            // Symbolic dims stay IntVar unconditionally — pins are
            // BINDING-side bounds seeds, never model content.
            [Term::Var(c)] => Ok(format!("(IntVar \"{c}\")")),
            // Compound dims render as full IntExpr trees (ruling
            // 2026-08-12: dims are any arbitrary IntExpr — extents are
            // structure, not identity; spellings that stall a
            // structural match surface as fail-closed refusals, never
            // as unsoundness). Coord atoms have no meaning in a shape
            // position and stay refused inside int_expr_term.
            _ => int_expr_term(expr, &[], "dim").map_err(|e| format!("dim render: {e:#}")),
        }
    }

    fn shape_term(dims: &[IntExpr]) -> Result<String, String> {
        let mut term = "(IntExprNil)".to_string();
        for dim in dims.iter().rev() {
            term = format!("(IntExprCons {} {term})", Self::dim_term(dim)?);
        }
        Ok(format!("(ShapeLit {term})"))
    }

    fn dtype_term(dtype: DType) -> String {
        format!("({dtype:?})")
    }

    /// `owner_shape` is the consuming view's OUT shape term — the box the
    /// map's coordinates are formals of (scoped CoordVar: ownership rides
    /// in the term; the extent field is gone, extents are the owner's own
    /// dims).
    fn entry_term(entry: &MapEntry, owner_shape: &str) -> Result<String, String> {
        Ok(match entry {
            MapEntry::Coord {
                from_end,
                extent: _,
            } => {
                format!("(CoordVar {owner_shape} {from_end})")
            }
            MapEntry::Lit(value) => Self::dim_term(value)?,
            MapEntry::Add(a, b) => format!(
                "(IntAdd {} {})",
                Self::entry_term(a, owner_shape)?,
                Self::entry_term(b, owner_shape)?
            ),
            MapEntry::Mul(a, e) => {
                format!(
                    "(IntMul {} {})",
                    Self::entry_term(a, owner_shape)?,
                    Self::dim_term(e)?
                )
            }
            MapEntry::Div(a, e) => format!(
                "(IntTruncDiv {} {})",
                Self::entry_term(a, owner_shape)?,
                Self::dim_term(e)?
            ),
            MapEntry::Rem(a, e) => format!(
                "(IntTruncRem {} {})",
                Self::entry_term(a, owner_shape)?,
                Self::dim_term(e)?
            ),
            MapEntry::Min(a, b) => format!(
                "(IntMin {} {})",
                Self::entry_term(a, owner_shape)?,
                Self::entry_term(b, owner_shape)?
            ),
            MapEntry::Max(a, b) => format!(
                "(IntMax {} {})",
                Self::entry_term(a, owner_shape)?,
                Self::entry_term(b, owner_shape)?
            ),
        })
    }

    fn push(
        &mut self,
        op: LogicalOp,
        operands: &[ValueId],
        dims: Vec<IntExpr>,
        dtype: DType,
    ) -> ValueId {
        if let Some(expected) = op.fixed_arity() {
            debug_assert_eq!(
                operands.len(),
                expected,
                "{} has the wrong operand count",
                op.constructor()
            );
        }
        let id = self.graph.add_node(LogicalNode { op, dims, dtype });
        for (port, operand) in operands.iter().copied().enumerate() {
            debug_assert!(
                self.graph.node_weight(operand).is_some(),
                "logical operand {operand:?} is not in the graph"
            );
            self.graph.add_edge(operand, id, InputPort(port));
        }
        id
    }

    fn operand_edges(&self, id: ValueId) -> Vec<(usize, ValueId)> {
        let mut operands: Vec<_> = self
            .graph
            .edges_directed(id, Direction::Incoming)
            .map(|edge| (edge.weight().0, edge.source()))
            .collect();
        operands.sort_unstable_by_key(|(port, _)| *port);
        debug_assert!(operands.iter().enumerate().all(|(i, (port, _))| i == *port));
        operands
    }

    fn operands(&self, id: ValueId) -> Vec<ValueId> {
        self.operand_edges(id)
            .into_iter()
            .map(|(_, operand)| operand)
            .collect()
    }

    /// Resolve an operand: it must be a value in the graph. (The
    /// tracker-vs-recorded dims tripwire is DELETED — R-D ruling
    /// 2026-08-26, reasserted 2026-09-01 over the petgraph backing:
    /// handle dims derive from the recorded value after every record
    /// call, so divergence is unrepresentable. The operand's dims
    /// payload is vestigial and ignored here.)
    fn resolve(&mut self, operand: &Operand, at: &str) -> Result<ValueId, String> {
        let (id, _vestigial_dims) = operand;
        if self.graph.node_weight(*id).is_none() {
            return Err(format!("{at}: operand {id:?} is not in the logical graph"));
        }
        Ok(*id)
    }

    /// Record an input declaration. The returned graph node is both its
    /// SSA identity and its staging key.
    /// TOTAL: an input is recorded even on a poisoned graph (it has no
    /// operands, so nothing about it can be wrong that the poison does
    /// not already cover), and a bad shape poisons AND still records —
    /// the handle stays usable and load() refuses with the reason.
    pub fn input(&mut self, label: &str, dims: &[IntExpr], dtype: DType) -> Option<ValueId> {
        let at = self.graph.node_count();
        if let Err(reason) = Self::shape_term(dims) {
            self.poison(format!("input t{at}: {reason}"));
        }
        // STAGE 3 (rulings 2026-08-13): every input has a unique
        // pristine label. Anonymous inputs auto-name "arg.{k}" in
        // declaration order (the ExportedProgram/ONNX convention for
        // positional user inputs); duplicate labels POISON at the one
        // choke point (uniqueness follows from hierarchical module namespaces,
        // and this tripwire catches hand-authored collisions); and the
        // label ALONE is the input's identity in the IR text — the
        // "{label}_{slot}" mangle is dead, its anti-hash-cons job done
        // by label uniqueness.
        let label = if label.is_empty() {
            let minted = format!("arg.{}", self.anon_inputs);
            self.anon_inputs += 1;
            minted
        } else {
            label.to_string()
        };
        if self.graph.node_weights().any(
            |node| matches!(&node.op, LogicalOp::Input { label: existing } if existing == &label),
        ) {
            self.poison(format!("duplicate input label \"{label}\""));
            // Keep the authored graph structurally complete even after a
            // fail-closed interface error. `model_text` will refuse this
            // graph, but the returned tensor still has a real SSA node.
            return Some(self.push(LogicalOp::Input { label }, &[], dims.to_vec(), dtype));
        }
        Some(self.push(LogicalOp::Input { label }, &[], dims.to_vec(), dtype))
    }

    /// Every bound input, in declaration order — the model's input
    /// interface, discoverable from the IR alone (checkpoint-name-
    /// driven staging: match `label` against checkpoint keys, stage by
    /// `id`). Labels are stored pristine; label uniqueness is an
    /// authoring obligation until the namespace tripwire lands.
    pub fn input_specs(&self) -> Vec<InputSpec> {
        self.graph
            .node_indices()
            .filter_map(|id| {
                let value = &self.graph[id];
                let LogicalOp::Input { label } = &value.op else {
                    return None;
                };
                Some(InputSpec {
                    label: label.clone(),
                    id,
                    dims: value.dims.clone(),
                    dtype: value.dtype,
                })
            })
            .collect()
    }

    /// Every output designation, in .output() order.
    pub fn output_specs(&self) -> Vec<OutputSpec> {
        self.outputs
            .iter()
            .map(|record| OutputSpec {
                label: record
                    .label
                    .clone()
                    .unwrap_or_else(|| format!("out_{}", record.value.index())),
                id: record.value,
            })
            .collect()
    }

    /// Record an op over operand values.
    pub fn op(
        &mut self,
        op: LogicalOp,
        operands: &[Operand],
        out_dims: Vec<IntExpr>,
        out_dtype: DType,
    ) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let constructor = op.constructor();
        let at = self.graph.node_count();
        if let Some(expected) = op.fixed_arity() {
            if operands.len() != expected {
                self.poison(format!(
                    "{constructor} at t{at}: expected {expected} operands, got {}",
                    operands.len()
                ));
                return None;
            }
        }
        let mut ids = Vec::with_capacity(operands.len());
        for operand in operands {
            match self.resolve(operand, &format!("{constructor} at t{at}")) {
                Ok(id) => ids.push(id),
                Err(reason) => {
                    self.poison(reason);
                    return None;
                }
            }
        }
        Some(self.push(op, &ids, out_dims, out_dtype))
    }

    /// Record a seam-node view: an IndexMapApply of the operand through
    /// entries built from the seam's own parameters.
    pub fn view_op(
        &mut self,
        operand: &Operand,
        entries: &[MapEntry],
        out_dims: Vec<IntExpr>,
        out_dtype: DType,
    ) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let at = self.graph.node_count();
        let base = match self.resolve(operand, &format!("view op at t{at}")) {
            Ok(id) => id,
            Err(reason) => {
                self.poison(reason);
                return None;
            }
        };
        self.push_view(base, entries.to_vec(), out_dims, out_dtype)
    }

    fn push_view(
        &mut self,
        base: ValueId,
        entries: Vec<MapEntry>,
        out_dims: Vec<IntExpr>,
        out_dtype: DType,
    ) -> Option<ValueId> {
        let at = self.graph.node_count();
        let shape = match Self::shape_term(&out_dims) {
            Ok(shape) => shape,
            Err(reason) => {
                self.poison(format!("view at t{at}: {reason}"));
                return None;
            }
        };
        // The map's DOMAIN TAG (ruling 2026-08-11): an IndexMapLit
        // carries the source shape it substitutes into — the parent's
        // own dims, written here at the single mint site so the
        // apply/map coherence tripwire can never fire on recorder output.
        let source_dims = self.graph[base].dims.clone();
        let source_shape = match Self::shape_term(&source_dims) {
            Ok(term) => term,
            Err(reason) => {
                self.poison(format!("view at t{at}: {reason}"));
                return None;
            }
        };
        let mut entries_term = "(IntExprNil)".to_string();
        for entry in entries.iter().rev() {
            match Self::entry_term(entry, &shape) {
                Ok(term) => entries_term = format!("(IntExprCons {term} {entries_term})"),
                Err(reason) => {
                    self.poison(format!("view at t{at}: {reason}"));
                    return None;
                }
            }
        }
        // Validate all structured render inputs at the insertion boundary.
        let _ = (shape, source_shape, entries_term);
        Some(self.push(
            LogicalOp::IndexMapApply { entries },
            &[base],
            out_dims,
            out_dtype,
        ))
    }

    /// Record a pad-mask indicator iota (see the pad seam): per padded
    /// axis, (before <= p) · (p < before + dim) as bool-bridge casts.
    /// Record a LogicalIota: the value expression is authored over the
    /// FLAT index (the frontend's `'z'`) and rewritten here into a true
    /// COORDINATE FUNCTION over the declared shape —
    /// `z := Σ CoordVar(shape, axis) · row-major-stride` — so the
    /// recorded model is per-coordinate at ANY rank (Design A, ruling
    /// 2026-08-06: the rank-1-plus-recorded-splits detour and its silent
    /// symbolic-total collapse are gone; symbolic dims render as IntVar
    /// through `shape_term`). Extent-1 axes contribute no summand (their
    /// coordinate is identically 0); a fully degenerate shape records
    /// `(IntLit 0)`. The authoring-contract bounds check pair rides
    /// every iota.
    /// Record a LogicalIota from a COORDINATE-FUNCTION expression (P1
    /// ruling 2026-08-07): the value expression is authored over
    /// `Term::Coord(k)` atoms — one per output axis, minted by
    /// `Graph::iota`'s closure — and lowers per-axis: coords become
    /// `(CoordVar shape axis)`, named symbols become `(IntVar "c")`.
    /// There is no flat-'z' form anymore ('z' in an iota expression is an
    /// ordinary named symbol; flat-index authoring is a rank-1 iota plus
    /// recorded reshapes). Extent-1 axes substitute `(IntLit 0)` (their
    /// coordinate is identically zero); a `Coord(k)` with `k >= rank` is
    /// a leaked atom and poisons loudly. The authoring-contract bounds
    /// pair rides every iota.
    pub fn record_iota(&mut self, expr: &IntExpr, dims: &[IntExpr]) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let at = self.graph.node_count();
        let shape = match Self::shape_term(dims) {
            Ok(term) => term,
            Err(reason) => {
                self.poison(format!("iota at t{at}: {reason}"));
                return None;
            }
        };
        let rank = dims.len();
        let coord_terms: Vec<String> = (0..rank)
            .map(|k| {
                if dims[k] == IntExpr::from(1) {
                    "(IntLit 0)".to_string()
                } else {
                    format!("(CoordVar {shape} {})", rank - 1 - k)
                }
            })
            .collect();
        let value_expr = match int_expr_term(expr, &coord_terms, &format!("recorder iota t{at}")) {
            Ok(text) => text,
            Err(err) => {
                self.poison(format!("iota at t{at}: {err}"));
                return None;
            }
        };
        let logical = Some(self.push(
            LogicalOp::Iota {
                value_expr: value_expr.clone(),
            },
            &[],
            dims.to_vec(),
            DType::Int,
        ));
        self.post_check(
            format!("iota value-bounds contract at t{at}"),
            &format!(
                "(check (= ?reclo{at} (lower-bound-of {value_expr})))\n\
             (check (= ?rechi{at} (upper-bound-of {value_expr})))\n"
            ),
        );
        logical
    }

    pub fn record_mask_iota(
        &mut self,
        befores: &[IntExpr],
        afters: &[IntExpr],
        in_dims: &[IntExpr],
    ) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let at = self.graph.node_count();
        let rank = in_dims.len();
        let mut out_dims = Vec::with_capacity(rank);
        let mut out_terms = Vec::with_capacity(rank);
        for k in 0..rank {
            // Frontend simplification restored (Austin's revert ruling
            // 2026-08-27): recorder-authored expressions are
            // construction-simplified, as pre-R-C.
            let out_dim = (befores[k] + in_dims[k] + afters[k]).simplify();
            match Self::dim_term(&out_dim) {
                Ok(term) => out_terms.push(term),
                Err(reason) => {
                    self.poison(format!("mask iota at t{at}: {reason}"));
                    return None;
                }
            }
            out_dims.push(out_dim);
        }
        let out_shape_term = match Self::shape_term(&out_dims) {
            Ok(term) => term,
            Err(reason) => {
                self.poison(format!("mask iota at t{at}: {reason}"));
                return None;
            }
        };
        let mut factors: Vec<String> = Vec::new();
        for k in 0..rank {
            let coord = format!("(CoordVar {out_shape_term} {})", rank - 1 - k);
            let before = befores[k];
            let after = afters[k];
            let (Ok(before_term), Ok(bound_term)) = (
                Self::dim_term(&before),
                Self::dim_term(&(before + in_dims[k]).simplify()),
            ) else {
                self.poison(format!("mask iota at t{at}: symbolic pad bound"));
                return None;
            };
            if before != IntExpr::from(0) {
                factors.push(format!(
                    "(IntCastFromBool (BoolLessThanInt {before_term} (IntAdd {coord} (IntLit 1))))"
                ));
            }
            if after != IntExpr::from(0) {
                factors.push(format!(
                    "(IntCastFromBool (BoolLessThanInt {coord} {bound_term}))"
                ));
            }
        }
        let mut expr = factors.pop().unwrap_or_else(|| "(IntLit 1)".to_string());
        for factor in factors {
            expr = format!("(IntMul {factor} {expr})");
        }
        let logical = Some(self.push(
            LogicalOp::Iota {
                value_expr: expr.clone(),
            },
            &[],
            out_dims,
            DType::Int,
        ));
        // The authoring-contract bounds pair — uniform with record_iota
        // (Design A fold-in, 2026-08-06): every recorded iota's value
        // expression must have derivable bounds, or the fixpoint refuses.
        self.post_check(
            format!("iota value-bounds contract at t{at}"),
            &format!(
                "(check (= ?reclo{at} (lower-bound-of {expr})))\n\
             (check (= ?rechi{at} (upper-bound-of {expr})))\n"
            ),
        );
        logical
    }

    /// Record a coordinate-form gather.
    pub fn record_gather(
        &mut self,
        data: &Operand,
        coords: &[Operand],
        out_dims: Vec<IntExpr>,
        out_dtype: DType,
    ) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let at = self.graph.node_count();
        let mut ids = Vec::with_capacity(coords.len() + 1);
        match self.resolve(data, &format!("gather at t{at}")) {
            Ok(id) => ids.push(id),
            Err(reason) => {
                self.poison(reason);
                return None;
            }
        }
        for coord in coords {
            match self.resolve(coord, &format!("gather at t{at}")) {
                Ok(id) => ids.push(id),
                Err(reason) => {
                    self.poison(reason);
                    return None;
                }
            }
        }
        Some(self.push(LogicalOp::Gather, &ids, out_dims, out_dtype))
    }

    /// Record a coordinate-form scatter (operands: init, coords..., src).
    pub fn record_scatter(
        &mut self,
        init: &Operand,
        coords: &[Operand],
        src: &Operand,
        out_dims: Vec<IntExpr>,
        out_dtype: DType,
    ) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let at = self.graph.node_count();
        let mut ids = Vec::with_capacity(coords.len() + 2);
        match self.resolve(init, &format!("scatter at t{at}")) {
            Ok(id) => ids.push(id),
            Err(reason) => {
                self.poison(reason);
                return None;
            }
        }
        for coord in coords {
            match self.resolve(coord, &format!("scatter at t{at}")) {
                Ok(id) => ids.push(id),
                Err(reason) => {
                    self.poison(reason);
                    return None;
                }
            }
        }
        match self.resolve(src, &format!("scatter at t{at}")) {
            Ok(id) => ids.push(id),
            Err(reason) => {
                self.poison(reason);
                return None;
            }
        }
        Some(self.push(LogicalOp::Scatter, &ids, out_dims, out_dtype))
    }

    /// Apply a movement: mints ONE view value per movement on the
    /// CURRENT value, carrying that single movement's map (fold-1
    /// removal, Austin's ruling 2026-08-26, REASSERTED 2026-09-01 over
    /// the petgraph backing: no base short-circuit, no entry
    /// composition — movement chains are recorded as chains and egglog
    /// composes the index-map applies).
    pub fn apply_movement(&mut self, current: &Operand, movement: Movement) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let at = self.graph.node_count();
        let current_id = match self.resolve(current, &format!("movement at t{at}")) {
            Ok(id) => id,
            Err(reason) => {
                self.poison(reason);
                return None;
            }
        };
        let value = &self.graph[current_id];
        let prev_dims = value.dims.clone();
        let out_dtype = value.dtype;

        let (replacement, new_dims) = match movement_entries(movement, &prev_dims) {
            Ok(pair) => pair,
            Err(reason) => {
                self.poison(reason);
                return None;
            }
        };

        // Fold-1 removed: `replacement` IS this movement's map (the
        // per-parent-axis entries over the new out space) — push it
        // directly on the current value.
        self.push_view(current_id, replacement, new_dims, out_dtype)
    }
}

/// One movement's index map, stated from the movement's own parameters:
/// per PARENT axis, an entry over the movement's OUT space, plus the out
/// dims. Shared by `apply_movement` (one view per movement) and
/// `ViewChain` (construction-time composition for one macro call).
pub(crate) fn movement_entries(
    movement: Movement,
    prev_dims: &[IntExpr],
) -> Result<(Vec<MapEntry>, Vec<IntExpr>), String> {
    let prev_rank = prev_dims.len();
    let pair: (Vec<MapEntry>, Vec<IntExpr>) = match movement {
        Movement::Permute(axes) => {
            if axes.len() != prev_rank {
                return Err(format!("permute arity {} vs rank {prev_rank}", axes.len()));
            }
            let mut replacement = vec![MapEntry::Lit(0.into()); prev_rank];
            for (q, &p) in axes.iter().enumerate() {
                replacement[p] = MapEntry::Coord {
                    from_end: prev_rank - 1 - q,
                    extent: prev_dims[p],
                };
            }
            let new_dims = axes.iter().map(|&p| prev_dims[p]).collect();
            (replacement, new_dims)
        }
        Movement::ExpandDim { axis, size } => {
            if axis > prev_rank {
                return Err(format!("expand_dim axis {axis} vs rank {prev_rank}"));
            }
            let new_rank = prev_rank + 1;
            let replacement = (0..prev_rank)
                .map(|p| {
                    let q = if p < axis { p } else { p + 1 };
                    MapEntry::Coord {
                        from_end: new_rank - 1 - q,
                        extent: prev_dims[p],
                    }
                })
                .collect();
            let mut new_dims = prev_dims.to_vec();
            new_dims.insert(axis, size);
            (replacement, new_dims)
        }
        Movement::RemoveDim { axis } => {
            if axis >= prev_rank || prev_dims[axis].to_usize().is_some_and(|d| d != 1) {
                return Err(format!(
                    "remove_dim axis {axis} of dims {prev_dims:?} (must be a size-1 axis)"
                ));
            }
            let new_rank = prev_rank - 1;
            let replacement = (0..prev_rank)
                .map(|p| {
                    if p == axis {
                        MapEntry::Lit(0.into())
                    } else {
                        let q = if p < axis { p } else { p - 1 };
                        MapEntry::Coord {
                            from_end: new_rank - 1 - q,
                            extent: prev_dims[p],
                        }
                    }
                })
                .collect();
            let mut new_dims = prev_dims.to_vec();
            new_dims.remove(axis);
            (replacement, new_dims)
        }
        Movement::SplitDims { axis, inner } => {
            if axis >= prev_rank {
                return Err(format!("split_dims axis {axis} vs rank {prev_rank}"));
            }
            // Frontend simplification restored (revert ruling 2026-08-27).
            let outer = (prev_dims[axis] / inner).simplify();
            let new_rank = prev_rank + 1;
            let replacement = (0..prev_rank)
                .map(|p| {
                    if p == axis {
                        MapEntry::Add(
                            Box::new(MapEntry::Mul(
                                Box::new(MapEntry::Coord {
                                    from_end: new_rank - 1 - axis,
                                    extent: outer,
                                }),
                                inner,
                            )),
                            Box::new(MapEntry::Coord {
                                from_end: new_rank - 1 - (axis + 1),
                                extent: inner,
                            }),
                        )
                    } else {
                        let q = if p < axis { p } else { p + 1 };
                        MapEntry::Coord {
                            from_end: new_rank - 1 - q,
                            extent: prev_dims[p],
                        }
                    }
                })
                .collect();
            let mut new_dims = prev_dims.to_vec();
            new_dims[axis] = outer;
            new_dims.insert(axis + 1, inner);
            (replacement, new_dims)
        }
        Movement::MergeDims { axis1, axis2 } => {
            if axis1 >= axis2 || axis2 >= prev_rank {
                return Err(format!("merge_dims ({axis1},{axis2}) vs rank {prev_rank}"));
            }
            let inner = prev_dims[axis2];
            // Frontend simplification restored (revert ruling 2026-08-27).
            let merged = (prev_dims[axis1] * prev_dims[axis2]).simplify();
            let new_rank = prev_rank - 1;
            let merged_coord = MapEntry::Coord {
                from_end: new_rank - 1 - axis1,
                extent: merged,
            };
            let replacement = (0..prev_rank)
                .map(|p| {
                    if p == axis1 {
                        MapEntry::Div(Box::new(merged_coord.clone()), inner)
                    } else if p == axis2 {
                        MapEntry::Rem(Box::new(merged_coord.clone()), inner)
                    } else {
                        let q = if p < axis2 { p } else { p - 1 };
                        MapEntry::Coord {
                            from_end: new_rank - 1 - q,
                            extent: prev_dims[p],
                        }
                    }
                })
                .collect();
            let mut new_dims = prev_dims.to_vec();
            new_dims[axis1] = merged;
            new_dims.remove(axis2);
            (replacement, new_dims)
        }
        Movement::Repeat(repeats) => {
            if repeats.len() != prev_rank {
                return Err(format!(
                    "repeat arity {} vs rank {prev_rank}",
                    repeats.len()
                ));
            }
            let replacement = (0..prev_rank)
                .map(|p| {
                    if repeats[p].to_usize() == Some(1) {
                        MapEntry::Coord {
                            from_end: prev_rank - 1 - p,
                            extent: prev_dims[p],
                        }
                    } else {
                        // Frontend simplification restored (revert
                        // ruling 2026-08-27).
                        let tiled = (prev_dims[p] * repeats[p]).simplify();
                        MapEntry::Rem(
                            Box::new(MapEntry::Coord {
                                from_end: prev_rank - 1 - p,
                                extent: tiled,
                            }),
                            prev_dims[p],
                        )
                    }
                })
                .collect();
            let new_dims = prev_dims
                .iter()
                .zip(&repeats)
                .map(|(d, r)| (*d * *r).simplify())
                .collect();
            (replacement, new_dims)
        }
        Movement::Shrink { new_dims } => {
            if new_dims.len() != prev_rank {
                return Err(format!(
                    "shrink arity {} vs rank {prev_rank}",
                    new_dims.len()
                ));
            }
            let replacement = (0..prev_rank)
                .map(|p| MapEntry::Coord {
                    from_end: prev_rank - 1 - p,
                    extent: new_dims[p],
                })
                .collect();
            (replacement, new_dims)
        }
    };
    Ok(pair)
}

impl LogicalGraph {
    /// Append post-schedule authoring checks (iota bounds pairs).
    /// SHAPE-CONTRACT INVARIANTS (ruling 2026-08-13, squeeze "option
    /// 3"): record always; validity is a POST-SATURATION check against
    /// the bounds lattice, so the BINDING decides per bucket — a
    /// runtime that pins/buckets the extent appropriately passes (the
    /// [n,n] pin-collapse discharges the check), any other bucket
    /// refuses loudly. Static extents discharge trivially.
    pub(crate) fn require_extent_eq_one(&mut self, at: usize, dim: &IntExpr, what: &str) {
        match Self::dim_term(dim) {
            Ok(term) => self.post_check(
                format!(
                    "{what} at t{at}: axis extent must be exactly 1 — \
                     bind or bucket the dim to [1,1]"
                ),
                &format!("(check (= {term} (IntLit 1)))\n"),
            ),
            Err(reason) => self.poison(format!("{what} at t{at}: {reason}")),
        }
    }

    /// The ≥-form of the same contract (empty-axis refusals:
    /// reduce_max/argmax need at least one element; unfold windows need
    /// a positive count).
    pub(crate) fn require_extent_at_least(
        &mut self,
        at: usize,
        dim: &IntExpr,
        min: i64,
        what: &str,
    ) {
        match Self::dim_term(dim) {
            Ok(term) => self.post_check(
                format!(
                    "{what} at t{at}: extent lower bound must reach {min} — \
                     bind the dim's range to exclude smaller values"
                ),
                &format!("(check (>= (lower-bound-of {term}) (bigint {min})))\n"),
            ),
            Err(reason) => self.poison(format!("{what} at t{at}: {reason}")),
        }
    }

    pub fn output(&mut self, operand: &Operand, label: Option<&str>) {
        if self.poisoned.is_some() {
            return;
        }
        if let Some(name) = label {
            if self
                .outputs
                .iter()
                .any(|record| record.label.as_deref() == Some(name))
            {
                return self.poison(format!("duplicate output name \"{name}\""));
            }
        }
        let id = match self.resolve(operand, "output") {
            Ok(id) => id,
            Err(reason) => return self.poison(reason),
        };
        // Outputs of view VALUES are fine — the binding puts a contiguous
        // boundary on the value and search prices the materialization.
        // (The genuinely divergent case — their pipeline's non-contiguous
        // materialize path — already poisons via its gather1d.)
        self.outputs.push(OutputRecord {
            value: id,
            label: label.map(str::to_string),
        });
    }

    /// The live set: every value transitively reachable from the outputs,
    /// plus every input declaration (bindings enumerate all inputs).
    pub(crate) fn live_set(&self) -> FxHashSet<ValueId> {
        let mut live = FxHashSet::default();
        let mut stack: Vec<ValueId> = self.outputs.iter().map(|record| record.value).collect();
        for id in self.graph.node_indices() {
            if matches!(self.graph[id].op, LogicalOp::Input { .. }) {
                stack.push(id);
            }
        }
        while let Some(id) = stack.pop() {
            if !live.insert(id) {
                continue;
            }
            stack.extend(self.operands(id));
        }
        live
    }

    fn render_value(&self, id: ValueId) -> Result<String, String> {
        let value = &self.graph[id];
        let operands = self.operands(id);
        let name = |id: &ValueId| format!("v{}", id.index());
        let shape = Self::shape_term(&value.dims)?;

        if let LogicalOp::Input { label } = &value.op {
            let wire_dtype = if value.dtype == DType::Bool {
                "(Bool8)".to_string()
            } else {
                Self::dtype_term(value.dtype)
            };
            let literal =
                format!("(LogicalTensorInputLit (LogicalIdLit \"{label}\") {shape} {wire_dtype})");
            return if value.dtype == DType::Bool {
                Ok(format!(
                    "(let input_wire_v{} {literal})\n(let v{} (LogicalCast input_wire_v{} (Bool)))\n",
                    id.index(),
                    id.index(),
                    id.index()
                ))
            } else {
                Ok(format!("(let v{} {literal})\n", id.index()))
            };
        }

        match value.op.render_form() {
            RenderForm::Plain => {
                let mut parts: Vec<String> = operands.iter().map(name).collect();
                match &value.op {
                    LogicalOp::Constant(constant) => parts.push(format!("{constant:?}")),
                    LogicalOp::Iota { value_expr } => {
                        parts.push(value_expr.clone());
                        parts.push(shape);
                    }
                    LogicalOp::Cast(dtype) => parts.push(Self::dtype_term(*dtype)),
                    LogicalOp::ReduceSum { axis_from_end }
                    | LogicalOp::ReduceMax { axis_from_end } => {
                        parts.push(axis_from_end.to_string());
                    }
                    LogicalOp::IndexMapApply { entries } => {
                        let source_shape = Self::shape_term(&self.graph[operands[0]].dims)?;
                        let mut entries_term = "(IntExprNil)".to_string();
                        for entry in entries.iter().rev() {
                            entries_term = format!(
                                "(IntExprCons {} {entries_term})",
                                Self::entry_term(entry, &shape)?
                            );
                        }
                        parts.push(format!("(IndexMapLit {entries_term} {source_shape})"));
                        parts.push(shape);
                    }
                    _ => {}
                }
                Ok(format!(
                    "(let v{} ({} {}))\n",
                    id.index(),
                    value.op.constructor(),
                    parts.join(" ")
                ))
            }
            RenderForm::GatherList => {
                let data = name(&operands[0]);
                let mut list = "(LogicalTensorNil)".to_string();
                for coord in operands[1..].iter().rev() {
                    list = format!("(LogicalTensorCons {} {list})", name(coord));
                }
                Ok(format!(
                    "(let v{} ({} {data} {list}))\n",
                    id.index(),
                    value.op.constructor()
                ))
            }
            RenderForm::ScatterList => {
                let init = name(&operands[0]);
                let src = name(operands.last().unwrap());
                let mut list = "(LogicalTensorNil)".to_string();
                for coord in operands[1..operands.len() - 1].iter().rev() {
                    list = format!("(LogicalTensorCons {} {list})", name(coord));
                }
                Ok(format!(
                    "(let v{} ({} {init} {list} {src}))\n",
                    id.index(),
                    value.op.constructor()
                ))
            }
        }
    }

    /// The rendered MODEL: live values in SSA order (creation order is
    /// topological — operands precede consumers), dead values elided,
    /// plus the output NAME annotations.
    pub fn model_text(&self) -> Result<String, String> {
        if let Some(reason) = &self.poisoned {
            return Err(format!("logical graph poisoned: {reason}"));
        }
        let live = self.live_set();
        let mut text = String::new();
        for id in self.graph.node_indices() {
            if live.contains(&id) {
                text.push_str(&self.render_value(id)?);
            }
        }
        for record in &self.outputs {
            let name = match &record.label {
                Some(label) => label.clone(),
                None => format!("out_{}", record.value.index()),
            };
            text.push_str(&format!(
                "(union v{} (LogicalTensorNamed (LogicalIdLit \"{name}\")))\n",
                record.value.index()
            ));
        }
        Ok(text)
    }

    /// The post-schedule authoring checks.
    pub fn post_check(&mut self, label: impl Into<String>, text: &str) {
        self.labeled_checks.push((label.into(), text.to_string()));
        self.post_checks.push_str(text);
    }

    pub fn post_checks(&self) -> &str {
        &self.post_checks
    }

    /// The bound assembly SPLIT at the schedule (binding seeds inject
    /// before saturation): (pre-schedule text, input slots, output slots,
    /// post-schedule checks). The model text is runtime-neutral; every
    /// boundary statement comes from the caller's
    /// [`RuntimeBindingsGenerator`](crate::runtime_binding::RuntimeBindingsGenerator)
    /// — each runtime hands in its own (Step C, 2026-08-17).
    #[allow(clippy::type_complexity)]
    pub fn bound_parts(
        &self,
        bindings: &dyn crate::runtime_binding::RuntimeBindingsGenerator,
    ) -> Result<
        (
            String,
            Vec<InputSlot>,
            Vec<OutputSlot>,
            String,
            Vec<(String, String)>,
        ),
        String,
    > {
        let mut text = self.model_text()?;
        let mut input_slots = Vec::new();
        let mut input_buffer_tensors = Vec::new();
        let mut next_buffer: i64 = 0;
        for id in self.graph.node_indices() {
            let value = &self.graph[id];
            let LogicalOp::Input { .. } = &value.op else {
                continue;
            };
            let slot = id.index();
            let shape = Self::shape_term(&value.dims)?;
            let stem = format!("nat{slot}");
            let buffer = next_buffer;
            next_buffer += 1;
            let value_name = if value.dtype == DType::Bool {
                format!("input_wire_v{}", id.index())
            } else {
                format!("v{}", id.index())
            };
            text.push_str(&bindings.input_binding(
                &stem,
                buffer as usize,
                &value_name,
                &shape,
                &bindings.width_term(value.dtype),
            ));
            input_buffer_tensors.push(format!("{stem}_buffer_tensor"));
            input_slots.push(InputSlot {
                tensor: id,
                buffer,
                size: slot as u64,
                value_name,
            });
        }
        let mut output_slots = Vec::new();
        let mut output_buffer_tensors = Vec::new();
        for record in &self.outputs {
            let id = record.value;
            let key = id.index();
            let value = &self.graph[id];
            let shape = Self::shape_term(&value.dims)?;
            let stem = format!("natout{key}");
            let buffer = next_buffer;
            next_buffer += 1;
            text.push_str(&bindings.output_binding(
                &stem,
                buffer as usize,
                &format!("v{}", id.index()),
                &shape,
                value.dtype,
            ));
            output_buffer_tensors.push(format!("{stem}_buffer_tensor"));
            output_slots.push(OutputSlot {
                tensor: id,
                buffer,
                size: key as u64,
            });
        }
        text.push_str(&bindings.boundary_lists(
            &input_buffer_tensors,
            &output_buffer_tensors,
            "nat_input_boundary",
            "nat_output_boundary",
        ));
        Ok((
            text,
            input_slots,
            output_slots,
            self.post_checks.clone(),
            self.labeled_checks.clone(),
        ))
    }

    /// The assembled program under the given runtime's bindings.
    pub fn bound_program(
        &self,
        bindings: &dyn crate::runtime_binding::RuntimeBindingsGenerator,
    ) -> Result<LogicalProgram, String> {
        let (pre, input_slots, output_slots, post_checks, _labeled) = self.bound_parts(bindings)?;
        Ok(LogicalProgram {
            text: format!("{pre}{}{post_checks}", bindings.schedule()),
            input_slots,
            output_slots,
        })
    }
}

// ─── Survivors of the interim translator (M3 Topic D) ───

/// One bound input: the graph tensor it carries, the buffer the runtime
/// allocated for it, and its declared size. Buffer ids are an internal,
/// sequential, binding-time allocation — inputs first, outputs after —
/// never derived from graph node indices (the retired HLIR keyspace).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputSlot {
    pub tensor: petgraph::graph::NodeIndex,
    pub buffer: i64,
    pub size: u64,
    /// The input's SSA value name in the model text (`v{index}`) — the
    /// handle binding-time seeds (value ranges) attach to.
    pub value_name: String,
}

/// One bound output. See [`InputSlot`] for the allocation discipline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputSlot {
    pub tensor: petgraph::graph::NodeIndex,
    pub buffer: i64,
    pub size: u64,
}

/// The assembled program plus the I/O binding tables the runtime needs
/// (moved here when the interim translator was deleted, M3 Topic D).
/// Buffer ids are runtime-internal sequential allocations; the runtime's
/// role-split maps translate tensor identities to buffers.
#[derive(Debug, Clone)]
pub struct LogicalProgram {
    /// Model + binding + schedule + authoring-contract checks. Run as
    /// `format!("{}\n\n{}", <the runtime's assembled program>, text)` —
    /// e.g. `luminal_reference::assembled_program()`.
    pub text: String,
    /// Bound inputs in signature order.
    pub input_slots: Vec<InputSlot>,
    /// Bound outputs in output-slot order.
    pub output_slots: Vec<OutputSlot>,
}

/// Their RPN index expression rendered as OUR IntExpr term, with `z`
/// replaced by the given coordinate term and dyn vars resolved via the
/// pins. Add/Mul only for now (their slice path is affine); anything else
/// bails loudly.
pub(crate) fn int_expr_term(expr: &IntExpr, coord_terms: &[String], at: &str) -> AnyResult<String> {
    let mut stack: Vec<String> = Vec::new();
    for term in expr.terms.read().iter() {
        match term {
            Term::Num(n) => stack.push(format!("(IntLit {n})")),
            // Symbolic vars stay IntVar in the model — pins are
            // BINDING-side bounds seeds, never model content (same rule
            // as dim_term; the R3 fix, 2026-08-06). No character is
            // special: 'z' is an ordinary named symbol (P1, 2026-08-07).
            Term::Var(c) => stack.push(format!("(IntVar \"{c}\")")),
            // Coordinate atoms substitute their axis's CoordVar term; an
            // out-of-range axis is a coord IntExpr that leaked out of
            // its own iota — refuse loudly.
            Term::Coord(k) => match coord_terms.get(*k as usize) {
                Some(term) => stack.push(term.clone()),
                None => bail!(
                    "coordinate atom c{k} at {at}: out of range for rank {} — a \
                     coordinate IntExpr escaped its iota's value function",
                    coord_terms.len()
                ),
            },
            Term::Add
            | Term::Mul
            | Term::Sub
            | Term::Div
            | Term::Mod
            | Term::Min
            | Term::Max
            | Term::Gte
            | Term::Lt => {
                // Their builders emit RHS terms first, so the stack TOP is
                // the LEFT operand (verified against as_op + the Sub impl).
                let (Some(left), Some(right)) = (stack.pop(), stack.pop()) else {
                    bail!("hlir_to_logical: malformed index expression at {at}");
                };
                let rendered = match term {
                    Term::Add => format!("(IntAdd {left} {right})"),
                    Term::Mul => format!("(IntMul {left} {right})"),
                    Term::Sub => format!("(IntAdd {left} (IntMul (IntLit -1) {right}))"),
                    Term::Div => format!("(IntTruncDiv {left} {right})"),
                    Term::Mod => format!("(IntTruncRem {left} {right})"),
                    Term::CeilDiv => format!("(IntCeilDiv {left} {right})"),
                    Term::Min => format!("(IntMin {left} {right})"),
                    Term::Max => format!("(IntMax {left} {right})"),
                    // Comparisons arrive as 0/1 VALUES in their expressions;
                    // ours are the bool bridge's indicators. Over the discrete
                    // integers, a >= b is spelled b < a+1 — one constructor.
                    Term::Lt => {
                        format!("(IntCastFromBool (BoolLessThanInt {left} {right}))")
                    }
                    Term::Gte => format!(
                        "(IntCastFromBool (BoolLessThanInt {right} (IntAdd {left} (IntLit 1))))"
                    ),
                    _ => unreachable!(),
                };
                stack.push(rendered);
            }
            other => {
                bail!("hlir_to_logical: index-expression term {other:?} at {at} — later slice")
            }
        }
    }
    match (stack.pop(), stack.is_empty()) {
        (Some(result), true) => Ok(result),
        _ => bail!("hlir_to_logical: malformed index expression at {at}"),
    }
}

#[cfg(test)]
mod logical_petgraph_tests {
    use super::*;

    #[test]
    fn tensor_ids_are_petgraph_values_with_ported_operand_edges() {
        let mut cx = Graph::new();
        let lhs = cx.named_tensor("lhs", (2usize, 3usize), DType::F32);
        let rhs = cx.named_tensor("rhs", (2usize, 3usize), DType::F32);
        let sum = lhs + rhs;
        let viewed = sum.expand_dim(0, 4usize).output();

        let graph = cx.logical.petgraph();
        assert_eq!(graph.node_count(), 4);
        assert!(matches!(graph[lhs.id].op, LogicalOp::Input { .. }));
        assert!(matches!(graph[rhs.id].op, LogicalOp::Input { .. }));
        assert!(matches!(graph[sum.id].op, LogicalOp::Add));
        assert!(matches!(
            graph[viewed.id].op,
            LogicalOp::IndexMapApply { .. }
        ));

        let mut add_inputs: Vec<_> = graph
            .edges_directed(sum.id, Direction::Incoming)
            .map(|edge| (edge.weight().0, edge.source()))
            .collect();
        add_inputs.sort_unstable_by_key(|(port, _)| *port);
        assert_eq!(add_inputs, vec![(0, lhs.id), (1, rhs.id)]);

        let view_input: Vec<_> = graph
            .edges_directed(viewed.id, Direction::Incoming)
            .map(|edge| (edge.weight().0, edge.source()))
            .collect();
        assert_eq!(view_input, vec![(0, sum.id)]);

        assert_eq!(cx.logical.input_specs()[0].id, lhs.id);
        assert_eq!(cx.logical.output_specs()[0].id, viewed.id);
    }
}
