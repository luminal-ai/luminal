//! The CUDA-lite runtime's boundary bindings, hand-rolled.
//!
//! The model is boundary-free logical structure; which values enter and
//! leave, through which buffers, at which layout, under which contents
//! permission, and whether their storage survives an execution is this
//! runtime's statement, made at load and rendered here into the
//! preamble's boundary vocabulary. Aliasing has exactly one spelling:
//! two bindings naming the same buffer id. Every buffer declares its
//! access and who frees it; the declarations are re-asserted after
//! saturation as checks.
//!
//! WHERE THIS DIVERGES FROM THE REFERENCE RUNTIME'S BINDING, and why the
//! two are separate code: a CUDA boundary is not always dense row-major.
//! Every binding carries a [`BoundaryLayout`], so a caller can hand this
//! runtime storage it already has — a column-major matrix, a strided
//! slice of a larger allocation — without a host-side repack. Every
//! binding also carries a [`Placement`], which is the BUFFER's storage
//! statement: a resident buffer lives in the device arena across
//! executions, an external one is the caller's own device allocation.

use luminal::dtype::DType;
use luminal::graph::{LogicalGraph, ValueId};
use luminal::layout_ir::{Access, FreedBy};
use luminal::shape::{IntExpr, Term};
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;

/// How a bound value's elements sit in its buffer. Element strides, no
/// storage offset: a binding names the start of its own storage.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum BoundaryLayout {
    /// Contiguous, last axis fastest.
    #[default]
    RowMajor,
    /// Contiguous, first axis fastest.
    ColumnMajor,
    /// One element stride per axis, in shape order. A stride is any
    /// extent expression, so a caller whose storage is shaped by a
    /// symbolic dimension states that dimension here rather than a
    /// number it does not have.
    Strided { strides: Vec<IntExpr> },
}

impl BoundaryLayout {
    /// A strided boundary at literal element strides.
    pub fn strided_literal(strides: impl IntoIterator<Item = i64>) -> Self {
        Self::Strided {
            strides: strides.into_iter().map(IntExpr::from).collect(),
        }
    }

    /// The preamble's element-layout literal over `shape` at `width`.
    /// A strided layout renders the `IntAffineExpr` cons list the
    /// preamble's own `affine-zip` builds: one
    /// `(IntMul (CoordVar shape axis) stride)` summand per axis, axes
    /// counted FROM THE END (the last axis is 0). Each stride goes
    /// through the CORE extent renderer, so a symbolic one reaches the
    /// e-graph as the same `IntVar` the shape uses.
    fn term(&self, shape: &str, width: &str) -> Result<String, String> {
        Ok(match self {
            Self::RowMajor => format!("(RightMajorContiguousElementLayoutLit {shape} {width})"),
            Self::ColumnMajor => format!("(LeftMajorContiguousElementLayoutLit {shape} {width})"),
            Self::Strided { strides } => {
                let mut chain = "(IntAffineExprNil)".to_string();
                for (position, stride) in strides.iter().enumerate().rev() {
                    let axis = strides.len() - 1 - position;
                    let stride = LogicalGraph::dim_term(stride)?;
                    chain = format!(
                        "(IntAffineExprCons (IntMul (CoordVar {shape} {axis}) {stride}) {chain})"
                    );
                }
                format!("(StridedElementLayoutLit {shape} {chain} {width})")
            }
        })
    }
}

/// A stride's value where the caller stated a literal one; `None` for a
/// stride that carries a symbol, which only the runtime's dims decide.
fn literal_stride(stride: &IntExpr) -> Option<i64> {
    let folded = stride.simplify();
    let terms = folded.terms.read();
    match terms[..] {
        [Term::Num(n)] => Some(n),
        _ => None,
    }
}

/// Where a BUFFER's storage lives between executions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Placement {
    /// Staged from the host before each execution, and read back to the
    /// host after it (the default).
    #[default]
    Staged,
    /// Kept in the device arena for the runtime's life: uploaded on the
    /// first execution and then only when the caller restages it, and
    /// written in place by an output bound on the same buffer.
    Resident,
    /// Caller device memory for the runtime's life: never host-staged,
    /// never given an arena range; the caller supplies a device pointer
    /// for the buffer before each execute.
    External,
}

/// One boundary binding: a logical value on a buffer, at a layout.
/// `placement` is the BUFFER's statement, carried by every binding that
/// names it and meaningful for outputs too: a [`Placement::Staged`]
/// output is written in the arena and read back to the host, an
/// [`Placement::External`] one is written straight into the caller's
/// pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bound {
    pub value: ValueId,
    pub buffer: i64,
    pub layout: BoundaryLayout,
    pub placement: Placement,
}

/// A buffer's declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDecl {
    pub access: Access,
    pub freed_by: FreedBy,
}

/// The CUDA-lite runtime's binding vocabulary.
#[derive(Debug, Clone, Default)]
pub struct CudaBindings {
    inputs: Vec<Bound>,
    outputs: Vec<Bound>,
    buffers: BTreeMap<i64, BufferDecl>,
    next: i64,
}

/// The bound program's parts. The seeds a runtime binds after load
/// (dim ranges) go between `prefix` and the schedule.
#[derive(Debug, Clone)]
pub struct BoundProgram {
    /// Model text plus boundary text.
    pub prefix: String,
    /// Post-schedule checks: the recorder's shape contracts, then the
    /// per-buffer declaration invariants.
    pub post_checks: String,
    /// The same checks as labeled units, for naming a failing door.
    pub labeled_checks: Vec<(String, String)>,
    pub inputs: Vec<Bound>,
    pub outputs: Vec<Bound>,
    /// The egglog `let` each bound value's boundary attaches to.
    pub let_names: FxHashMap<ValueId, String>,
}

impl BoundProgram {
    pub fn text(&self) -> String {
        self.text_with_seeds("")
    }

    pub fn text_with_seeds(&self, seeds: &str) -> String {
        format!(
            "{}{seeds}{}{}",
            self.prefix,
            CudaBindings::SCHEDULE,
            self.post_checks
        )
    }

    /// The program without its post-schedule checks — the probe a
    /// runtime re-saturates to name which check failed.
    pub fn text_unchecked_with_seeds(&self, seeds: &str) -> String {
        format!("{}{seeds}{}", self.prefix, CudaBindings::SCHEDULE)
    }

    /// The buffers whose storage is not ordinary host staging: the
    /// arena's resident set, stated by the input bindings, and the
    /// caller-owned external set, stated by bindings of either side.
    pub fn residents(&self) -> crate::resident::ResidentBindings {
        crate::resident::ResidentBindings {
            inputs: self
                .inputs
                .iter()
                .filter(|bound| bound.placement == Placement::Resident)
                .map(|bound| bound.buffer)
                .collect(),
            externals: self
                .inputs
                .iter()
                .chain(&self.outputs)
                .filter(|bound| bound.placement == Placement::External)
                .map(|bound| bound.buffer)
                .collect(),
        }
    }
}

impl CudaBindings {
    /// The schedule tail this runtime appends to every assembled
    /// program: core rulesets and logical helpers saturate first, then
    /// everything including the `backend` matchers (the op matchers and
    /// the cuBLASLt estate) saturates together.
    pub const SCHEDULE: &'static str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (saturate (saturate (run) (run backend) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run cleanup)) (saturate (run fixpoint-invariants)))\n\n";

    pub fn new() -> Self {
        Self::default()
    }

    /// Every input read-only on its own buffer; every leaf on its own
    /// read-write buffer. The default `load` binding.
    pub fn leaves(graph: &LogicalGraph) -> Self {
        let leaves = graph.leaves();
        Self::dense(graph, &leaves)
    }

    /// Every input read-only on its own buffer; the given outputs each
    /// on their own read-write buffer. All row-major, all host-staged.
    pub fn dense(graph: &LogicalGraph, outputs: &[ValueId]) -> Self {
        let mut bindings = Self::new();
        for input in graph.inputs() {
            bindings.input(input);
        }
        for &output in outputs {
            bindings.output(output);
        }
        bindings
    }

    /// Declare a fresh buffer.
    pub fn buffer(&mut self, access: Access, freed_by: FreedBy) -> i64 {
        let id = self.next;
        self.next += 1;
        self.buffers.insert(id, BufferDecl { access, freed_by });
        id
    }

    /// Declare (or re-declare) a specific buffer id. Re-declaring is how
    /// an input's buffer becomes `ReadWrite` so an output may be bound
    /// on it; the bindings that name the buffer are untouched.
    pub fn declare(&mut self, buffer: i64, access: Access, freed_by: FreedBy) {
        self.buffers.insert(buffer, BufferDecl { access, freed_by });
        self.next = self.next.max(buffer + 1);
    }

    fn push_input(
        &mut self,
        value: ValueId,
        buffer: i64,
        layout: BoundaryLayout,
        placement: Placement,
    ) {
        self.inputs.push(Bound {
            value,
            buffer,
            layout,
            placement,
        });
    }

    fn push_output(
        &mut self,
        value: ValueId,
        buffer: i64,
        layout: BoundaryLayout,
        placement: Placement,
    ) {
        self.outputs.push(Bound {
            value,
            buffer,
            layout,
            placement,
        });
    }

    /// The placement the bindings already on `buffer` carry — placement
    /// is the buffer's property, so a binding added to an existing
    /// buffer inherits it. A buffer with no binding yet is host-staged.
    fn placement_on(&self, buffer: i64) -> Placement {
        self.inputs
            .iter()
            .chain(&self.outputs)
            .find(|bound| bound.buffer == buffer)
            .map_or(Placement::default(), |bound| bound.placement)
    }

    /// Bind an input on a fresh read-only, caller-owned buffer,
    /// row-major and host-staged.
    pub fn input(&mut self, value: ValueId) -> i64 {
        self.input_with(value, BoundaryLayout::RowMajor)
    }

    /// [`Self::input`] at the caller's layout.
    pub fn input_with(&mut self, value: ValueId, layout: BoundaryLayout) -> i64 {
        let buffer = self.buffer(Access::ReadOnly, FreedBy::Caller);
        self.push_input(value, buffer, layout, Placement::Staged);
        buffer
    }

    /// Bind an input on an existing buffer, row-major. The binding
    /// INHERITS the buffer's placement.
    pub fn input_on(&mut self, value: ValueId, buffer: i64) {
        self.input_on_with(value, buffer, BoundaryLayout::RowMajor);
    }

    /// [`Self::input_on`] at the caller's layout.
    pub fn input_on_with(&mut self, value: ValueId, buffer: i64, layout: BoundaryLayout) {
        let placement = self.placement_on(buffer);
        self.push_input(value, buffer, layout, placement);
    }

    /// Bind an input on a fresh DEVICE-RESIDENT buffer: its storage is
    /// allocated once in the arena and survives every execution, so a
    /// weight is uploaded once, and a state bound as an output on this
    /// same buffer is mutated in place with no readback.
    pub fn input_resident(&mut self, value: ValueId) -> i64 {
        self.input_resident_with(value, BoundaryLayout::RowMajor)
    }

    /// [`Self::input_resident`] at the caller's layout.
    pub fn input_resident_with(&mut self, value: ValueId, layout: BoundaryLayout) -> i64 {
        let buffer = self.buffer(Access::ReadOnly, FreedBy::Caller);
        self.push_input(value, buffer, layout, Placement::Resident);
        buffer
    }

    /// Bind an input on a fresh CALLER-OWNED DEVICE buffer: the storage
    /// is the caller's live allocation, so it is never host-staged and
    /// never given an arena range, and the caller supplies its address
    /// before each execution.
    pub fn input_external(&mut self, value: ValueId) -> i64 {
        self.input_external_with(value, BoundaryLayout::RowMajor)
    }

    /// [`Self::input_external`] at the caller's layout.
    pub fn input_external_with(&mut self, value: ValueId, layout: BoundaryLayout) -> i64 {
        let buffer = self.buffer(Access::ReadOnly, FreedBy::Caller);
        self.push_input(value, buffer, layout, Placement::External);
        buffer
    }

    /// Bind an output on a fresh read-write, caller-owned buffer,
    /// row-major.
    pub fn output(&mut self, value: ValueId) -> i64 {
        self.output_with(value, BoundaryLayout::RowMajor)
    }

    /// [`Self::output`] at the caller's layout.
    pub fn output_with(&mut self, value: ValueId, layout: BoundaryLayout) -> i64 {
        let buffer = self.buffer(Access::ReadWrite, FreedBy::Caller);
        self.output_on_with(value, buffer, layout);
        buffer
    }

    /// Bind an output on a fresh read-write CALLER-OWNED DEVICE buffer:
    /// the producing op writes straight into the caller's allocation,
    /// with no arena range and no host readback.
    pub fn output_external(&mut self, value: ValueId) -> i64 {
        self.output_external_with(value, BoundaryLayout::RowMajor)
    }

    /// [`Self::output_external`] at the caller's layout.
    pub fn output_external_with(&mut self, value: ValueId, layout: BoundaryLayout) -> i64 {
        let buffer = self.buffer(Access::ReadWrite, FreedBy::Caller);
        self.push_output(value, buffer, layout, Placement::External);
        buffer
    }

    /// Bind an output on an existing buffer — naming an input's buffer
    /// here is the one way to state that the output writes the input's
    /// storage; the buffer must have been declared `ReadWrite`. The
    /// binding INHERITS the buffer's placement.
    pub fn output_on(&mut self, value: ValueId, buffer: i64) {
        self.output_on_with(value, buffer, BoundaryLayout::RowMajor);
    }

    /// [`Self::output_on`] at the caller's layout.
    pub fn output_on_with(&mut self, value: ValueId, buffer: i64, layout: BoundaryLayout) {
        let placement = self.placement_on(buffer);
        self.push_output(value, buffer, layout, placement);
    }

    pub fn inputs(&self) -> &[Bound] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Bound] {
        &self.outputs
    }

    pub fn buffers(&self) -> &BTreeMap<i64, BufferDecl> {
        &self.buffers
    }

    pub fn buffer_of_input(&self, value: ValueId) -> Option<i64> {
        self.inputs
            .iter()
            .find(|b| b.value == value)
            .map(|b| b.buffer)
    }

    /// The boundary element width: booleans cross as Bool8 bytes.
    /// (Half/fp8 boundary widths land with the device dtype work, and
    /// land here.)
    pub fn width_term(dtype: DType) -> String {
        match dtype {
            DType::Bool => "(bits-of (Bool8))".to_string(),
            other => format!("(bits-of ({other:?}))"),
        }
    }

    /// Render the bound program: the model's cone over the bound values,
    /// then the boundary. Refuses, by name, an input binding on a
    /// non-input value, a binding on an undeclared buffer, a reachable
    /// input left unbound, an input value bound twice on one buffer, an
    /// empty output set, a strided binding whose stride count is not the
    /// value's rank or whose literal strides include a negative one,
    /// and two bindings on one buffer that disagree about placement.
    pub fn bind(&self, graph: &LogicalGraph) -> Result<BoundProgram, String> {
        if self.outputs.is_empty() {
            return Err("bindings name no output".to_string());
        }
        for bound in &self.inputs {
            if !graph.is_input(bound.value) {
                return Err(format!(
                    "input binding on v{}, which is not an input",
                    bound.value.index()
                ));
            }
        }
        for bound in self.inputs.iter().chain(&self.outputs) {
            if !self.buffers.contains_key(&bound.buffer) {
                return Err(format!(
                    "v{} is bound on undeclared buffer {}",
                    bound.value.index(),
                    bound.buffer
                ));
            }
            if let BoundaryLayout::Strided { strides } = &bound.layout {
                let rank = graph.value_dims(bound.value).len();
                if strides.len() != rank {
                    return Err(format!(
                        "v{} is bound at {} strides but has rank {rank}: a strided \
                         boundary states one element stride per axis",
                        bound.value.index(),
                        strides.len()
                    ));
                }
                // A stride the caller spelled symbolically is taken as
                // stated: which number it is, is the runtime's dims to
                // say. A literal one is judged here.
                for (axis, stride) in strides.iter().enumerate() {
                    let Some(stride) = literal_stride(stride) else {
                        continue;
                    };
                    if stride < 0 {
                        return Err(format!(
                            "v{}'s strided boundary has stride {stride} on axis {axis}: \
                             boundary element strides are never negative",
                            bound.value.index()
                        ));
                    }
                }
            }
        }
        // An input value is bound once per buffer. An output may repeat an
        // input's binding — the value passes through, the caller receives
        // the storage it handed in — or another output's: two returned
        // names for one value on one buffer are two slots on one content.
        let mut seen = std::collections::HashSet::new();
        for bound in &self.inputs {
            if !seen.insert((bound.value, bound.buffer)) {
                return Err(format!(
                    "v{} is bound twice as an input on buffer {}",
                    bound.value.index(),
                    bound.buffer
                ));
            }
        }
        // Placement is the BUFFER's property — the arena allocates homes
        // by buffer id and the caller supplies one device pointer per
        // buffer — so every binding sharing a buffer must agree.
        let mut placement_of: BTreeMap<i64, Placement> = BTreeMap::new();
        for bound in self.inputs.iter().chain(&self.outputs) {
            if let Some(other) = placement_of.insert(bound.buffer, bound.placement)
                && other != bound.placement
            {
                return Err(format!(
                    "buffer {} carries both a {other:?} and a {:?} binding",
                    bound.buffer, bound.placement
                ));
            }
        }
        let roots: Vec<ValueId> = self
            .outputs
            .iter()
            .chain(&self.inputs)
            .map(|b| b.value)
            .collect();
        let cone = graph.cone(&roots);
        for input in graph.inputs() {
            if cone.contains(&input) && self.buffer_of_input(input).is_none() {
                return Err(format!(
                    "input v{} feeds a bound output but has no input binding",
                    input.index()
                ));
            }
        }

        let mut prefix = graph.render(&roots)?;
        let mut let_names: FxHashMap<ValueId, String> = FxHashMap::default();
        for (k, decl) in &self.buffers {
            let access = match decl.access {
                Access::ReadOnly => "ReadOnly",
                Access::ReadWrite => "ReadWrite",
            };
            let freed = match decl.freed_by {
                FreedBy::Caller => "CallerFrees",
                FreedBy::Program => "ProgramFrees",
            };
            prefix.push_str(&format!(
                "(let buf{k}_id (BufferLit {k}))\n\
                 (set (buffer-access-of buf{k}_id) ({access}))\n\
                 (set (buffer-freed-by buf{k}_id) ({freed}))\n"
            ));
        }
        prefix.push('\n');
        let mut input_tensors = Vec::new();
        for bound in &self.inputs {
            let idx = bound.value.index();
            let stem = format!("nat{idx}");
            let logical = graph.let_name(bound.value);
            let shape = graph.value_shape_term(bound.value)?;
            let width = Self::width_term(graph.value_dtype(bound.value));
            prefix.push_str(&format!(
                "(let {stem}_layout {})\n\
                 (let {stem}_layout_tensor (LayoutTensorLit {logical} {stem}_layout))\n\
                 (let {stem}_buffer_tensor (BufferTensorLit {stem}_layout_tensor buf{}_id))\n\n",
                bound.layout.term(&shape, &width)?,
                bound.buffer
            ));
            input_tensors.push(format!("{stem}_buffer_tensor"));
            let_names.insert(bound.value, logical);
        }
        let mut output_tensors = Vec::new();
        let mut output_stems: FxHashMap<ValueId, usize> = FxHashMap::default();
        for bound in &self.outputs {
            let idx = bound.value.index();
            let repeat = output_stems.entry(bound.value).or_insert(0);
            let stem = if *repeat == 0 {
                format!("natout{idx}")
            } else {
                format!("natout{idx}_b{}", bound.buffer)
            };
            *repeat += 1;
            let value_name = graph.let_name(bound.value);
            let dtype = graph.value_dtype(bound.value);
            // Bool crosses as Bool8 through an explicit boundary cast
            // (the Bool8 ruling): the two legal codes are a storage
            // statement, never a width override.
            let (boundary_name, cast_text) = if dtype == DType::Bool {
                let bool8 = format!("{stem}_bool8");
                (
                    bool8.clone(),
                    format!("(let {bool8} (LogicalCast {value_name} (Bool8)))\n"),
                )
            } else {
                (value_name.clone(), String::new())
            };
            let shape = graph.value_shape_term(bound.value)?;
            let width = Self::width_term(dtype);
            prefix.push_str(&format!(
                "{cast_text}\
                 (let {stem}_layout {})\n\
                 (let {stem}_layout_tensor (LayoutTensorLit {boundary_name} {stem}_layout))\n\
                 (let {stem}_buffer_tensor (BufferTensorLit {stem}_layout_tensor buf{}_id))\n\n",
                bound.layout.term(&shape, &width)?,
                bound.buffer
            ));
            output_tensors.push(format!("{stem}_buffer_tensor"));
            let_names.entry(bound.value).or_insert(value_name);
        }
        let join = |items: &[String]| {
            let mut term = "(BufferTensorNil)".to_string();
            for item in items.iter().rev() {
                term = format!("(BufferTensorCons {item} {term})");
            }
            term
        };
        prefix.push_str(&format!(
            "(let nat_input_boundary (BufferInputLit {}))\n(let nat_output_boundary (BufferOutputLit {}))\n\n",
            join(&input_tensors),
            join(&output_tensors)
        ));

        // Post-schedule checks: the recorder's shape contracts, then the
        // declaration invariants — every buffer states its access and its
        // deallocation responsibility, re-asserted at the end of saturation.
        let mut post_checks = String::new();
        let mut labeled_checks = Vec::new();
        for k in self.buffers.keys() {
            let text = format!(
                "(check (= ?access{k} (buffer-access-of buf{k}_id)))\n\
                 (check (= ?freed{k} (buffer-freed-by buf{k}_id)))\n"
            );
            post_checks.push_str(&text);
            labeled_checks.push((format!("buffer {k} declares access and freed-by"), text));
        }
        Ok(BoundProgram {
            prefix,
            post_checks,
            labeled_checks,
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            let_names,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A graph with a mutation shape: two inputs and one sum, so a sink
    /// can be bound on an input's buffer.
    fn pair() -> (luminal::graph::Graph, ValueId, ValueId, ValueId) {
        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor(4, DType::F32);
        let b = cx.tensor(4, DType::F32);
        let sum = a + b;
        (cx, a.id, b.id, sum.id)
    }

    /// PLACEMENT IS THE BUFFER'S: an output bound on an External input's
    /// buffer inherits External, and `residents()` reports that buffer
    /// once, over both sides of the boundary.
    #[test]
    fn a_sink_inherits_its_buffers_external_placement() {
        let (cx, a, b, sum) = pair();
        let mut bindings = CudaBindings::new();
        let home = bindings.input_external(a);
        bindings.declare(home, Access::ReadWrite, FreedBy::Caller);
        bindings.input(b);
        bindings.output_on(sum, home);
        let bound = bindings.bind(&cx.logical).unwrap();
        assert_eq!(bound.outputs[0].placement, Placement::External);
        let residents = bound.residents();
        assert_eq!(residents.externals, [home].into_iter().collect());
        assert!(residents.inputs.is_empty());
    }

    /// A fresh external output owns its own buffer, and a staged input
    /// beside it stays out of the external set.
    #[test]
    fn an_external_output_declares_its_own_placement() {
        let (cx, a, b, sum) = pair();
        let mut bindings = CudaBindings::new();
        bindings.input(a);
        let staged = bindings.input(b);
        let out = bindings.output_external(sum);
        let residents = bindings.bind(&cx.logical).unwrap().residents();
        assert_eq!(residents.externals, [out].into_iter().collect());
        assert!(!residents.externals.contains(&staged));
    }

    /// A SYMBOLIC STRIDE REACHES THE PREAMBLE AS THE DIM ITSELF: the
    /// boundary's element strides go through the core extent renderer,
    /// so the stride and the shape name one `IntVar`.
    #[test]
    fn a_symbolic_stride_renders_as_the_dim() {
        let mut cx = luminal::graph::Graph::new();
        let x = cx.tensor(('n', 4usize), DType::F32);
        let out = x + 1.;
        let mut bindings = CudaBindings::new();
        bindings.input_with(
            x.id,
            BoundaryLayout::Strided {
                strides: vec![IntExpr::from(1i64), IntExpr::from('n')],
            },
        );
        bindings.output(out.id);
        let prefix = bindings.bind(&cx.logical).unwrap().prefix;
        let shape = cx.logical.value_shape_term(x.id).unwrap();
        let expected = format!(
            "(StridedElementLayoutLit {shape} \
             (IntAffineExprCons (IntMul (CoordVar {shape} 1) (IntLit 1)) \
             (IntAffineExprCons (IntMul (CoordVar {shape} 0) (IntVar \"n\")) \
             (IntAffineExprNil))) (bits-of (F32)))"
        );
        assert!(prefix.contains(&expected), "{prefix}");
    }

    /// TWO PLACEMENTS ON ONE BUFFER ARE REFUSED. The constructors cannot
    /// state it (every binding on a buffer inherits its placement), so
    /// the disagreement is pushed directly here — this is the door that
    /// keeps the arena and the pointer table from disagreeing about who
    /// owns a buffer's bytes.
    #[test]
    fn two_placements_on_one_buffer_are_refused() {
        let (cx, a, b, sum) = pair();
        let mut bindings = CudaBindings::new();
        let home = bindings.input_external(a);
        bindings.push_input(b, home, BoundaryLayout::RowMajor, Placement::Staged);
        bindings.output(sum);
        let refusal = bindings.bind(&cx.logical).unwrap_err();
        assert!(
            refusal.contains(&format!("buffer {home} carries both"))
                && refusal.contains("External")
                && refusal.contains("Staged"),
            "{refusal}"
        );
    }
}
