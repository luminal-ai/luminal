//! The TestRuntime's boundary bindings, hand-rolled.
//!
//! Core keeps no binding vocabulary at all; every runtime owns its own
//! (runtime-ownership ruling 2026-08-17). This crate is a runtime, so it
//! states its own — it does not borrow the reference runtime's, and
//! nothing here is kept in sync with it.
//!
//! Models are boundary-free logical structure (the 2026-07-24 ruling);
//! everything about REPRESENTATION at the boundary — layouts, buffers,
//! access, freed-by, dynamic-dim seeds, the Bool8 boundary cast — is
//! binding vocabulary, stated per runtime at load time. WHICH values
//! enter and leave, and through which buffers, is this runtime's
//! statement too: aliasing has exactly one spelling, two bindings naming
//! the same buffer id. Every buffer declares its access and who frees
//! it, and the declarations are re-asserted after saturation as checks.
//!
//! THIS runtime's contract: row-major contiguous boundaries at the
//! dtype's own width, byte-code booleans (Bool8), caller-owned buffers.
//! It coincides with the reference runtime's contract today because both
//! are dense out-of-place runtimes describing the same fixtures — but the
//! two are stated separately ON PURPOSE. This runtime is plan-level and
//! carries op shapes the reference registry does not (views, the mutating
//! forms, the fused pair); when its boundary needs to describe one of
//! those, it changes HERE and the reference contract does not move.

use luminal::dtype::DType;
use luminal::graph::{LogicalGraph, ValueId};
use luminal::layout_ir::{Access, FreedBy};
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;

/// One boundary binding: a logical value on a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bound {
    pub value: ValueId,
    pub buffer: i64,
}

/// A buffer's declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDecl {
    pub access: Access,
    pub freed_by: FreedBy,
}

/// The TestRuntime's binding vocabulary: input and output bindings over
/// the buffers they name.
#[derive(Debug, Clone, Default)]
pub struct TestRuntimeBindings {
    inputs: Vec<Bound>,
    outputs: Vec<Bound>,
    buffers: BTreeMap<i64, BufferDecl>,
    next: i64,
}

/// The bound program's parts. The seeds a runtime binds after load
/// (dim ranges, value ranges) go between `prefix` and the schedule.
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
            TestRuntimeBindings::SCHEDULE,
            self.post_checks
        )
    }

    /// The program without its post-schedule checks — the probe a
    /// runtime re-saturates to name which check failed.
    pub fn text_unchecked_with_seeds(&self, seeds: &str) -> String {
        format!("{}{seeds}{}", self.prefix, TestRuntimeBindings::SCHEDULE)
    }
}

impl TestRuntimeBindings {
    /// The standard schedule tail shared by every assembled program on
    /// this runtime: core saturates first, then everything with the
    /// `backend` matchers.
    pub const SCHEDULE: &'static str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (saturate (saturate (run) (run backend) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run cleanup)) (saturate (run fixpoint-invariants)))\n\n";

    pub fn new() -> Self {
        Self::default()
    }

    /// Every input read-only on its own buffer; every leaf on its own
    /// read-write buffer.
    pub fn leaves(graph: &LogicalGraph) -> Self {
        let leaves = graph.leaves();
        Self::dense(graph, &leaves)
    }

    /// Every input read-only on its own buffer; the given outputs each
    /// on their own read-write buffer.
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

    /// Declare (or re-declare) a specific buffer id.
    pub fn declare(&mut self, buffer: i64, access: Access, freed_by: FreedBy) {
        self.buffers.insert(buffer, BufferDecl { access, freed_by });
        self.next = self.next.max(buffer + 1);
    }

    /// Bind an input on a fresh read-only, caller-owned buffer.
    pub fn input(&mut self, value: ValueId) -> i64 {
        let buffer = self.buffer(Access::ReadOnly, FreedBy::Caller);
        self.inputs.push(Bound { value, buffer });
        buffer
    }

    /// Bind an input on an existing buffer.
    pub fn input_on(&mut self, value: ValueId, buffer: i64) {
        self.inputs.push(Bound { value, buffer });
    }

    /// Bind an output on a fresh read-write, caller-owned buffer.
    pub fn output(&mut self, value: ValueId) -> i64 {
        let buffer = self.buffer(Access::ReadWrite, FreedBy::Caller);
        self.outputs.push(Bound { value, buffer });
        buffer
    }

    /// Bind an output on an existing buffer — naming an input's buffer
    /// here is the one way to state that the output writes the input's
    /// storage; the buffer must have been declared `ReadWrite`.
    pub fn output_on(&mut self, value: ValueId, buffer: i64) {
        self.outputs.push(Bound { value, buffer });
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

    /// The boundary element width for a dtype under this binding.
    /// Boolean values cross as Bool8 (the two-legal-codes byte type —
    /// see the preamble's Dtype contract); everything else at its own
    /// bits-of width.
    pub fn width_term(dtype: DType) -> String {
        match dtype {
            DType::Bool => "(bits-of (Bool8))".to_string(),
            other => format!("(bits-of ({other:?}))"),
        }
    }

    /// Render the bound program: the model's cone over the bound values,
    /// then the boundary. Refuses, by name, an input binding on a
    /// non-input value, an output on an undeclared buffer, a reachable
    /// input left unbound, an output bound twice on one buffer, and an
    /// empty output set.
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
        }
        let mut seen = std::collections::HashSet::new();
        for bound in self.inputs.iter().chain(&self.outputs) {
            if !seen.insert((bound.value, bound.buffer)) {
                return Err(format!(
                    "v{} is bound twice on buffer {}",
                    bound.value.index(),
                    bound.buffer
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
                "(let {stem}_layout (RightMajorContiguousElementLayoutLit {shape} {width}))\n\
                 (let {stem}_layout_tensor (LayoutTensorLit {logical} {stem}_layout))\n\
                 (let {stem}_buffer_tensor (BufferTensorLit {stem}_layout_tensor buf{}_id))\n\n",
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
            // A Bool value crosses as Bool8: the BINDING states the byte
            // representation with a cast; the model's value stays logical.
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
                 (let {stem}_layout (RightMajorContiguousElementLayoutLit {shape} {width}))\n\
                 (let {stem}_layout_tensor (LayoutTensorLit {boundary_name} {stem}_layout))\n\
                 (let {stem}_buffer_tensor (BufferTensorLit {stem}_layout_tensor buf{}_id))\n\n",
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
