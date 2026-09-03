//! `ReferenceRuntime`: the CPU reference executor for logical-SSA
//! bufferized plans (ruling 2026-07-28: lives in luminal; once the path is
//! COMPLETE it replaces `ReferenceRuntime` — not before).
//!
//! Executes a [`BufferIrGraph`] directly: every buffer is a [`TypedBuffer`]
//! sized by ASSIGNMENT LOOKUP (corrected contract, 2026-08-31) — the plan
//! says which tensor a buffer backs, the carried [`RefLayout`] is that
//! tensor's elected layout, and allocation is span-of-layout elements in
//! the layout's own dtype. No sizing walk, no voting: every consumer
//! takes the BufferId blindly. Compute nodes dispatch through THIS
//! runtime's kernel registry ([`kernels`]) by concrete op type — ops
//! carry no execution of their own (ruling 2026-08-06), and an op
//! without a kernel here refuses by name. Caller data binds by the
//! numeric `BufferLit` id, the same key `hlir_to_logical` derives from
//! HLIR node indices, so differential tests against `ReferenceRuntime`
//! bind identically on both sides.

use anyhow::{anyhow, ensure, Context, Result};
use petgraph::algo::toposort;
use rustc_hash::FxHashMap;

use luminal::buffer_tensor_ir::{ReferenceKernelCtx, TypedBuffer};
use luminal::bufferize::{BufferId, BufferIrGraph, BufferNode, OutputBinding};

use crate::layouts::RefLayout;

/// The reference backend's implementation inventory, DERIVED from the
/// kernel registry: a matcher's op is claimed iff a kernel bearing its
/// label exists in [`kernels`] (ruling 2026-08-06 — the runtime can
/// neither over-claim, turning kernel gaps into search refusals, nor
/// silently under-claim). The view op and the Mutating family fall out
/// naturally: neither has a kernel (the reference runtime materializes
/// views and mutates nothing in place — rulings 2026-07-28 / 2026-08-05).
pub fn reference_allow_list() -> Vec<&'static str> {
    let implemented: rustc_hash::FxHashSet<&'static str> = crate::kernels::reference_kernels()
        .iter()
        .map(|kernel| kernel.label)
        .collect();
    crate::ops::built_in_matchers()
        .iter()
        .map(|matcher| matcher.egglog_constructor())
        .filter(|constructor| {
            let label = constructor
                .strip_prefix("LayoutTensorOp")
                .unwrap_or(constructor);
            implemented.contains(label)
        })
        .collect()
}

/// M3 Step 2: what `load` captured from a natively-recorded Graph — the
/// pre-schedule program text (model + reference-binding defaults), the
/// I/O slots, the post-schedule authoring checks, plus whatever the
/// binding calls accumulate before `search` assembles and saturates.
struct NativeSpec {
    pre_schedule: String,
    input_slots: Vec<luminal::graph::InputSlot>,
    output_slots: Vec<luminal::graph::OutputSlot>,
    post_checks: String,
    labeled_checks: Vec<(String, String)>,
    binding_seeds: String,
    ops: Option<Vec<&'static str>>,
}

#[derive(Default)]
pub struct ReferenceRuntime {
    plan: Option<BufferIrGraph<RefLayout>>,
    /// Caller-staged data by numeric `BufferLit` id, consumed at `execute`.
    staged: FxHashMap<i64, TypedBuffer>,
    /// Post-execute storage, kept for `get_f32` / `get_bool`.
    storage: FxHashMap<BufferId, TypedBuffer>,
    /// `BufferLit` id → plan buffer, built at `load_plan`.
    lit_index: FxHashMap<i64, BufferId>,
    /// Role-split tensor→buffer maps (the retired-HLIR-keyspace design,
    /// 2026-08-05): `set_data` consults inputs ONLY, `get_*` consults
    /// outputs ONLY. When one tensor is both (an input passed straight
    /// to output), writes stage the input buffer and reads see the
    /// output buffer — two buffers, no ambiguity, no fallback.
    input_buffers: FxHashMap<petgraph::graph::NodeIndex, i64>,
    output_buffers: FxHashMap<petgraph::graph::NodeIndex, i64>,
    /// M3 Step 2 native-ladder state (`load` → bind → `with_ops` → `search`).
    native: Option<NativeSpec>,
}

impl ReferenceRuntime {
    /// Register the tensor→buffer role maps from a program's slots.
    pub fn stage_slots(
        &mut self,
        inputs: &[luminal::graph::InputSlot],
        outputs: &[luminal::graph::OutputSlot],
    ) {
        self.input_buffers = inputs.iter().map(|s| (s.tensor, s.buffer)).collect();
        self.output_buffers = outputs.iter().map(|s| (s.tensor, s.buffer)).collect();
    }

    /// Load a plan for execution.
    ///
    /// CORRECTION 7, and OPTION B's answer to it. A plan built by this
    /// runtime's own `search` needs nothing else: the runtime owns the
    /// recorder, the e-graph and the elections, so it knew every boundary
    /// binding and every layout before it called `bufferize`. An
    /// EXTERNALLY LOADED or HAND-BUILT plan has no such runtime behind it,
    /// and the corrected contract requires that the boundary/layout
    /// knowledge be supplied explicitly, with a loud bail when absent and
    /// never a guess.
    ///
    /// Under Option B that argument is already in the plan, so this
    /// signature stays one-argument: `Buffer::layout` carries every
    /// backed tensor's elected layout (span + dtype ⇒ allocation) and
    /// `OutputBinding::layout` carries every delivery's. The "loud bail
    /// when absent" lives at USE: `execute` refuses a buffer whose layout
    /// has no literal span or no dtype fact, naming the buffer and the
    /// tensor it backs. Nothing is defaulted.
    ///
    /// Boundary bindings likewise ride the plan: `Buffer::lit` is the
    /// numeric `BufferLit` key caller data binds by, indexed here.
    pub fn load_plan(&mut self, plan: BufferIrGraph<RefLayout>) {
        self.lit_index = plan
            .buffers
            .values()
            .filter_map(|buffer| buffer.lit.map(|lit| (lit, buffer.id.clone())))
            .collect();
        self.plan = Some(plan);
        self.storage.clear();
    }

    /// M3 Step 2, the native entry ladder: LOAD a natively-recorded graph
    /// (the model + reference-binding defaults; loud if the recorder is
    /// poisoned) — then bind, choose allowable ops, and `search`.
    pub fn load(graph: &luminal::graph::Graph) -> Result<Self> {
        let (pre_schedule, input_slots, output_slots, post_checks, labeled_checks) = graph
            .logical
            .bound_parts(&crate::bindings::ReferenceBindings)
            .map_err(|reason| anyhow!("native load refused: {reason}"))?;
        Ok(Self {
            native: Some(NativeSpec {
                pre_schedule,
                input_slots,
                output_slots,
                post_checks,
                labeled_checks,
                binding_seeds: String::new(),
                ops: None,
            }),
            ..Self::default()
        })
    }

    /// BINDING: seed a dynamic dim's range (bounds-on-vars — never a pin).
    pub fn bind_dyn_range(
        &mut self,
        var: impl Into<luminal::shape::Symbol>,
        lower: u64,
        upper: u64,
    ) -> Result<()> {
        let var = var.into();
        let spec = self
            .native
            .as_mut()
            .ok_or_else(|| anyhow!("bind before load"))?;
        spec.binding_seeds.push_str(&format!(
            "(set (lower-bound-of (IntVar \"{var}\")) (bigint {lower}))\n\
             (set (upper-bound-of (IntVar \"{var}\")) (bigint {upper}))\n"
        ));
        Ok(())
    }

    /// BINDING: declare an Int input tensor's VALUE range (typed-buffers
    /// landing D). Ints are non-wrapping, so plain Int arithmetic only
    /// implements under value-bounds proofs — for arithmetic over
    /// caller data the proof starts here, with the caller declaring
    /// what the data can hold (token ids in [0, vocab), etc.). Bounds
    /// facts, never pins: the lattice tightens monotonically.
    pub fn bind_value_range(
        &mut self,
        tensor: petgraph::graph::NodeIndex,
        lower: i64,
        upper: i64,
    ) -> Result<()> {
        let spec = self
            .native
            .as_mut()
            .ok_or_else(|| anyhow!("bind before load"))?;
        anyhow::ensure!(lower <= upper, "empty value range [{lower}, {upper}]");
        let name = spec
            .input_slots
            .iter()
            .find(|slot| slot.tensor == tensor)
            .map(|slot| slot.value_name.clone())
            .ok_or_else(|| anyhow!("tensor {tensor:?} is not a bound input"))?;
        spec.binding_seeds.push_str(&format!(
            "(set (value-lower-bound-of {name}) (bigint {lower}))
             (set (value-upper-bound-of {name}) (bigint {upper}))
"
        ));
        Ok(())
    }

    /// The ALLOWABLE-OPS inventory for this runtime (per-runtime API,
    /// deliberately unstandardized — ruling 2026-07-30).
    pub fn with_ops(&mut self, ops: Vec<&'static str>) -> Result<()> {
        let spec = self
            .native
            .as_mut()
            .ok_or_else(|| anyhow!("with_ops before load"))?;
        spec.ops = Some(ops);
        Ok(())
    }

    /// SEARCH: one saturation to fixpoint discovers the implementations;
    /// selection then prices every candidate by EXECUTING its bufferized
    /// plan on this runtime with the given data; the winner loads.
    pub fn search(
        &mut self,
        input_data: &FxHashMap<petgraph::graph::NodeIndex, TypedBuffer>,
        options: &luminal::implementation_search::ImplementationSearchOptions,
    ) -> Result<luminal::implementation_search::SearchOutcome<RefLayout>> {
        let spec = self
            .native
            .take()
            .ok_or_else(|| anyhow!("search before load"))?;
        let text = format!(
            "{}{}{}{}",
            spec.pre_schedule,
            spec.binding_seeds,
            crate::bindings::ReferenceBindings::SCHEDULE,
            spec.post_checks
        );
        let program = luminal::graph::LogicalProgram {
            text,
            input_slots: spec.input_slots,
            output_slots: spec.output_slots,
        };
        let full = format!("{}\n\n{}", crate::assembled_program(), program.text);
        let mut egraph = luminal::egglog_snippet::new_egraph();
        let saturation_start = std::time::Instant::now();
        if let Err(err) = egraph.parse_and_run_program(None, &full) {
            // NAME THE DOOR (ruling 2026-08-13): a failed authoring
            // contract must never surface as a bare saturation error.
            // Re-saturate WITHOUT the checks, then run each labeled
            // check alone to isolate the culprits — the label carries
            // what failed and how to unblock it. (Failure path only;
            // the green path pays nothing.)
            let unchecked = format!(
                "{}\n\n{}{}{}",
                crate::assembled_program(),
                spec.pre_schedule,
                spec.binding_seeds,
                crate::bindings::ReferenceBindings::SCHEDULE,
            );
            let mut probe = luminal::egglog_snippet::new_egraph();
            if probe.parse_and_run_program(None, &unchecked).is_ok() {
                let mut failed: Vec<&str> = Vec::new();
                for (label, text) in &spec.labeled_checks {
                    if probe.parse_and_run_program(None, text).is_err() {
                        failed.push(label);
                    }
                }
                if !failed.is_empty() {
                    return Err(anyhow!(
                        "shape contracts failed:\n  - {}",
                        failed.join("\n  - ")
                    ));
                }
            }
            return Err(anyhow!("native saturation failed: {err}"));
        }
        let saturation_nanos = saturation_start.elapsed().as_nanos();
        let serialize_start = std::time::Instant::now();
        let serialized = egraph.serialize(egglog::SerializeConfig::default()).egraph;
        let serialize_nanos = serialize_start.elapsed().as_nanos();
        let mut outcome = crate::search::search_implementations_with_ops(
            &serialized,
            &program,
            input_data,
            options,
            spec.ops,
        )?;
        outcome.timings.saturation_nanos = saturation_nanos;
        outcome.timings.serialize_nanos = serialize_nanos;
        self.stage_slots(&program.input_slots, &program.output_slots);
        self.load_plan(outcome.best_plan.clone());
        Ok(outcome)
    }

    /// Stage caller data for an INPUT tensor — TYPED (2026-08-11): the
    /// payload's variant must match the buffer's dtype at execute;
    /// there is no conversion at this boundary, ever. `Vec<f32>`,
    /// `Vec<i32>`, and `Vec<i64>` convert via `From`; boolean data must
    /// come through the validated [`TypedBuffer::bool8`] constructor.
    /// Loud if the tensor is not a bound input of the loaded program.
    pub fn set_data(&mut self, tensor: petgraph::graph::NodeIndex, data: impl Into<TypedBuffer>) {
        let buffer = *self
            .input_buffers
            .get(&tensor)
            .unwrap_or_else(|| panic!("tensor {tensor:?} is not a bound input"));
        self.staged.insert(buffer, data.into());
    }

    /// Buffer-id staging for search internals (the slots carry the ids).
    pub fn set_data_buffer(&mut self, buffer: i64, data: impl Into<TypedBuffer>) {
        self.staged.insert(buffer, data.into());
    }

    pub fn execute(&mut self) -> Result<()> {
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| anyhow!("no plan loaded"))?;

        // ESCAPE GUARD (ruling 2026-08-27): an output slot's backing
        // storage must SURVIVE the call — FreedBy::Caller, whatever the
        // owner. FreedBy::Program backing an output means the caller
        // would receive bytes the program destroys: minted non-escaping
        // storage (Owner::System) and DONATED boundary storage
        // (Owner::Caller — validate()'s donated arm forbids exactly this
        // plan shape) alike. The pre-lowering certificate rejects such
        // plans, but hand-built / load_plan plans never pass through it —
        // so the executor re-checks, loudly.
        for node in plan.dag.node_weights() {
            if let BufferNode::BufferOutput { slots } = node {
                for slot in slots {
                    let Some(buffer) = plan.buffers.get(&slot.buffer) else {
                        anyhow::bail!(
                            "output slot {} names unknown buffer {:?}",
                            slot.index,
                            slot.buffer
                        );
                    };
                    ensure!(
                        buffer.freed_by == luminal::layout_ir::FreedBy::Caller,
                        "output slot {} is backed by NON-ESCAPING buffer {} \
                         (FreedBy::Program, {:?}-owned) — escaped output storage \
                         must be FreedBy::Caller; refusing to hand the caller bytes \
                         the program destroys",
                        slot.index,
                        buffer.label,
                        buffer.owner,
                    );
                }
            }
        }

        // Materialize every buffer by ASSIGNMENT LOOKUP: the buffer backs
        // one tensor, whose carried layout gives the span (elements) and
        // the typed representation (the dtype fact rides the runtime's
        // own RefLayout — width alone cannot pick a variant:
        // bits-of(Int) == bits-of(F32)). Staged caller data is
        // variant-checked against that dtype and length-checked against
        // the span; zeros otherwise. A staged payload of the wrong
        // variant is a loud refusal, never a conversion.
        let mut storage: FxHashMap<BufferId, TypedBuffer> = FxHashMap::default();
        for (id, buffer) in &plan.buffers {
            let numel = buffer
                .layout
                .mirror
                .literal_span_elements()
                .ok_or_else(|| {
                    anyhow!(
                        "buffer {} (backing {}) has no literal span — symbolic \
                     or undisclosed-reach layouts are not executable",
                        buffer.label,
                        buffer.backs
                    )
                })?;
            let dtype = buffer.layout.dtype.ok_or_else(|| {
                anyhow!(
                    "buffer {} (backing {}) carries no dtype fact — cannot \
                     pick a typed representation",
                    buffer.label,
                    buffer.backs
                )
            })?;
            let staged = buffer.lit.and_then(|lit| self.staged.get(&lit));
            if let Some(staged) = staged {
                ensure!(
                    staged.len() == numel,
                    "staged data for {} has {} elements, buffer holds {numel}",
                    buffer.label,
                    staged.len()
                );
            }
            use luminal::dtype::PlanDtype;
            let data = match (dtype, staged) {
                (PlanDtype::F32, Some(TypedBuffer::F32(values))) => {
                    TypedBuffer::F32(values.clone())
                }
                (PlanDtype::F32, None) => TypedBuffer::F32(vec![0.0; numel]),
                // F64 is EXECUTABLE here (ruling 2026-09-02, main #398's
                // f64 unary kernels re-expressed): a double-precision
                // model runs in double precision, never on an F32 bridge.
                (PlanDtype::F64, Some(TypedBuffer::F64(values))) => {
                    TypedBuffer::F64(values.clone())
                }
                (PlanDtype::F64, None) => TypedBuffer::F64(vec![0.0; numel]),
                (PlanDtype::Int, Some(TypedBuffer::I32(values))) => {
                    TypedBuffer::I32(values.clone())
                }
                (PlanDtype::Int, None) => TypedBuffer::I32(vec![0; numel]),
                (PlanDtype::Int64, Some(TypedBuffer::I64(values))) => {
                    TypedBuffer::I64(values.clone())
                }
                (PlanDtype::Int64, None) => TypedBuffer::I64(vec![0; numel]),
                // Narrow ints (ruling 2026-09-02, main #399): stored at
                // their OWN width, never widened to i32 on the way in.
                (PlanDtype::I8, Some(TypedBuffer::I8(values))) => TypedBuffer::I8(values.clone()),
                (PlanDtype::I8, None) => TypedBuffer::I8(vec![0; numel]),
                (PlanDtype::U8, Some(TypedBuffer::U8(values))) => TypedBuffer::U8(values.clone()),
                (PlanDtype::U8, None) => TypedBuffer::U8(vec![0; numel]),
                (PlanDtype::I16, Some(TypedBuffer::I16(values))) => {
                    TypedBuffer::I16(values.clone())
                }
                (PlanDtype::I16, None) => TypedBuffer::I16(vec![0; numel]),
                (PlanDtype::F8E4M3, Some(TypedBuffer::F8E4M3(codes))) => {
                    TypedBuffer::F8E4M3(codes.clone())
                }
                (PlanDtype::F8E4M3, None) => {
                    TypedBuffer::F8E4M3(vec![float8::F8E4M3::from_bits(0); numel])
                }
                // 1-bit logical Bool and byte-code Bool8 both live as
                // Bool8 codes in reference storage; staged codes were
                // validated at the TypedBuffer::bool8 door.
                (PlanDtype::Bool | PlanDtype::Bool8, Some(TypedBuffer::Bool8(codes))) => {
                    TypedBuffer::Bool8(codes.clone())
                }
                (PlanDtype::Bool | PlanDtype::Bool8, None) => TypedBuffer::Bool8(vec![0u8; numel]),
                (expected, Some(staged)) => anyhow::bail!(
                    "buffer {} is {expected:?}; staged {} data is the wrong \
                     type (staging never converts)",
                    buffer.label,
                    staged.type_name()
                ),
                (other, None) => anyhow::bail!(
                    "buffer {} has dtype {other:?}, which the reference \
                     runtime cannot execute (f32, f64, i8, u8, i16, i32, \
                     i64, bool only)",
                    buffer.label
                ),
            };
            storage.insert(id.clone(), data);
        }

        // Inputs that the plan reads MUST have been staged — zeros would be
        // silently wrong numbers, and silence is the one forbidden failure.
        for node in plan.dag.node_weights() {
            if let BufferNode::BufferInput { slots } = node {
                for slot in slots {
                    let buffer = plan
                        .buffers
                        .get(&slot.buffer)
                        .ok_or_else(|| anyhow!("input slot references unknown buffer"))?;
                    let lit = buffer.lit.ok_or_else(|| {
                        anyhow!(
                            "input buffer {} has no BufferLit id to bind by",
                            buffer.label
                        )
                    })?;
                    ensure!(
                        self.staged.contains_key(&lit),
                        "input buffer {} (BufferLit {lit}) was never set_data",
                        buffer.label
                    );
                }
            }
        }

        // Execute in dependency order (anti-edges are real edges, so WAR
        // ordering rides the same toposort).
        let order =
            toposort(&plan.dag, None).map_err(|_| anyhow!("bufferized plan has a cycle"))?;
        for index in order {
            match &plan.dag[index] {
                BufferNode::BufferInput { .. } | BufferNode::BufferOutput { .. } => {}
                BufferNode::BufferCopy { src, dst } => {
                    // THE BUFFERCOPY CONTRACT, executor side (Austin, ruled
                    // 2026-08-31 — see `bufferize::BufferNode::BufferCopy`):
                    //
                    // * The node carries ONLY {src, dst}. Nothing else is
                    //   read here, because nothing else exists.
                    // * Semantics: a DUMB EXACT-SIZE WHOLE-BUFFER copy.
                    //   "If a runtime chooses to do resource reuse and do
                    //   unequal sized buffer that is an entirely runtime
                    //   owned choice" — THIS runtime makes no such choice,
                    //   so it holds itself to exact size and refuses
                    //   otherwise (the length/type check below is this
                    //   executor's own discipline, not a re-check of an
                    //   e-graph premise: copies are bufferizer-authored).
                    // * ORDERING IS THIS RUNTIME'S OBLIGATION. The plan gave
                    //   us dependency structure only (data edges + WAR
                    //   anti-edges); we discharge the obligation by
                    //   executing the toposort of that dag above, which puts
                    //   every dependent op after this copy and every prior
                    //   reader of `dst` before it. A runtime with real
                    //   concurrency would need barriers here; this one is
                    //   sequential, and that IS its scheduling answer.
                    // * The three causes a copy exists (conflict repair,
                    //   boundary placement, lifetime repair) are the
                    //   bufferizer's business; the executor treats all three
                    //   identically — move the bytes.
                    let data = storage
                        .get(src)
                        .ok_or_else(|| anyhow!("copy reads unknown buffer"))?
                        .clone();
                    let dest = storage
                        .get_mut(dst)
                        .ok_or_else(|| anyhow!("copy writes unknown buffer"))?;
                    ensure!(data.len() == dest.len(), "copy length mismatch");
                    ensure!(
                        data.type_name() == dest.type_name(),
                        "copy between {} and {} buffers",
                        data.type_name(),
                        dest.type_name()
                    );
                    *dest = data;
                }
                BufferNode::Compute {
                    op,
                    reads,
                    writes,
                    operand_info,
                    ..
                } => {
                    let mut operands = Vec::with_capacity(reads.len());
                    let mut operand_dims = Vec::with_capacity(reads.len());
                    for (k, id) in reads.iter().enumerate() {
                        operands.push(
                            storage
                                .get(id)
                                .ok_or_else(|| anyhow!("{} reads unknown buffer", op.label()))?
                                .clone(),
                        );
                        // Per-slot VALUE geometry from the slot's own
                        // carried layout — the layout's DOMAIN is the
                        // value's shape, which is all a flat kernel needs.
                        //
                        // WHY NO LAYOUT-SPELLING FENCE HERE. This executor
                        // is layout-agnostic BY CONSISTENCY, not by
                        // assumption: it allocates one span-of-layout
                        // buffer per BACKED TENSOR and both writes and
                        // reads that buffer in the backed tensor's own
                        // element order. Whatever function the e-graph
                        // elected — right-major, left-major, strided — the
                        // producer and every consumer of that same tensor
                        // agree on it, so the numbers are right. Demanding
                        // RightMajor would refuse perfectly consistent
                        // plans (a left-major class with no right-major
                        // spelling is a legal election, not an error).
                        //
                        // THE REAL HAZARD is reading someone ELSE's bytes
                        // through a DIFFERENT function — a FOLDED operand.
                        // The plan does not (and should not) label folds:
                        // a folded view and an in-place cohabitant are
                        // both just "this value lives in this buffer". The
                        // difference is entirely in the LAYOUT, so that is
                        // the test: the operand's carried `L` must be the
                        // one the buffer was allocated for. A DPS
                        // in-place cohabitant passes by construction (the
                        // poison destination clones its tied result's
                        // layout); a view does not (its composed layout is
                        // a different function over the same bytes) and is
                        // a LOUD capability refusal — this executor lowers
                        // no composed read path.
                        //
                        // Layout EQUALITY is a runtime-side operation on
                        // the runtime's OWN type. Core never compares
                        // layouts (`PlanLayout` has no `PartialEq`).
                        let slot = operand_info.get(k).ok_or_else(|| {
                            anyhow!("{} operand {k} lacks its slot descriptor", op.label())
                        })?;
                        if op.operand_reads_memory(k) && slot.layout != plan.buffers[id].layout {
                            anyhow::bail!(
                                "{} operand {k} READS value {} through a layout that is not \
                                 the one buffer {} was allocated for — a folded read this \
                                 executor does not lower (it reads each buffer in one \
                                 element order). Fail-closed, never a silent flat misread.",
                                op.label(),
                                slot.value,
                                plan.buffers[id].label,
                            );
                        }
                        let dims = slot.layout.mirror.literal_extents().ok_or_else(|| {
                            anyhow!("{} operand {k} has symbolic extents", op.label())
                        })?;
                        operand_dims.push(dims);
                    }
                    let mut dests = Vec::with_capacity(writes.len());
                    for id in writes {
                        let existing = storage
                            .get(id)
                            .ok_or_else(|| anyhow!("{} writes unknown buffer", op.label()))?;
                        dests.push(existing.zeroed_like());
                    }
                    let mut ctx = ReferenceKernelCtx {
                        operands,
                        operand_dims,
                        dests,
                    };
                    match crate::kernels::kernel_for(op.as_ref()) {
                        Some(kernel) => (kernel.execute)(op.as_ref(), &mut ctx)
                            .with_context(|| format!("executing {}", op.label()))?,
                        None => anyhow::bail!("no reference kernel for {}", op.label()),
                    }
                    for (id, data) in writes.iter().zip(ctx.dests) {
                        *storage.get_mut(id).expect("write buffer exists") = data;
                    }
                }
            }
        }

        self.storage = storage;
        Ok(())
    }

    /// The f32 contents of an OUTPUT tensor's buffer. Loud if the tensor
    /// is not a bound output, and loud on a boolean buffer — use
    /// [`Self::get_bool8`] for those. Returns a borrow: reads never
    /// mutate or consume runtime state.
    pub fn get_f32(&self, tensor: petgraph::graph::NodeIndex) -> Result<&Vec<f32>> {
        self.get_typed(self.output_buffer(tensor)?)?.as_f32()
    }

    /// The Bool8 codes of an OUTPUT tensor's buffer (each element exactly
    /// 0 or 1 — the two legal codes).
    pub fn get_bool8(&self, tensor: petgraph::graph::NodeIndex) -> Result<&Vec<u8>> {
        self.get_typed(self.output_buffer(tensor)?)?.as_bool8()
    }

    /// The f64 twin of [`Self::get_f32`] — native double-precision
    /// output readback (ruling 2026-09-02: F64 executes, so it also
    /// reads back as f64 and never through an f32 narrowing).
    pub fn get_f64(&self, tensor: petgraph::graph::NodeIndex) -> Result<&Vec<f64>> {
        self.get_typed(self.output_buffer(tensor)?)?.as_f64()
    }

    /// The i32 twin of [`Self::get_f32`] — native Int output readback
    /// (typed buffers 2026-08-11; Int results no longer need an
    /// observe-only cast to F32).
    pub fn get_i32(&self, tensor: petgraph::graph::NodeIndex) -> Result<&Vec<i32>> {
        self.get_typed(self.output_buffer(tensor)?)?.as_i32()
    }

    /// The i64 twin of [`Self::get_f32`].
    pub fn get_i64(&self, tensor: petgraph::graph::NodeIndex) -> Result<&Vec<i64>> {
        self.get_typed(self.output_buffer(tensor)?)?.as_i64()
    }

    /// The narrow-integer readers (ruling 2026-09-02, main #399's
    /// `get_output_i8` / `get_output_u8` / `get_output_i16`). STRICTLY
    /// NON-WIDENING, which is main's whole point and this branch's
    /// typed-readback contract both: an I8 output reads back as `i8`,
    /// and asking for it as `i32` refuses by name rather than quietly
    /// promoting.
    pub fn get_i8(&self, tensor: petgraph::graph::NodeIndex) -> Result<&Vec<i8>> {
        self.get_typed(self.output_buffer(tensor)?)?.as_i8()
    }

    /// The u8 twin of [`Self::get_i8`]. Distinct from
    /// [`Self::get_bool8`]: same storage width, different dtype, and a
    /// U8 buffer has no two-legal-codes invariant.
    pub fn get_u8(&self, tensor: petgraph::graph::NodeIndex) -> Result<&Vec<u8>> {
        self.get_typed(self.output_buffer(tensor)?)?.as_u8()
    }

    /// The i16 twin of [`Self::get_i8`].
    pub fn get_i16(&self, tensor: petgraph::graph::NodeIndex) -> Result<&Vec<i16>> {
        self.get_typed(self.output_buffer(tensor)?)?.as_i16()
    }

    /// Buffer-id read for search internals.
    pub fn get_f32_buffer(&self, buffer: i64) -> Result<&Vec<f32>> {
        self.get_typed(buffer)?.as_f32()
    }

    fn output_buffer(&self, tensor: petgraph::graph::NodeIndex) -> Result<i64> {
        self.output_buffers
            .get(&tensor)
            .copied()
            .ok_or_else(|| anyhow!("tensor {tensor:?} is not a bound output of this program"))
    }

    /// The escape-and-disclose fetch (ruling 2026-08-27), universal over
    /// elections: output slot `index`'s BACKING buffer contents plus its
    /// [`OutputBinding`] — the elected layout the caller interprets those
    /// bytes under. A dense election returns the slot's boundary buffer
    /// and that tensor's own layout; a VIEW election returns the escaped
    /// backing buffer (possibly parent-sized) and the view's COMPOSED
    /// layout — zero-copy by construction, completely legal, never a
    /// refusal (correction 5).
    ///
    /// The layout is TOTAL on the binding, so there is nothing to test
    /// for presence. Element readback through the returned (buffer,
    /// layout) pair is a TEST concern and lives in the testing crate
    /// (`test_runtime::test_equality`); core's ex-`walk_layout_index`
    /// died with the hop machinery. This runtime's own searches are
    /// materialize-only (they never elect views), so its outputs are
    /// de-facto dense and `get_f32` semantics are unchanged; view-elected
    /// slots arrive only via externally loaded plans.
    pub fn output_slot(
        &self,
        index: usize,
    ) -> Result<(&TypedBuffer, &OutputBinding<crate::layouts::RefLayout>)> {
        let binding = self.output_layout(index)?;
        let data = self
            .storage
            .get(&binding.buffer)
            .ok_or_else(|| anyhow!("output slot {index} has no contents (execute first)"))?;
        Ok((data, binding))
    }

    /// Output slot `index`'s binding — buffer identity plus the elected
    /// layout (see [`Self::output_slot`]).
    pub fn output_layout(&self, index: usize) -> Result<&OutputBinding<crate::layouts::RefLayout>> {
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| anyhow!("no plan loaded"))?;
        for node in plan.dag.node_weights() {
            if let BufferNode::BufferOutput { slots } = node {
                if let Some(slot) = slots.iter().find(|slot| slot.index == index) {
                    return Ok(slot);
                }
            }
        }
        Err(anyhow!("no output slot {index} in the loaded plan"))
    }

    fn get_typed(&self, id: i64) -> Result<&TypedBuffer> {
        let buffer = self
            .lit_index
            .get(&id)
            .ok_or_else(|| anyhow!("no boundary buffer with BufferLit {id}"))?;
        self.storage
            .get(buffer)
            .ok_or_else(|| anyhow!("buffer for BufferLit {id} has no contents (execute first)"))
    }
}

#[cfg(test)]
mod tests {
    use crate::harness::run_reference;
    use crate::ReferenceRuntime;
    use luminal::buffer_tensor_ir::TypedBuffer;
    use luminal::dtype::DType;
    use luminal::graph::Graph;
    use rustc_hash::FxHashMap;

    /// The allow list is DERIVED from the kernel registry — which is
    /// itself derived from the op rows (runtime split, PR #425), so
    /// "registered" and "executable" cannot drift apart by construction.
    /// This pins the RESOLVED claim set, so a registry edit that silently
    /// grows or shrinks the runtime's claims is still loud: the two
    /// derivations agreeing does not tell you they agree on the right
    /// list. (Div/Exp are claimed because their kernels exist — the
    /// 2026-08-06 relocation closed the over-claim the old hardcoded
    /// filter hid.)
    #[test]
    fn allow_list_matches_the_kernel_registry() {
        let mut allow = crate::reference_allow_list();
        allow.sort_unstable();
        let expected = vec![
            "LayoutTensorOpAddFunctionalGeneric",
            "LayoutTensorOpCastGeneric",
            "LayoutTensorOpConstantGeneric",
            // No CopyGeneric: `materialize_layout_copy` left this runtime
            // with the split (PR #425). Its kernel only ever copied under
            // IDENTICAL geometry — it asserted that rather than assuming
            // it — so a runtime moving toward canonical-layout-only has no
            // layout copy left to make. The op lives on the TestRuntime,
            // which reasons about plans instead of executing them.
            "LayoutTensorOpDivFunctionalGeneric",
            "LayoutTensorOpExp2FunctionalGeneric",
            "LayoutTensorOpExpFunctionalGeneric",
            "LayoutTensorOpGatherGeneric",
            "LayoutTensorOpIndexMapApplyMaterialize",
            "LayoutTensorOpIotaGeneric",
            "LayoutTensorOpLessThanGeneric",
            "LayoutTensorOpLog2FunctionalGeneric",
            "LayoutTensorOpModFunctionalGeneric",
            "LayoutTensorOpMulFunctionalGeneric",
            "LayoutTensorOpRecipFunctionalGeneric",
            "LayoutTensorOpReduceMaxGeneric",
            "LayoutTensorOpReduceSumGeneric",
            "LayoutTensorOpScatterFunctionalGeneric",
            "LayoutTensorOpSinFunctionalGeneric",
            "LayoutTensorOpSqrtFunctionalGeneric",
            "LayoutTensorOpTruncDivFunctionalGeneric",
            "LayoutTensorOpTruncRemFunctionalGeneric",
        ];
        assert_eq!(allow, expected, "derived allow list drifted");
        // Registry coherence: no two rows claim one concrete type, and no
        // Mutating/View label carries a kernel.
        let table = crate::kernels::reference_kernels();
        let mut seen = std::collections::HashSet::new();
        for kernel in table {
            assert!(
                seen.insert(kernel.op_type),
                "duplicate registry row for {}",
                kernel.label
            );
            assert!(
                !kernel.label.contains("Mutating") && !kernel.label.contains("View"),
                "the reference runtime is out-of-place and view-free; {} cannot have a kernel",
                kernel.label
            );
        }
    }

    fn assert_close(ours: &[f32], theirs: &[f32]) {
        assert_eq!(ours.len(), theirs.len(), "length mismatch");
        for (index, (a, b)) in ours.iter().zip(theirs).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                "element {index}: ours {a} vs theirs {b}"
            );
        }
    }

    /// THE DIFFERENTIAL: their `simple`-test graph (a = b*c + g and
    /// d = sin(b*c / e)) through BOTH pipelines — their egglog search +
    /// ReferenceRuntime vs our translation + saturation + extraction +
    /// bufferization + ReferenceRuntime — must agree numerically.
    #[test]
    fn differential_simple_elementwise_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let b = cx.tensor(3);
            let c = cx.tensor(3);
            let g = cx.tensor(3);
            let e = cx.tensor(3);
            let a = (b * c + g).output();
            let d = (b * c / e).sin().output();
            (cx, b, c, g, e, a, d)
        };
        let b_data = vec![1.0, 2.0, 3.0];
        let c_data = vec![4.0, 5.0, 6.0];
        let g_data = vec![0.5, -1.5, 2.5];
        let e_data = vec![2.0, 4.0, 8.0];

        // GOLDEN (pinned from their ReferenceRuntime before its deletion).
        let expected_a = vec![4.5, 8.5, 20.5];
        let expected_d = vec![0.9092974, 0.5984721, 0.7780732];
        let (cx2, b2, c2, g2, e2, a2, d2) = build();
        let ours = run_reference(
            &cx2,
            &[
                (b2.id, b_data.into()),
                (c2.id, c_data.into()),
                (g2.id, g_data.into()),
                (e2.id, e_data.into()),
            ],
        );
        assert_close(ours.get_f32(a2.id).unwrap(), &expected_a);
        assert_close(ours.get_f32(d2.id).unwrap(), &expected_d);
    }

    /// Slice-2 differential: a permuted operand (transpose view) through
    /// both pipelines.
    #[test]
    fn differential_permuted_mul_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor((2, 3));
            let y = cx.tensor((3, 2));
            let out = (x.permute((1, 0)) * y).output();
            (cx, x, y, out)
        };
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected = vec![10.0, 80.0, 60.0, 200.0, 150.0, 360.0];
        let (cx2, x2, y2, out2) = build();
        let ours = run_reference(&cx2, &[(x2.id, x_data.into()), (y2.id, y_data.into())]);
        assert_close(ours.get_f32(out2.id).unwrap(), &expected);
    }

    /// Slice-2 differential: subtraction routes through their Neg (a
    /// broadcast constant) — rank-0 LogicalConstant + lifted broadcast view.
    #[test]
    fn differential_subtraction_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor(4);
            let y = cx.tensor(4);
            let out = (x - y).output();
            (cx, x, y, out)
        };
        let x_data = vec![10.0, 20.0, 30.0, 40.0];
        let y_data = vec![1.0, 2.0, 3.0, 4.0];

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected = vec![9.0, 18.0, 27.0, 36.0];
        let (cx2, x2, y2, out2) = build();
        let ours = run_reference(&cx2, &[(x2.id, x_data.into()), (y2.id, y_data.into())]);
        assert_close(ours.get_f32(out2.id).unwrap(), &expected);
    }

    /// THE MATMUL DIFFERENTIAL: their fully-decomposed frontend matmul
    /// (movement views + Mul + SumReduce) through our whole pipeline —
    /// slice-2 lifting translating their expand/permute stride patterns.
    #[test]
    fn differential_matmul_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let a = cx.tensor((2, 3));
            let b = cx.tensor((3, 4));
            let c = a.matmul(b).output();
            (cx, a, b, c)
        };
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = (1..=12).map(|v| v as f32).collect();

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected = vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0];
        let (cx2, a2, b2, c2) = build();
        let ours = run_reference(&cx2, &[(a2.id, a_data.into()), (b2.id, b_data.into())]);
        assert_close(ours.get_f32(c2.id).unwrap(), &expected);
    }

    /// DYNAMIC DIMS over the bounds interface: the model declares
    /// `(IntVar "a")`, the binding seeds tight bounds from set_dim, the
    /// [n,n] collapse delivers the literal to the geometry walk — and the
    /// SAME symbolic graph shape re-renders per pin (the per-bucket model).
    #[test]
    fn differential_dynamic_dim_against_reference_runtime() {
        for pin in [3usize, 5usize] {
            let build = |dim: usize| {
                let mut cx = Graph::new();
                cx.set_dim('a', dim);
                let x = cx.tensor(('a', 2));
                let y = cx.tensor(('a', 2));
                let out = (x * y).output();
                (cx, x, y, out)
            };
            let data_x: Vec<f32> = (0..pin * 2).map(|v| v as f32 + 1.0).collect();
            let data_y: Vec<f32> = (0..pin * 2).map(|v| (v as f32) * 0.5 - 1.0).collect();

            // GOLDEN per pin (pinned from their ReferenceRuntime — Step 4b).
            let expected = match pin {
                3 => vec![-1.0, -1.0, 0.0, 2.0, 5.0, 9.0],
                5 => vec![-1.0, -1.0, 0.0, 2.0, 5.0, 9.0, 14.0, 20.0, 27.0, 35.0],
                _ => unreachable!(),
            };
            let (cx2, x2, y2, out2) = build(pin);
            let program = cx2
                .logical
                .bound_program(&crate::bindings::ReferenceBindings)
                .expect("native program");
            assert!(
                program.text.contains("(IntVar \"a\")"),
                "the model must stay symbolic:\n{}",
                program.text
            );
            // The pin arrives as BINDING seeds, not model content: run_reference
            // injects (bigint {pin}) bounds from the graph's dyn_map — the
            // execution below at both pins is the proof.
            let ours = run_reference(&cx2, &[(x2.id, data_x.into()), (y2.id, data_y.into())]);
            assert_close(ours.get_f32(out2.id).unwrap(), &expected);
        }
    }

    /// SLICE differential: their nonzero-start slice lowers to
    /// iota(z + start) + flat gather — the general-iota expression walker,
    /// the coordinate-form gather bridge (rank-1 data), and both kernels,
    /// against their runtime.
    #[test]
    fn differential_slice_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor(8);
            let out = (x.slice(2..6) + x.slice(1..5)).output();
            (cx, x, out)
        };
        let x_data: Vec<f32> = (0..8).map(|v| (v * v) as f32).collect();

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected = vec![5.0, 13.0, 25.0, 41.0];
        let (cx2, x2, out2) = build();
        let program = cx2
            .logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .expect("native program");
        assert!(
            program.text.contains("LogicalIndexMapApply"),
            "the slice must arrive as a view:\n{}",
            program.text
        );
        let ours = run_reference(&cx2, &[(x2.id, x_data.into())]);
        assert_close(ours.get_f32(out2.id).unwrap(), &expected);
    }

    /// THE SEAM PAYOFF: a 2-D nonzero-start slice arrives structure-intact
    /// (SliceView) and translates as the view it is — while THEIR side of
    /// this same test runs the SliceView's legacy iota+gather lowering, so
    /// this differential proves BOTH halves of the seam at once.
    #[test]
    fn differential_two_dim_slice_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor((4, 5));
            let out = x.slice((1..3, 2..5)).output();
            (cx, x, out)
        };
        let x_data: Vec<f32> = (0..20).map(|v| v as f32 * 1.5).collect();

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected = vec![10.5, 12.0, 13.5, 18.0, 19.5, 21.0];
        let (cx2, x2, out2) = build();
        let program = cx2
            .logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .expect("native program");
        assert!(
            program.text.contains("LogicalIndexMapApply"),
            "the slice must arrive as a view:\n{}",
            program.text
        );
        let ours = run_reference(&cx2, &[(x2.id, x_data.into())]);
        assert_close(ours.get_f32(out2.id).unwrap(), &expected);
    }

    /// UNFOLD differential through the seam: sliding windows (with a
    /// dilated variant) arrive structure-intact as UnfoldView and translate
    /// as two-coordinate affine view entries; their side runs the legacy
    /// flat iota+gather lowering.
    #[test]
    fn differential_unfold_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor(8);
            let plain = x.unfold(3, 2, 1).output(); // windows at 0,2,4
            let y = cx.tensor(10);
            let dilated = y.unfold(3, 2, 2).output(); // effective window 5
            (cx, x, y, plain, dilated)
        };
        let x_data: Vec<f32> = (0..8).map(|v| (v * v) as f32).collect();
        let y_data: Vec<f32> = (0..10).map(|v| v as f32 * 3.0 - 5.0).collect();

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected_plain = vec![0.0, 1.0, 4.0, 4.0, 9.0, 16.0, 16.0, 25.0, 36.0];
        let expected_dilated = vec![-5.0, 1.0, 7.0, 1.0, 7.0, 13.0, 7.0, 13.0, 19.0];
        let (cx2, x2, y2, plain2, dilated2) = build();
        let program = cx2
            .logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .expect("native program");
        assert!(
            program.text.contains("LogicalIndexMapApply"),
            "unfold must arrive as a view:\n{}",
            program.text
        );
        let ours = run_reference(&cx2, &[(x2.id, x_data.into()), (y2.id, y_data.into())]);
        assert_close(ours.get_f32(plain2.id).unwrap(), &expected_plain);
        assert_close(ours.get_f32(dilated2.id).unwrap(), &expected_dilated);
    }

    /// PAD differential, 1-D, THROUGH THE BOOL BRIDGE with zero frontend
    /// changes: their clamp iota (Min/Max terms) + mask iota (Gte/Lt as
    /// indicator values) + cast + blend translate directly — comparisons
    /// become IntCastFromBool(BoolLessThanInt ...) indicators, decided
    /// masks collapse via bounds, undecided ones evaluate in the kernels.
    /// Zero fill and nonzero fill both compared against their runtime.
    #[test]
    fn differential_pad_against_reference_runtime() {
        for fill in [0.0f32, 2.5f32] {
            let build = |fill: f32| {
                let mut cx = Graph::new();
                let x = cx.tensor(4);
                let out = x.pad((1, 2), fill).output();
                (cx, x, out)
            };
            let x_data = vec![10.0, 20.0, 30.0, 40.0];

            // GOLDEN per fill (pinned from their ReferenceRuntime — Step 4b).
            let expected = if fill == 0.0 {
                vec![0.0, 10.0, 20.0, 30.0, 40.0, 0.0, 0.0]
            } else {
                vec![2.5, 10.0, 20.0, 30.0, 40.0, 2.5, 2.5]
            };
            let (cx2, x2, out2) = build(fill);
            let program = cx2
                .logical
                .bound_program(&crate::bindings::ReferenceBindings)
                .expect("native program");
            assert!(
                program.text.contains("IntCastFromBool"),
                "the mask must ride the bool bridge:\n{}",
                program.text
            );
            let ours = run_reference(&cx2, &[(x2.id, x_data.into())]);
            assert_close(ours.get_f32(out2.id).unwrap(), &expected);
        }
    }

    /// RANK-2 PAD differential through the seam nodes: asymmetric padding
    /// on both axes (one axis before-only, one both sides), both fills —
    /// the case the flat lowering made untranslatable. Their side runs the
    /// legacy lowerings out of the seam nodes' to_egglog.
    #[test]
    fn differential_rank2_pad_against_reference_runtime() {
        for fill in [0.0f32, -1.5f32] {
            let build = |fill: f32| {
                let mut cx = Graph::new();
                let x = cx.tensor((3, 4));
                let out = x.pad(((1, 0), (2, 1)), fill).output();
                (cx, x, out)
            };
            let x_data: Vec<f32> = (0..12).map(|v| v as f32 + 1.0).collect();

            // GOLDEN per fill (pinned from their ReferenceRuntime — Step 4b).
            let expected = if fill == 0.0 {
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0,
                    5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 9.0, 10.0, 11.0, 12.0, 0.0,
                ]
            } else {
                vec![
                    -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 1.0, 2.0, 3.0, 4.0, -1.5,
                    -1.5, -1.5, 5.0, 6.0, 7.0, 8.0, -1.5, -1.5, -1.5, 9.0, 10.0, 11.0, 12.0, -1.5,
                ]
            };
            let (cx2, x2, out2) = build(fill);
            let program = cx2
                .logical
                .bound_program(&crate::bindings::ReferenceBindings)
                .expect("native program");
            assert!(
                program.text.contains("IntMax") && program.text.contains("IntCastFromBool"),
                "clamp view + indicator mask expected:\n{}",
                program.text
            );
            let ours = run_reference(&cx2, &[(x2.id, x_data.into())]);
            assert_close(ours.get_f32(out2.id).unwrap(), &expected);
        }
    }

    /// COORDINATE-FORM GATHER differential (ruling 2026-07-31): the
    /// primary gather — one Int coordinate tensor per data axis — records
    /// LogicalGather directly (rank-N native, no flatten trick); their
    /// runtime executes the transitional flat-index HLIR lowering as the
    /// oracle.
    #[test]
    fn differential_native_coordinate_gather() {
        let build = || {
            let mut cx = Graph::new();
            let data = cx.tensor((3, 4));
            let row = cx.tensor_dtyped((2, 3), luminal::dtype::DType::Int);
            let col = cx.tensor_dtyped((2, 3), luminal::dtype::DType::Int);
            let out = data.gather(&[row, col]).output();
            (cx, data, row, col, out)
        };
        let data_vals: Vec<f32> = (0..12).map(|v| v as f32 * 1.5 + 1.0).collect();
        let row_ints = [0i32, 2, 1, 2, 0, 1];
        let col_ints = [3i32, 0, 2, 3, 1, 0];
        let row_vals: Vec<i32> = row_ints.to_vec();
        let col_vals: Vec<i32> = col_ints.to_vec();

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected = vec![5.5, 13.0, 10.0, 17.5, 2.5, 7.0];
        let (cx2, data2, row2, col2, out2) = build();
        let program = cx2
            .logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .expect("native program");
        assert!(
            program.text.contains("(LogicalGather v"),
            "coordinate-form gather expected in the model:\n{}",
            program.text
        );
        let ours = run_reference(
            &cx2,
            &[
                (data2.id, data_vals.into()),
                (row2.id, row_vals.into()),
                (col2.id, col_vals.into()),
            ],
        );
        assert_close(ours.get_f32(out2.id).unwrap(), &expected);
    }

    /// COORDINATE-FORM SCATTER differential: dest updated at (row, col)
    /// coordinate positions with src — value semantics, against their
    /// flat-Scatter lowering.
    #[test]
    fn differential_native_coordinate_scatter() {
        let build = || {
            let mut cx = Graph::new();
            let dest = cx.tensor((3, 4));
            let row = cx.tensor_dtyped(4, luminal::dtype::DType::Int);
            let col = cx.tensor_dtyped(4, luminal::dtype::DType::Int);
            let src = cx.tensor(4);
            let out = dest.scatter(&[row, col], src).output();
            (cx, dest, row, col, src, out)
        };
        let dest_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let row_ints = [0i32, 1, 2, 1];
        let col_ints = [1i32, 3, 0, 0];
        let row_vals: Vec<i32> = row_ints.to_vec();
        let col_vals: Vec<i32> = col_ints.to_vec();
        let src_vals = vec![100.0, 200.0, 300.0, 400.0];

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected = vec![
            0.0, 100.0, 2.0, 3.0, 400.0, 5.0, 6.0, 200.0, 300.0, 9.0, 10.0, 11.0,
        ];
        let (cx2, dest2, row2, col2, src2, out2) = build();
        let program = cx2
            .logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .expect("native program");
        assert!(
            program.text.contains("(LogicalScatter v"),
            "coordinate-form scatter expected in the model:\n{}",
            program.text
        );
        let ours = run_reference(
            &cx2,
            &[
                (dest2.id, dest_vals.into()),
                (row2.id, row_vals.into()),
                (col2.id, col_vals.into()),
                (src2.id, src_vals.into()),
            ],
        );
        assert_close(ours.get_f32(out2.id).unwrap(), &expected);
    }

    /// FLAT gather1d sugar (B-tail 2026-08-06): out[c] = data.flat[idx[c]]
    /// — delegates to flatten + coordinate gather, records natively.
    #[test]
    fn differential_flat_gather1d_sugar() {
        let mut cx = Graph::new();
        let data = cx.tensor((3, 4));
        let idx = cx.tensor_dtyped((2, 3), luminal::dtype::DType::Int);
        let out = data.gather1d(idx).output();
        assert_eq!(out.dims(), idx.dims(), "out shape = index shape");

        let data_vals: Vec<f32> = (0..12).map(|v| v as f32 * 1.5 + 1.0).collect();
        let idx_ints = [0i32, 5, 11, 7, 3, 2];
        let idx_vals: Vec<i32> = idx_ints.to_vec();
        // Hand golden: data.flat[i] = i*1.5 + 1.
        let expected = vec![1.0, 8.5, 17.5, 11.5, 5.5, 4.0];

        let program = cx
            .logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .expect("native program");
        assert!(
            program.text.contains("(LogicalGather v"),
            "flat sugar lowers to coordinate gather:\n{}",
            program.text
        );
        let ours = run_reference(
            &cx,
            &[(data.id, data_vals.into()), (idx.id, idx_vals.into())],
        );
        assert_close(ours.get_f32(out.id).unwrap(), &expected);
    }

    /// FLAT scatter1d sugar (B-tail 2026-08-06): copy dest, write src at
    /// flat positions, rebuild dest's shape with recorded splits.
    #[test]
    fn differential_flat_scatter1d_sugar() {
        let mut cx = Graph::new();
        let dest = cx.tensor((2, 6));
        let idx = cx.tensor_dtyped(4, luminal::dtype::DType::Int);
        let src = cx.tensor(4);
        let out = src.scatter1d(idx, dest).output();
        assert_eq!(out.dims(), dest.dims(), "out shape = dest shape");

        let dest_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let idx_vals = vec![3i32, 0, 11, 6];
        let src_vals = vec![100.0f32, 200.0, 300.0, 400.0];
        let expected = vec![
            200.0, 1.0, 2.0, 100.0, 4.0, 5.0, 400.0, 7.0, 8.0, 9.0, 10.0, 300.0,
        ];

        let program = cx
            .logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .expect("native program");
        assert!(
            program.text.contains("(LogicalScatter v"),
            "flat sugar lowers to coordinate scatter:\n{}",
            program.text
        );
        let ours = run_reference(
            &cx,
            &[
                (dest.id, dest_vals.into()),
                (idx.id, idx_vals.into()),
                (src.id, src_vals.into()),
            ],
        );
        assert_close(ours.get_f32(out.id).unwrap(), &expected);
    }

    /// NATIVE RANK-N IOTA (Design A, 2026-08-06): a multi-dim iota
    /// records as ONE LogicalIota over its true shape — a per-coordinate
    /// function, no flat-then-reshape view detour in the model.
    #[test]
    fn differential_rank2_iota_records_natively() {
        let mut cx = Graph::new();
        // out[r, c] = (r·3 + c)·2 over (2, 3) — read back NATIVE i32
        // (the observe-only cast to F32 died with typed buffers).
        let out = cx.iota((2, 3), |c| (c[0] * 3 + c[1]) * 2).output();

        let program = cx
            .logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .expect("native program");
        assert!(
            program.text.contains("(LogicalIota"),
            "iota expected in the model:\n{}",
            program.text
        );
        assert!(
            !program.text.contains("LogicalIndexMapApply"),
            "no view detour for a bare multi-dim iota:\n{}",
            program.text
        );
        let expected = vec![0i32, 2, 4, 6, 8, 10];
        let ours = run_reference(&cx, &[]);
        assert_eq!(ours.get_i32(out.id).unwrap(), &expected);
    }

    /// DYNAMIC-EXTENT ARANGE: a symbolic iota total previously collapsed
    /// silently to shape/coordinate literals of 0 (the
    /// `to_usize().unwrap_or(0)` hole); the shape now renders as IntVar
    /// and the dyn pin delivers the extent at binding time.
    #[test]
    fn differential_dynamic_arange() {
        let mut cx = Graph::new();
        cx.set_dim('a', 5);
        let out = cx.arange(luminal::shape::IntExpr::from('a')).output();

        let expected = vec![0i32, 1, 2, 3, 4];
        let ours = run_reference(&cx, &[]);
        assert_eq!(ours.get_i32(out.id).unwrap(), &expected);
    }

    /// A symbolic var inside the iota EXPRESSION (not just the shape)
    /// records as IntVar and resolves at binding time (the R3 fix,
    /// 2026-08-06 — previously any dyn var in an iota expression
    /// poisoned; this is the paged-attention `z + prev_seq` shape).
    #[test]
    fn differential_dynamic_offset_iota() {
        let mut cx = Graph::new();
        cx.set_dim('a', 10);
        let out = cx.iota(3, |c| c[0] + 'a').output();

        let expected = vec![10i32, 11, 12];
        let ours = run_reference(&cx, &[]);
        assert_eq!(ours.get_i32(out.id).unwrap(), &expected);
    }

    /// ONNX GatherElements over axis 1 (rides the flat sugar + the
    /// iota/normalization index arithmetic), including a negative index.
    #[test]
    fn differential_gather_elements_axis1() {
        let mut cx = Graph::new();
        let data = cx.tensor((2, 3));
        let idx = cx.tensor_dtyped((2, 2), luminal::dtype::DType::Int);
        let out = data.gather_elements(idx, 1).output();

        let data_vals: Vec<f32> = (0..6).map(|v| v as f32 * 10.0).collect();
        // out[i, j] = data[i, idx[i, j]]; -1 normalizes to axis extent - 1 = 2.
        let idx_vals = vec![2i32, 0, -1, 1];
        let expected = vec![20.0, 0.0, 50.0, 40.0];

        cx.logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .expect("native program");
        // Landing D: plain Int assembly is proof-gated — the caller
        // ATTESTS the index range (gather semantics require it in
        // [-d, d) anyway; the attestation states the contract).
        let ours = crate::harness::run_reference_with_ranges(
            &cx,
            &[(data.id, data_vals.into()), (idx.id, idx_vals.into())],
            &[(idx.id, -3, 2)],
        );
        assert_close(ours.get_f32(out.id).unwrap(), &expected);
    }

    /// ONNX ScatterElements over axis 0: updates land at per-element row
    /// targets; everywhere else the data copies through.
    #[test]
    fn differential_scatter_elements_axis0() {
        let mut cx = Graph::new();
        let data = cx.tensor((3, 2));
        let idx = cx.tensor_dtyped((1, 2), luminal::dtype::DType::Int);
        let upd = cx.tensor((1, 2));
        let out = data.scatter_elements(idx, upd, 0).output();
        assert_eq!(out.dims(), data.dims());

        let data_vals: Vec<f32> = (0..6).map(|v| v as f32).collect();
        // out[idx[0, j], j] = upd[0, j]: column 0 → row 2, column 1 → row 0.
        let idx_vals = vec![2i32, 0];
        let upd_vals = vec![100.0f32, 200.0];
        let expected = vec![0.0, 200.0, 2.0, 3.0, 100.0, 5.0];

        cx.logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .expect("native program");
        let ours = crate::harness::run_reference_with_ranges(
            &cx,
            &[
                (data.id, data_vals.into()),
                (idx.id, idx_vals.into()),
                (upd.id, upd_vals.into()),
            ],
            &[(idx.id, -3, 2)],
        );
        assert_close(ours.get_f32(out.id).unwrap(), &expected);
    }

    /// ONNX ScatterND, K=1 row scatter: indices (2,1) select rows of a
    /// (3,2) data tensor, updates (2,2) replace them wholesale.
    #[test]
    fn differential_scatter_nd_row_case() {
        let mut cx = Graph::new();
        let data = cx.tensor((3, 2));
        let idx = cx.tensor_dtyped((2, 1), luminal::dtype::DType::Int);
        let upd = cx.tensor((2, 2));
        let out = data.scatter_nd(idx, upd).output();
        assert_eq!(out.dims(), data.dims());

        let data_vals: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let idx_vals = vec![2i32, 0];
        let upd_vals = vec![100.0f32, 101.0, 200.0, 201.0];
        let expected = vec![200.0, 201.0, 2.0, 3.0, 100.0, 101.0];

        cx.logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .expect("native program");
        let ours = crate::harness::run_reference_with_ranges(
            &cx,
            &[
                (data.id, data_vals.into()),
                (idx.id, idx_vals.into()),
                (upd.id, upd_vals.into()),
            ],
            &[(idx.id, 0, 2)],
        );
        assert_close(ours.get_f32(out.id).unwrap(), &expected);
    }

    /// CONFLICTING SCATTER WRITES are a deterministic runtime panic in the
    /// checked functional reference kernel (ruling 2026-08-06): duplicate
    /// flat targets refuse loudly instead of silently picking a winner.
    #[test]
    fn scatter_conflicting_writes_panic() {
        let mut cx = Graph::new();
        let dest = cx.tensor(6);
        let idx = cx.tensor_dtyped(3, luminal::dtype::DType::Int);
        let src = cx.tensor(3);
        let out = src.scatter1d(idx, dest).output();

        let (pre, input_slots, output_slots, post, _labeled) = cx
            .logical
            .bound_parts(&crate::bindings::ReferenceBindings)
            .expect("recorder clean");
        let program = luminal::graph::LogicalProgram {
            text: format!(
                "{pre}{}{post}",
                crate::bindings::ReferenceBindings::SCHEDULE
            ),
            input_slots,
            output_slots,
        };
        let text = format!("{}\n\n{}", crate::assembled_program(), program.text);
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &text)
            .expect("program runs");
        let serialized = egraph.serialize(egglog::SerializeConfig::default()).egraph;
        let allow = crate::reference_allow_list();
        let extracted = luminal::extractor::extract_layout_ir_with_ops_and_matchers(
            &serialized,
            Some(&allow),
            crate::ops::built_in_matchers(),
        )
        .expect("extracts")
        .expect("plan");
        let dps = luminal::dps::dps_rewrite(&extracted);
        let layouts = luminal::extractor::decoded_layout_table(
            &serialized,
            &dps,
            &crate::layouts::ReferenceLayoutDecoder,
            &mut std::collections::HashMap::new(),
        )
        .expect("layouts decode");
        let plan = luminal::bufferize::bufferize(&dps, &layouts).expect("bufferizes");
        let mut rt = crate::ReferenceRuntime::default();
        rt.stage_slots(&program.input_slots, &program.output_slots);
        rt.load_plan(plan);
        rt.set_data(dest.id, (0..6).map(|v| v as f32).collect::<Vec<f32>>());
        rt.set_data(idx.id, vec![1i32, 4, 1]); // 1 appears twice — conflict
        rt.set_data(src.id, vec![10.0, 20.0, 30.0]);
        let err = rt.execute().expect_err("duplicate targets must refuse");
        assert!(
            format!("{err:#}").contains("conflicting scatter writes"),
            "attributable conflict message, got: {err:#}"
        );

        let _ = out;
    }

    /// The poison MECHANISM: a poisoned recorder refuses the native path
    /// loudly with its attributable reason — never mistranslates. (As of
    /// the persist deletion, 2026-08-06, NO public frontend construct
    /// poisons — the recorder covers the whole live surface — so this
    /// pokes the mechanism directly; internal guards like the multi-dim
    /// iota tripwire still route through it.)
    #[test]
    fn recorder_poisons_refuse_loudly() {
        let mut cx = Graph::new();
        let x = cx.tensor((2, 3));
        let _out = x.output();
        cx.logical
            .poison("synthetic guard tripped at t0 (mechanism test)".to_string());
        let reason = cx.logical.poisoned().expect("poison recorded");
        assert!(
            reason.contains("synthetic guard"),
            "attributable reason: {reason}"
        );
        assert!(cx
            .logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .is_err());
    }

    /// M3 STEP 1: THE FIRST NATIVE DIFFERENTIAL — the recorder's model +
    /// the reference binding generator, with NO translator anywhere,
    /// against their full search + runtime.
    #[test]
    fn differential_native_recorder_simple_elementwise() {
        let build = || {
            let mut cx = Graph::new();
            let b = cx.tensor(3);
            let c = cx.tensor(3);
            let g = cx.tensor(3);
            let e = cx.tensor(3);
            let a = (b * c + g).output();
            let d = (b * c / e).sin().output();
            (cx, b, c, g, e, a, d)
        };
        let b_data = vec![1.0, 2.0, 3.0];
        let c_data = vec![4.0, 5.0, 6.0];
        let g_data = vec![0.5, -1.5, 2.5];
        let e_data = vec![2.0, 4.0, 8.0];

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected_a = vec![4.5, 8.5, 20.5];
        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected_d = vec![0.9092974, 0.5984721, 0.7780732];
        let (cx2, b2, c2, g2, e2, a2, d2) = build();
        cx2.logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .expect("native program");
        let ours = run_reference(
            &cx2,
            &[
                (b2.id, b_data.into()),
                (c2.id, c_data.into()),
                (g2.id, g_data.into()),
                (e2.id, e_data.into()),
            ],
        );
        assert_close(ours.get_f32(a2.id).unwrap(), &expected_a);
        assert_close(ours.get_f32(d2.id).unwrap(), &expected_d);
    }

    /// TYPED-BUFFERS differential: lt produces a genuinely BOOLEAN
    /// intermediate (byte-backed u8 in reference storage; the logical dtype
    /// stays 1-bit), cast bridges it back to f32 as exact 0/1 indicators,
    /// and blend arithmetic runs downstream — element-for-element against
    /// their full search + ReferenceRuntime.
    #[test]
    fn differential_less_than_cast_against_reference_runtime() {
        use luminal::dtype::DType;
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor((2, 3));
            let y = cx.tensor((2, 3));
            let out = (x.lt(y).cast(DType::F32) * 3.0 + 1.0).output();
            (cx, x, y, out)
        };
        let x_data = vec![1.0, 5.0, 2.0, 8.0, -1.0, 0.0];
        let y_data = vec![2.0, 4.0, 2.0, 9.0, -2.0, 0.5];

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected = vec![4.0, 1.0, 1.0, 4.0, 1.0, 4.0];
        let (cx2, x2, y2, out2) = build();
        let program = cx2
            .logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .expect("native program");
        assert!(
            program.text.contains("LogicalLessThan") && program.text.contains("LogicalCast"),
            "comparison + cast expected in the model:\n{}",
            program.text
        );
        let ours = run_reference(&cx2, &[(x2.id, x_data.into()), (y2.id, y_data.into())]);
        assert_close(ours.get_f32(out2.id).unwrap(), &expected);
    }

    /// BOOL8 BOUNDARY differential (ruling 2026-07-30): a bare lt output
    /// crosses the boundary as Bool8 — the translator inserts the
    /// LogicalCast to Bool8, the boundary layout speaks (bits-of (Bool8)),
    /// and get_bool8 yields exactly the two legal codes — against their
    /// runtime's native Vec<bool> for the same graph.
    #[test]
    fn differential_bool8_boundary_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor((2, 3));
            let y = cx.tensor((2, 3));
            let out = x.lt(y).output();
            (cx, x, y, out)
        };
        let x_data = vec![1.0, 5.0, 2.0, 8.0, -1.0, 0.0];
        let y_data = vec![2.0, 4.0, 2.0, 9.0, -2.0, 0.5];

        // GOLDEN (pinned; x.lt(y) elementwise on the fixed data).
        let expected: Vec<bool> = vec![true, false, false, true, false, true];
        let (cx2, x2, y2, out2) = build();
        let program = cx2
            .logical
            .bound_program(&crate::bindings::ReferenceBindings)
            .expect("native program");
        assert!(
            program.text.contains("(LogicalCast v") && program.text.contains("(Bool8)"),
            "boundary Bool8 cast expected in the binding:\n{}",
            program.text
        );
        assert!(
            program.text.contains("(bits-of (Bool8))"),
            "Bool8 boundary layout width expected:\n{}",
            program.text
        );
        let ours = run_reference(&cx2, &[(x2.id, x_data.into()), (y2.id, y_data.into())]);
        let codes = ours.get_bool8(out2.id).expect("bool8 boundary");
        assert_eq!(codes.len(), expected.len());
        for (index, (code, truth)) in codes.iter().zip(&expected).enumerate() {
            assert!(*code <= 1, "ill-formed Bool8 code {code} at {index}");
            assert_eq!(
                *code == 1,
                *truth,
                "element {index}: our code {code} vs their {truth}"
            );
        }
    }

    /// RESHAPE differentials: split (mixed-radix group entries), merge
    /// (div/rem digit entries), and flatten (a multi-axis merge run) — all
    /// read structurally off the tracker strides, no seam nodes needed.
    #[test]
    fn differential_reshapes_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let a = cx.tensor(12);
            let b = cx.tensor((3, 4));
            let split_out = (a.split_dims(0, 4) * b).output(); // [12] -> [3,4]
            let c = cx.tensor((3, 4));
            let d = cx.tensor(12);
            let merge_out = (c.merge_dims(0, 1) * d).output(); // [3,4] -> [12]
            let e = cx.tensor((2, 3, 2));
            let f = cx.tensor(12);
            let flatten_out = (e.flatten() * f).output(); // [2,3,2] -> [12]
            (cx, a, b, c, d, e, f, split_out, merge_out, flatten_out)
        };
        let v12a: Vec<f32> = (0..12).map(|v| v as f32 + 1.0).collect();
        let v12b: Vec<f32> = (0..12).map(|v| v as f32 * 0.5 - 2.0).collect();
        let v12c: Vec<f32> = (0..12).map(|v| (v * v) as f32).collect();
        let v12d: Vec<f32> = (0..12).map(|v| v as f32 - 6.0).collect();
        let v12e: Vec<f32> = (0..12).map(|v| v as f32 * 1.5).collect();
        let v12f: Vec<f32> = (0..12).map(|v| 12.0 - v as f32).collect();

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected_flatten = vec![
            0.0, 16.5, 30.0, 40.5, 48.0, 52.5, 54.0, 52.5, 48.0, 40.5, 30.0, 16.5,
        ];
        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected_merge = vec![
            -0.0, -5.0, -16.0, -27.0, -32.0, -25.0, 0.0, 49.0, 128.0, 243.0, 400.0, 605.0,
        ];
        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected_split = vec![
            -2.0, -3.0, -3.0, -2.0, 0.0, 3.0, 7.0, 12.0, 18.0, 25.0, 33.0, 42.0,
        ];
        let (cx2, a2, b2, c2, d2, e2, f2, split2, merge2, flatten2) = build();
        let ours = run_reference(
            &cx2,
            &[
                (a2.id, v12a.into()),
                (b2.id, v12b.into()),
                (c2.id, v12c.into()),
                (d2.id, v12d.into()),
                (e2.id, v12e.into()),
                (f2.id, v12f.into()),
            ],
        );
        assert_close(ours.get_f32(split2.id).unwrap(), &expected_split);
        assert_close(ours.get_f32(merge2.id).unwrap(), &expected_merge);
        assert_close(ours.get_f32(flatten2.id).unwrap(), &expected_flatten);
    }

    /// REPEAT differential: tiling strides (z % d) lift into IntTruncRem
    /// map entries.
    #[test]
    fn differential_repeat_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor(3);
            let y = cx.tensor(12);
            let out = (x.repeat(4) * y).output();
            (cx, x, y, out)
        };
        let x_data = vec![1.0, 2.0, 3.0];
        let y_data: Vec<f32> = (0..12).map(|v| v as f32 + 0.5).collect();

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected = vec![
            0.5, 3.0, 7.5, 3.5, 9.0, 16.5, 6.5, 15.0, 25.5, 9.5, 21.0, 34.5,
        ];
        let (cx2, x2, y2, out2) = build();
        let ours = run_reference(&cx2, &[(x2.id, x_data.into()), (y2.id, y_data.into())]);
        assert_close(ours.get_f32(out2.id).unwrap(), &expected);
    }

    /// Reduction differential: sum over the front axis of a [2, 3] tensor,
    /// crossing the axis-convention flip and the reduce kernel.
    #[test]
    fn differential_sum_reduce_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor((2, 3));
            let s = x.sum(0).output();
            (cx, x, s)
        };
        let x_data = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected = vec![11.0, 22.0, 33.0];
        let (cx2, x2, s2) = build();
        let ours = run_reference(&cx2, &[(x2.id, x_data.into())]);
        assert_close(ours.get_f32(s2.id).unwrap(), &expected);
    }

    /// NARROW INTEGERS WRAP AT THEIR OWN WIDTH (carve-out 2026-09-02,
    /// main #399's `reference_narrow_integer_add_wraps_in_declared_dtype`
    /// re-expressed end to end). Main asserted this against a bare
    /// `ReferenceData` op; here the same values go through the real
    /// recorder / search / execute ladder and read back through the
    /// STRICTLY NON-WIDENING getters, which is the other half of main's
    /// claim: an I8 result is `i8`, not an i32 that happens to be small.
    ///
    /// This is the CARVE-OUT under review: I32 and I64 keep the
    /// non-wrapping ruling of 2026-08-11 and would refuse these same
    /// operands loudly.
    #[test]
    fn narrow_int_add_wraps_at_its_own_width() {
        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor_dtyped(2, DType::I8);
        let b = cx.tensor_dtyped(2, DType::I8);
        let out = (a + b).output();
        let rt = crate::harness::run_reference(
            &cx,
            &[
                (a.id, TypedBuffer::I8(vec![127, -128])),
                (b.id, TypedBuffer::I8(vec![1, -1])),
            ],
        );
        assert_eq!(rt.get_i8(out.id).unwrap(), &vec![-128i8, 127]);

        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor_dtyped(2, DType::U8);
        let b = cx.tensor_dtyped(2, DType::U8);
        let out = (a + b).output();
        let rt = crate::harness::run_reference(
            &cx,
            &[
                (a.id, TypedBuffer::U8(vec![255, 0])),
                (b.id, TypedBuffer::U8(vec![1, 255])),
            ],
        );
        assert_eq!(rt.get_u8(out.id).unwrap(), &vec![0u8, 255]);

        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor_dtyped(2, DType::I16);
        let b = cx.tensor_dtyped(2, DType::I16);
        let out = (a + b).output();
        let rt = crate::harness::run_reference(
            &cx,
            &[
                (a.id, TypedBuffer::I16(vec![32_767, -32_768])),
                (b.id, TypedBuffer::I16(vec![1, -1])),
            ],
        );
        assert_eq!(rt.get_i16(out.id).unwrap(), &vec![-32_768i16, 32_767]);
    }

    /// NARROW-INT CASTS TRUNCATE — main #399's
    /// `reference_narrow_integer_casts_preserve_native_widths`, same
    /// source values and the same expected results, through this
    /// branch's cast kernel. The wide targets are NOT part of the
    /// carve-out: `I64 -> I32` still refuses out of range, which the
    /// last act pins.
    #[test]
    fn narrow_int_casts_truncate_and_wide_casts_stay_checked() {
        let source = vec![-32_769i32, -129, -128, -1, 0, 127, 128, 255, 256];

        let mut cx = luminal::graph::Graph::new();
        let x = cx.tensor_dtyped(9, DType::Int);
        let out = x.cast(DType::I8).output();
        let rt = crate::harness::run_reference(&cx, &[(x.id, source.clone().into())]);
        assert_eq!(
            rt.get_i8(out.id).unwrap(),
            &vec![-1i8, 127, -128, -1, 0, 127, -128, -1, 0]
        );

        let mut cx = luminal::graph::Graph::new();
        let x = cx.tensor_dtyped(9, DType::Int);
        let out = x.cast(DType::U8).output();
        let rt = crate::harness::run_reference(&cx, &[(x.id, source.clone().into())]);
        assert_eq!(
            rt.get_u8(out.id).unwrap(),
            &vec![255u8, 127, 128, 255, 0, 127, 128, 255, 0]
        );

        let mut cx = luminal::graph::Graph::new();
        let x = cx.tensor_dtyped(9, DType::Int);
        let out = x.cast(DType::I16).output();
        let rt = crate::harness::run_reference(&cx, &[(x.id, source.clone().into())]);
        assert_eq!(
            rt.get_i16(out.id).unwrap(),
            &vec![32_767i16, -129, -128, -1, 0, 127, 128, 255, 256]
        );

        // The wide half of the policy, unchanged by the carve-out.
        let mut cx = luminal::graph::Graph::new();
        let x = cx.tensor_dtyped(1, DType::I64);
        let _out = x.cast(DType::Int).output();
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        let mut data = FxHashMap::default();
        data.insert(x.id, TypedBuffer::I64(vec![i64::from(i32::MAX) + 1]));
        let err = rt
            .search(&data, &luminal::test_support::harness_search_options())
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("cast i64 -> i32 out of range at value 2147483648"),
            "i64 -> i32 must stay CHECKED (ruling 2026-08-11), got: {message}"
        );
    }

    /// INTEGER `abs()` EXECUTES (main #399's dtype-aware `abs`,
    /// re-expressed). Before this, `abs()` on any integer went through
    /// `relu` -> `maximum_f32` -> `constant_float(0.0).cast(Int)`, and
    /// that F32 -> Int cast is REFUSED at authoring, so integer `abs`
    /// panicked before it recorded anything. Now it is
    /// `x * (1 - 2*(x < 0))` built from INT constants.
    ///
    /// At the signed minimum the result is the signed minimum: |i16::MIN|
    /// is not representable in i16, and under the narrow-int carve-out
    /// the multiplication wraps, which is exactly what torch reports.
    #[test]
    fn integer_abs_executes_and_wraps_at_the_signed_minimum() {
        let mut cx = luminal::graph::Graph::new();
        let x = cx.tensor_dtyped(4, DType::I16);
        let out = x.abs().output();
        let rt = crate::harness::run_reference(
            &cx,
            &[(x.id, TypedBuffer::I16(vec![-3, 0, 5, i16::MIN]))],
        );
        assert_eq!(rt.get_i16(out.id).unwrap(), &vec![3i16, 0, 5, i16::MIN]);

        // Int stays proof-gated (2026-08-11), so the caller attests the
        // range; inside it the answer is exact.
        let mut cx = luminal::graph::Graph::new();
        let x = cx.tensor_dtyped(4, DType::Int);
        let out = x.abs().output();
        let rt = crate::harness::run_reference_with_ranges(
            &cx,
            &[(x.id, vec![-3i32, 0, 5, -7].into())],
            &[(x.id, -10, 10)],
        );
        assert_eq!(rt.get_i32(out.id).unwrap(), &vec![3i32, 0, 5, 7]);
    }

    /// A FLOAT -> NARROW-INT CAST IS STILL REFUSED. Main #399's
    /// `to_i8_vec` family truncates from any source, floats included;
    /// this branch does not carry that half. The carve-out is about
    /// integer WIDTH, not about making a lossy float read implicit
    /// (cast policy 2026-08-11), and the refusal is at AUTHORING so the
    /// model's author sees it rather than the search.
    #[test]
    fn float_to_narrow_int_cast_is_refused_at_authoring() {
        let refusal = std::panic::catch_unwind(|| {
            let mut cx = luminal::graph::Graph::new();
            let x = cx.tensor_dtyped(4, DType::F32);
            let _ = x.cast(DType::I8);
        })
        .unwrap_err();
        let message = refusal
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default();
        assert!(
            message.contains("F32") && message.contains("I8") && message.contains("refused"),
            "expected the float -> int cast refusal, got: {message:?}"
        );
    }

    /// F64 IS A REAL EXECUTABLE DTYPE (ruling 2026-09-02, main #398's
    /// `f64_fn` unary kernels re-expressed). Main's
    /// `reference_unary_ops_execute_f64_natively` rewritten against
    /// this branch's runtime: an F64 input through `sqrt` on the real
    /// search-and-execute ladder, read back as `f64`, BIT-EXACT
    /// against `f64::sqrt`.
    ///
    /// The assertion is exact equality, not a tolerance, and `1e300`
    /// is the load-bearing value: it has no f32 representation at all,
    /// so any bridge through F32 anywhere in staging, execution or
    /// readback turns it into an infinity and this test fails. `0.1`
    /// covers the quieter direction — its f32 round trip differs from
    /// its f64 value in the 9th significant digit.
    #[test]
    fn f64_unary_round_trips_exactly() {
        let mut cx = luminal::graph::Graph::new();
        let x = cx.tensor_dtyped(4, DType::F64);
        let out = x.sqrt().output();

        let values = vec![2.0f64, 3.0, 0.1, 1e300];
        let rt = crate::harness::run_reference(&cx, &[(x.id, TypedBuffer::F64(values.clone()))]);

        let expected: Vec<f64> = values.iter().map(|v| v.sqrt()).collect();
        assert_eq!(
            rt.get_f64(out.id).unwrap(),
            &expected,
            "F64 sqrt must be bit-exact double precision, never an f32 bridge"
        );
        assert!(
            expected[3].is_finite(),
            "sqrt(1e300) is finite in f64 and infinite in f32 — the whole point"
        );
        // And the readback is TYPED: asking for f32 refuses by name
        // rather than narrowing.
        let err = rt.get_f32(out.id).unwrap_err();
        assert!(
            err.to_string()
                .contains("expected an f32 buffer, found f64"),
            "typed readback must refuse, got: {err}"
        );
    }

    /// TYPED-BUFFERS LANDING B PINS (2026-08-11).
    /// Bool8 INPUT staging end to end: a Bool input tensor takes caller
    /// codes through the validated constructor, the indicator cast
    /// turns them into exact 0/1 weights, and (a) ill-formed codes are
    /// refused at the door, (b) an f32 payload staged into the boolean
    /// buffer is a loud variant refusal at execute — staging never
    /// converts.
    #[test]
    fn differential_bool8_input_staging() {
        let mut cx = luminal::graph::Graph::new();
        let mask = cx.tensor_dtyped(4, DType::Bool);
        let x = cx.tensor(4);
        let out = (mask.cast(DType::F32) * x).output();

        let x_vals = vec![2.0f32, 3.0, 5.0, 7.0];
        let codes = TypedBuffer::bool8(vec![1u8, 0, 1, 0]).expect("legal codes");
        let rt =
            crate::harness::run_reference(&cx, &[(mask.id, codes), (x.id, x_vals.clone().into())]);
        assert_close(rt.get_f32(out.id).unwrap(), &[2.0, 0.0, 5.0, 0.0]);

        // (a) the two-legal-codes door
        let err = TypedBuffer::bool8(vec![0u8, 2]).unwrap_err();
        assert!(err.to_string().contains("ill-formed code 2"), "{err}");

        // (b) staging never converts: f32 into the boolean buffer refuses
        let mut cx2 = luminal::graph::Graph::new();
        let mask2 = cx2.tensor_dtyped(4, DType::Bool);
        let x2 = cx2.tensor(4);
        let out2 = (mask2.cast(DType::F32) * x2).output();
        let _ = out2;
        let mut rt2 = ReferenceRuntime::load(&cx2).expect("native load");
        let mut data = FxHashMap::default();
        data.insert(mask2.id, TypedBuffer::bool8(vec![1u8, 0, 1, 0]).unwrap());
        data.insert(x2.id, x_vals.clone().into());
        rt2.search(&data, &luminal::test_support::harness_search_options())
            .expect("search finds a plan");
        rt2.set_data(mask2.id, vec![1.0f32, 0.0, 1.0, 0.0]);
        rt2.set_data(x2.id, x_vals);
        let err = rt2.execute().unwrap_err();
        assert!(
            err.to_string().contains("staging never converts"),
            "expected the variant refusal, got: {err}"
        );
    }

    /// Native Int output readback through get_i32, with the landing-D
    /// contract in play: the mul over caller data implements because
    /// the caller DECLARED the data's range.
    #[test]
    fn differential_int_output_reads_native() {
        let mut cx = luminal::graph::Graph::new();
        let idx = cx.tensor_dtyped(5, DType::Int);
        let out = (idx * 3usize).output();
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.bind_value_range(idx.id, 0, 4).expect("range binds");
        let mut data = FxHashMap::default();
        data.insert(idx.id, vec![0i32, 1, 2, 3, 4].into());
        rt.search(&data, &luminal::test_support::harness_search_options())
            .expect("proven mul implements");
        rt.set_data(idx.id, vec![0i32, 1, 2, 3, 4]);
        rt.execute().expect("executes");
        assert_eq!(rt.get_i32(out.id).unwrap(), &vec![0i32, 3, 6, 9, 12]);
    }

    /// LANDING D, the whole non-wrapping story: (1) a plain Int add
    /// over UNATTESTED caller data is UNPROVABLE — no implementation
    /// mints and the search refuses loudly (there is no Strict escape
    /// hatch, by ruling: reject, never check-and-hope); (2)
    /// `bind_value_range` supplies the caller's attestation and the
    /// SAME graph proves, executes, and reads back exactly — while the
    /// kernel keeps its checked arithmetic as the proof's tripwire.
    #[test]
    fn int_add_proof_gating() {
        // Act 1: unproven plain add refuses at search.
        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor_dtyped(1, DType::Int);
        let b = cx.tensor_dtyped(1, DType::Int);
        let _out = (a + b).output();
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        let mut data = FxHashMap::default();
        data.insert(a.id, vec![1i32].into());
        data.insert(b.id, vec![2i32].into());
        let err = rt
            .search(&data, &luminal::test_support::harness_search_options())
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("no candidate genome"),
            "expected the unproven refusal, got: {message}"
        );
        assert!(
            message.contains("UNPROVEN") && message.contains("bind_value_range"),
            "the refusal must name the missing proof and the attestation \
             door, got: {message}"
        );

        // Act 2: the same graph under declared value ranges proves and runs.
        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor_dtyped(1, DType::Int);
        let b = cx.tensor_dtyped(1, DType::Int);
        let out = (a + b).output();
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.bind_value_range(a.id, 0, 1000).expect("range binds");
        rt.bind_value_range(b.id, 0, 1000).expect("range binds");
        rt.search(&data, &luminal::test_support::harness_search_options())
            .expect("proven add implements");
        rt.set_data(a.id, vec![700i32]);
        rt.set_data(b.id, vec![300i32]);
        rt.execute().expect("proven add executes");
        assert_eq!(rt.get_i32(out.id).unwrap(), &vec![1000i32]);
    }

    /// TruncDiv is proof-gated on the divisor excluding zero: with an
    /// attested positive divisor range it implements and truncates
    /// toward zero; without the attestation the divisor might be zero
    /// and the search refuses to find a plan at all.
    #[test]
    fn trunc_div_gating() {
        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor_dtyped(4, DType::Int);
        let b = cx.tensor_dtyped(4, DType::Int);
        let out = a.trunc_div(b).output();
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.bind_value_range(a.id, -100, 100).expect("range binds");
        rt.bind_value_range(b.id, 2, 4).expect("range binds");
        let mut data = FxHashMap::default();
        data.insert(a.id, vec![7i32, -7, 100, -1].into());
        data.insert(b.id, vec![2i32, 2, 3, 4].into());
        rt.search(&data, &luminal::test_support::harness_search_options())
            .expect("proven trunc-div implements");
        rt.set_data(a.id, vec![7i32, -7, 100, -1]);
        rt.set_data(b.id, vec![2i32, 2, 3, 4]);
        rt.execute().expect("proven trunc-div executes");
        assert_eq!(rt.get_i32(out.id).unwrap(), &vec![3i32, -3, 33, 0]);

        // Without the divisor attestation the same graph REFUSES: the
        // bounds admit zero, so no implementation exists to find.
        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor_dtyped(1, DType::Int);
        let b = cx.tensor_dtyped(1, DType::Int);
        let _out = a.trunc_div(b).output();
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        let mut data = FxHashMap::default();
        data.insert(a.id, vec![7i32].into());
        data.insert(b.id, vec![2i32].into());
        let err = rt
            .search(&data, &luminal::test_support::harness_search_options())
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("no candidate genome"),
            "expected the unattested refusal, got: {err:#}"
        );
    }
}
