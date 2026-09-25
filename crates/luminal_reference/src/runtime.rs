//! `ReferenceRuntime`: the CPU reference executor for logical-SSA
//! bufferized plans (ruling 2026-07-28: lives in luminal; once the path is
//! COMPLETE it replaces `ReferenceRuntime` — not before).
//!
//! Executes a [`BufferIrGraph`] directly: every buffer is a [`TypedBuffer`]
//! sized by ASSIGNMENT LOOKUP (corrected contract, 2026-08-31) — the plan
//! says which tensor a buffer backs, the carried [`DecodedLayout`] is that
//! tensor's elected layout, and allocation is span-of-layout elements in
//! the layout's own dtype. No sizing walk, no voting: every consumer
//! takes the BufferId blindly. Compute nodes dispatch through THIS
//! runtime's kernel registry ([`kernels`]) by concrete op type — ops
//! carry no execution of their own (ruling 2026-08-06), and an op
//! without a kernel here refuses by name. Caller data binds by the
//! numeric `BufferLit` id, the same key `hlir_to_logical` derives from
//! HLIR node indices, so differential tests against `ReferenceRuntime`
//! bind identically on both sides.

use anyhow::{Context, Result, anyhow, ensure};
use petgraph::algo::toposort;
use rustc_hash::FxHashMap;
use std::borrow::Cow;

use crate::typed_buffer::{ReferenceKernelCtx, TypedBuffer};
use luminal::bufferize::{BufferId, BufferIrGraph, BufferNode, OutputBinding};

use luminal::layouts::DecodedLayout;

/// Default live tensor payload and scratch budget (8 GiB), excluding compiler state.
pub const DEFAULT_MEMORY_BUDGET_BYTES: usize = 8 * 1024 * 1024 * 1024;

/// Default per-intermediate allocation ceiling (2 GiB), excluding boundary buffers.
pub const DEFAULT_MAX_INTERMEDIATE_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Conservative tensor-payload scratch bound for the unchanged reference
/// kernels. All currently registered kernels either clone operands, allocate
/// index/coordinate columns, or use only rank-sized metadata. This is allocation
/// accounting, never op selection or graph rewriting.
fn kernel_scratch_bytes(label: &str, ctx: &ReferenceKernelCtx) -> Result<usize> {
    let sum = |values: Vec<usize>| {
        values.into_iter().try_fold(0usize, |a, b| {
            a.checked_add(b)
                .ok_or_else(|| anyhow!("kernel scratch size overflow"))
        })
    };
    let mut bytes = sum(ctx.operands.iter().map(TypedBuffer::byte_len).collect())?;
    let n = ctx.operands.first().map_or(0, TypedBuffer::len);
    let out = ctx.dests.first().map_or(0, TypedBuffer::len);
    let extra = match label {
        "IndexMapApplyMaterialize" => out.checked_mul(std::mem::size_of::<usize>()),
        "GatherGeneric" => {
            let columns = sum(ctx.operands.iter().skip(1).map(TypedBuffer::len).collect())?;
            columns.checked_mul(8).and_then(|v| {
                out.checked_mul(std::mem::size_of::<usize>())
                    .and_then(|w| v.checked_add(w))
            })
        }
        "ScatterFunctionalGeneric" => {
            let columns = sum(ctx.operands.iter().skip(2).map(TypedBuffer::len).collect())?;
            let src = ctx.operands.get(1).map_or(0, TypedBuffer::len);
            columns
                .checked_mul(8)
                .and_then(|v| {
                    src.checked_mul(std::mem::size_of::<usize>())
                        .and_then(|w| v.checked_add(w))
                })
                .and_then(|v| v.checked_add(out))
        }
        "CastGeneric" => n.checked_mul(8),

        "AddFunctionalGeneric"
        | "CeilFunctionalGeneric"
        | "ConstantGeneric"
        | "DivFunctionalGeneric"
        | "Exp2FunctionalGeneric"
        | "ExpFunctionalGeneric"
        | "FloorFunctionalGeneric"
        | "IotaGeneric"
        | "LessThanGeneric"
        | "Log2FunctionalGeneric"
        | "ModFunctionalGeneric"
        | "MulFunctionalGeneric"
        | "RecipFunctionalGeneric"
        | "ReduceMaxGeneric"
        | "ReduceSumGeneric"
        | "RoundFunctionalGeneric"
        | "SelectFunctionalGeneric"
        | "SinFunctionalGeneric"
        | "SqrtFunctionalGeneric"
        | "TruncCastGeneric"
        | "TruncDivFunctionalGeneric"
        | "TruncFunctionalGeneric"
        | "TruncRemFunctionalGeneric"
        | "BufferAlloc"
        | "BufferFree" => Some(0),
        _ => anyhow::bail!("reference kernel {label} has no scratch allocation bound"),
    }
    .ok_or_else(|| anyhow!("kernel scratch size overflow"))?;
    bytes = bytes
        .checked_add(extra)
        .ok_or_else(|| anyhow!("kernel scratch size overflow"))?;
    Ok(bytes)
}

fn is_storage_marker(op: &dyn luminal::buffer_tensor_ir::BufferTensorIrOp) -> bool {
    op.as_any().is::<luminal::buffer_tensor_ir::BufferAlloc>()
        || op.as_any().is::<luminal::buffer_tensor_ir::BufferFree>()
}

fn reserve_live(live: &mut usize, bytes: usize, budget: usize, peak: &mut usize) -> Result<()> {
    let needed = live
        .checked_add(bytes)
        .ok_or_else(|| anyhow!("live allocation size overflow"))?;
    ensure!(
        needed <= budget,
        "reference live memory budget exceeded: {live} live + {bytes} requested = {needed} bytes, budget={budget}"
    );
    *live = needed;
    *peak = (*peak).max(needed);
    Ok(())
}

fn depth_first_order(
    plan: &BufferIrGraph<DecodedLayout>,
) -> Result<Vec<petgraph::graph::NodeIndex>> {
    toposort(&plan.dag, None).map_err(|_| anyhow!("bufferized plan has a cycle"))?;
    let roots = plan
        .dag
        .node_indices()
        .filter(|i| matches!(plan.dag[*i], BufferNode::BufferOutput { .. }))
        .chain(plan.dag.node_indices());
    let mut done = rustc_hash::FxHashSet::default();
    let mut order = Vec::new();
    for root in roots {
        let mut stack = vec![(root, false)];
        while let Some((node, exiting)) = stack.pop() {
            if done.contains(&node) {
                continue;
            }
            if exiting {
                done.insert(node);
                order.push(node);
            } else {
                stack.push((node, true));
                let mut deps: Vec<_> = plan
                    .dag
                    .neighbors_directed(node, petgraph::Direction::Incoming)
                    .collect();
                deps.sort_by_key(|n| n.index());
                stack.extend(deps.into_iter().rev().map(|n| (n, false)));
            }
        }
    }
    Ok(order)
}

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

/// What `load` captured from a recorded Graph: the bound program (model
/// text, this runtime's boundary, the post-schedule checks) plus whatever
/// the binding calls accumulate before `search` assembles and saturates.
struct NativeSpec {
    bound: crate::bindings::BoundProgram,
    binding_seeds: String,
    ops: Option<Vec<&'static str>>,
}

#[derive(Default)]
pub struct ReferenceRuntime {
    memory_budget_bytes: Option<usize>,
    peak_live_bytes: usize,
    plan: Option<BufferIrGraph<DecodedLayout>>,
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
    /// Every buffer a value is bound to as an output. A value bound on
    /// two buffers is read back by buffer, not by tensor.
    output_buffers: FxHashMap<petgraph::graph::NodeIndex, Vec<i64>>,
    /// M3 Step 2 native-ladder state (`load` → bind → `with_ops` → `search`).
    native: Option<NativeSpec>,
    /// BUCKETS (D7, 2026-09-03): per-dim intervals a single search
    /// covers, bound before `search_buckets`. Empty = the ordinary
    /// single-pin ladder, unchanged in every respect.
    dim_buckets: std::collections::BTreeMap<luminal::shape::Symbol, Vec<luminal::graph::DimBucket>>,
    /// One finished plan per Cartesian bucket combination.
    bucket_plans: Vec<crate::search::BucketPlan>,
    /// Plans received from a peer. These are executable without a search.
    loaded_bucket_plans: Vec<crate::compiled_artifact::CompiledBucket>,
    /// The dim values this runtime currently holds — every `[n, n]`
    /// `bind_dyn_range` pin, plus whatever [`Self::set_dim`] sets. With
    /// buckets bound this is what picks the plan at execute time.
    dims: luminal::shape::DynMap,
    /// EVERY dim [`Self::bind_dyn_range`] has bound, tight or not, with
    /// the interval it was given. `dims` records only the `[n, n]` pins,
    /// so it cannot answer the exclusivity question: buckets and range
    /// bindings must refuse each other in BOTH orders, and a non-tight
    /// range under a later bucket would otherwise seed the same `IntVar`
    /// twice and INTERSECT under the bounds lattice's merge rather than
    /// refuse.
    range_bound: std::collections::BTreeMap<luminal::shape::Symbol, (u64, u64)>,
}

impl ReferenceRuntime {
    /// Bound live tensor payloads: staged inputs, live buffers, destinations,
    /// and tensor-sized kernel scratch. Does not include compiler/Python memory.
    pub fn set_memory_budget_bytes(&mut self, bytes: usize) {
        self.memory_budget_bytes = Some(bytes);
    }

    pub fn peak_live_bytes(&self) -> usize {
        self.peak_live_bytes
    }

    /// Register the tensor→buffer role maps from the boundary bindings.
    pub fn stage_bindings(
        &mut self,
        inputs: &[crate::bindings::Bound],
        outputs: &[crate::bindings::Bound],
    ) {
        self.input_buffers = inputs.iter().map(|b| (b.value, b.buffer)).collect();
        self.output_buffers.clear();
        for bound in outputs {
            self.output_buffers
                .entry(bound.value)
                .or_default()
                .push(bound.buffer);
        }
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
    pub fn load_plan(&mut self, plan: BufferIrGraph<DecodedLayout>) {
        self.lit_index = plan
            .buffers
            .values()
            .filter_map(|buffer| buffer.lit.map(|lit| (lit, buffer.id.clone())))
            .collect();
        self.plan = Some(plan);
        self.storage.clear();
    }

    /// LOAD a recorded graph under the default binding: every input on
    /// its own read-only buffer, every leaf on its own read-write buffer.
    pub fn load(graph: &luminal::graph::Graph) -> Result<Self> {
        Self::load_with(
            graph,
            crate::bindings::ReferenceBindings::leaves(&graph.logical),
        )
    }

    /// LOAD a recorded graph under the caller's binding — which values
    /// enter and leave through which buffers. The tensor→buffer maps are
    /// live from here, so `set_data` needs no search first.
    pub fn load_with(
        graph: &luminal::graph::Graph,
        bindings: crate::bindings::ReferenceBindings,
    ) -> Result<Self> {
        let bound = bindings
            .bind(&graph.logical)
            .map_err(|reason| anyhow!("load refused: {reason}"))?;
        let mut runtime = Self::default();
        runtime.stage_bindings(&bound.inputs, &bound.outputs);
        runtime.native = Some(NativeSpec {
            bound,
            binding_seeds: String::new(),
            ops: None,
        });
        Ok(runtime)
    }

    /// BINDING: seed a dynamic dim's range (bounds-on-vars — never a pin).
    pub fn bind_dyn_range(
        &mut self,
        var: impl Into<luminal::shape::Symbol>,
        lower: u64,
        upper: u64,
    ) -> Result<()> {
        let var = var.into();
        ensure!(
            !self.dim_buckets.contains_key(&var),
            "dim `{var}` has buckets bound; a bucketed dim is seeded per bucket \
             and must not carry a second range binding"
        );
        let spec = self
            .native
            .as_mut()
            .ok_or_else(|| anyhow!("bind before load"))?;
        spec.binding_seeds.push_str(&format!(
            "(set (lower-bound-of (IntVar \"{var}\")) (bigint {lower}))\n\
             (set (upper-bound-of (IntVar \"{var}\")) (bigint {upper}))\n"
        ));
        // EVERY range binding is remembered, so `bind_dim_buckets` can
        // refuse this dim whatever the interval was.
        self.range_bound.insert(var, (lower, upper));
        // A tight [n, n] binding IS a pin: remember it too, so a bucketed
        // plan's representative map records the whole assignment and
        // `select_bucket` sees every dim.
        if lower == upper {
            self.dims.insert(var, lower as usize);
        }
        Ok(())
    }

    /// BIND BUCKETS for a dynamic dimension (D7, 2026-09-03): a set of
    /// disjoint intervals, each of which gets its own searched plan.
    /// `search_buckets` then runs one search per Cartesian combination
    /// and `execute` picks the covering plan from the current dims.
    ///
    /// THE BUCKETS MUST PARTITION CLEANLY: non-empty, sorted by `min`,
    /// and pairwise disjoint. Overlap is REFUSED rather than resolved
    /// first-wins — two plans that both claim a value is an ambiguity in
    /// the caller's model, and picking one silently is how a graph ends
    /// up running the plan its author did not mean.
    pub fn bind_dim_buckets(
        &mut self,
        dim: impl Into<luminal::shape::Symbol>,
        buckets: Vec<luminal::graph::DimBucket>,
    ) -> Result<()> {
        let dim = dim.into();
        ensure!(!buckets.is_empty(), "dim `{dim}` was given no buckets");
        if let Some((lo, hi)) = self.range_bound.get(&dim) {
            anyhow::bail!(
                "dim `{dim}` already carries a range binding [{lo}, {hi}] from \
                 bind_dyn_range; a bucketed dim is seeded per bucket and must not \
                 carry a second range binding"
            );
        }
        ensure!(
            !self.dims.contains_key(&dim),
            "dim `{dim}` already has a value from set_dim; bind buckets before \
             setting the execution dim"
        );
        for pair in buckets.windows(2) {
            ensure!(
                pair[0].max < pair[1].min,
                "dim `{dim}` buckets must be sorted and disjoint, but [{}, {}] and \
                 [{}, {}] are not",
                pair[0].min,
                pair[0].max,
                pair[1].min,
                pair[1].max
            );
        }
        self.dim_buckets.insert(dim, buckets);
        Ok(())
    }

    /// Set a dynamic dimension's value for EXECUTION (D7). With buckets
    /// bound this is what selects the plan; without them it is a
    /// record-keeping no-op on a runtime whose plan is already pinned.
    pub fn set_dim(&mut self, dim: impl Into<luminal::shape::Symbol>, value: usize) {
        self.dims.insert(dim.into(), value);
    }

    /// The finished per-bucket plans (empty until `search_buckets`).
    pub fn bucket_plans(&self) -> &[crate::search::BucketPlan] {
        &self.bucket_plans
    }

    /// Serialize selected reference plans using caller-provided boundary slot
    /// order. Internal node IDs never cross the process boundary.
    pub fn serialize_compiled(
        &self,
        inputs: &[petgraph::graph::NodeIndex],
        outputs: &[i64],
    ) -> Result<Vec<u8>> {
        let buckets = if self.bucket_plans.is_empty() {
            let plan = self
                .plan
                .as_ref()
                .ok_or_else(|| anyhow!("no selected plan"))?;
            vec![crate::compiled_artifact::CompiledBucket {
                ranges: Default::default(),
                input_buffers: inputs
                    .iter()
                    .map(|id| {
                        self.input_buffers
                            .get(id)
                            .copied()
                            .ok_or_else(|| anyhow!("input {id:?} has no boundary binding"))
                    })
                    .collect::<Result<_>>()?,
                output_buffers: outputs.to_vec(),
                plan: plan.clone(),
            }]
        } else {
            self.bucket_plans
                .iter()
                .map(|bucket| {
                    Ok(crate::compiled_artifact::CompiledBucket {
                        ranges: bucket
                            .ranges
                            .iter()
                            .map(|(s, r)| (s.to_string(), *r))
                            .collect(),
                        input_buffers: bucket.program.inputs.iter().map(|b| b.buffer).collect(),
                        output_buffers: bucket.program.outputs.iter().map(|b| b.buffer).collect(),
                        plan: bucket.outcome.best_plan.clone(),
                    })
                })
                .collect::<Result<Vec<_>>>()?
        };
        crate::compiled_artifact::serialize(&buckets)
    }

    /// Install selected plans without saturation or profiling. The lists map
    /// artifact boundary positions onto this process's translated values.
    pub fn deserialize_compiled(
        &mut self,
        bytes: &[u8],
        inputs: &[petgraph::graph::NodeIndex],
        outputs: &[petgraph::graph::NodeIndex],
        memory_budget_bytes: usize,
    ) -> Result<Vec<i64>> {
        let buckets = crate::compiled_artifact::deserialize(bytes)?;
        let first = &buckets[0];
        ensure!(
            first.input_buffers.len() == inputs.len()
                && first.output_buffers.len() == outputs.len(),
            "reference artifact boundary arity differs from local translation"
        );
        for bucket in &buckets {
            ensure!(
                bucket.input_buffers == first.input_buffers
                    && bucket.output_buffers == first.output_buffers,
                "reference artifact buckets have inconsistent boundary assignments"
            );
        }
        let input_bindings: Vec<_> = inputs
            .iter()
            .zip(&first.input_buffers)
            .map(|(&value, &buffer)| crate::bindings::Bound { value, buffer })
            .collect();
        let output_bindings: Vec<_> = outputs
            .iter()
            .zip(&first.output_buffers)
            .map(|(&value, &buffer)| crate::bindings::Bound { value, buffer })
            .collect();
        self.stage_bindings(&input_bindings, &output_bindings);
        self.set_memory_budget_bytes(memory_budget_bytes);
        let output_buffers = first.output_buffers.clone();
        self.native = None;
        self.bucket_plans.clear();
        self.loaded_bucket_plans = buckets;
        // A tracing hint need not fall inside any requested bucket. Select
        // symbolic plans only after the caller binds its actual input dims.
        if self.loaded_bucket_plans.len() == 1 && self.loaded_bucket_plans[0].ranges.is_empty() {
            self.select_bucket_plan()?;
        }
        Ok(output_buffers)
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
            .bound
            .inputs
            .iter()
            .find(|bound| bound.value == tensor)
            .and_then(|bound| spec.bound.let_names.get(&bound.value).cloned())
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
        options: &crate::search::CompileOptions,
    ) -> Result<crate::search::SearchOutcome> {
        ensure!(
            self.dim_buckets.is_empty(),
            "dim buckets are bound: call search_buckets instead. Each bucket is \
             searched at its OWN representative, so its inputs are a different \
             SIZE — one fixed data map cannot stage them all, and staging the \
             wrong size is exactly the silent mis-fit this refuses"
        );
        self.set_memory_budget_bytes(options.memory_budget_bytes);
        let spec = self
            .native
            .take()
            .ok_or_else(|| anyhow!("search before load"))?;
        let program = crate::search::SearchProgram {
            text: spec.bound.text_with_seeds(&spec.binding_seeds),
            inputs: spec.bound.inputs.clone(),
            outputs: spec.bound.outputs.clone(),
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
                "{}\n\n{}",
                crate::assembled_program(),
                spec.bound.text_unchecked_with_seeds(&spec.binding_seeds)
            );
            let mut probe = luminal::egglog_snippet::new_egraph();
            if probe.parse_and_run_program(None, &unchecked).is_ok() {
                let mut failed: Vec<&str> = Vec::new();
                for (label, text) in &spec.bound.labeled_checks {
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
        crate::search::check_interrupt()?;
        // THE ASSEMBLY TRIPWIRE: every constructor of a decoded sort in
        // this program has exactly one decoder, checked against the LIVE
        // schema before anything reads a serialized class.
        crate::decoder_registry().check(&egraph)?;
        let saturation_nanos = saturation_start.elapsed().as_nanos();
        let serialize_start = std::time::Instant::now();
        let mut serialized = egraph
            .serialize(luminal::prelude::egglog::SerializeConfig::default())
            .egraph;
        let serialize_nanos = serialize_start.elapsed().as_nanos();
        let mut outcome = crate::search::search_implementations_with_ops(
            &mut serialized,
            &program,
            input_data,
            &self.dims,
            &self.dims,
            options,
            spec.ops,
        )?;
        outcome.timings.saturation_nanos = saturation_nanos;
        outcome.timings.serialize_nanos = serialize_nanos;
        self.stage_bindings(&program.inputs, &program.outputs);
        self.load_plan(outcome.best_plan.clone());
        Ok(outcome)
    }

    /// BUCKETED SEARCH (D7, 2026-09-03): one search per Cartesian
    /// combination of the bound [`Self::bind_dim_buckets`] intervals.
    /// Each combination is rendered RANGE-seeded: its whole fixpoint
    /// (authoring checks included) must pass, proving the base logical
    /// program valid over the WHOLE interval, and the plan is extracted
    /// from that range-valid fixpoint so its spans/extents stay symbolic.
    /// The representative assignment is used only to STAGE and PRICE the
    /// plan during profiling; the resulting plan executes at every value
    /// in the bucket.
    ///
    /// `input_data` is a FUNCTION of the pins because it has to be: a
    /// bucket profiled at `a = 3` and one profiled at `a = 7` want
    /// differently sized payloads. It is called once per combination with
    /// that combination's representative map.
    ///
    /// The plans are kept; [`Self::execute`] selects among them from the
    /// current dims. Nothing is loaded here unless the runtime's dims
    /// already name a covering bucket.
    pub fn search_buckets(
        &mut self,
        input_data: impl Fn(
            &luminal::shape::DynMap,
        ) -> FxHashMap<petgraph::graph::NodeIndex, TypedBuffer>,
        options: &crate::search::CompileOptions,
    ) -> Result<&[crate::search::BucketPlan]> {
        ensure!(
            !self.dim_buckets.is_empty(),
            "no dim buckets are bound: call search"
        );
        self.set_memory_budget_bytes(options.memory_budget_bytes);
        let spec = self
            .native
            .take()
            .ok_or_else(|| anyhow!("search_buckets before load"))?;
        let assembly = crate::search::BucketAssembly {
            assembled_program: crate::assembled_program(),
            prefix: &spec.bound.prefix,
            binding_seeds: &spec.binding_seeds,
            schedule: crate::bindings::ReferenceBindings::SCHEDULE,
            post_checks: &spec.bound.post_checks,
            inputs: &spec.bound.inputs,
            outputs: &spec.bound.outputs,
            base_dims: &self.dims,
        };
        self.bucket_plans = crate::search::bucketed_search_implementations(
            &assembly,
            &self.dim_buckets,
            input_data,
            options,
            spec.ops.clone(),
        )?;
        self.stage_bindings(&spec.bound.inputs, &spec.bound.outputs);
        // Load eagerly when the runtime already sits inside a bucket at
        // its representative; otherwise `execute` will select.
        let _ = self.select_bucket_plan();
        Ok(&self.bucket_plans)
    }

    /// Pick and load the bucket plan covering the current dims.
    ///
    /// SYMBOLIC PLANS (the Phase 1 limitation, lifted): a bucket's winning
    /// plan is searched over the bucket's RANGE-seeded render, so its
    /// spans and extents stay expressions (`Var("a")`) rather than the
    /// representative's literals. `execute` evaluates them under the
    /// runtime's current [`Self::dims`], so the one plan runs correctly at
    /// every value in the bucket — no re-search. The representative still
    /// picks the plan by bucket coverage and prices it during search, but
    /// it no longer constrains what the loaded plan can execute.
    fn select_bucket_plan(&mut self) -> Result<()> {
        if !self.loaded_bucket_plans.is_empty() {
            let bucket = self
                .loaded_bucket_plans
                .iter()
                .find(|bucket| {
                    bucket.ranges.iter().all(|(name, &(lo, hi))| {
                        self.dims
                            .get(&luminal::shape::Symbol::from(name.as_str()))
                            .is_some_and(|&value| lo <= value && value <= hi)
                    })
                })
                .ok_or_else(|| {
                    anyhow!("no serialized reference plan covers dims {:?}", self.dims)
                })?;
            self.load_plan(bucket.plan.clone());
            return Ok(());
        }
        let Some(plan) = crate::search::select_bucket(&self.bucket_plans, &self.dims) else {
            let covered: Vec<_> = self.bucket_plans.iter().map(|p| p.ranges.clone()).collect();
            anyhow::bail!(
                "no bucket covers dims {:?}; the searched buckets are {covered:?}",
                self.dims
            );
        };
        let chosen = plan.outcome.best_plan.clone();
        let (inputs, outputs) = (plan.program.inputs.clone(), plan.program.outputs.clone());
        self.stage_bindings(&inputs, &outputs);
        self.load_plan(chosen);
        Ok(())
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
        // With buckets bound, the plan is chosen HERE, from the current
        // dims (see [`Self::select_bucket_plan`] for the static-plan
        // refusal). Without them nothing changes.
        if !self.bucket_plans.is_empty() || !self.loaded_bucket_plans.is_empty() {
            self.select_bucket_plan()?;
        }
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

        // Evaluate all geometry and dtype contracts without allocating tensor
        // payloads. A depth-first walk of prerequisites retains every DAG edge,
        // including WAR anti-dependencies and effects.
        let order = depth_first_order(plan)?;
        let mut geometry = FxHashMap::default();
        for (id, buffer) in &plan.buffers {
            let numel = buffer.layout.span_with(&self.dims)?;
            let dtype = buffer
                .layout
                .dtype
                .ok_or_else(|| anyhow!("buffer {} has no dtype", buffer.label))?;
            let empty = TypedBuffer::zeroed(dtype, 0)?;
            let bytes = numel
                .checked_mul(usize::try_from(dtype.egglog_bits())?.div_ceil(8))
                .ok_or_else(|| anyhow!("buffer {} size overflow", buffer.label))?;
            geometry.insert(id.clone(), (numel, dtype, bytes, empty.type_name()));
        }
        let mut outputs = rustc_hash::FxHashSet::default();
        let mut last_use = FxHashMap::default();
        let mut storage: FxHashMap<BufferId, Cow<'_, TypedBuffer>> = FxHashMap::default();
        for (step, index) in order.iter().enumerate() {
            match &plan.dag[*index] {
                BufferNode::BufferInput { slots } => {
                    for slot in slots {
                        let buffer = &plan.buffers[&slot.buffer];
                        let data = buffer
                            .lit
                            .and_then(|lit| self.staged.get(&lit))
                            .ok_or_else(|| {
                                anyhow!("input buffer {} was never set_data", buffer.label)
                            })?;
                        let (numel, _, _, name) = geometry[&slot.buffer];
                        ensure!(
                            data.len() == numel,
                            "staged data for {} has {} elements, buffer holds {numel}",
                            buffer.label,
                            data.len()
                        );
                        ensure!(
                            data.type_name() == name,
                            "buffer {} expects {name}; staged {} data is the wrong type (staging never converts)",
                            buffer.label,
                            data.type_name()
                        );
                        storage.insert(slot.buffer.clone(), Cow::Borrowed(data));
                        last_use.insert(slot.buffer.clone(), step);
                    }
                }
                BufferNode::BufferOutput { slots } => {
                    outputs.extend(slots.iter().map(|slot| slot.buffer.clone()));
                    for slot in slots {
                        last_use.insert(slot.buffer.clone(), step);
                    }
                }
                BufferNode::Compute {
                    op, reads, writes, ..
                } => {
                    if is_storage_marker(op.as_ref()) {
                        continue;
                    }
                    for id in reads
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| op.operand_reads_memory(*k))
                        .map(|(_, id)| id)
                        .chain(writes)
                    {
                        last_use.insert(id.clone(), step);
                    }
                }
                BufferNode::BufferCopy { src, dst } => {
                    last_use.insert(src.clone(), step);
                    last_use.insert(dst.clone(), step);
                }
            }
        }
        let mut releases: Vec<Vec<BufferId>> = vec![Vec::new(); order.len()];
        for (id, step) in last_use {
            if !outputs.contains(&id) {
                releases[step].push(id);
            }
        }
        let budget = self
            .memory_budget_bytes
            .unwrap_or(DEFAULT_MEMORY_BUDGET_BYTES);
        let staged_bytes = self.staged.values().try_fold(0usize, |n, data| {
            n.checked_add(data.byte_len())
                .ok_or_else(|| anyhow!("staged byte count overflow"))
        })?;
        ensure!(
            staged_bytes <= budget,
            "reference live memory budget exceeded: staged inputs require {staged_bytes} bytes, budget={budget}"
        );
        // Old outputs are invalidated on execution; keeping them would double
        // storage across calls. Inputs remain borrowed from self.staged.
        self.storage.clear();
        let mut live = staged_bytes;
        self.peak_live_bytes = live;
        for (step, index) in order.into_iter().enumerate() {
            crate::search::check_interrupt()?;
            match &plan.dag[index] {
                BufferNode::BufferInput { .. } | BufferNode::BufferOutput { .. } => {}
                BufferNode::BufferCopy { src, dst } => {
                    let source = storage
                        .get(src)
                        .ok_or_else(|| anyhow!("copy reads missing buffer"))?;
                    let (n, _, bytes, name) = geometry[dst];
                    ensure!(
                        source.len() == n && source.type_name() == name,
                        "copy length/type mismatch"
                    );
                    reserve_live(&mut live, bytes, budget, &mut self.peak_live_bytes)?;
                    let copy = source.as_ref().clone();
                    if let Some(Cow::Owned(old)) = storage.insert(dst.clone(), Cow::Owned(copy)) {
                        live -= old.byte_len();
                    }
                }
                BufferNode::Compute { op, .. } if is_storage_marker(op.as_ref()) => {
                    // Allocation/free nodes retain their ordering edges. Physical
                    // storage is allocated at the writer and freed at last use.
                }
                BufferNode::Compute {
                    op,
                    reads,
                    writes,
                    operand_info,
                    ..
                } => {
                    let mut operands: Vec<TypedBuffer> = Vec::with_capacity(reads.len());
                    let mut restore = Vec::with_capacity(reads.len());
                    let mut taken: FxHashMap<BufferId, usize> = FxHashMap::default();
                    let mut operand_dims = Vec::with_capacity(reads.len());
                    for (k, id) in reads.iter().enumerate() {
                        let slot = operand_info.get(k).ok_or_else(|| {
                            anyhow!("{} operand {k} lacks its slot descriptor", op.label())
                        })?;
                        if op.operand_reads_memory(k) {
                            ensure!(
                                slot.layout == plan.buffers[id].layout,
                                "{} operand {k} reads through a folded read this executor does not lower",
                                op.label()
                            );
                            if let Some(&first) = taken.get(id) {
                                reserve_live(
                                    &mut live,
                                    operands[first].byte_len(),
                                    budget,
                                    &mut self.peak_live_bytes,
                                )?;
                                operands.push(operands[first].clone());
                                restore.push(None);
                            } else {
                                let data = storage.remove(id).ok_or_else(|| {
                                    anyhow!("{} reads missing buffer {id:?}", op.label())
                                })?;
                                let borrowed = match &data {
                                    Cow::Borrowed(data) => {
                                        reserve_live(
                                            &mut live,
                                            data.byte_len(),
                                            budget,
                                            &mut self.peak_live_bytes,
                                        )?;
                                        Some(*data)
                                    }
                                    Cow::Owned(_) => None,
                                };
                                taken.insert(id.clone(), operands.len());
                                operands.push(data.into_owned());
                                restore.push(Some((id.clone(), borrowed)));
                            }
                        } else {
                            operands.push(TypedBuffer::F32(Vec::new()));
                            restore.push(None);
                        }
                        operand_dims.push(slot.layout.extents_with(&self.dims)?);
                    }
                    let mut dests = Vec::with_capacity(writes.len());
                    for id in writes {
                        let (n, dtype, bytes, _) = geometry[id];
                        reserve_live(&mut live, bytes, budget, &mut self.peak_live_bytes)?;
                        dests.push(TypedBuffer::zeroed(dtype, n)?);
                    }
                    let mut ctx = ReferenceKernelCtx {
                        operands,
                        operand_dims,
                        dests,
                        dims: self.dims.clone(),
                    };
                    // Kernels keep their existing owned-buffer ABI and bodies.
                    // Reserve a conservative bound for their tensor-sized
                    // scratch before entering them; metadata is not payload.
                    let scratch = kernel_scratch_bytes(op.label(), &ctx)?;
                    reserve_live(&mut live, scratch, budget, &mut self.peak_live_bytes)?;
                    let kernel = crate::kernels::kernel_for(op.as_ref())
                        .ok_or_else(|| anyhow!("no reference kernel for {}", op.label()))?;
                    (kernel.execute)(op.as_ref(), &mut ctx)
                        .with_context(|| format!("executing {}", op.label()))?;
                    live -= scratch;
                    let ReferenceKernelCtx {
                        operands,
                        dests: results,
                        ..
                    } = ctx;
                    for (data, slot) in operands.into_iter().zip(restore) {
                        match slot {
                            Some((id, None)) => {
                                storage.insert(id, Cow::Owned(data));
                            }
                            Some((id, Some(original))) => {
                                live -= data.byte_len();
                                storage.insert(id, Cow::Borrowed(original));
                            }
                            None => {
                                live -= data.byte_len();
                            }
                        }
                    }
                    for (id, data) in writes.iter().zip(results) {
                        if let Some(Cow::Owned(old)) = storage.insert(id.clone(), Cow::Owned(data))
                        {
                            live -= old.byte_len();
                        }
                    }
                }
            }
            for id in &releases[step] {
                if let Some(Cow::Owned(data)) = storage.remove(id) {
                    live -= data.byte_len();
                }
            }
        }
        let mut result = FxHashMap::default();
        for id in outputs {
            let data = storage
                .remove(&id)
                .ok_or_else(|| anyhow!("output buffer {id:?} was not produced"))?;
            // A passthrough output borrowed from staging needs an owned copy.
            if let Cow::Borrowed(data) = &data {
                reserve_live(
                    &mut live,
                    data.byte_len(),
                    budget,
                    &mut self.peak_live_bytes,
                )?;
            }
            result.insert(id, data.into_owned());
        }
        self.storage = result;
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

    fn output_buffer(&self, tensor: petgraph::graph::NodeIndex) -> Result<i64> {
        match self.output_buffers.get(&tensor).map(Vec::as_slice) {
            Some([buffer]) => Ok(*buffer),
            Some(buffers) => Err(anyhow!(
                "tensor {tensor:?} is bound as an output on {} buffers ({buffers:?}); read it by buffer",
                buffers.len()
            )),
            None => Err(anyhow!(
                "tensor {tensor:?} is not a bound output of this program"
            )),
        }
    }

    /// The typed contents of an output BUFFER — for values bound on more
    /// than one buffer, or for callers that think in buffers.
    pub fn get_buffer(&self, buffer: i64) -> Result<&TypedBuffer> {
        self.get_typed(buffer)
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
    ) -> Result<(
        &TypedBuffer,
        &OutputBinding<luminal::layouts::DecodedLayout>,
    )> {
        let binding = self.output_layout(index)?;
        let data = self
            .storage
            .get(&binding.buffer)
            .ok_or_else(|| anyhow!("output slot {index} has no contents (execute first)"))?;
        Ok((data, binding))
    }

    /// Output slot `index`'s binding — buffer identity plus the elected
    /// layout (see [`Self::output_slot`]).
    pub fn output_layout(
        &self,
        index: usize,
    ) -> Result<&OutputBinding<luminal::layouts::DecodedLayout>> {
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| anyhow!("no plan loaded"))?;
        for node in plan.dag.node_weights() {
            if let BufferNode::BufferOutput { slots } = node
                && let Some(slot) = slots.iter().find(|slot| slot.index == index)
            {
                return Ok(slot);
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
    use crate::ReferenceRuntime;
    use crate::harness::run_reference;
    use crate::typed_buffer::TypedBuffer;
    use luminal::dtype::DType;
    use luminal::graph::Graph;
    use rustc_hash::FxHashMap;

    /// `constant_f64` crosses the pipeline as an exact double: the value
    /// read back must be bit-identical to the f64 literal, and distinct
    /// from the f32-rounded constant the old single-constant path would
    /// have produced. (CUDA Lite and Metal have no double device type
    /// yet, so the reference is the semantic authority here.)
    #[test]
    fn f64_constant_keeps_double_precision() {
        let exact = 0.1f64 + 0.2f64;
        let mut cx = Graph::new();
        let constant = cx.constant_f64(exact);
        let out = constant;

        let mut runtime = ReferenceRuntime::load(&cx).expect("load");
        runtime
            .search(&FxHashMap::default(), &crate::harness_search_options())
            .expect("search a constant-only graph");
        runtime.execute().expect("execute");

        let value = runtime.get_f64(out.id).expect("F64 readback");
        assert_eq!(value.len(), 1);
        assert_eq!(value[0].to_bits(), exact.to_bits());
        assert_ne!(
            value[0],
            f64::from(0.1f32) + f64::from(0.2f32),
            "the F64 constant must not round through F32"
        );
    }

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
            "LayoutTensorOpCeilFunctionalGeneric",
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
            "LayoutTensorOpFloorFunctionalGeneric",
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
            "LayoutTensorOpRoundFunctionalGeneric",
            "LayoutTensorOpScatterFunctionalGeneric",
            "LayoutTensorOpSelectFunctionalGeneric",
            "LayoutTensorOpSinFunctionalGeneric",
            "LayoutTensorOpSqrtFunctionalGeneric",
            "LayoutTensorOpTruncCastGeneric",
            "LayoutTensorOpTruncDivFunctionalGeneric",
            "LayoutTensorOpTruncFunctionalGeneric",
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

    #[test]
    fn intermediate_limit_prunes_before_search() {
        let mut graph = Graph::new();
        let x = graph.tensor(4, DType::F32);
        let out = x.sin().cos();
        let data = FxHashMap::from_iter([(x.id, vec![0.0f32; 4].into())]);
        let mut options = crate::search::harness_search_options();
        options.max_intermediate_bytes = 15;
        let mut rejected = ReferenceRuntime::load(&graph).unwrap();
        let error = rejected.search(&data, &options).unwrap_err().to_string();
        assert!(error.contains("max_intermediate_bytes=15"), "{error}");

        options.max_intermediate_bytes = 16;
        options.memory_budget_bytes = 31;
        let mut rejected = ReferenceRuntime::load(&graph).unwrap();
        assert!(
            rejected
                .search(&data, &options)
                .unwrap_err()
                .to_string()
                .contains("live memory budget exceeded")
        );
        options.memory_budget_bytes = super::DEFAULT_MEMORY_BUDGET_BYTES;

        let mut runtime = ReferenceRuntime::load(&graph).unwrap();
        runtime.search(&data, &options).unwrap();
        runtime.set_data(x.id, vec![0.0f32; 4]);
        runtime.execute().unwrap();
        assert_eq!(runtime.get_f32(out.id).unwrap(), &vec![1.0; 4]);
    }

    #[test]
    fn intermediate_pruning_uses_bucket_capacity() {
        use luminal::{graph::DimBucket, shape::Symbol};
        let mut graph = Graph::new();
        graph.set_dim('a', 2);
        let x = graph.tensor('a', DType::F32);
        let out = x.sin().cos();
        let mut runtime = ReferenceRuntime::load(&graph).unwrap();
        runtime
            .bind_dim_buckets('a', vec![DimBucket::new(1, 8).representative(2)])
            .unwrap();
        let mut options = crate::search::harness_search_options();
        options.max_intermediate_bytes = 16;
        // A plan must serve the entire bucket. The 8-element intermediate
        // needs 32 bytes even though the representative needs only 8.
        let mut rejected = ReferenceRuntime::load(&graph).unwrap();
        rejected
            .bind_dim_buckets('a', vec![DimBucket::new(1, 8).representative(2)])
            .unwrap();
        let error = rejected
            .search_buckets(
                |dims| {
                    FxHashMap::from_iter([(x.id, vec![0.0f32; dims[&Symbol::from('a')]].into())])
                },
                &options,
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("max_intermediate_bytes=16"), "{error}");
        options.max_intermediate_bytes = 32;
        runtime
            .search_buckets(
                |dims| {
                    FxHashMap::from_iter([(x.id, vec![0.0f32; dims[&Symbol::from('a')]].into())])
                },
                &options,
            )
            .unwrap();
        runtime.set_dim('a', 4);
        runtime.set_data(x.id, vec![0.0f32; 4]);
        runtime.execute().unwrap();
        assert_eq!(runtime.get_f32(out.id).unwrap(), &vec![1.0; 4]);
        // The aggregate budget is also re-evaluated at the current shape.
        runtime.set_memory_budget_bytes(runtime.peak_live_bytes());
        runtime.set_dim('a', 5);
        runtime.set_data(x.id, vec![0.0f32; 5]);
        assert!(
            runtime
                .execute()
                .unwrap_err()
                .to_string()
                .contains("live memory budget exceeded")
        );
    }

    #[test]
    fn intermediate_pruning_preserves_boundary_buffers() {
        let mut graph = Graph::new();
        let x = graph.tensor(4, DType::F32);
        let out = x.sin();
        let data = FxHashMap::from_iter([(x.id, vec![0.0f32; 4].into())]);
        let mut options = crate::search::harness_search_options();
        options.max_intermediate_bytes = 0;
        let mut runtime = ReferenceRuntime::load(&graph).unwrap();
        runtime.search(&data, &options).unwrap();
        runtime.set_data(x.id, vec![0.0f32; 4]);
        runtime.execute().unwrap();
        assert_eq!(runtime.get_f32(out.id).unwrap(), &vec![0.0; 4]);
    }

    #[test]
    fn live_budget_releases_a_long_chain_and_repeated_outputs() {
        let mut graph = Graph::new();
        let x = graph.tensor(4, DType::F32);
        let mut out = x;
        for _ in 0..20 {
            out = out.sin();
        }
        let data = FxHashMap::from_iter([(x.id, vec![0.3f32; 4].into())]);
        let mut rt = ReferenceRuntime::load(&graph).unwrap();
        rt.search(&data, &crate::search::harness_search_options())
            .unwrap();
        let all_bytes: usize = rt
            .plan
            .as_ref()
            .unwrap()
            .buffers
            .values()
            .map(|b| b.layout.span_with(&rt.dims).unwrap() * 4)
            .sum();
        assert!(all_bytes > 128);
        rt.set_memory_budget_bytes(128);
        let mut expected = 0.3f32;
        for _ in 0..20 {
            expected = expected.sin();
        }
        for _ in 0..3 {
            rt.set_data(x.id, vec![0.3f32; 4]);
            rt.execute().unwrap();
            assert_eq!(rt.get_f32(out.id).unwrap(), &vec![expected; 4]);
            assert!(rt.peak_live_bytes() <= 128);
            assert_eq!(rt.storage.len(), 1, "only output survives execution");
        }
        rt.set_memory_budget_bytes(31);
        assert!(
            rt.execute()
                .unwrap_err()
                .to_string()
                .contains("live memory budget exceeded")
        );
    }

    #[test]
    fn depth_first_schedule_preserves_all_edges_and_shared_values() {
        use petgraph::visit::EdgeRef;
        let mut graph = Graph::new();
        let x = graph.tensor(4, DType::F32);
        let shared = x.sin();
        let out = shared.cos() + shared.sin();
        let data = FxHashMap::from_iter([(x.id, vec![0.3f32; 4].into())]);
        let mut rt = ReferenceRuntime::load(&graph).unwrap();
        rt.search(&data, &crate::search::harness_search_options())
            .unwrap();
        let plan = rt.plan.as_ref().unwrap();
        let order = super::depth_first_order(plan).unwrap();
        let positions: FxHashMap<_, _> = order.iter().enumerate().map(|(i, n)| (*n, i)).collect();
        for edge in plan.dag.edge_references() {
            assert!(positions[&edge.source()] < positions[&edge.target()]);
        }
        rt.set_data(x.id, vec![0.3f32; 4]);
        rt.execute().unwrap();
        let v = 0.3f32.sin();
        assert_eq!(rt.get_f32(out.id).unwrap(), &vec![v.cos() + v.sin(); 4]);
    }

    #[test]
    fn every_registered_kernel_has_a_scratch_bound() {
        let ctx = crate::typed_buffer::ReferenceKernelCtx {
            operands: Vec::new(),
            operand_dims: Vec::new(),
            dests: Vec::new(),
            dims: Default::default(),
        };
        for kernel in crate::kernels::reference_kernels() {
            super::kernel_scratch_bytes(kernel.label, &ctx).unwrap();
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
            let b = cx.tensor(3, DType::F32);
            let c = cx.tensor(3, DType::F32);
            let g = cx.tensor(3, DType::F32);
            let e = cx.tensor(3, DType::F32);
            let a = b * c + g;
            let d = (b * c / e).sin();
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
            let x = cx.tensor((2, 3), DType::F32);
            let y = cx.tensor((3, 2), DType::F32);
            let out = x.permute((1, 0)) * y;
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
            let x = cx.tensor(4, DType::F32);
            let y = cx.tensor(4, DType::F32);
            let out = x - y;
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
            let a = cx.tensor((2, 3), DType::F32);
            let b = cx.tensor((3, 4), DType::F32);
            let c = a.matmul(b);
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
                let x = cx.tensor(('a', 2), DType::F32);
                let y = cx.tensor(('a', 2), DType::F32);
                let out = x * y;
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
            let bound = crate::bindings::ReferenceBindings::leaves(&cx2.logical)
                .bind(&cx2.logical)
                .expect("native program");
            assert!(
                bound.text().contains("(IntVar \"a\")"),
                "the model must stay symbolic:\n{}",
                bound.text()
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
            let x = cx.tensor(8, DType::F32);
            let out = x.slice(2..6) + x.slice(1..5);
            (cx, x, out)
        };
        let x_data: Vec<f32> = (0..8).map(|v| (v * v) as f32).collect();

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected = vec![5.0, 13.0, 25.0, 41.0];
        let (cx2, x2, out2) = build();
        let bound = crate::bindings::ReferenceBindings::leaves(&cx2.logical)
            .bind(&cx2.logical)
            .expect("native program");
        assert!(
            bound.text().contains("LogicalIndexMapApply"),
            "the slice must arrive as a view:\n{}",
            bound.text()
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
            let x = cx.tensor((4, 5), DType::F32);
            let out = x.slice((1..3, 2..5));
            (cx, x, out)
        };
        let x_data: Vec<f32> = (0..20).map(|v| v as f32 * 1.5).collect();

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected = vec![10.5, 12.0, 13.5, 18.0, 19.5, 21.0];
        let (cx2, x2, out2) = build();
        let bound = crate::bindings::ReferenceBindings::leaves(&cx2.logical)
            .bind(&cx2.logical)
            .expect("native program");
        assert!(
            bound.text().contains("LogicalIndexMapApply"),
            "the slice must arrive as a view:\n{}",
            bound.text()
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
            let x = cx.tensor(8, DType::F32);
            let plain = x.unfold(3, 2, 1); // windows at 0,2,4
            let y = cx.tensor(10, DType::F32);
            let dilated = y.unfold(3, 2, 2); // effective window 5
            (cx, x, y, plain, dilated)
        };
        let x_data: Vec<f32> = (0..8).map(|v| (v * v) as f32).collect();
        let y_data: Vec<f32> = (0..10).map(|v| v as f32 * 3.0 - 5.0).collect();

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected_plain = vec![0.0, 1.0, 4.0, 4.0, 9.0, 16.0, 16.0, 25.0, 36.0];
        let expected_dilated = vec![-5.0, 1.0, 7.0, 1.0, 7.0, 13.0, 7.0, 13.0, 19.0];
        let (cx2, x2, y2, plain2, dilated2) = build();
        let bound = crate::bindings::ReferenceBindings::leaves(&cx2.logical)
            .bind(&cx2.logical)
            .expect("native program");
        assert!(
            bound.text().contains("LogicalIndexMapApply"),
            "unfold must arrive as a view:\n{}",
            bound.text()
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
                let x = cx.tensor(4, DType::F32);
                let out = x.pad((1, 2), fill);
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
            let bound = crate::bindings::ReferenceBindings::leaves(&cx2.logical)
                .bind(&cx2.logical)
                .expect("native program");
            assert!(
                bound.text().contains("IntCastFromBool"),
                "the mask must ride the bool bridge:\n{}",
                bound.text()
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
                let x = cx.tensor((3, 4), DType::F32);
                let out = x.pad(((1, 0), (2, 1)), fill);
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
            let bound = crate::bindings::ReferenceBindings::leaves(&cx2.logical)
                .bind(&cx2.logical)
                .expect("native program");
            assert!(
                bound.text().contains("IntMax") && bound.text().contains("IntCastFromBool"),
                "clamp view + indicator mask expected:\n{}",
                bound.text()
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
            let data = cx.tensor((3, 4), DType::F32);
            let row = cx.tensor((2, 3), luminal::dtype::DType::Int);
            let col = cx.tensor((2, 3), luminal::dtype::DType::Int);
            let out = data.gather(&[row, col]);
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
        let bound = crate::bindings::ReferenceBindings::leaves(&cx2.logical)
            .bind(&cx2.logical)
            .expect("native program");
        assert!(
            bound.text().contains("(LogicalGather v"),
            "coordinate-form gather expected in the model:\n{}",
            bound.text()
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
            let dest = cx.tensor((3, 4), DType::F32);
            let row = cx.tensor(4, luminal::dtype::DType::Int);
            let col = cx.tensor(4, luminal::dtype::DType::Int);
            let src = cx.tensor(4, DType::F32);
            let out = dest.scatter(&[row, col], src);
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
        let bound = crate::bindings::ReferenceBindings::leaves(&cx2.logical)
            .bind(&cx2.logical)
            .expect("native program");
        assert!(
            bound.text().contains("(LogicalScatter v"),
            "coordinate-form scatter expected in the model:\n{}",
            bound.text()
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
        let data = cx.tensor((3, 4), DType::F32);
        let idx = cx.tensor((2, 3), luminal::dtype::DType::Int);
        let out = data.gather1d(idx);
        assert_eq!(out.dims(), idx.dims(), "out shape = index shape");

        let data_vals: Vec<f32> = (0..12).map(|v| v as f32 * 1.5 + 1.0).collect();
        let idx_ints = [0i32, 5, 11, 7, 3, 2];
        let idx_vals: Vec<i32> = idx_ints.to_vec();
        // Hand golden: data.flat[i] = i*1.5 + 1.
        let expected = vec![1.0, 8.5, 17.5, 11.5, 5.5, 4.0];

        let bound = crate::bindings::ReferenceBindings::leaves(&cx.logical)
            .bind(&cx.logical)
            .expect("native program");
        assert!(
            bound.text().contains("(LogicalGather v"),
            "flat sugar lowers to coordinate gather:\n{}",
            bound.text()
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
        let dest = cx.tensor((2, 6), DType::F32);
        let idx = cx.tensor(4, luminal::dtype::DType::Int);
        let src = cx.tensor(4, DType::F32);
        let out = src.scatter1d(idx, dest);
        assert_eq!(out.dims(), dest.dims(), "out shape = dest shape");

        let dest_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let idx_vals = vec![3i32, 0, 11, 6];
        let src_vals = vec![100.0f32, 200.0, 300.0, 400.0];
        let expected = vec![
            200.0, 1.0, 2.0, 100.0, 4.0, 5.0, 400.0, 7.0, 8.0, 9.0, 10.0, 300.0,
        ];

        let bound = crate::bindings::ReferenceBindings::leaves(&cx.logical)
            .bind(&cx.logical)
            .expect("native program");
        assert!(
            bound.text().contains("(LogicalScatter v"),
            "flat sugar lowers to coordinate scatter:\n{}",
            bound.text()
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
        let out = cx.iota((2, 3), |c| (c[0] * 3 + c[1]) * 2);

        let bound = crate::bindings::ReferenceBindings::leaves(&cx.logical)
            .bind(&cx.logical)
            .expect("native program");
        assert!(
            bound.text().contains("(LogicalIota"),
            "iota expected in the model:\n{}",
            bound.text()
        );
        assert!(
            !bound.text().contains("LogicalIndexMapApply"),
            "no view detour for a bare multi-dim iota:\n{}",
            bound.text()
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
        let out = cx.arange(luminal::shape::IntExpr::from('a'));

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
        let out = cx.iota(3, |c| c[0] + 'a');

        let expected = vec![10i32, 11, 12];
        let ours = run_reference(&cx, &[]);
        assert_eq!(ours.get_i32(out.id).unwrap(), &expected);
    }

    /// ONNX GatherElements over axis 1 (rides the flat sugar + the
    /// iota/normalization index arithmetic), including a negative index.
    #[test]
    fn differential_gather_elements_axis1() {
        let mut cx = Graph::new();
        let data = cx.tensor((2, 3), DType::F32);
        let idx = cx.tensor((2, 2), luminal::dtype::DType::Int);
        let out = data.gather_elements(idx, 1);

        let data_vals: Vec<f32> = (0..6).map(|v| v as f32 * 10.0).collect();
        // out[i, j] = data[i, idx[i, j]]; -1 normalizes to axis extent - 1 = 2.
        let idx_vals = vec![2i32, 0, -1, 1];
        let expected = vec![20.0, 0.0, 50.0, 40.0];

        crate::bindings::ReferenceBindings::leaves(&cx.logical)
            .bind(&cx.logical)
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
        let data = cx.tensor((3, 2), DType::F32);
        let idx = cx.tensor((1, 2), luminal::dtype::DType::Int);
        let upd = cx.tensor((1, 2), DType::F32);
        let out = data.scatter_elements(idx, upd, 0);
        assert_eq!(out.dims(), data.dims());

        let data_vals: Vec<f32> = (0..6).map(|v| v as f32).collect();
        // out[idx[0, j], j] = upd[0, j]: column 0 → row 2, column 1 → row 0.
        let idx_vals = vec![2i32, 0];
        let upd_vals = vec![100.0f32, 200.0];
        let expected = vec![0.0, 200.0, 2.0, 3.0, 100.0, 5.0];

        crate::bindings::ReferenceBindings::leaves(&cx.logical)
            .bind(&cx.logical)
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
        let data = cx.tensor((3, 2), DType::F32);
        let idx = cx.tensor((2, 1), luminal::dtype::DType::Int);
        let upd = cx.tensor((2, 2), DType::F32);
        let out = data.scatter_nd(idx, upd);
        assert_eq!(out.dims(), data.dims());

        let data_vals: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let idx_vals = vec![2i32, 0];
        let upd_vals = vec![100.0f32, 101.0, 200.0, 201.0];
        let expected = vec![200.0, 201.0, 2.0, 3.0, 100.0, 101.0];

        crate::bindings::ReferenceBindings::leaves(&cx.logical)
            .bind(&cx.logical)
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
        let dest = cx.tensor(6, DType::F32);
        let idx = cx.tensor(3, luminal::dtype::DType::Int);
        let src = cx.tensor(3, DType::F32);
        let out = src.scatter1d(idx, dest);

        let bound = crate::bindings::ReferenceBindings::leaves(&cx.logical)
            .bind(&cx.logical)
            .expect("recorder clean");
        let text = format!("{}\n\n{}", crate::assembled_program(), bound.text());
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &text)
            .expect("program runs");
        let serialized = egraph
            .serialize(luminal::prelude::egglog::SerializeConfig::default())
            .egraph;
        let allow = crate::reference_allow_list();
        let extracted = crate::extractor::extract_layout_ir_with_ops_and_matchers(
            &serialized,
            Some(&allow),
            &crate::ops::built_in_matchers(),
        )
        .expect("extracts")
        .expect("plan");
        let dps = luminal::dps::dps_rewrite(&extracted);
        let view =
            luminal::egglog_utils::eclass::EGraphView::new(&serialized, crate::decoder_registry());
        let layouts = luminal::layouts::decode_layout_table(
            &view,
            &dps,
            "test",
            &mut luminal::layouts::LayoutDecodeCache::new(),
        )
        .expect("layouts decode");
        let plan = luminal::bufferize::bufferize(&dps, &layouts).expect("bufferizes");
        let mut rt = crate::ReferenceRuntime::default();
        rt.stage_bindings(&bound.inputs, &bound.outputs);
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

    /// The REFUSAL MECHANISM: a construct the recorder cannot lower
    /// refuses at the construction site, loudly, carrying its
    /// attributable reason — never mistranslates, and never leaves a
    /// half-recorded graph behind to be discovered later. (NO public
    /// frontend construct refuses here — the recorder covers the whole
    /// live surface — so this pokes the mechanism directly; internal
    /// guards like the multi-dim iota tripwire go through the same
    /// door.)
    #[test]
    #[should_panic(expected = "synthetic guard tripped at t0 (mechanism test)")]
    fn recorder_refusals_are_loud() {
        let mut cx = Graph::new();
        let x = cx.tensor((2, 3), DType::F32);
        let _out = x;
        cx.logical
            .refuse("synthetic guard tripped at t0 (mechanism test)");
    }

    /// M3 STEP 1: THE FIRST NATIVE DIFFERENTIAL — the recorder's model +
    /// the reference binding generator, with NO translator anywhere,
    /// against their full search + runtime.
    #[test]
    fn differential_native_recorder_simple_elementwise() {
        let build = || {
            let mut cx = Graph::new();
            let b = cx.tensor(3, DType::F32);
            let c = cx.tensor(3, DType::F32);
            let g = cx.tensor(3, DType::F32);
            let e = cx.tensor(3, DType::F32);
            let a = b * c + g;
            let d = (b * c / e).sin();
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
        crate::bindings::ReferenceBindings::leaves(&cx2.logical)
            .bind(&cx2.logical)
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
            let x = cx.tensor((2, 3), DType::F32);
            let y = cx.tensor((2, 3), DType::F32);
            let out = x.lt(y).cast(DType::F32) * 3.0 + 1.0;
            (cx, x, y, out)
        };
        let x_data = vec![1.0, 5.0, 2.0, 8.0, -1.0, 0.0];
        let y_data = vec![2.0, 4.0, 2.0, 9.0, -2.0, 0.5];

        // GOLDEN (pinned from their ReferenceRuntime before its deletion — Step 4b ruling).
        let expected = vec![4.0, 1.0, 1.0, 4.0, 1.0, 4.0];
        let (cx2, x2, y2, out2) = build();
        let bound = crate::bindings::ReferenceBindings::leaves(&cx2.logical)
            .bind(&cx2.logical)
            .expect("native program");
        assert!(
            bound.text().contains("LogicalLessThan") && bound.text().contains("LogicalCast"),
            "comparison + cast expected in the model:\n{}",
            bound.text()
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
            let x = cx.tensor((2, 3), DType::F32);
            let y = cx.tensor((2, 3), DType::F32);
            let out = x.lt(y);
            (cx, x, y, out)
        };
        let x_data = vec![1.0, 5.0, 2.0, 8.0, -1.0, 0.0];
        let y_data = vec![2.0, 4.0, 2.0, 9.0, -2.0, 0.5];

        // GOLDEN (pinned; x.lt(y) elementwise on the fixed data).
        let expected: Vec<bool> = vec![true, false, false, true, false, true];
        let (cx2, x2, y2, out2) = build();
        let bound = crate::bindings::ReferenceBindings::leaves(&cx2.logical)
            .bind(&cx2.logical)
            .expect("native program");
        assert!(
            bound.text().contains("(LogicalCast v") && bound.text().contains("(Bool8)"),
            "boundary Bool8 cast expected in the binding:\n{}",
            bound.text()
        );
        assert!(
            bound.text().contains("(bits-of (Bool8))"),
            "Bool8 boundary layout width expected:\n{}",
            bound.text()
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
            let a = cx.tensor(12, DType::F32);
            let b = cx.tensor((3, 4), DType::F32);
            let split_out = a.split_dims(0, 4) * b; // [12] -> [3,4]
            let c = cx.tensor((3, 4), DType::F32);
            let d = cx.tensor(12, DType::F32);
            let merge_out = c.merge_dims(0, 1) * d; // [3,4] -> [12]
            let e = cx.tensor((2, 3, 2), DType::F32);
            let f = cx.tensor(12, DType::F32);
            let flatten_out = e.flatten() * f; // [2,3,2] -> [12]
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
            let x = cx.tensor(3, DType::F32);
            let y = cx.tensor(12, DType::F32);
            let out = x.repeat(4) * y;
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
            let x = cx.tensor((2, 3), DType::F32);
            let s = x.sum(0);
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
        let a = cx.tensor(2, DType::I8);
        let b = cx.tensor(2, DType::I8);
        let out = a + b;
        let rt = crate::harness::run_reference(
            &cx,
            &[
                (a.id, TypedBuffer::I8(vec![127, -128])),
                (b.id, TypedBuffer::I8(vec![1, -1])),
            ],
        );
        assert_eq!(rt.get_i8(out.id).unwrap(), &vec![-128i8, 127]);

        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor(2, DType::U8);
        let b = cx.tensor(2, DType::U8);
        let out = a + b;
        let rt = crate::harness::run_reference(
            &cx,
            &[
                (a.id, TypedBuffer::U8(vec![255, 0])),
                (b.id, TypedBuffer::U8(vec![1, 255])),
            ],
        );
        assert_eq!(rt.get_u8(out.id).unwrap(), &vec![0u8, 255]);

        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor(2, DType::I16);
        let b = cx.tensor(2, DType::I16);
        let out = a + b;
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
        let x = cx.tensor(9, DType::Int);
        let out = x.cast(DType::I8);
        let rt = crate::harness::run_reference(&cx, &[(x.id, source.clone().into())]);
        assert_eq!(
            rt.get_i8(out.id).unwrap(),
            &vec![-1i8, 127, -128, -1, 0, 127, -128, -1, 0]
        );

        let mut cx = luminal::graph::Graph::new();
        let x = cx.tensor(9, DType::Int);
        let out = x.cast(DType::U8);
        let rt = crate::harness::run_reference(&cx, &[(x.id, source.clone().into())]);
        assert_eq!(
            rt.get_u8(out.id).unwrap(),
            &vec![255u8, 127, 128, 255, 0, 127, 128, 255, 0]
        );

        let mut cx = luminal::graph::Graph::new();
        let x = cx.tensor(9, DType::Int);
        let out = x.cast(DType::I16);
        let rt = crate::harness::run_reference(&cx, &[(x.id, source.clone().into())]);
        assert_eq!(
            rt.get_i16(out.id).unwrap(),
            &vec![32_767i16, -129, -128, -1, 0, 127, 128, 255, 256]
        );

        // The wide half of the policy, unchanged by the carve-out.
        let mut cx = luminal::graph::Graph::new();
        let x = cx.tensor(1, DType::I64);
        let _out = x.cast(DType::Int);
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        let mut data = FxHashMap::default();
        data.insert(x.id, TypedBuffer::I64(vec![i64::from(i32::MAX) + 1]));
        let err = rt
            .search(&data, &crate::search::harness_search_options())
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("cast i64 -> i32 out of range at value 2147483648"),
            "i64 -> i32 must stay CHECKED (ruling 2026-08-11), got: {message}"
        );
    }

    /// INTEGER `abs()` EXECUTES (main #399's dtype-aware `abs`,
    /// re-expressed). Before this, `abs()` on any integer went through
    /// `relu` -> `maximum_f32` -> `constant_f32(0.0).cast(Int)`, and
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
        let x = cx.tensor(4, DType::I16);
        let out = x.abs();
        let rt = crate::harness::run_reference(
            &cx,
            &[(x.id, TypedBuffer::I16(vec![-3, 0, 5, i16::MIN]))],
        );
        assert_eq!(rt.get_i16(out.id).unwrap(), &vec![3i16, 0, 5, i16::MIN]);

        // Int stays proof-gated (2026-08-11), so the caller attests the
        // range; inside it the answer is exact.
        let mut cx = luminal::graph::Graph::new();
        let x = cx.tensor(4, DType::Int);
        let out = x.abs();
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
            let x = cx.tensor(4, DType::F32);
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
        let x = cx.tensor(4, DType::F64);
        let out = x.sqrt();

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
        let mask = cx.tensor(4, DType::Bool);
        let x = cx.tensor(4, DType::F32);
        let out = mask.cast(DType::F32) * x;

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
        let mask2 = cx2.tensor(4, DType::Bool);
        let x2 = cx2.tensor(4, DType::F32);
        let out2 = mask2.cast(DType::F32) * x2;
        let _ = out2;
        let mut rt2 = ReferenceRuntime::load(&cx2).expect("native load");
        let mut data = FxHashMap::default();
        data.insert(mask2.id, TypedBuffer::bool8(vec![1u8, 0, 1, 0]).unwrap());
        data.insert(x2.id, x_vals.clone().into());
        rt2.search(&data, &crate::search::harness_search_options())
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
        let idx = cx.tensor(5, DType::Int);
        let out = idx * 3usize;
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.bind_value_range(idx.id, 0, 4).expect("range binds");
        let mut data = FxHashMap::default();
        data.insert(idx.id, vec![0i32, 1, 2, 3, 4].into());
        rt.search(&data, &crate::search::harness_search_options())
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
        let a = cx.tensor(1, DType::Int);
        let b = cx.tensor(1, DType::Int);
        let _out = a + b;
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        let mut data = FxHashMap::default();
        data.insert(a.id, vec![1i32].into());
        data.insert(b.id, vec![2i32].into());
        let err = rt
            .search(&data, &crate::search::harness_search_options())
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
        let a = cx.tensor(1, DType::Int);
        let b = cx.tensor(1, DType::Int);
        let out = a + b;
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.bind_value_range(a.id, 0, 1000).expect("range binds");
        rt.bind_value_range(b.id, 0, 1000).expect("range binds");
        rt.search(&data, &crate::search::harness_search_options())
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
        let a = cx.tensor(4, DType::Int);
        let b = cx.tensor(4, DType::Int);
        let out = a.trunc_div(b);
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.bind_value_range(a.id, -100, 100).expect("range binds");
        rt.bind_value_range(b.id, 2, 4).expect("range binds");
        let mut data = FxHashMap::default();
        data.insert(a.id, vec![7i32, -7, 100, -1].into());
        data.insert(b.id, vec![2i32, 2, 3, 4].into());
        rt.search(&data, &crate::search::harness_search_options())
            .expect("proven trunc-div implements");
        rt.set_data(a.id, vec![7i32, -7, 100, -1]);
        rt.set_data(b.id, vec![2i32, 2, 3, 4]);
        rt.execute().expect("proven trunc-div executes");
        assert_eq!(rt.get_i32(out.id).unwrap(), &vec![3i32, -3, 33, 0]);

        // Without the divisor attestation the same graph REFUSES: the
        // bounds admit zero, so no implementation exists to find.
        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor(1, DType::Int);
        let b = cx.tensor(1, DType::Int);
        let _out = a.trunc_div(b);
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        let mut data = FxHashMap::default();
        data.insert(a.id, vec![7i32].into());
        data.insert(b.id, vec![2i32].into());
        let err = rt
            .search(&data, &crate::search::harness_search_options())
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("no candidate genome"),
            "expected the unattested refusal, got: {err:#}"
        );
    }
    /// BUCKETED (D7, moved here from core's `implementation_search` as a
    /// RUNTIME-LEVEL test, #420/#422 rejoin Phase 1): two buckets over
    /// 'a', each validated bucket-wide (range seeds) and searched at its
    /// representative; selection covers runtime dims; each bucket's plan
    /// agrees with the runtime at its representative.
    #[test]
    fn bucketed_search_validates_searches_and_selects() {
        use luminal::graph::DimBucket;
        use luminal::shape::Symbol;

        let mut cx = Graph::new();
        cx.set_dim('a', 3);
        let x = cx.tensor(('a', 2), DType::F32);
        let y = cx.tensor(('a', 2), DType::F32);
        let out = x * y;

        let data_for = |rep: &luminal::shape::DynMap| {
            let n = rep[&Symbol::from('a')] * 2;
            let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
            data.insert(
                x.id,
                (0..n).map(|v| v as f32 + 1.0).collect::<Vec<f32>>().into(),
            );
            data.insert(
                y.id,
                (0..n).map(|v| v as f32 * 0.5).collect::<Vec<f32>>().into(),
            );
            data
        };

        let mut rt = ReferenceRuntime::load(&cx).expect("records + loads");
        rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4), DimBucket::new(5, 9)])
            .expect("disjoint sorted buckets bind");
        let plans = rt
            .search_buckets(data_for, &crate::search::harness_search_options())
            .expect("bucketed search completes");
        assert_eq!(plans.len(), 2, "one plan per bucket");

        // Selection covers each bucket; out-of-range dims select nothing.
        let ranges: Vec<_> = plans.iter().map(|plan| plan.ranges.clone()).collect();
        let mut dims = luminal::shape::DynMap::default();
        dims.insert(Symbol::from('a'), 3usize);
        assert_eq!(
            crate::search::select_bucket(rt.bucket_plans(), &dims)
                .unwrap()
                .ranges[&Symbol::from('a')],
            (2, 4),
            "{ranges:?}"
        );
        dims.insert(Symbol::from('a'), 7usize);
        assert_eq!(
            crate::search::select_bucket(rt.bucket_plans(), &dims)
                .unwrap()
                .ranges[&Symbol::from('a')],
            (5, 9)
        );
        dims.insert(Symbol::from('a'), 20usize);
        assert!(crate::search::select_bucket(rt.bucket_plans(), &dims).is_none());

        // Numeric agreement at each bucket's representative, through the
        // ladder: set_dim picks the plan, execute runs it.
        let representatives: Vec<usize> = rt
            .bucket_plans()
            .iter()
            .map(|plan| plan.representative[&Symbol::from('a')])
            .collect();
        for rep in representatives {
            // GOLDEN (computed: out = x * y with x[i] = i+1, y[i] = i*0.5
            // — the data_for closure's values at this representative).
            let n = rep * 2;
            let expected: Vec<f32> = (0..n)
                .map(|v| (v as f32 + 1.0) * (v as f32 * 0.5))
                .collect();

            rt.set_dim('a', rep);
            let mut pins = luminal::shape::DynMap::default();
            pins.insert(Symbol::from('a'), rep);
            for (id, values) in data_for(&pins) {
                rt.set_data(id, values);
            }
            rt.execute()
                .expect("bucket plan executes at representative");
            let ours = rt.get_f32(out.id).unwrap();
            assert_eq!(ours.len(), expected.len());
            for (index, (lhs, rhs)) in ours.iter().zip(&expected).enumerate() {
                assert!(
                    (lhs - rhs).abs() <= 1e-5 * rhs.abs().max(1.0),
                    "representative {rep} element {index}: ours {lhs} vs theirs {rhs}"
                );
            }
        }

        // SYMBOLIC PLANS (the Phase 1 limitation, lifted): a bucket's
        // plan stays an expression in `a`, so a NON-representative value
        // inside the same bucket now executes correctly instead of being
        // refused. `a = 4` sits in the first bucket [2, 4] but is not its
        // representative (3).
        let non_representative = 4usize;
        rt.set_dim('a', non_representative);
        let mut pins = luminal::shape::DynMap::default();
        pins.insert(Symbol::from('a'), non_representative);
        for (id, values) in data_for(&pins) {
            rt.set_data(id, values);
        }
        rt.execute()
            .expect("a symbolic bucket plan executes at a non-representative value");
        let n = non_representative * 2;
        let expected: Vec<f32> = (0..n)
            .map(|v| (v as f32 + 1.0) * (v as f32 * 0.5))
            .collect();
        let ours = rt.get_f32(out.id).unwrap();
        assert_eq!(ours.len(), expected.len());
        for (index, (lhs, rhs)) in ours.iter().zip(&expected).enumerate() {
            assert!(
                (lhs - rhs).abs() <= 1e-5 * rhs.abs().max(1.0),
                "a = {non_representative} element {index}: ours {lhs} vs theirs {rhs}"
            );
        }
    }

    /// DEFINITION OF DONE for symbolic plans: ONE bucketed search produces
    /// a plan whose spans/extents stay expressions in the symbolic dim,
    /// and that single plan then executes at SEVERAL dim values with NO
    /// re-search. The values cross bucket boundaries, so this also proves
    /// [`crate::search::select_bucket`] picks the covering symbolic plan
    /// per execution while the searched plans stay fixed.
    ///
    /// Golden values are computed INDEPENDENTLY from the scalar formula
    /// `out[i] = x[i] * y[i] + x[i]` with `x[i] = i + 1`, `y[i] = i / 2`.
    #[test]
    fn one_searched_symbolic_plan_executes_at_many_dim_values() {
        use luminal::graph::DimBucket;
        use luminal::shape::Symbol;

        let mut cx = Graph::new();
        cx.set_dim('a', 3);
        let x = cx.tensor(('a', 2), DType::F32);
        let y = cx.tensor(('a', 2), DType::F32);
        let out = x * y + x;

        let data_for = |n: usize| {
            let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
            data.insert(
                x.id,
                (0..n).map(|v| v as f32 + 1.0).collect::<Vec<f32>>().into(),
            );
            data.insert(
                y.id,
                (0..n).map(|v| v as f32 * 0.5).collect::<Vec<f32>>().into(),
            );
            data
        };

        let mut rt = ReferenceRuntime::load(&cx).expect("records + loads");
        rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4), DimBucket::new(5, 9)])
            .expect("disjoint sorted buckets bind");
        rt.search_buckets(
            |rep| data_for(rep[&Symbol::from('a')] * 2),
            &crate::search::harness_search_options(),
        )
        .expect("bucketed search completes ONCE");
        assert_eq!(rt.bucket_plans().len(), 2, "one searched plan per bucket");

        // SEVERAL values per bucket, representatives and non-representatives
        // alike. Nothing below searches again: set_dim + set_data + execute
        // only, so every value after the initial search reuses a searched
        // symbolic plan.
        for a in [2usize, 3, 4, 5, 6, 7, 8, 9] {
            rt.set_dim('a', a);
            for (id, values) in data_for(a * 2) {
                rt.set_data(id, values);
            }
            rt.execute()
                .expect("the searched symbolic plan executes at every dim value");
            assert_eq!(rt.bucket_plans().len(), 2, "execution must never re-search");
            // INDEPENDENT GOLDEN: scalar formula, not another runtime.
            let expected: Vec<f32> = (0..a * 2)
                .map(|v| {
                    let xv = v as f32 + 1.0;
                    let yv = v as f32 * 0.5;
                    xv * yv + xv
                })
                .collect();
            let ours = rt.get_f32(out.id).unwrap();
            assert_eq!(ours.len(), expected.len(), "a = {a}");
            for (index, (lhs, rhs)) in ours.iter().zip(&expected).enumerate() {
                assert!(
                    (lhs - rhs).abs() <= 1e-5 * rhs.abs().max(1.0),
                    "a = {a} element {index}: ours {lhs} vs theirs {rhs}"
                );
            }
        }
    }

    /// OP-RECORD GEOMETRY audit: an `arange` whose extent (and iota
    /// expression) is the symbolic dim retains `Var("a")` in the searched
    /// op record. One bucketed search then serves every value because the
    /// kernel evaluates that expression against the runtime's PER-CALL
    /// dims instead of a representative's literals.
    #[test]
    fn symbolic_iota_reuses_one_plan_across_dims() {
        use luminal::graph::DimBucket;
        use luminal::shape::IntExpr;

        let mut cx = Graph::new();
        cx.set_dim('a', 5);
        let out = cx.arange(IntExpr::from('a'));

        let mut rt = ReferenceRuntime::load(&cx).expect("records + loads");
        rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4), DimBucket::new(5, 9)])
            .expect("buckets bind");
        rt.search_buckets(
            |_| FxHashMap::default(),
            &crate::search::harness_search_options(),
        )
        .expect("bucketed search completes once");
        assert_eq!(rt.bucket_plans().len(), 2, "one plan per bucket");
        // The searched plan must actually be symbolic, or this test would
        // pass by way of a literal plan and prove nothing about the
        // op-record path.
        assert!(
            rt.bucket_plans()[0]
                .outcome
                .best_plan
                .buffers
                .values()
                .any(|buffer| buffer.layout.literal_span_elements().is_none()),
            "the searched arange plan must keep a symbolic span"
        );

        for a in [2usize, 3, 4, 5, 7, 9] {
            rt.set_dim('a', a);
            rt.execute().expect("symbolic iota plan executes");
            // INDEPENDENT GOLDEN: arange(a) is 0..a.
            let expected: Vec<i32> = (0..a as i32).collect();
            assert_eq!(rt.get_i32(out.id).unwrap(), &expected, "a = {a}");
        }
    }

    /// Buckets must partition: overlap is refused, not resolved
    /// first-wins, and neither is an unsorted pair.
    #[test]
    fn overlapping_or_unsorted_buckets_are_refused() {
        use luminal::graph::DimBucket;

        let mut cx = Graph::new();
        cx.set_dim('a', 3);
        let x = cx.tensor(('a', 2), DType::F32);
        let _out = x * x;

        let mut rt = ReferenceRuntime::load(&cx).expect("records + loads");
        let err = rt
            .bind_dim_buckets('a', vec![DimBucket::new(2, 6), DimBucket::new(5, 9)])
            .expect_err("overlapping buckets must refuse");
        assert!(
            format!("{err:#}").contains("sorted and disjoint"),
            "{err:#}"
        );

        let err = rt
            .bind_dim_buckets('a', vec![DimBucket::new(5, 9), DimBucket::new(2, 4)])
            .expect_err("unsorted buckets must refuse");
        assert!(
            format!("{err:#}").contains("sorted and disjoint"),
            "{err:#}"
        );

        let err = rt
            .bind_dim_buckets('a', vec![])
            .expect_err("an empty bucket list must refuse");
        assert!(format!("{err:#}").contains("no buckets"), "{err:#}");
    }

    /// BUCKETS AND RANGE BINDINGS ARE EXCLUSIVE PER DIM, BOTH ORDERS,
    /// LOUDLY — and the range need not be tight. Both seed the same
    /// `IntVar`'s `lower-bound-of` / `upper-bound-of`, which MERGE
    /// (`max` / `min`) rather than error, so two seed sets on one dim
    /// would silently INTERSECT: a bucket [5, 9] under a prior range
    /// [2, 8] would be validated over [5, 8] while the plan claims 9.
    #[test]
    fn a_dim_cannot_be_both_range_bound_and_bucketed() {
        use luminal::graph::DimBucket;

        let graph = || {
            let mut cx = Graph::new();
            cx.set_dim('a', 3);
            let x = cx.tensor(('a', 2), DType::F32);
            let _out = x * x;
            cx
        };

        // A NON-TIGHT range, then buckets: the case `dims` could not see.
        let mut rt = ReferenceRuntime::load(&graph()).expect("records + loads");
        rt.bind_dyn_range('a', 2, 8).expect("range binds");
        let err = rt
            .bind_dim_buckets('a', vec![DimBucket::new(5, 9)])
            .expect_err("a range-bound dim must not take buckets");
        assert!(
            format!("{err:#}").contains("already carries a range binding [2, 8]"),
            "{err:#}"
        );

        // A tight [n, n] pin is a range binding too.
        let mut rt = ReferenceRuntime::load(&graph()).expect("records + loads");
        rt.bind_dyn_range('a', 3, 3).expect("pin binds");
        let err = rt
            .bind_dim_buckets('a', vec![DimBucket::new(2, 4)])
            .expect_err("a pinned dim must not take buckets");
        assert!(
            format!("{err:#}").contains("already carries a range binding [3, 3]"),
            "{err:#}"
        );

        // And the other order.
        let mut rt = ReferenceRuntime::load(&graph()).expect("records + loads");
        rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4)])
            .expect("buckets bind");
        let err = rt
            .bind_dyn_range('a', 2, 8)
            .expect_err("a bucketed dim must not take a range binding");
        assert!(format!("{err:#}").contains("has buckets bound"), "{err:#}");
    }
}
