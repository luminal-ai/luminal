//! The CUDA-lite runtime: the reference ladder
//! (`load → bind_* → search → set_data → execute → get_*`) with the
//! search claiming only this backend's codegen inventory and execution
//! delegated to the `device` module.
//!
//! Everything up to `execute` is device-free and runs anywhere: load
//! accumulates the native program parts, bind_* appends bounds seeds,
//! search assembles + saturates + runs THIS crate's genetic search
//! ([`crate::search`]) with OUR allow list, ranking candidates by the
//! device-free heuristic ([`crate::heuristic`] — a weak static prior,
//! not a measurement). Only `execute` requires the `device` feature and
//! a CUDA device.

use crate::host_buffer::HostBuffer;
use anyhow::{anyhow, bail, Context, Result};
use luminal::bufferize::BufferIrGraph;

use crate::search::{CompileOptions, SearchOutcome};
use luminal::graph;
use luminal::layouts::DecodedLayout;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal::shape;

/// The accumulated pre-search program parts (the reference runtime's
/// NativeSpec is private; this is the same accumulation rebuilt from
/// the public `bound_parts` seam).
struct NativeParts {
    pre_schedule: String,
    input_slots: Vec<graph::InputSlot>,
    output_slots: Vec<graph::OutputSlot>,
    post_checks: String,
    labeled_checks: Vec<(String, String)>,
    binding_seeds: String,
}

/// A `Default` instance holds NO program and NO op vocabulary: every
/// ladder method past `load` refuses it by name (`load before search`),
/// so the empty registry a default carries is never the thing a caller
/// searches under. `load`/`load_with_cublaslt`/`load_with_registry` are
/// the only ways to get a usable one.
#[derive(Default)]
pub struct CudaRuntime {
    native: Option<NativeParts>,
    /// THE INSTANCE'S OP VOCABULARY (Phase 2, 2026-09-03): the matcher
    /// column of the registry this runtime was LOADED with, and the
    /// allow list derived from that same registry. Both are decided once,
    /// at [`CudaRuntime::load_with_registry`], and never consulted from a
    /// crate-level constant again — which op set a runtime assembles,
    /// saturates, searches and claims with is a property of the instance,
    /// selectable by its caller.
    ///
    /// The matchers are HELD rather than rebuilt because `dyn OpMatcher`
    /// is not clonable and one instance runs many extractions (every
    /// genome, and with buckets every Cartesian combination); everything
    /// downstream borrows this slice.
    matchers: Vec<Box<dyn luminal::layout_ir::OpMatcher>>,
    /// The claim set derived from that same registry — see
    /// [`CudaRuntime::allow_list_over`] for the three classes.
    allow: Vec<&'static str>,
    plan: Option<BufferIrGraph<DecodedLayout>>,
    /// Host-staged input payloads by BufferLit id, H2D'd at execute.
    staged: FxHashMap<i64, HostBuffer>,
    /// Host copies of each output slot's BACKING buffer plus its elected
    /// layout, filled by execute (D2H) — the escape-and-disclose fetch,
    /// keyed by slot index (an escaped slot's backing buffer is a minted
    /// allocation with no BufferLit, so slot order is the stable key).
    outputs_host: FxHashMap<usize, (HostBuffer, luminal::bufferize::OutputBinding<DecodedLayout>)>,
    input_buffers: FxHashMap<NodeIndex, i64>,
    /// Bound output tensor → its slot index (program slot order).
    output_index: FxHashMap<NodeIndex, usize>,
    /// BUCKETS (D7, 2026-09-03): per-dim intervals one search covers.
    /// Empty = the ordinary single-pin ladder, unchanged.
    dim_buckets: std::collections::BTreeMap<shape::Symbol, Vec<graph::DimBucket>>,
    /// One finished plan per Cartesian bucket combination.
    bucket_plans: Vec<crate::search::BucketPlan>,
    /// The dim values this runtime currently holds — every `[n, n]`
    /// `bind_dyn_range` pin plus whatever [`Self::set_dim`] sets. With
    /// buckets bound this is what picks the plan at execute time.
    dims: shape::DynMap,
    /// EVERY dim [`Self::bind_dyn_range`] has bound, tight or not, with
    /// the interval it was given. `dims` records only the `[n, n]` pins,
    /// so it cannot answer the exclusivity question: buckets and range
    /// bindings must refuse each other in BOTH orders, and a non-tight
    /// range under a later bucket would otherwise seed the same `IntVar`
    /// twice and INTERSECT under the bounds lattice's merge rather than
    /// refuse.
    range_bound: std::collections::BTreeMap<shape::Symbol, (u64, u64)>,
    /// THE PERSISTENT DEVICE (#422, rejoin Phase 3): context, stream,
    /// NVRTC module cache and the arena slab, created on the first
    /// [`Self::execute`] and kept for the runtime's life — "each runtime
    /// remembers its own buffer hygiene". `None` until then, so
    /// `Default` still gives a device-free runtime that plans and
    /// searches on any host.
    #[cfg(feature = "device")]
    device: Option<crate::device::CudaDevice>,
}

impl CudaRuntime {
    /// Record the graph's native program under the DEFAULT op registry
    /// ([`crate::ops::cuda_registry`]). Saturation happens in
    /// [`CudaRuntime::search`].
    pub fn load(graph: &graph::Graph) -> Result<Self> {
        Self::load_with_registry(graph, crate::ops::cuda_registry())
    }

    /// [`CudaRuntime::load`] with the cuBLASLt marker vocabulary
    /// enabled: search assembles the marker's egg snippets and may
    /// elect the four host-call contracts. EXPLICIT OPT-IN for now: the
    /// view-arity tripwire that used to kill saturation on real graphs
    /// is fixed (it now checks the map literal); at the 2×4 harness
    /// budget the search can still die of exhaustion on the collapse's
    /// re-description 2-cycle — loudly, never a wrong plan. The 2D
    /// canonical matmul form searches and elects green; real graphs
    /// elect at 12×16.
    ///
    /// It is now exactly [`CudaRuntime::load_with_registry`] over the
    /// [`crate::ops::cuda_registry_with_cublaslt`] preset — a named
    /// convenience, not a mode.
    pub fn load_with_cublaslt(graph: &graph::Graph) -> Result<Self> {
        Self::load_with_registry(graph, crate::ops::cuda_registry_with_cublaslt())
    }

    /// THE CONFIGURABLE LOAD (ruling 2026-09-03: *"you should select the
    /// allowed ops when you initialize the runtime ... You should not
    /// need to edit CL in order to modify this"*): record the graph's
    /// native program and FIX this instance's op vocabulary to the given
    /// registry. Everything downstream — the assembled egglog preamble,
    /// the saturation, the extraction matcher set, and the derived allow
    /// list the search claims through — reads that registry and nothing
    /// else, so two runtimes in one process may hold different op sets.
    ///
    /// Build the argument with [`crate::ops::cuda_registry_filtered`]
    /// (narrow either preset by label or constructor) or by pushing
    /// [`crate::ops::RegisteredOp::new`] rows onto one. A row whose op
    /// is neither kernel-bearing, plan-transparent, nor host-dispatchable
    /// is simply not claimable — it never reaches the allow list, so the
    /// search refuses loudly instead of electing something the device
    /// cannot run.
    pub fn load_with_registry(
        graph: &graph::Graph,
        registry: Vec<crate::ops::RegisteredOp>,
    ) -> Result<Self> {
        let (pre_schedule, input_slots, output_slots, post_checks, labeled_checks) = graph
            .logical
            .bound_parts(&crate::bindings::CudaBindings)
            .map_err(|e| anyhow!(e))?;
        // Derive the claim set BEFORE the rows are consumed: the allow
        // list reads the prototypes, the search reads the matchers.
        let allow = Self::allow_list_over(&registry);
        let matchers = registry.into_iter().map(|entry| entry.matcher).collect();
        Ok(Self {
            native: Some(NativeParts {
                pre_schedule,
                input_slots,
                output_slots,
                post_checks,
                labeled_checks,
                binding_seeds: String::new(),
            }),
            matchers,
            allow,
            ..Self::default()
        })
    }

    /// The matcher vocabulary this instance assembles/searches with —
    /// LENT, never rebuilt (see the field's note).
    fn matchers(&self) -> &[Box<dyn luminal::layout_ir::OpMatcher>] {
        &self.matchers
    }

    /// The claim set THIS instance searches under — the allow list
    /// derived from the registry it was loaded with. Named
    /// `active_allow_list` because the static
    /// [`CudaRuntime::allow_list`] (the default preset's) already owns
    /// the plain name and inherent methods may not share one.
    pub fn active_allow_list(&self) -> &[&'static str] {
        &self.allow
    }

    /// Seed interval bounds for a dynamic dimension (facts, never pins:
    /// `[n, n]` is how a caller pins).
    pub fn bind_dyn_range(
        &mut self,
        var: impl Into<shape::Symbol>,
        lower: u64,
        upper: u64,
    ) -> Result<()> {
        let name = var.into();
        anyhow::ensure!(
            !self.dim_buckets.contains_key(&name),
            "dim `{name}` has buckets bound; a bucketed dim is seeded per bucket \
             and must not carry a second range binding"
        );
        let native = self
            .native
            .as_mut()
            .ok_or_else(|| anyhow!("load before bind"))?;
        native.binding_seeds.push_str(&format!(
            "(set (lower-bound-of (IntVar \"{name}\")) (bigint {lower}))\n\
             (set (upper-bound-of (IntVar \"{name}\")) (bigint {upper}))\n"
        ));
        // EVERY range binding is remembered, so `bind_dim_buckets` can
        // refuse this dim whatever the interval was.
        self.range_bound.insert(name, (lower, upper));
        // A tight [n, n] binding IS a pin: remember it too, so a bucketed
        // plan's representative records the whole assignment.
        if lower == upper {
            self.dims.insert(name, lower as usize);
        }
        Ok(())
    }

    /// BIND BUCKETS for a dynamic dimension (D7, 2026-09-03): a set of
    /// disjoint intervals, each of which gets its own searched plan.
    /// `search` then runs one search per Cartesian combination and
    /// `execute` picks the covering plan from the current dims.
    ///
    /// THE BUCKETS MUST PARTITION CLEANLY: non-empty, sorted by `min`,
    /// and pairwise disjoint. Overlap is REFUSED rather than resolved
    /// first-wins — two plans that both claim a value is an ambiguity in
    /// the caller's model, and picking one silently is how a graph ends
    /// up running the plan its author did not mean.
    pub fn bind_dim_buckets(
        &mut self,
        dim: impl Into<shape::Symbol>,
        buckets: Vec<graph::DimBucket>,
    ) -> Result<()> {
        let dim = dim.into();
        anyhow::ensure!(!buckets.is_empty(), "dim `{dim}` was given no buckets");
        if let Some((lo, hi)) = self.range_bound.get(&dim) {
            anyhow::bail!(
                "dim `{dim}` already carries a range binding [{lo}, {hi}] from \
                 bind_dyn_range; a bucketed dim is seeded per bucket and must not \
                 carry a second range binding"
            );
        }
        anyhow::ensure!(
            !self.dims.contains_key(&dim),
            "dim `{dim}` already has a value from set_dim; bind buckets before \
             setting the execution dim"
        );
        for pair in buckets.windows(2) {
            anyhow::ensure!(
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
    /// bound this is what selects the plan.
    pub fn set_dim(&mut self, dim: impl Into<shape::Symbol>, value: usize) {
        self.dims.insert(dim.into(), value);
    }

    /// The finished per-bucket plans (empty until a bucketed `search`).
    pub fn bucket_plans(&self) -> &[crate::search::BucketPlan] {
        &self.bucket_plans
    }

    /// Pick and load the bucket plan covering the current dims.
    ///
    /// THE STATIC-PLAN REFUSAL (the Phase 1 limitation, stated rather
    /// than solved): a bucket's winning plan was searched at ONE pin and
    /// carries LITERAL spans, so it allocates and indexes for that pin
    /// and nothing else. Executing it at another value inside the same
    /// bucket would silently run the representative's geometry over the
    /// caller's data, so it is refused by name. Lifting this needs
    /// symbolic plans (spans as expressions) and the capacity contract
    /// that goes with them.
    fn select_bucket_plan(&mut self) -> Result<()> {
        let Some(plan) = crate::search::select_bucket(&self.bucket_plans, &self.dims) else {
            let covered: Vec<_> = self.bucket_plans.iter().map(|p| p.ranges.clone()).collect();
            bail!(
                "no bucket covers dims {:?}; the searched buckets are {covered:?}",
                self.dims
            );
        };
        for (dim, representative) in &plan.representative {
            if let Some(value) = self.dims.get(dim) {
                anyhow::ensure!(
                    value == representative,
                    "bucket {:?} was searched at `{dim} = {representative}` and its plan is \
                     STATIC at that pin (plan spans are literals), but this runtime is set \
                     to `{dim} = {value}`. Re-search at this pin, or pick a bucket whose \
                     representative is it. Running the representative's plan here would \
                     silently use the wrong geometry — the open item is symbolic plans \
                     (spans as expressions) and the capacity contract that goes with them.",
                    plan.ranges
                );
            }
        }
        self.plan = Some(plan.plan.clone());
        Ok(())
    }

    /// The ops this runtime claims: the CUDA analogue of
    /// `reference_allow_list()` — three classes, all derived, never
    /// name-listed (M4 Phase 5 + Train 3):
    ///
    ///  * KERNEL-BEARING: matcher constructors whose label has a
    ///    codegen row — claimable because the device can execute them.
    ///  * PLAN-TRANSPARENT: constructors whose registered PROTOTYPE's
    ///    declared effects prove the planner folds them before any
    ///    kernel is needed (see [`crate::plan_transparent`]) —
    ///    claimable because nothing ever executes.
    ///  * HOST-CALL DISPATCHABLE (Train 3): constructors whose
    ///    prototype the executor dispatches as a host library call
    ///    (`cublasLtMatmul`) — claimable because the device runs them
    ///    without any NVRTC kernel (see
    ///    [`crate::ops::cublaslt::host_dispatchable`]).
    ///
    /// THIS STATIC IS THE DEFAULT PRESET'S claim set — the same
    /// derivation over [`crate::ops::cuda_registry`], for callers with no
    /// graph in hand. A LOADED instance claims what its own registry
    /// derives: [`CudaRuntime::active_allow_list`].
    pub fn allow_list() -> Vec<&'static str> {
        Self::allow_list_over(&crate::ops::cuda_registry())
    }

    /// [`CudaRuntime::allow_list`] over the marker-enabled registry —
    /// the claim set a [`CudaRuntime::load_with_cublaslt`] search uses.
    pub fn allow_list_with_cublaslt() -> Vec<&'static str> {
        Self::allow_list_over(&crate::ops::cuda_registry_with_cublaslt())
    }

    fn allow_list_over(registry: &[crate::ops::RegisteredOp]) -> Vec<&'static str> {
        let labels: Vec<&'static str> = crate::kernels::cuda_kernels()
            .iter()
            .map(|k| k.label)
            .collect();
        registry
            .iter()
            .filter(|entry| {
                let ctor = entry.matcher.egglog_constructor();
                let stripped = ctor.trim_start_matches("LayoutTensorOp");
                let kernel_bearing = labels.iter().any(|label| {
                    stripped == *label || stripped.trim_end_matches("Generic") == *label
                });
                kernel_bearing
                    || crate::plan_transparent(entry.prototype.as_ref())
                    || crate::ops::cublaslt::host_dispatchable(entry.prototype.as_ref())
            })
            .map(|entry| entry.matcher.egglog_constructor())
            .collect()
    }

    /// The SATURATED, SERIALIZED e-graph this runtime's search reads —
    /// exactly the assembly [`CudaRuntime::search`] performs (this
    /// backend's matcher vocabulary + the bound program + the schedule),
    /// run to saturation and serialized, WITHOUT the genetic search. A
    /// test seam: estate pins assert on the e-graph the search sees
    /// (which constructors were minted, which spellings a layout class
    /// holds) rather than on an election that depends on the budget.
    pub fn saturated_egraph(&self) -> Result<luminal::prelude::egraph_serialize::EGraph> {
        let (serialized, _program) = self.assemble_and_saturate()?;
        Ok(serialized)
    }

    /// Assemble the program under this runtime's bindings and matcher
    /// vocabulary, run it to saturation, and serialize. Shared by
    /// [`CudaRuntime::search`] and [`CudaRuntime::saturated_egraph`] so
    /// the two can never see different programs. On saturation failure
    /// the labeled post-checks are re-run in isolation to name the door,
    /// mirroring the reference runtime.
    fn assemble_and_saturate(
        &self,
    ) -> Result<(
        luminal::prelude::egraph_serialize::EGraph,
        graph::LogicalProgram,
    )> {
        let native = self
            .native
            .as_ref()
            .ok_or_else(|| anyhow!("load before search"))?;
        let program = graph::LogicalProgram {
            text: format!(
                "{}{}{}{}",
                native.pre_schedule,
                native.binding_seeds,
                crate::bindings::CudaBindings::SCHEDULE,
                native.post_checks
            ),
            input_slots: native.input_slots.clone(),
            output_slots: native.output_slots.clone(),
        };
        let full = format!(
            "{}\n\n{}",
            luminal::egglog_snippet::assembled_program_for(self.matchers()),
            program.text
        );
        let mut egraph = luminal::egglog_snippet::new_egraph();
        if let Err(err) = egraph.parse_and_run_program(None, &full) {
            // Name the door: re-saturate without checks, then probe each
            // labeled check alone.
            let mut doors = Vec::new();
            let unchecked = format!(
                "{}\n\n{}\n{}\n{}",
                luminal::egglog_snippet::assembled_program_for(self.matchers()),
                native.pre_schedule,
                native.binding_seeds,
                crate::bindings::CudaBindings::SCHEDULE
            );
            let mut probe = luminal::egglog_snippet::new_egraph();
            if probe.parse_and_run_program(None, &unchecked).is_ok() {
                for (label, text) in &native.labeled_checks {
                    if probe.parse_and_run_program(None, text).is_err() {
                        doors.push(label.clone());
                    }
                }
            }
            if doors.is_empty() {
                return Err(err).context("cuda-lite saturation failed");
            }
            bail!("shape contracts failed:\n  - {}", doors.join("\n  - "));
        }
        let serialized = egraph.serialize(luminal::prelude::egglog::SerializeConfig::default());
        Ok((serialized.egraph, program))
    }

    /// Assemble, saturate, and search — with THIS backend's allow list.
    /// On saturation failure the labeled post-checks are re-run in
    /// isolation to name the door, mirroring the reference runtime.
    pub fn search(
        &mut self,
        input_data: &FxHashMap<NodeIndex, HostBuffer>,
        options: &CompileOptions,
    ) -> Result<SearchOutcome> {
        let (serialized, program) = self.assemble_and_saturate()?;
        let native = self
            .native
            .as_ref()
            .ok_or_else(|| anyhow!("load before search"))?;

        // The caller's payloads are CHECKED here always, because binding
        // a tensor that is not an input of the loaded program is a
        // caller bug under either evaluator. Whether they go FURTHER
        // depends on how the search prices candidates: the device-free
        // heuristic (D6, 2026-09-03) never runs anything and so needs
        // nothing staged, while device profiling (Phase 4) executes each
        // candidate and needs exactly these bytes.
        for tensor in input_data.keys() {
            assert!(
                program
                    .input_slots
                    .iter()
                    .any(|slot| slot.tensor == *tensor),
                "tensor {tensor:?} is not a bound input"
            );
        }
        #[cfg(not(feature = "device"))]
        anyhow::ensure!(
            !options.profile_on_device,
            "device profiling requested but cuda-lite was built without the `device` \
             feature: this host can search by the heuristic, but a request to MEASURE \
             must not be answered with a prior"
        );

        // THE SEARCH-TIME STAGING (Phase 4), by BufferLit id and BY
        // REFERENCE — the ladder's `set_data` staging is a separate,
        // later step and is untouched. A full-size model's weights are
        // gigabytes; the search must borrow them, never copy them.
        #[cfg(feature = "device")]
        let staged_for_search: FxHashMap<i64, &HostBuffer> = if options.profile_on_device {
            program
                .input_slots
                .iter()
                .filter_map(|slot| input_data.get(&slot.tensor).map(|data| (slot.buffer, data)))
                .collect()
        } else {
            FxHashMap::default()
        };
        // THE DEVICE IS CREATED LAZILY, here or at the first `execute`
        // (Phase 3's persistent device, unchanged): a search that ranks
        // by the heuristic still touches no CUDA API at all.
        #[cfg(feature = "device")]
        if options.profile_on_device && self.device.is_none() {
            self.device = Some(crate::device::CudaDevice::new(0)?);
        }

        // THIS INSTANCE's claim set, derived at load from THIS
        // instance's registry — no crate-level default is consulted.
        let allow = self.allow.clone();
        // FIELD BORROWS, not `self.matchers()`: the device evaluator
        // holds `&mut self.device` at the same time, and only disjoint
        // FIELD borrows can coexist — a `&self` method would borrow the
        // whole runtime.
        let matchers = &self.matchers;
        let mut evaluator = {
            #[cfg(feature = "device")]
            {
                if options.profile_on_device {
                    crate::search::Evaluator::Device {
                        device: self
                            .device
                            .as_mut()
                            .expect("the device was just created if it was missing"),
                        staged: &staged_for_search,
                    }
                } else {
                    crate::search::Evaluator::Heuristic
                }
            }
            #[cfg(not(feature = "device"))]
            {
                crate::search::Evaluator::Heuristic
            }
        };

        // Own matchers, own allow list, own ranking: nothing in this
        // search touches another runtime.
        //
        // WHAT `search` RETURNS is a pair: the outcome to report, and the
        // plan to install. They are no longer the same thing (Phase 5):
        // the outcome is the genetic search's report, while the installed
        // plan is whichever FINALIST the bucket lattice selected under the
        // aggregate device budget. With no budget set they coincide,
        // which is why every existing caller sees the trajectory it had.
        let (outcome, unbucketed_plan, searched_buckets) = if self.dim_buckets.is_empty() {
            let mut outcome = crate::search::search_implementations(
                &serialized,
                &program,
                options,
                Some(allow.clone()),
                matchers,
                evaluator.reborrow(),
            )?;
            // THE UNBUCKETED LATTICE (Phase 5) — a lattice over ONE
            // bucket, so unbucketed and bucketed installs run the same
            // code. Main's "one designed difference" from its pre-#420
            // behaviour, adopted for the same reason: whether the
            // installed plan fits the caller's device budget is a
            // property of what is installed, and an unbucketed install is
            // a set of one.
            let finalists = vec![crate::finalists::Finalists::new(
                "the search",
                &serialized,
                Some(allow.clone()),
                matchers,
                outcome.ranked.clone(),
                Some(outcome.best_plan.clone()),
            )];
            let (selected, rejections) =
                crate::search::select_finalist_set(finalists, options, &mut evaluator)?;
            outcome.lattice_rejections = rejections;
            let (_, finalist) = selected
                .into_iter()
                .next()
                .expect("a one-bucket lattice selects exactly one finalist");
            (outcome, Some(finalist.plan), Vec::new())
        } else {
            // BUCKETED (D7): one search per Cartesian combination, each
            // validated bucket-wide before its representative is
            // searched. The caller's data is staged ONCE and every
            // bucket's search borrows the same map — a bucket only
            // changes the dim seeds, never the payloads.
            let assembly = crate::search::BucketAssembly {
                assembled_program: &luminal::egglog_snippet::assembled_program_for(matchers),
                pre_schedule: &native.pre_schedule,
                binding_seeds: &native.binding_seeds,
                schedule: crate::bindings::CudaBindings::SCHEDULE,
                post_checks: &native.post_checks,
                input_slots: &native.input_slots,
                output_slots: &native.output_slots,
                base_dims: &self.dims,
            };
            let plans = crate::search::bucketed_search_implementations(
                &assembly,
                &self.dim_buckets,
                options,
                Some(allow),
                matchers,
                evaluator,
            )?;
            let first = plans
                .first()
                .map(|plan| plan.outcome.clone())
                .ok_or_else(|| anyhow!("bucketed search produced no plans"))?;
            (first, None, plans)
        };
        if !searched_buckets.is_empty() {
            self.bucket_plans = searched_buckets;
        }

        let native = self
            .native
            .as_ref()
            .ok_or_else(|| anyhow!("load before search"))?;
        self.input_buffers = native
            .input_slots
            .iter()
            .map(|slot| (slot.tensor, slot.buffer))
            .collect();
        self.output_index = native
            .output_slots
            .iter()
            .enumerate()
            .map(|(index, slot)| (slot.tensor, index))
            .collect();
        if let Some(plan) = unbucketed_plan {
            // THE LATTICE'S CHOICE, not `outcome.best_plan` (Phase 5).
            // Unconstrained they are the same plan — the rank-0 finalist
            // re-extracts the winning genome — but the installed one is
            // the one that passed the aggregate check.
            self.plan = Some(plan);
        } else {
            // With buckets the plan is chosen at execute time; load
            // eagerly only if the runtime already sits at a covered pin.
            let _ = self.select_bucket_plan();
        }
        Ok(outcome)
    }

    /// Stage input payload for a bound tensor (host side; H2D happens
    /// inside execute).
    pub fn set_data(&mut self, tensor: NodeIndex, data: impl Into<HostBuffer>) {
        let Some(&buffer) = self.input_buffers.get(&tensor) else {
            panic!("set_data on a tensor with no input binding");
        };
        self.staged.insert(buffer, data.into());
    }

    /// Run the plan on the CUDA device. Requires the `device` feature
    /// and an available device; refuses loudly otherwise.
    pub fn execute(&mut self) -> Result<()> {
        // With buckets bound, the plan is chosen HERE, from the current
        // dims (see [`Self::select_bucket_plan`] for the static-plan
        // refusal). Without them nothing changes.
        if !self.bucket_plans.is_empty() {
            self.select_bucket_plan()?;
        }
        #[cfg(feature = "device")]
        {
            // The device is created ONCE and reused: the module cache
            // keeps every NVRTC compilation from the previous calls, and
            // the arena slab keeps the bytes (grow-only, never parked).
            if self.device.is_none() {
                self.device = Some(crate::device::CudaDevice::new(0)?);
            }
            let plan = self
                .plan
                .as_ref()
                .ok_or_else(|| anyhow!("search before execute"))?;
            // SERVING KEEPS THE SLAB (#422 policy, Phase 4): nothing
            // here releases it — only the search does, between
            // candidates.
            let staged: FxHashMap<i64, &HostBuffer> =
                self.staged.iter().map(|(lit, data)| (*lit, data)).collect();
            let device = self
                .device
                .as_mut()
                .expect("the device was just created if it was missing");
            let outputs = crate::device::execute_plan(device, plan, &staged)?;
            self.outputs_host = outputs;
            Ok(())
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = self
                .plan
                .as_ref()
                .ok_or_else(|| anyhow!("search before execute"))?;
            bail!(
                "cuda-lite built without the `device` feature: plans can be \
                 searched and inspected but not executed on this host"
            )
        }
    }

    /// Read back an output tensor's f32 payload (already D2H'd by
    /// execute), interpreting it as row-major over the value's dims.
    ///
    /// NO DENSENESS CHECK HERE (ruling 2026-09-01). This is the record.
    ///
    /// What was checked: that the output binding's elected layout is the
    /// flat index over the value's dims, so element `k` of the value is
    /// at flat index `k` of the backing and this `Vec<f32>` IS the value.
    /// A view-elected (escaped) output was refused loudly and directed to
    /// [`Self::fetch`] + [`Self::output_layout`].
    ///
    /// Why it went. Austin, 2026-09-01, ruling the CL-4b write fence out
    /// of the backend: "this is something that needs to be expressed in
    /// egglog by matching only only to right major contiguous layouts
    /// ouputs or something, we should not have it in the codebase here.
    /// delete it. same with the get_f32 path."
    ///
    /// WHAT THE LANDED EGGLOG CONSTRAINT DOES AND DOES NOT COVER. The
    /// write-capability guard (same day) makes non-dense KERNEL
    /// destinations unelectable. It deliberately does NOT constrain
    /// output slots: a view remains electable as an output
    /// (escape-and-disclose), and on such an output this dense-shaped
    /// signature hands over the BACKING bytes silently — a same-numel
    /// weld such as a transpose has the right LENGTH and the wrong
    /// ORDER, so the caller reads plausible, wrong numbers. The
    /// escape-and-disclose path ([`Self::fetch`] under
    /// [`Self::output_layout`], read by [`crate::layouts::dense_f32`])
    /// remains correct for every layout and is what callers that cannot
    /// assume a dense output should use.
    pub fn get_f32(&self, tensor: NodeIndex) -> Result<Vec<f32>> {
        let (payload, _) = self.fetch(tensor)?;
        payload.as_f32()
    }

    /// [`Self::get_f32`] for 32-bit integer outputs.
    pub fn get_i32(&self, tensor: NodeIndex) -> Result<Vec<i32>> {
        let (payload, _) = self.fetch(tensor)?;
        payload.as_i32()
    }

    /// [`Self::get_f32`] for 64-bit integer outputs.
    pub fn get_i64(&self, tensor: NodeIndex) -> Result<Vec<i64>> {
        let (payload, _) = self.fetch(tensor)?;
        payload.as_i64()
    }

    /// [`Self::get_f32`] for boolean outputs: the two-legal-code bytes.
    pub fn get_bool8(&self, tensor: NodeIndex) -> Result<&[u8]> {
        let (payload, _) = self.fetch(tensor)?;
        payload.as_bool8()
    }

    /// The universal escape-and-disclose fetch: the output slot's backing
    /// bytes plus its [`luminal::bufferize::OutputBinding`] (the elected
    /// layout).
    pub fn fetch(
        &self,
        tensor: NodeIndex,
    ) -> Result<(
        &HostBuffer,
        &luminal::bufferize::OutputBinding<DecodedLayout>,
    )> {
        let index = self
            .output_index
            .get(&tensor)
            .ok_or_else(|| anyhow!("tensor has no output binding"))?;
        match self.outputs_host.get(index) {
            Some((data, binding)) => Ok((data, binding)),
            None => bail!("execute before fetch"),
        }
    }

    /// The slot's elected layout alone (see [`Self::fetch`]).
    pub fn output_layout(
        &self,
        tensor: NodeIndex,
    ) -> Result<&luminal::bufferize::OutputBinding<DecodedLayout>> {
        Ok(self.fetch(tensor)?.1)
    }

    /// The searched plan, for inspection and tests.
    pub fn plan(&self) -> Option<&BufferIrGraph<DecodedLayout>> {
        self.plan.as_ref()
    }
}
