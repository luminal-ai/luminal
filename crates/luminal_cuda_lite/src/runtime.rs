//! The CUDA-lite runtime: the reference ladder
//! (`load → bind_* → search → set_data → execute → get_*`) with the
//! search claiming only this backend's codegen inventory and execution
//! delegated to the `device` module.
//!
//! Everything up to `execute` is device-free BY DEFAULT and runs
//! anywhere: load accumulates the native program parts, bind_* appends
//! bounds seeds, search assembles + saturates + runs THIS crate's
//! genetic search ([`crate::search`]) with OUR allow list, ranking
//! candidates by the device-free heuristic ([`crate::heuristic`] — a
//! weak static prior, not a measurement). Two things need the `device`
//! feature and a CUDA device: `execute`, and a `search` with
//! [`crate::search::CompileOptions::profile_on_device`] set (Phase 4),
//! which creates the device lazily, stages the caller's payloads, and
//! ranks by measured time (the `profile` module, `device` only) — on a
//! device-free build it refuses by name rather than falling back to the
//! prior.

use crate::host_buffer::HostBuffer;
use anyhow::{Context, Result, anyhow, bail};
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
/// searches under. `load`/`load_with_registry` are the only ways to get
/// a usable one.
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
    /// The `(sort, constructor)` DECODERS for that same matcher set:
    /// core's built-ins plus whatever the instance's matchers declare.
    /// Every layout decode in this runtime reads through these, and the
    /// assembly tripwire checks each saturated program against them. A
    /// `Default` runtime carries an empty registry, like `matchers`.
    decoders: luminal::egglog_utils::eclass::ConstructorRegistry,
    plan: Option<BufferIrGraph<DecodedLayout>>,
    /// Host-staged input payloads by BufferLit id, H2D'd at execute.
    staged: FxHashMap<i64, HostBuffer>,
    residents: crate::resident::ResidentBindings,
    device_budget_bytes: Option<usize>,
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
    selected_bucket: Option<usize>,
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
    /// NVRTC module cache and the arena slab, created lazily — by the
    /// first device-profiled [`Self::search`] (Phase 4) or by the first
    /// [`Self::execute`], whichever comes first — and kept for the
    /// runtime's life, "each runtime remembers its own buffer hygiene".
    /// `None` until then, so `Default` still gives a device-free runtime
    /// that plans and searches by the heuristic on any host.
    #[cfg(feature = "device")]
    device: Option<crate::device::CudaDevice>,
}

impl CudaRuntime {
    /// Record the graph's native program under the DEFAULT op registry
    /// ([`crate::ops::cuda_registry`]) — which since the 2026-09-04
    /// ruling INCLUDES the four cuBLASLt marker contracts, so a default
    /// search assembles the marker's egg snippets and may elect the
    /// host-call route. Saturation happens in [`CudaRuntime::search`].
    ///
    /// For the DECOMPOSED route on purpose (every matmul as CL's own
    /// multiply/reduce kernels), load with
    /// [`crate::ops::cuda_registry_without_cublaslt`].
    pub fn load(graph: &graph::Graph) -> Result<Self> {
        Self::load_with_registry(graph, crate::ops::cuda_registry())
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
    /// [`crate::ops::RegisteredOp::new`] rows onto one. An op is claimable
    /// if its prototype is plan-transparent or its executable DPS form
    /// exposes [`crate::KernelOp`] or [`crate::HostOp`]. A matching label
    /// alone does not grant a claim. External ops supply the same interfaces
    /// as built-in ops without changing this runtime's dispatch code.
    ///
    /// ONE EXCEPTION TO ROW-BY-ROW SELECTION: the four cuBLASLt marker
    /// rows are ONE vocabulary, declared and minted by the Base row's
    /// snippets. A registry holding a non-Base marker row without Base is
    /// REFUSED here — such a row would be claimed but never declared, an
    /// op that cannot be elected under a claim set that says it can.
    pub fn load_with_registry(
        graph: &graph::Graph,
        registry: Vec<crate::ops::RegisteredOp>,
    ) -> Result<Self> {
        let (pre_schedule, input_slots, output_slots, post_checks, labeled_checks) = graph
            .logical
            .bound_parts(&crate::bindings::CudaBindings)
            .map_err(|e| anyhow!(e))?;
        // THE FOUR cuBLASLt MARKER ROWS ARE ONE VOCABULARY. Only the
        // Base row emits snippets, and that one snippet set declares all
        // four constructors and every minting rule. A registry holding a
        // non-Base marker WITHOUT Base would derive a claim for an op the
        // assembled program never declares and never mints: claimed,
        // un-electable, and `active_allow_list()` — the check this
        // module's doc recommends — would say it is available. Refuse the
        // configuration at load instead, keyed on constructor names.
        {
            use crate::ops::cublaslt::CublasLtForm;
            let has = |ctor: &str| {
                registry
                    .iter()
                    .any(|entry| entry.matcher.egglog_constructor() == ctor)
            };
            if let Some(orphan) = CublasLtForm::ALL
                .into_iter()
                .filter(|form| *form != CublasLtForm::Base)
                .find(|form| has(form.constructor_name()))
            {
                anyhow::ensure!(
                    has(CublasLtForm::Base.constructor_name()),
                    "registry holds the cuBLASLt `{}` row without the Base row `{}`: \
                     the Base row declares and mints the whole marker vocabulary, so \
                     the four marker rows must be kept or dropped together",
                    orphan.constructor_name(),
                    CublasLtForm::Base.constructor_name()
                );
            }
        }
        // Derive the claim set BEFORE the rows are consumed: the allow
        // list reads the prototypes, the search reads the matchers.
        let allow = Self::allow_list_over(&registry);
        let matchers: Vec<Box<dyn luminal::layout_ir::OpMatcher>> =
            registry.into_iter().map(|entry| entry.matcher).collect();
        // The decoders for that same vocabulary. Refused here if two
        // matchers claim one `(sort, constructor)` — a registration bug,
        // named at load rather than at the first decode.
        let decoders = luminal::egglog_snippet::decoder_registry_for(&matchers)?;
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
            decoders,
            ..Self::default()
        })
    }

    /// The matcher vocabulary this instance assembles/searches with —
    /// LENT, never rebuilt (see the field's note).
    fn matchers(&self) -> &[Box<dyn luminal::layout_ir::OpMatcher>] {
        &self.matchers
    }

    /// THIS INSTANCE'S CONSTRUCTOR DECODERS — what to build an
    /// [`luminal::egglog_utils::eclass::EGraphView`] over
    /// [`Self::saturated_egraph`] with, so a caller can ask a class
    /// which spellings it holds.
    pub fn decoders(&self) -> &luminal::egglog_utils::eclass::ConstructorRegistry {
        &self.decoders
    }

    /// The claim set THIS instance searches under — the allow list
    /// derived from the registry it was loaded with. Named
    /// `active_allow_list` because the static
    /// [`CudaRuntime::allow_list`] (the default preset's) already owns
    /// the plain name and inherent methods may not share one.
    pub fn active_allow_list(&self) -> &[&'static str] {
        &self.allow
    }

    fn invalidate_plans(&mut self) {
        self.plan = None;
        self.bucket_plans.clear();
        self.selected_bucket = None;
        self.outputs_host.clear();
        #[cfg(feature = "device")]
        if let Some(device) = &mut self.device {
            device.release_slab();
        }
    }

    /// Seed interval bounds for a dynamic dimension (facts, never pins:
    /// `[n, n]` is how a caller pins).
    pub fn bind_dyn_range(
        &mut self,
        var: impl Into<shape::Symbol>,
        lower: u64,
        upper: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.residents.inputs.is_empty(),
            "configure dimension bounds before residency"
        );
        let name = var.into();
        let (lower, upper) = self
            .range_bound
            .get(&name)
            .map(|(lo, hi)| (lower.max(*lo), upper.min(*hi)))
            .unwrap_or((lower, upper));
        anyhow::ensure!(
            lower <= upper,
            "empty dimension range for `{name}`: [{lower}, {upper}]"
        );
        anyhow::ensure!(upper <= i64::MAX as u64, "dimension range exceeds i64");
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
        self.invalidate_plans();
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
        anyhow::ensure!(
            self.residents.inputs.is_empty(),
            "configure dimension bounds before residency"
        );
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
        self.invalidate_plans();
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

    /// Pick the range-valid plan covering the current dimensions.
    fn select_bucket_plan(&mut self) -> Result<()> {
        let index = self
            .bucket_plans
            .iter()
            .position(|p| {
                p.ranges
                    .iter()
                    .all(|(s, (lo, hi))| self.dims.get(s).is_some_and(|v| v >= lo && v <= hi))
            })
            .ok_or_else(|| anyhow!("no bucket covers dims {:?}", self.dims))?;
        self.selected_bucket = Some(index);
        Ok(())
    }

    /// Cumulative graph/arena counters, available after first device use.
    #[cfg(feature = "device")]
    pub fn graph_stats(&self) -> Option<crate::device::GraphStats> {
        self.device.as_ref().map(|d| d.stats())
    }

    /// The ops this runtime claims: the CUDA analogue of
    /// `reference_allow_list()` — three classes, all derived, never
    /// name-listed (M4 Phase 5 + Train 3):
    ///
    ///  * KERNEL-BEARING: the registered prototype's DPS form exposes
    ///    [`crate::KernelOp`], which supplies its codegen implementation.
    ///  * PLAN-TRANSPARENT: the prototype's declared effects prove the
    ///    planner folds it (see [`crate::plan_transparent`]).
    ///  * HOST-CALL DISPATCHABLE: the prototype's DPS form exposes
    ///    [`crate::HostOp`], which supplies its host launch implementation.
    ///
    /// THIS STATIC IS THE DEFAULT PRESET'S claim set — the same
    /// derivation over [`crate::ops::cuda_registry`] (markers included),
    /// for callers with no graph in hand. A LOADED instance claims what
    /// its own registry derives: [`CudaRuntime::active_allow_list`]. For
    /// the decomposed preset's claim set, load with
    /// [`crate::ops::cuda_registry_without_cublaslt`] and read that
    /// instance's `active_allow_list`.
    pub fn allow_list() -> Vec<&'static str> {
        Self::allow_list_over(&crate::ops::cuda_registry())
    }

    fn allow_list_over(registry: &[crate::ops::RegisteredOp]) -> Vec<&'static str> {
        registry
            .iter()
            .filter(|entry| {
                let prototype = entry.prototype.as_ref();
                if crate::plan_transparent(prototype) {
                    return true;
                }
                // Execution belongs to the DPS form carried by buffer plans.
                // Derive claims from that same interface, never from a label.
                let dps = prototype.to_dps();
                let executable = dps.as_deref().unwrap_or(prototype);
                crate::as_kernel_op(executable).is_some() || crate::as_host_op(executable).is_some()
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
        // THE ASSEMBLY TRIPWIRE: this program's every constructor of a
        // decoded sort has exactly one decoder, checked against the LIVE
        // schema before anything reads a serialized class.
        self.decoders.check(&egraph)?;
        let serialized = egraph.serialize(luminal::prelude::egglog::SerializeConfig::default());
        Ok((serialized.egraph, program))
    }

    /// Assemble, saturate, and search — with THIS backend's allow list.
    /// On saturation failure the labeled post-checks are re-run in
    /// isolation to name the door, mirroring the reference runtime.
    ///
    /// With `options.profile_on_device` this needs the `device` feature
    /// and a CUDA device: it creates the device lazily and ranks by
    /// measured time (see the `profile` module, `device` only).
    pub fn search(
        &mut self,
        input_data: &FxHashMap<NodeIndex, HostBuffer>,
        options: &CompileOptions,
    ) -> Result<SearchOutcome> {
        anyhow::ensure!(
            self.residents.inputs.is_empty(),
            "create a new runtime to re-search a resident program"
        );
        self.invalidate_plans();
        let mut resolved_options = options.clone();
        resolved_options.shapes.bounds = self
            .range_bound
            .iter()
            .map(|(s, (lo, hi))| Ok((*s, (usize::try_from(*lo)?, usize::try_from(*hi)?))))
            .collect::<Result<_>>()?;
        resolved_options.shapes.values = self.dims.clone();
        for (s, (lo, hi)) in &resolved_options.shapes.bounds {
            resolved_options
                .shapes
                .values
                .entry(*s)
                .or_insert(lo + (hi - lo) / 2);
        }
        let options = &resolved_options;
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
        //
        // The slot list is the LOAD-TIME one, `native.input_slots` — the
        // same list a rendered program's `input_slots` is cloned from, and
        // the one the bucketed ladder has, which renders no base program
        // at all.
        for tensor in input_data.keys() {
            assert!(
                native.input_slots.iter().any(|slot| slot.tensor == *tensor),
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
            native
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

        // THE BASE PROGRAM IS RENDERED AND SATURATED ONLY FOR THE
        // SINGLE-PIN LADDER. A bucketed dim carries no seeds in this
        // render — `bind_dyn_range` is refused on it, and the intervals
        // are seeded per bucket inside `bucketed_search_implementations`
        // — so an unbucketed render validates nothing the bucketed
        // search will use, and an authoring check that NEEDS bounds
        // (`reduce_max`'s `require_extent_at_least`, an iota's value
        // bounds) would refuse the whole bucketed search over a program
        // every per-bucket render accepts. It is also a full fixpoint
        // whose result the bucketed arm discards. The reference ladder's
        // `search_buckets` never rendered one.
        //
        // Computed HERE, before the evaluator borrows `self.device`
        // mutably: `assemble_and_saturate` takes `&self`.
        let base = if self.dim_buckets.is_empty() {
            Some(self.assemble_and_saturate()?)
        } else {
            None
        };

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
        let (outcome, unbucketed_plan, searched_buckets) = if let Some((serialized, program)) = base
        {
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
            let finalists = vec![
                crate::finalists::Finalists::new(
                    "the search",
                    &serialized,
                    Some(allow.clone()),
                    matchers,
                    outcome.ranked.clone(),
                    Some(outcome.best_plan.clone()),
                )
                .with_shapes(options.shapes.clone()),
            ];
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
            // searched and validated over the complete interval. The caller's data is staged ONCE and every
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
                base_dims: &options.shapes.values,
                decoders: &self.decoders,
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
        self.device_budget_bytes = options.device_budget_bytes;
        self.bucket_plans = searched_buckets;
        self.selected_bucket = None;

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

    /// Resolve public graph handles to this compiled program's boundary IDs.
    pub fn input_buffer(&self, tensor: NodeIndex) -> Result<i64> {
        self.input_buffers
            .get(&tensor)
            .copied()
            .ok_or_else(|| anyhow!("no input binding for {tensor:?}"))
    }
    pub fn output_slot_index(&self, tensor: NodeIndex) -> Result<usize> {
        self.output_index
            .get(&tensor)
            .copied()
            .ok_or_else(|| anyhow!("no output binding for {tensor:?}"))
    }

    /// Keep this input in the shared device arena between executions. Its
    /// shape must be static and its boundary must be read-only. Call after
    /// search and before the first execute; set_data uploads it only when changed.
    pub fn retain_input(&mut self, tensor: NodeIndex) -> Result<()> {
        #[cfg(feature = "device")]
        anyhow::ensure!(
            self.device.as_ref().is_none_or(|d| !d.is_installed()),
            "configure residency before execution"
        );
        let lit = *self
            .input_buffers
            .get(&tensor)
            .ok_or_else(|| anyhow!("no input binding for {tensor:?}"))?;
        self.residents.inputs.insert(lit);
        Ok(())
    }

    /// Route an output into a resident input after each execution. Feedback
    /// outputs stay on device and are unavailable through fetch. Their elected
    /// layout must equal the static, contiguous input boundary layout.
    pub fn bind_feedback(&mut self, input: NodeIndex, output: NodeIndex) -> Result<()> {
        let slot = *self
            .output_index
            .get(&output)
            .ok_or_else(|| anyhow!("no output binding for {output:?}"))?;
        let lit = *self
            .input_buffers
            .get(&input)
            .ok_or_else(|| anyhow!("no input binding for {input:?}"))?;
        anyhow::ensure!(
            !self.residents.feedback.contains_key(&slot)
                && !self.residents.feedback.values().any(|v| *v == lit),
            "duplicate feedback endpoint"
        );
        self.retain_input(input)?;
        self.residents.feedback.insert(slot, lit);
        Ok(())
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
        // Select a range-valid plan using the current dimensions.
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
            anyhow::ensure!(
                self.plan.is_some() || !self.bucket_plans.is_empty(),
                "search before execute"
            );
            let device = self.device.as_mut().unwrap();
            if !device.is_installed() {
                let base_bounds: crate::symbolic::Bounds = self
                    .range_bound
                    .iter()
                    .map(|(s, (lo, hi))| Ok((*s, (usize::try_from(*lo)?, usize::try_from(*hi)?))))
                    .collect::<Result<_>>()?;

                let plans = if self.bucket_plans.is_empty() {
                    vec![(self.plan.as_ref().unwrap().clone(), base_bounds)]
                } else {
                    self.bucket_plans
                        .iter()
                        .map(|p| {
                            let mut bounds = base_bounds.clone();
                            bounds.extend(p.ranges.iter().map(|(k, v)| (*k, *v)));
                            (p.plan.clone(), bounds)
                        })
                        .collect()
                };
                device.install_resident_with_budget(
                    plans,
                    self.residents.clone(),
                    self.device_budget_bytes,
                )?;
            }
            let bucket = self.selected_bucket.unwrap_or(0);
            let staged = self.staged.iter().map(|(lit, data)| (*lit, data)).collect();
            let outputs = device.execute(bucket, &staged, &self.dims)?;
            self.outputs_host = outputs;
            self.staged
                .retain(|lit, _| !self.residents.inputs.contains(lit));
            Ok(())
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = self
                .plan()
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
        self.selected_bucket
            .and_then(|i| self.bucket_plans.get(i))
            .map(|p| &p.plan)
            .or(self.plan.as_ref())
    }
}
