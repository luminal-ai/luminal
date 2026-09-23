//! The CUDA-lite runtime: the reference ladder
//! (`load → bind_* → search → set_data → execute → get_*`) with the
//! search claiming only this backend's codegen inventory and execution
//! delegated to the `device` module.
//!
//! Loading, shape binding and saturation can run on any host. Search and
//! execution require the `device` feature and a CUDA GPU. Search compiles,
//! warms and measures every distinct candidate using the caller's inputs.

use crate::host_buffer::HostBuffer;
use anyhow::{Context, Result, anyhow, bail, ensure};
use luminal::bufferize::BufferIrGraph;

use crate::search::{CompileOptions, SearchOutcome};
use luminal::graph;
use luminal::layouts::DecodedLayout;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal::shape;

/// What `load` captured: the bound program (model text, this runtime's
/// boundary, the post-schedule checks) plus whatever the `bind_*` calls
/// accumulate before `search` assembles and saturates.
struct NativeParts {
    bound: crate::bindings::BoundProgram,
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
    /// WHERE EACH BOUNDARY BUFFER'S STORAGE LIVES, stated by the bindings
    /// at load ([`crate::bindings::Placement`]): the arena's resident set,
    /// and the buffers that are the caller's own device memory.
    residents: crate::resident::ResidentBindings,
    device_budget_bytes: Option<usize>,
    /// The caller's device address for each External buffer, supplied before
    /// each execution ([`Self::set_device_ptr`]). Keyed by buffer id, which
    /// is also how the bindings spell aliasing: a mutation sink and its
    /// target share one buffer and therefore one pointer.
    device_ptrs: FxHashMap<i64, (u64, usize)>,
    /// Host copies of each output slot's BACKING buffer plus its elected
    /// layout, filled by execute (D2H) — the escape-and-disclose fetch,
    /// keyed by slot index (an escaped slot's backing buffer is a minted
    /// allocation with no BufferLit, so slot order is the stable key).
    outputs_host: FxHashMap<usize, (HostBuffer, luminal::bufferize::OutputBinding<DecodedLayout>)>,
    input_buffers: FxHashMap<NodeIndex, i64>,
    /// Bound output tensor → its buffer ids, in binding order. A value bound
    /// on two buffers has two, and [`Self::output_buffer`] refuses rather
    /// than picking one.
    output_buffers: FxHashMap<NodeIndex, Vec<i64>>,
    /// Bound output tensor → its slot indices, in binding order. A value
    /// bound on two buffers has two slots and is read back by slot, not
    /// by tensor: [`Self::output_slot_index`] refuses the ambiguity
    /// rather than picking one.
    output_slots: FxHashMap<NodeIndex, Vec<usize>>,
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
    /// that can load, bind and inspect saturated graphs on any host.
    #[cfg(feature = "device")]
    device: Option<crate::device::CudaDevice>,
    /// A caller-allocated arena for the NEXT execution. `None` means the
    /// device allocates and owns its slab, the pre-existing standalone
    /// behaviour.
    #[cfg(feature = "device")]
    external_arena: Option<(u64, usize)>,
    /// Raw `CUstream` to run on (the caller's current stream). `None` runs on
    /// a stream this runtime owns.
    #[cfg(feature = "device")]
    borrowed_stream: Option<u64>,
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
    ///
    /// THE DEFAULT BINDING is dense: every input read-only and
    /// host-staged on its own row-major buffer, every leaf read-write on
    /// its own. [`CudaRuntime::load_with`] takes the caller's instead.
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
        Self::load_with(
            graph,
            crate::bindings::CudaBindings::leaves(&graph.logical),
            registry,
        )
    }

    /// LOAD a recorded graph under the CALLER's binding — which values
    /// enter and leave, through which buffers, at which layout, and
    /// which of them stay device-resident. The tensor→buffer maps are
    /// live from here, so [`Self::set_data`] needs no search first, and
    /// the resident set is known before the first plan is priced.
    pub fn load_with(
        graph: &graph::Graph,
        bindings: crate::bindings::CudaBindings,
        registry: Vec<crate::ops::RegisteredOp>,
    ) -> Result<Self> {
        let bound = bindings
            .bind(&graph.logical)
            .map_err(|reason| anyhow!("load refused: {reason}"))?;
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
        // THE BOUNDARY MAPS ARE LIVE AT LOAD, not at search: which
        // tensor stages onto which buffer, which slot reads a value
        // back, and which buffers the arena keeps are all statements the
        // bindings already made.
        let input_buffers = bound.inputs.iter().map(|b| (b.value, b.buffer)).collect();
        let mut output_slots: FxHashMap<NodeIndex, Vec<usize>> = FxHashMap::default();
        let mut output_buffers: FxHashMap<NodeIndex, Vec<i64>> = FxHashMap::default();
        for (index, bound) in bound.outputs.iter().enumerate() {
            output_slots.entry(bound.value).or_default().push(index);
            output_buffers
                .entry(bound.value)
                .or_default()
                .push(bound.buffer);
        }
        let residents = bound.residents();
        Ok(Self {
            native: Some(NativeParts {
                bound,
                binding_seeds: String::new(),
            }),
            matchers,
            allow,
            decoders,
            input_buffers,
            output_slots,
            output_buffers,
            residents,
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

    /// Refuse a reconfiguration that would throw away installed device
    /// state. The arena slab holds the installed graphs and every
    /// resident input's uploaded home, and anything that invalidates
    /// plans releases it — so once the device is installed, a caller who
    /// wants a different program wants a different runtime.
    fn ensure_not_installed(&self, what: &str) -> Result<()> {
        #[cfg(feature = "device")]
        anyhow::ensure!(
            self.device.as_ref().is_none_or(|d| !d.is_installed()),
            "{what} after execution: create a new runtime — the device arena holds \
             the installed plans and the resident inputs' data"
        );
        #[cfg(not(feature = "device"))]
        let _ = what;
        Ok(())
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
        self.ensure_not_installed("binding a dimension range")?;
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
        self.ensure_not_installed("binding dimension buckets")?;
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
        crate::search::SearchProgram,
    )> {
        let native = self
            .native
            .as_ref()
            .ok_or_else(|| anyhow!("load before search"))?;
        let program = crate::search::SearchProgram {
            text: native.bound.text_with_seeds(&native.binding_seeds),
            inputs: native.bound.inputs.clone(),
            outputs: native.bound.outputs.clone(),
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
                "{}\n\n{}",
                luminal::egglog_snippet::assembled_program_for(self.matchers()),
                native
                    .bound
                    .text_unchecked_with_seeds(&native.binding_seeds)
            );
            let mut probe = luminal::egglog_snippet::new_egraph();
            if probe.parse_and_run_program(None, &unchecked).is_ok() {
                for (label, text) in &native.bound.labeled_checks {
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
    /// Requires the `device` feature and a CUDA GPU; all candidates rank
    /// by measured execution time.
    pub fn search(
        &mut self,
        input_data: &FxHashMap<NodeIndex, HostBuffer>,
        options: &CompileOptions,
    ) -> Result<SearchOutcome> {
        self.search_with_profile_inputs(input_data, &[], options)
    }

    /// Search with host input overrides for each complete profiling dimension
    /// assignment. Entries borrow the shared inputs (including weights) and
    /// override only the supplied tensors. Every representative, including
    /// finalist validation, must have exactly one entry when overrides are used.
    pub fn search_with_profile_inputs(
        &mut self,
        input_data: &FxHashMap<NodeIndex, HostBuffer>,
        profile_inputs: &[(shape::DynMap, FxHashMap<NodeIndex, HostBuffer>)],
        options: &CompileOptions,
    ) -> Result<SearchOutcome> {
        ensure!(
            cfg!(feature = "device"),
            "candidate search requires the `device` feature and a CUDA GPU"
        );
        self.ensure_not_installed("re-searching")?;
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

        // Check the caller's payloads against the load-time input bindings.
        for tensor in input_data.keys() {
            assert!(
                native.bound.inputs.iter().any(|b| b.value == *tensor),
                "tensor {tensor:?} is not a bound input"
            );
        }

        // THE SEARCH-TIME STAGING (Phase 4), by BufferLit id and BY
        // REFERENCE — the ladder's `set_data` staging is a separate,
        // later step and is untouched. A full-size model's weights are
        // gigabytes; the search must borrow them, never copy them.
        #[cfg(feature = "device")]
        let staged_for_search: FxHashMap<i64, &HostBuffer> = {
            native
                .bound
                .inputs
                .iter()
                .filter_map(|bound| {
                    input_data
                        .get(&bound.value)
                        .map(|data| (bound.buffer, data))
                })
                .collect()
        };
        #[cfg(feature = "device")]
        let profile_inputs: Vec<crate::search::ProfileInputs<'_>> = profile_inputs
            .iter()
            .map(|(dims, inputs)| {
                let inputs = inputs
                    .iter()
                    .map(|(tensor, data)| {
                        let bound = native
                            .bound
                            .inputs
                            .iter()
                            .find(|bound| bound.value == *tensor)
                            .ok_or_else(|| {
                                anyhow!("profiling tensor {tensor:?} is not a bound input")
                            })?;
                        Ok((bound.buffer, data))
                    })
                    .collect::<Result<_>>()?;
                Ok((dims.clone(), inputs))
            })
            .collect::<Result<_>>()?;
        #[cfg(not(feature = "device"))]
        let _ = profile_inputs;
        // Keep the device and compiled modules across candidate measurements.
        #[cfg(feature = "device")]
        if self.device.is_none() {
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

        // A search that finds nothing has found nothing that WRITES these,
        // so a refusal states them: whether a kernel can write a bound
        // layout is the search's question, never bind's.
        let bound_outputs = native
            .bound
            .outputs
            .iter()
            .map(|bound| {
                format!(
                    "v{} at {:?} on buffer {}",
                    bound.value.index(),
                    bound.layout,
                    bound.buffer
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        let no_plan = || format!("no plan writes the bound outputs: {bound_outputs}");

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
                crate::search::Evaluator::Device {
                    device: self.device.as_mut().expect("device initialized"),
                    staged: &staged_for_search,
                    residents: &self.residents,
                    profile_inputs: &profile_inputs,
                }
            }
            #[cfg(not(feature = "device"))]
            {
                crate::search::Evaluator::NoDevice(std::marker::PhantomData)
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
        let (outcome, unbucketed_plan, searched_buckets) =
            if let Some((mut serialized, program)) = base {
                let mut outcome = crate::search::search_implementations(
                    &mut serialized,
                    &program,
                    options,
                    Some(allow.clone()),
                    matchers,
                    evaluator.reborrow(),
                )
                .with_context(no_plan)?;
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
                    .with_shapes(options.shapes.clone())
                    .with_resident_bindings(self.residents.clone()),
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
                    prefix: &native.bound.prefix,
                    binding_seeds: &native.binding_seeds,
                    schedule: crate::bindings::CudaBindings::SCHEDULE,
                    post_checks: &native.bound.post_checks,
                    inputs: &native.bound.inputs,
                    outputs: &native.bound.outputs,
                    residents: &self.residents,
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
                )
                .with_context(no_plan)?;
                let first = plans
                    .first()
                    .map(|plan| plan.outcome.clone())
                    .ok_or_else(|| anyhow!(no_plan()))?;
                (first, None, plans)
            };
        // CALLER STORAGE BECOMES THE ESCAPE CELL, before anything reads the
        // plan: every later guard (`check_external_outputs`, the arena's
        // external set, execute's pointer map) reads the retargeted plan.
        let (mut unbucketed_plan, mut searched_buckets) = (unbucketed_plan, searched_buckets);
        if let Some(plan) = unbucketed_plan.as_mut() {
            Self::retarget_external_outputs(plan, &native.bound.outputs)?;
        }
        for bucket in &mut searched_buckets {
            Self::retarget_external_outputs(&mut bucket.plan, &native.bound.outputs)?;
        }
        self.device_budget_bytes = options.device_budget_bytes;
        self.bucket_plans = searched_buckets;
        self.selected_bucket = None;
        // The tensor→buffer and tensor→slot maps were built at LOAD from
        // the same bindings every bucket renders; a search changes which
        // implementation runs, never which value crosses where.
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

    /// The buffers the bindings declared device-resident — the arena's
    /// homes, readable on any host.
    pub fn residents(&self) -> &std::collections::BTreeSet<i64> {
        &self.residents.inputs
    }

    /// The buffers the bindings declared CALLER-OWNED device memory — the
    /// ones [`Self::set_device_ptr`] must address before every execute.
    pub fn externals(&self) -> &std::collections::BTreeSet<i64> {
        &self.residents.externals
    }

    /// Resolve public graph handles to this compiled program's boundary IDs.
    pub fn input_buffer(&self, tensor: NodeIndex) -> Result<i64> {
        self.input_buffers
            .get(&tensor)
            .copied()
            .ok_or_else(|| anyhow!("no input binding for {tensor:?}"))
    }
    /// The buffer a value is bound to as an output — the key a device
    /// pointer is supplied under. A value bound as an output on TWO buffers
    /// has no answer here: the ambiguity is refused by name rather than
    /// resolved first-wins.
    pub fn output_buffer(&self, tensor: NodeIndex) -> Result<i64> {
        match self.output_buffers.get(&tensor).map(Vec::as_slice) {
            Some([buffer]) => Ok(*buffer),
            Some(many) => bail!(
                "{tensor:?} is bound as an output on {} buffers ({many:?}); \
                 name the buffer, not the tensor",
                many.len()
            ),
            _ => bail!("no output binding for {tensor:?}"),
        }
    }

    /// The output slot a value is read back through. A value bound as an
    /// output on TWO buffers has two slots and no answer here: the
    /// ambiguity is refused by name rather than resolved first-wins.
    pub fn output_slot_index(&self, tensor: NodeIndex) -> Result<usize> {
        match self.output_slots.get(&tensor).map(Vec::as_slice) {
            Some([index]) => Ok(*index),
            Some(many) => bail!(
                "{tensor:?} is bound as an output on {} buffers (slots {many:?}); \
                 read it back by slot, not by tensor",
                many.len()
            ),
            _ => bail!("no output binding for {tensor:?}"),
        }
    }

    /// Stage input payload for a bound tensor (host side; H2D happens
    /// inside execute). Refused by name for a tensor with no input
    /// binding, and for one bound on caller device memory: an External
    /// buffer has no staging step, so bytes handed over here would
    /// silently never reach the device.
    pub fn set_data(&mut self, tensor: NodeIndex, data: impl Into<HostBuffer>) -> Result<()> {
        let buffer = *self
            .input_buffers
            .get(&tensor)
            .ok_or_else(|| anyhow!("set_data on {tensor:?}, which has no input binding"))?;
        anyhow::ensure!(
            !self.residents.externals.contains(&buffer),
            "v{} is bound on External buffer {buffer}: its storage is the caller's, so \
             it is addressed with set_device_ptr, never staged",
            tensor.index()
        );
        self.staged.insert(buffer, data.into());
        Ok(())
    }

    /// Address an EXTERNAL buffer for the next execution: the storage every
    /// binding on that buffer names lives at `ptr`, so the kernels read and
    /// write it directly and neither H2D nor D2H runs for it. One pointer per
    /// buffer — an output bound on an input's buffer is the same storage and
    /// the same pointer. Refused for a buffer the bindings did not declare
    /// [`crate::bindings::Placement::External`].
    ///
    /// # Safety
    ///
    /// `ptr` must be a live device allocation of at least `bytes` bytes that
    /// stays valid until the next `execute` completes, holding the buffer's
    /// bound layout; the runtime reads it, and writes it where an output is
    /// bound on the buffer.
    pub unsafe fn set_device_ptr(&mut self, buffer: i64, ptr: u64, bytes: usize) -> Result<()> {
        anyhow::ensure!(
            self.residents.externals.contains(&buffer),
            "buffer {buffer} is not bound External; a device pointer may only \
             be supplied for an External buffer"
        );
        anyhow::ensure!(
            ptr != 0 || bytes == 0,
            "buffer {buffer} was given the null device pointer for {bytes} bytes: \
             a null address is not storage"
        );
        self.device_ptrs.insert(buffer, (ptr, bytes));
        Ok(())
    }

    /// Forget an External buffer's address. The next `execute` refuses until
    /// one is supplied again.
    pub fn clear_device_ptr(&mut self, buffer: i64) {
        self.device_ptrs.remove(&buffer);
    }

    /// The External buffers with no address for the next execution — what
    /// `execute` refuses on, in buffer order.
    pub fn missing_external_pointers(&self) -> Vec<i64> {
        self.residents
            .externals
            .iter()
            .filter(|buffer| !self.device_ptrs.contains_key(buffer))
            .copied()
            .collect()
    }

    /// Refuse an execution whose boundary is not addressable, naming the
    /// buffer and the values bound on it.
    fn ensure_external_pointers(&self) -> Result<()> {
        let Some(&buffer) = self.missing_external_pointers().first() else {
            return Ok(());
        };
        let native = self
            .native
            .as_ref()
            .ok_or_else(|| anyhow!("load before execute"))?;
        let values: Vec<String> = native
            .bound
            .inputs
            .iter()
            .chain(&native.bound.outputs)
            .filter(|bound| bound.buffer == buffer)
            .map(|bound| format!("v{}", bound.value.index()))
            .collect();
        bail!(
            "External buffer {buffer} ({}) has no device pointer for this execute",
            values.join(", ")
        )
    }

    /// Run on a stream owned by another library. Rebind each call if the
    /// caller's current stream can change.
    #[cfg(feature = "device")]
    pub fn use_borrowed_stream(&mut self, raw_stream: u64) {
        self.borrowed_stream = Some(raw_stream);
    }

    #[cfg(feature = "device")]
    pub fn use_owned_stream(&mut self) {
        self.borrowed_stream = None;
    }

    /// Bind a caller-allocated arena for subsequent executions. Called once
    /// per execution when the arena is allocated and freed per call;
    /// `clear_arena` reverts to the owned slab.
    #[cfg(feature = "device")]
    pub fn set_arena(&mut self, ptr: u64, bytes: usize) {
        self.external_arena = Some((ptr, bytes));
    }

    #[cfg(feature = "device")]
    pub fn clear_arena(&mut self) {
        self.external_arena = None;
    }

    /// The slab size the currently selected plan set requires. The caller
    /// allocates at least this many bytes and passes the pointer to
    /// [`Self::set_arena`]. Device-free: it packs lifetimes but touches no
    /// CUDA API.
    #[cfg(feature = "device")]
    pub fn arena_bytes(&self) -> Result<usize> {
        let plans = self.install_plans()?;
        Ok(crate::resident::allocate(plans, self.residents.clone())?.bytes)
    }

    /// The searched plan re-asked of the External output bindings, on any
    /// host: [`Self::ensure_external_output_slots_are_literal`] over the
    /// plans `execute` would install.
    pub fn check_external_outputs(&self) -> Result<()> {
        self.ensure_external_output_slots_are_literal(&self.install_plans()?)
    }

    /// THE CALLER'S STORAGE IS THE ESCAPE CELL: an External output elected on
    /// a planner-minted cell takes the caller's buffer id, so the arena leaves
    /// it out of the slab and execute addresses it through the caller's
    /// pointer. Two External outputs on one cell, or an output elected on a
    /// different caller buffer, are refused by name.
    fn retarget_external_outputs(
        plan: &mut crate::layouts::CudaPlan,
        outputs: &[crate::bindings::Bound],
    ) -> Result<()> {
        // Cell → the bound buffer it was given, and the value that gave it.
        let mut retargeted: FxHashMap<luminal::bufferize::BufferId, (i64, usize)> =
            FxHashMap::default();
        for node in plan.dag.node_weights() {
            let luminal::bufferize::BufferNode::BufferOutput { slots } = node else {
                continue;
            };
            for slot in slots {
                let bound = outputs
                    .get(slot.index)
                    .ok_or_else(|| anyhow!("plan output slot {} has no binding", slot.index))?;
                if bound.placement != crate::bindings::Placement::External {
                    continue;
                }
                let cell = plan.buffers.get_mut(&slot.buffer).ok_or_else(|| {
                    anyhow!(
                        "plan output slot {} names buffer {:?}, which the plan has no entry for",
                        slot.index,
                        slot.buffer
                    )
                })?;
                match cell.lit {
                    Some(lit) if lit == bound.buffer => {}
                    Some(other) => {
                        if let Some((_, first)) = retargeted.get(&slot.buffer) {
                            bail!(
                                "outputs v{first} and v{} are both bound External and share \
                                 escape cell {:?}, which can carry only one caller buffer id \
                                 ({other} and {})",
                                bound.value.index(),
                                slot.buffer,
                                bound.buffer
                            );
                        }
                        bail!(
                            "output v{} is bound External on buffer {} but the searched plan \
                             elected a view of caller buffer {other}; aliasing between caller \
                             buffers is out of scope (LUM-825)",
                            bound.value.index(),
                            bound.buffer
                        );
                    }
                    None => {
                        cell.lit = Some(bound.buffer);
                        retargeted.insert(slot.buffer.clone(), (bound.buffer, bound.value.index()));
                    }
                }
            }
        }
        Ok(())
    }

    /// ESCAPE-AND-DISCLOSE ON CALLER STORAGE: an output bound External sits
    /// on the caller's buffer id, which an escape cell also carries because
    /// [`Self::retarget_external_outputs`] gave it that id after the search.
    /// What is left to refuse is a view of a DIFFERENT bound buffer: it has
    /// another buffer's bytes under the caller's tensor.
    fn ensure_external_output_slots_are_literal(
        &self,
        plans: &[(crate::layouts::CudaPlan, crate::symbolic::Bounds)],
    ) -> Result<()> {
        let native = self
            .native
            .as_ref()
            .ok_or_else(|| anyhow!("load before execute"))?;
        for (plan, _) in plans {
            for node in plan.dag.node_weights() {
                let luminal::bufferize::BufferNode::BufferOutput { slots } = node else {
                    continue;
                };
                for slot in slots {
                    let bound =
                        native.bound.outputs.get(slot.index).ok_or_else(|| {
                            anyhow!("plan output slot {} has no binding", slot.index)
                        })?;
                    if !self.residents.externals.contains(&bound.buffer) {
                        continue;
                    }
                    anyhow::ensure!(
                        plan.buffers[&slot.buffer].lit == Some(bound.buffer),
                        "output v{} is bound External on buffer {} but the searched plan \
                         elected a view of it (escape-and-disclose); bind it Staged and read \
                         it back through fetch/output_layout",
                        bound.value.index(),
                        bound.buffer
                    );
                }
            }
        }
        Ok(())
    }

    /// The `(plan, bounds)` set `execute` installs — the single unpinned plan,
    /// or one per bucket. Factored out so `arena_bytes` can size the slab
    /// without a device.
    fn install_plans(&self) -> Result<Vec<(crate::layouts::CudaPlan, crate::symbolic::Bounds)>> {
        let base_bounds: crate::symbolic::Bounds = self
            .range_bound
            .iter()
            .map(|(s, (lo, hi))| Ok((*s, (usize::try_from(*lo)?, usize::try_from(*hi)?))))
            .collect::<Result<_>>()?;
        Ok(if self.bucket_plans.is_empty() {
            let plan = self
                .plan
                .as_ref()
                .ok_or_else(|| anyhow!("search before reading the installed plans"))?;
            vec![(plan.clone(), base_bounds)]
        } else {
            self.bucket_plans
                .iter()
                .map(|p| {
                    let mut bounds = base_bounds.clone();
                    bounds.extend(p.ranges.iter().map(|(k, v)| (*k, *v)));
                    (p.plan.clone(), bounds)
                })
                .collect()
        })
    }

    /// Run the plan on the CUDA device. Requires the `device` feature
    /// and an available device; refuses loudly otherwise.
    pub fn execute(&mut self) -> Result<()> {
        self.execute_mode(false)
    }

    /// Enqueue an already-warmed, zero-copy plan without synchronizing the
    /// borrowed stream. The caller keeps every boundary and arena alive.
    pub fn execute_async(&mut self) -> Result<()> {
        self.execute_mode(true)
    }

    fn execute_mode(&mut self, asynchronous: bool) -> Result<()> {
        // A boundary this runtime cannot address is refused before anything
        // else: the statement is the bindings', not the device's.
        self.ensure_external_pointers()?;
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
            {
                let device = self.device.as_mut().unwrap();
                // The borrowed stream is rebound every execution because
                // the caller's current stream is thread-local and may change.
                if let Some(raw) = self.borrowed_stream {
                    device.use_borrowed_stream(raw)?;
                } else if device.stream_is_borrowed() {
                    device.use_owned_stream()?;
                }
                if let Some((ptr, bytes)) = self.external_arena {
                    device.set_external_arena(ptr, bytes)?;
                } else {
                    device.clear_external_arena();
                }
            }
            if !self.device.as_ref().unwrap().is_installed() {
                let plans = self.install_plans()?;
                self.ensure_external_output_slots_are_literal(&plans)?;
                self.device.as_mut().unwrap().install_resident_with_budget(
                    plans,
                    self.residents.clone(),
                    self.device_budget_bytes,
                )?;
            }
            let device = self.device.as_mut().unwrap();
            let bucket = self.selected_bucket.unwrap_or(0);
            let staged = self.staged.iter().map(|(lit, data)| (*lit, data)).collect();
            let external: FxHashMap<i64, crate::device::ExternalPtr> = self
                .device_ptrs
                .iter()
                .map(|(buffer, (ptr, bytes))| {
                    (
                        *buffer,
                        crate::device::ExternalPtr {
                            ptr: *ptr,
                            bytes: *bytes,
                        },
                    )
                })
                .collect();
            let outputs = device.execute_external_mode(
                bucket,
                &staged,
                &self.dims,
                &external,
                asynchronous,
            )?;
            self.outputs_host = outputs;
            // Zero-copy inputs are not staged, so nothing to clear; host-staged
            // residents are kept for the (owned-slab) reuse path.
            self.staged
                .retain(|lit, _| !self.residents.inputs.contains(lit));
            Ok(())
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = asynchronous;
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
        let index = self.output_slot_index(tensor)?;
        if let Some((data, binding)) = self.outputs_host.get(&index) {
            return Ok((data, binding));
        }
        // An output written straight into caller device memory is never
        // read back, so it is absent here on a perfectly good execution.
        if let Some(bound) = self
            .native
            .as_ref()
            .and_then(|native| native.bound.outputs.get(index))
            && self.residents.externals.contains(&bound.buffer)
        {
            bail!(
                "output v{} is bound External on buffer {}: its bytes are in the \
                 caller's device memory, not readable through fetch",
                bound.value.index(),
                bound.buffer
            );
        }
        bail!("execute before fetch")
    }

    /// The slot's elected layout alone (see [`Self::fetch`]).
    pub fn output_layout(
        &self,
        tensor: NodeIndex,
    ) -> Result<&luminal::bufferize::OutputBinding<DecodedLayout>> {
        Ok(self.fetch(tensor)?.1)
    }

    /// The buffer id the installed plan writes this output's bytes into —
    /// the caller's own id for an output bound External. Valid after search.
    pub fn output_backing_buffer(&self, tensor: NodeIndex) -> Result<i64> {
        let (plan, slot) = self.installed_output_slot(tensor)?;
        plan.buffers[&slot.buffer].lit.ok_or_else(|| {
            anyhow!(
                "output v{}'s backing buffer is a program allocation with no buffer id",
                tensor.index()
            )
        })
    }

    /// The bytes that backing buffer spans at the runtime's current dims.
    pub fn output_span_bytes(&self, tensor: NodeIndex) -> Result<usize> {
        let (plan, slot) = self.installed_output_slot(tensor)?;
        crate::symbolic::bytes(&plan.buffers[&slot.buffer].layout, &self.dims)
    }

    /// The output's ELECTED element strides, one per axis, at the current dims.
    pub fn output_elected_strides(&self, tensor: NodeIndex) -> Result<Vec<i64>> {
        let (_, slot) = self.installed_output_slot(tensor)?;
        let layout = crate::symbolic::resolve_layout(&slot.layout, &self.dims)?;
        elected_strides(&layout).ok_or_else(|| {
            anyhow!(
                "output v{}'s elected layout {:?} has no strides",
                tensor.index(),
                layout.present()
            )
        })
    }

    /// [`Self::output_backing_buffer`] by slot index: the answer for a value
    /// bound as an output on two buffers, which has two slots.
    pub fn output_slot_backing_buffer(&self, slot: usize) -> Result<i64> {
        let (plan, binding) = self.installed_output_slot_at(slot)?;
        plan.buffers[&binding.buffer].lit.ok_or_else(|| {
            anyhow!("output slot {slot}'s backing buffer is a program allocation with no buffer id")
        })
    }

    /// [`Self::output_span_bytes`] by slot index.
    pub fn output_slot_span_bytes(&self, slot: usize) -> Result<usize> {
        let (plan, binding) = self.installed_output_slot_at(slot)?;
        crate::symbolic::bytes(&plan.buffers[&binding.buffer].layout, &self.dims)
    }

    /// [`Self::output_elected_strides`] by slot index.
    pub fn output_slot_elected_strides(&self, slot: usize) -> Result<Vec<i64>> {
        let (_, binding) = self.installed_output_slot_at(slot)?;
        let layout = crate::symbolic::resolve_layout(&binding.layout, &self.dims)?;
        elected_strides(&layout).ok_or_else(|| {
            anyhow!(
                "output slot {slot}'s elected layout {:?} has no strides",
                layout.present()
            )
        })
    }

    /// A bound output's slot in the plan `execute` would run.
    fn installed_output_slot(
        &self,
        tensor: NodeIndex,
    ) -> Result<(
        &crate::layouts::CudaPlan,
        &luminal::bufferize::OutputBinding<DecodedLayout>,
    )> {
        self.installed_output_slot_at(self.output_slot_index(tensor)?)
    }

    /// Output slot `index` (binding order) in the plan `execute` would run.
    fn installed_output_slot_at(
        &self,
        index: usize,
    ) -> Result<(
        &crate::layouts::CudaPlan,
        &luminal::bufferize::OutputBinding<DecodedLayout>,
    )> {
        let plan = self
            .selected_bucket
            .and_then(|i| self.bucket_plans.get(i))
            .or_else(|| self.bucket_plans.first())
            .map(|bucket| &bucket.plan)
            .or(self.plan.as_ref())
            .ok_or_else(|| anyhow!("search before reading an output's backing storage"))?;
        let slot = plan
            .dag
            .node_weights()
            .filter_map(|node| match node {
                luminal::bufferize::BufferNode::BufferOutput { slots } => Some(slots),
                _ => None,
            })
            .flatten()
            .find(|slot| slot.index == index)
            .ok_or_else(|| anyhow!("the installed plan has no output slot {index}"))?;
        Ok((plan, slot))
    }

    /// The searched plan, for inspection and tests.
    pub fn plan(&self) -> Option<&BufferIrGraph<DecodedLayout>> {
        self.selected_bucket
            .and_then(|i| self.bucket_plans.get(i))
            .map(|p| &p.plan)
            .or(self.plan.as_ref())
    }
}

/// One element stride per axis of a RESOLVED layout, taken from the
/// constructor's own meaning: the contiguous forms state theirs by shape
/// order, a strided chain by its three canonical residues (`coord * stride`,
/// the bare coordinate, the dead axis's zero). `None` for a layout that
/// states an offset function instead of strides, and for a chain summand
/// outside those residues — a stride is never recovered by evaluation.
fn elected_strides(layout: &DecodedLayout) -> Option<Vec<i64>> {
    use luminal::layouts::{
        IntExprTerm as T, LeftMajorContiguousElementLayout as LM,
        RightMajorContiguousElementLayout as RM, StridedElementLayout as ST,
    };
    let extents: Vec<i64> = layout
        .literal_extents()?
        .into_iter()
        .map(|e| i64::try_from(e).ok())
        .collect::<Option<_>>()?;
    let rank = extents.len();
    if layout.has::<RM>() {
        let mut strides = vec![1i64; rank];
        for axis in (0..rank.saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1].checked_mul(extents[axis + 1])?;
        }
        return Some(strides);
    }
    if layout.has::<LM>() {
        let mut strides = vec![1i64; rank];
        for axis in 1..rank {
            strides[axis] = strides[axis - 1].checked_mul(extents[axis - 1])?;
        }
        return Some(strides);
    }
    let chain = &layout.first::<ST>()?.chain;
    if chain.len() != rank {
        return None;
    }
    // A summand names its own axis FROM THE END; only the dead axis's bare
    // zero has none, and it sits at that axis's position in the chain.
    let axis_of = |axis_from_end: &i64| -> Option<usize> {
        let axis = usize::try_from(*axis_from_end).ok()?;
        (axis < rank).then_some(rank - 1 - axis)
    };
    let mut strides: Vec<Option<i64>> = vec![None; rank];
    for (position, summand) in chain.iter().enumerate() {
        let (axis, stride) = match summand {
            T::Lit(0) => (position, 0),
            T::Coord { axis_from_end } => (axis_of(axis_from_end)?, 1),
            T::Mul(a, b) => match (a.as_ref(), b.as_ref()) {
                (T::Coord { axis_from_end }, k) | (k, T::Coord { axis_from_end }) => {
                    (axis_of(axis_from_end)?, k.eval_literal()?)
                }
                _ => return None,
            },
            _ => return None,
        };
        if strides[axis].replace(stride).is_some() {
            return None;
        }
    }
    strides.into_iter().collect()
}
