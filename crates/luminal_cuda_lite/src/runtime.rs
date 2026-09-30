//! The CUDA-lite runtime: the reference ladder
//! (`load → search → bind device storage → execute`) with the
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

/// A slab borrowed for one synchronous execution. The owner determines whether
/// the same allocation survives between calls.
#[derive(Debug)]
pub struct CudaArena<'a> {
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    pub(crate) ptr: u64,
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    pub(crate) bytes: usize,
    _borrow: std::marker::PhantomData<&'a mut [u8]>,
}
impl<'a> CudaArena<'a> {
    /// An empty arena. Execution rejects it if the program needs any storage.
    pub fn empty() -> Self {
        Self {
            ptr: 0,
            bytes: 0,
            _borrow: std::marker::PhantomData,
        }
    }
    /// Borrow foreign device memory (for example, PyTorch allocator storage).
    ///
    /// # Safety
    /// `ptr` must be a live, 256-byte-aligned allocation of at least `bytes` on
    /// the execution device, exclusively available for the returned lifetime.
    /// For asynchronous execution it must remain live until GPU completion.
    pub unsafe fn from_raw(ptr: u64, bytes: usize) -> Self {
        Self {
            ptr,
            bytes,
            _borrow: std::marker::PhantomData,
        }
    }
}

/// What `load` captured: the bound program (model text, this runtime's
/// boundary, the post-schedule checks) plus whatever the `bind_*` calls
/// accumulate before `search` assembles and saturates.
#[derive(Clone)]
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
    /// genome); everything
    /// downstream borrows this slice.
    matchers: std::rc::Rc<Vec<Box<dyn luminal::layout_ir::OpMatcher>>>,
    /// The claim set derived from that same registry — see
    /// [`CudaRuntime::allow_list_over`] for the three classes.
    allow: Vec<&'static str>,
    /// The `(sort, constructor)` DECODERS for that same matcher set:
    /// core's built-ins plus whatever the instance's matchers declare.
    /// Every layout decode in this runtime reads through these, and the
    /// assembly tripwire checks each saturated program against them. A
    /// `Default` runtime carries an empty registry, like `matchers`.
    decoders: std::rc::Rc<luminal::egglog_utils::eclass::ConstructorRegistry>,
    plan: Option<BufferIrGraph<DecodedLayout>>,
    storage: Option<crate::arena::ArenaPlan>,
    /// Boundaries supplied as caller-owned device pointers.
    externals: std::collections::BTreeSet<i64>,
    device_budget_bytes: Option<usize>,
    /// The caller's device address for each External buffer, supplied before
    /// each execution ([`Self::set_device_ptr`]). Keyed by buffer id, which
    /// is also how the bindings spell aliasing: a mutation sink and its
    /// target share one buffer and therefore one pointer.
    device_ptrs: FxHashMap<i64, (u64, usize)>,
    /// Logical input values mapped to device boundary buffers.
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
    bounds: shape::DimensionBounds,
    dims: shape::DynMap,
    /// Live execution context and code/graph caches, attached explicitly or lazily.
    /// This is runtime state, not a portable serialized program.
    #[cfg(feature = "device")]
    device: Option<crate::device::CudaExecutable>,
    /// Raw `CUstream` to run on (the caller's current stream). `None` runs on
    /// a stream this runtime owns.
    #[cfg(feature = "device")]
    borrowed_stream: Option<u64>,
}

impl CudaRuntime {
    /// Bind the selected program to independent execution state. Device code
    /// caches are shared; inputs, output buffers, arenas and captures are fresh.
    pub fn fork(&self) -> Result<Self> {
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| anyhow!("search before fork"))?;
        Ok(Self {
            native: self.native.clone(),
            matchers: self.matchers.clone(),
            allow: self.allow.clone(),
            decoders: self.decoders.clone(),
            plan: Some(plan.clone()),
            storage: self.storage.clone(),
            bounds: self.bounds.clone(),
            dims: self.dims.clone(),
            input_buffers: self.input_buffers.clone(),
            output_buffers: self.output_buffers.clone(),
            output_slots: self.output_slots.clone(),
            externals: self.externals.clone(),
            device_budget_bytes: self.device_budget_bytes,
            #[cfg(feature = "device")]
            device: self
                .device
                .as_ref()
                .map(|device| device.fork())
                .transpose()?,
            ..Default::default()
        })
    }

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
    /// stored in its own row-major arena range, every leaf read-write on
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
    /// which of them use caller device storage. The tensor→buffer maps are
    /// available immediately, and
    /// external placement is known before the first plan is priced.
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
        let externals = bound.externals();
        Ok(Self {
            native: Some(NativeParts {
                bound,
                binding_seeds: String::new(),
            }),
            matchers: std::rc::Rc::new(matchers),
            allow,
            decoders: std::rc::Rc::new(decoders),
            input_buffers,
            output_slots,
            output_buffers,
            externals,
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

    /// Use a shared device context/cache while keeping this program's state separate.
    #[cfg(feature = "device")]
    pub fn with_device(mut self, device: &crate::device::CudaDevice) -> Result<Self> {
        self.device = Some(device.executable());
        self.device_ptrs.clear();
        self.borrowed_stream = None;
        Ok(self)
    }

    /// Execution values never change a program's compiled domain.
    pub fn set_dim(&mut self, dim: impl Into<shape::Symbol>, value: usize) {
        self.dims.insert(dim.into(), value);
    }

    pub fn bounds(&self) -> &shape::DimensionBounds {
        &self.bounds
    }

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
    pub fn saturated_egraph(
        &self,
        bounds: &shape::DimensionBounds,
    ) -> Result<luminal::prelude::egraph_serialize::EGraph> {
        let (serialized, _program) = self.assemble_and_saturate(bounds)?;
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
        bounds: &shape::DimensionBounds,
    ) -> Result<(
        luminal::prelude::egraph_serialize::EGraph,
        crate::search::SearchProgram,
    )> {
        let native = self
            .native
            .as_ref()
            .ok_or_else(|| anyhow!("load before search"))?;
        bounds.validate_symbols(&shape::program_dimensions(&format!(
            "{}{}",
            native.bound.prefix, native.bound.post_checks
        ))?)?;
        let seeds = format!("{}{}", native.binding_seeds, bounds.egglog_seeds());
        let program = crate::search::SearchProgram {
            text: native.bound.text_with_seeds(&seeds),
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
                native.bound.text_unchecked_with_seeds(&seeds)
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
        bounds: &shape::DimensionBounds,
        profile_dims: &shape::DynMap,
        input_data: &FxHashMap<NodeIndex, HostBuffer>,
        options: &CompileOptions,
    ) -> Result<SearchOutcome> {
        #[cfg(feature = "device")]
        if let Some(device) = self.device.as_mut() {
            device.uninstall();
        }
        bounds.validate_values(profile_dims)?;
        let native = self
            .native
            .as_ref()
            .ok_or_else(|| anyhow!("load before search"))?;
        bounds.validate_symbols(&shape::program_dimensions(&format!(
            "{}{}",
            native.bound.prefix, native.bound.post_checks
        ))?)?;
        ensure!(
            cfg!(feature = "device"),
            "candidate search requires a CUDA GPU"
        );
        let (mut serialized, program) = self.assemble_and_saturate(bounds)?;
        let shapes = crate::symbolic::ShapeEnv {
            bounds: bounds.ranges(),
            values: profile_dims.clone(),
        };
        let native = self
            .native
            .as_ref()
            .ok_or_else(|| anyhow!("load before search"))?;
        for tensor in input_data.keys() {
            ensure!(
                native.bound.inputs.iter().any(|b| b.value == *tensor),
                "tensor {tensor:?} is not a bound input"
            );
        }
        #[cfg(feature = "device")]
        let staged: FxHashMap<i64, &HostBuffer> = native
            .bound
            .inputs
            .iter()
            .filter_map(|bound| {
                input_data
                    .get(&bound.value)
                    .map(|data| (bound.buffer, data))
            })
            .collect();
        #[cfg(feature = "device")]
        if self.device.is_none() {
            self.device = Some(crate::device::CudaDevice::new(0)?.executable());
        }
        let mut evaluator = {
            #[cfg(feature = "device")]
            {
                crate::search::Evaluator::Device {
                    device: self.device.as_mut().expect("device initialized"),
                    staged: &staged,
                }
            }
            #[cfg(not(feature = "device"))]
            {
                crate::search::Evaluator::NoDevice(std::marker::PhantomData)
            }
        };
        let mut outcome = crate::search::search_implementations(
            &mut serialized,
            &program,
            &shapes,
            options,
            Some(self.allow.clone()),
            &self.matchers,
            evaluator.reborrow(),
        )?;
        let finalists = crate::finalists::Finalists::new(
            "program",
            &serialized,
            Some(self.allow.clone()),
            &self.matchers,
            outcome.ranked.clone(),
            Some(outcome.best_plan.clone()),
        )
        .with_shapes(shapes)
        .with_external_bindings(self.externals.clone());
        let (finalist, rejections) =
            crate::search::select_finalist(finalists, options, &mut evaluator)?;
        outcome.finalist_rejections = rejections;
        outcome.finalist_rank = finalist.rank;
        outcome.best_plan = finalist.plan;
        outcome.best_genome = finalist.genome;
        outcome.best_nanos = finalist.metric;
        Self::retarget_external_outputs(&mut outcome.best_plan, &native.bound.outputs)?;
        self.storage = Some(crate::storage::plan_storage(
            &outcome.best_plan,
            &bounds.ranges(),
            &self.externals,
        )?);
        self.plan = Some(outcome.best_plan.clone());
        self.bounds = bounds.clone();
        self.dims = profile_dims.clone();
        self.device_budget_bytes = options.device_budget_bytes;
        Ok(outcome)
    }

    /// The buffers the bindings declared CALLER-OWNED device memory — the
    /// ones [`Self::set_device_ptr`] must address before every execute.
    pub fn externals(&self) -> &std::collections::BTreeSet<i64> {
        &self.externals
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
            self.externals.contains(&buffer),
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
        self.externals
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

    /// The slab size the currently selected program requires. The caller
    /// allocates at least this many bytes and passes the pointer to
    /// [`Self::execute`]. Device-free: it packs lifetimes but touches no
    /// CUDA API.
    #[cfg(feature = "device")]
    pub fn arena_bytes(&self) -> Result<usize> {
        let program = self.install_program()?;
        Ok(crate::storage::plan_storage(&program.0, &program.1, &self.externals)?.slab_bytes)
    }

    /// The searched plan re-asked of the External output bindings, on any
    /// host: [`Self::ensure_external_output_slots_are_literal`] over the
    /// plans `execute` would install.
    pub fn check_external_outputs(&self) -> Result<()> {
        self.ensure_external_output_slots_are_literal(&self.install_program()?)
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
        program: &(crate::layouts::CudaPlan, crate::symbolic::Bounds),
    ) -> Result<()> {
        let native = self
            .native
            .as_ref()
            .ok_or_else(|| anyhow!("load before execute"))?;
        let (plan, _) = program;
        {
            for node in plan.dag.node_weights() {
                let luminal::bufferize::BufferNode::BufferOutput { slots } = node else {
                    continue;
                };
                for slot in slots {
                    let bound =
                        native.bound.outputs.get(slot.index).ok_or_else(|| {
                            anyhow!("plan output slot {} has no binding", slot.index)
                        })?;
                    if !self.externals.contains(&bound.buffer) {
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

    fn install_program(&self) -> Result<(crate::layouts::CudaPlan, crate::symbolic::Bounds)> {
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| anyhow!("search before installing a program"))?;
        Ok((plan.clone(), self.bounds.ranges()))
    }

    /// Native CUDA stream after attaching a device or searching. Tensor storage
    /// and CPU/device transfers are entirely the caller's responsibility.
    #[cfg(feature = "device")]
    pub fn cuda_stream(&self) -> Result<&std::sync::Arc<cudarc::driver::CudaStream>> {
        Ok(self
            .device
            .as_ref()
            .ok_or_else(|| anyhow!("attach a device or search first"))?
            .stream())
    }
    /// Pinned host bytes needed for this program's runtime dimension parameters.
    pub fn staging_bytes(&self) -> Result<usize> {
        let program = self.install_program()?;
        Ok(
            crate::storage::plan_storage(&program.0, &program.1, &self.externals)?
                .staging_bytes
                .max(1),
        )
    }

    #[cfg(feature = "device")]
    pub fn allocate_staging(&mut self) -> Result<crate::device::CudaStaging> {
        let bytes = self.staging_bytes()?;
        if self.device.is_none() {
            self.device = Some(crate::device::CudaDevice::new(0)?.executable());
        }
        self.device.as_ref().unwrap().allocate_staging(bytes)
    }

    /// Run with borrowed device and host storage; neither is retained or freed.
    pub fn execute(
        &mut self,
        arena: CudaArena<'_>,
        #[cfg(feature = "device")] staging: &mut crate::device::CudaStaging,
    ) -> Result<()> {
        self.execute_mode(
            arena,
            #[cfg(feature = "device")]
            staging,
            false,
        )
    }

    /// Enqueue an already-warmed, zero-copy plan without synchronizing the
    /// borrowed stream. The caller keeps every boundary and arena alive.
    /// # Safety
    /// The caller must retain all memory and the stream until GPU completion,
    /// and must not modify or reuse that storage while GPU work is in flight.
    /// Outer graph captures require these bindings to remain valid for replay.
    pub unsafe fn execute_async(
        &mut self,
        arena: CudaArena<'_>,
        #[cfg(feature = "device")] staging: &mut crate::device::CudaStaging,
    ) -> Result<()> {
        self.execute_mode(
            arena,
            #[cfg(feature = "device")]
            staging,
            true,
        )
    }

    fn execute_mode(
        &mut self,
        arena: CudaArena<'_>,
        #[cfg(feature = "device")] staging: &mut crate::device::CudaStaging,
        asynchronous: bool,
    ) -> Result<()> {
        // A boundary this runtime cannot address is refused before anything
        // else: the statement is the bindings', not the device's.
        self.ensure_external_pointers()?;
        // Select a range-valid plan using the current dimensions.
        self.bounds.validate_values(&self.dims)?;
        #[cfg(feature = "device")]
        {
            // Cache live device modules and launch handles independently of storage.
            if self.device.is_none() {
                self.device = Some(crate::device::CudaDevice::new(0)?.executable());
            }
            anyhow::ensure!(self.plan.is_some(), "search before execute");
            {
                let device = self.device.as_mut().unwrap();
                // The borrowed stream is rebound every execution because
                // the caller's current stream is thread-local and may change.
                if let Some(raw) = self.borrowed_stream {
                    device.use_borrowed_stream(raw)?;
                } else if device.stream_is_borrowed() {
                    device.use_owned_stream()?;
                }
            }
            if !self.device.as_ref().unwrap().is_installed() {
                let program = self.install_program()?;
                self.ensure_external_output_slots_are_literal(&program)?;
                self.device.as_mut().unwrap().install_with_bindings(
                    program,
                    self.externals.clone(),
                    self.device_budget_bytes,
                )?;
            }
            let device = self.device.as_mut().unwrap();

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
            device.execute_external_mode(arena, staging, &self.dims, &external, asynchronous)?;
            Ok(())
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = (arena, asynchronous);
            let _ = self
                .plan()
                .ok_or_else(|| anyhow!("search before execute"))?;
            bail!(
                "cuda-lite built without the `device` feature: plans can be \
                 searched and inspected but not executed on this host"
            )
        }
    }

    /// Live input range within the supplied arena. External inputs have no range.
    pub fn input_arena_range(&self, tensor: NodeIndex) -> Result<crate::arena::ArenaSlice> {
        let lit = self.input_buffer(tensor)?;
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| anyhow!("search before inspecting storage"))?;
        let buffer = plan
            .buffers
            .values()
            .find(|b| b.lit == Some(lit))
            .ok_or_else(|| anyhow!("input is absent from the selected plan"))?;
        self.buffer_arena_range(&buffer.id)
    }

    /// Live output backing range. Its elected view is available through `output_layout`.
    pub fn output_arena_range(&self, tensor: NodeIndex) -> Result<crate::arena::ArenaSlice> {
        self.buffer_arena_range(&self.output_layout(tensor)?.buffer)
    }

    fn buffer_arena_range(
        &self,
        id: &luminal::bufferize::BufferId,
    ) -> Result<crate::arena::ArenaSlice> {
        self.bounds.validate_values(&self.dims)?;
        let memory = self.memory_plan()?;
        let mut range = *memory
            .slices
            .get(id)
            .ok_or_else(|| anyhow!("buffer has external device storage"))?;
        range.bytes =
            crate::symbolic::bytes(&self.plan.as_ref().unwrap().buffers[id].layout, &self.dims)?;
        Ok(range)
    }

    /// Current dimension assignment, used to resolve device buffer layouts.
    pub fn dimensions(&self) -> &shape::DynMap {
        &self.dims
    }

    /// Offsets/capacities in the caller's device arena. External bindings are excluded.
    pub fn memory_plan(&self) -> Result<&crate::arena::ArenaPlan> {
        self.storage
            .as_ref()
            .ok_or_else(|| anyhow!("search before inspecting storage"))
    }

    /// Elected output layout at the current dimensions. Does not access tensor data.
    pub fn output_layout(
        &self,
        tensor: NodeIndex,
    ) -> Result<luminal::bufferize::OutputBinding<DecodedLayout>> {
        let index = self.output_slot_index(tensor)?;
        let mut slot = self
            .plan
            .as_ref()
            .ok_or_else(|| anyhow!("search before inspecting output"))?
            .dag
            .node_weights()
            .filter_map(|node| match node {
                luminal::bufferize::BufferNode::BufferOutput { slots } => Some(slots.iter()),
                _ => None,
            })
            .flatten()
            .find(|slot| slot.index == index)
            .ok_or_else(|| anyhow!("missing output slot {index}"))?
            .clone();
        slot.layout = crate::symbolic::resolve_layout(&slot.layout, &self.dims)?;
        Ok(slot)
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
            .plan
            .as_ref()
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
        self.plan.as_ref()
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

#[cfg(all(test, feature = "device"))]
mod caller_memory_tests {
    use super::*;
    use luminal::layout_ir::{Access, FreedBy};
    use luminal::prelude::*;

    #[test]
    fn host_staging_can_be_shared_between_programs() {
        let mut graph = Graph::new();
        let output = graph.iota('n', |coords| coords[0]);
        let mut first = CudaRuntime::load(&graph).unwrap();
        first
            .search(
                &shape::DimensionBounds::from_ranges([('n'.into(), (1, 8))]).unwrap(),
                &[('n'.into(), 4)].into_iter().collect(),
                &Default::default(),
                &crate::harness_search_options(),
            )
            .unwrap();
        let mut second = first.fork().unwrap();
        first.set_dim('n', 4);
        second.set_dim('n', 7);
        let mut first_arena = crate::test_memory::Allocation::new(
            first.cuda_stream().unwrap().clone(),
            first.arena_bytes().unwrap(),
        )
        .unwrap();
        let mut second_arena = crate::test_memory::Allocation::new(
            second.cuda_stream().unwrap().clone(),
            second.arena_bytes().unwrap(),
        )
        .unwrap();
        let mut staging = first.allocate_staging().unwrap();
        first.execute(first_arena.arena(), &mut staging).unwrap();
        second.execute(second_arena.arena(), &mut staging).unwrap();
        first.execute(first_arena.arena(), &mut staging).unwrap();
        for (runtime, arena, expected) in [(&first, &first_arena, 4), (&second, &second_arena, 7)] {
            let range = runtime.output_arena_range(output.id).unwrap();
            let result = HostBuffer::new(
                luminal::dtype::PlanDtype::Int,
                arena.read(range.offset, range.bytes).unwrap(),
            )
            .unwrap();
            assert_eq!(result.as_i32().unwrap(), (0..expected).collect::<Vec<_>>());
        }
    }

    #[test]
    fn application_state_survives_replaced_execution_storage() {
        let mut graph = Graph::new();
        let state = graph.tensor(4, DType::F32);
        let increment = graph.tensor(4, DType::F32);
        let previous = state.sum(0);
        let next = state + increment;
        let mut bindings = crate::bindings::CudaBindings::new();
        let state_id = bindings.input_external(state.id);
        bindings.declare(state_id, Access::ReadWrite, FreedBy::Caller);
        bindings.input(increment.id);
        bindings.output(previous.id);
        bindings.output_on(next.id, state_id);
        let device = crate::device::CudaDevice::new(0).unwrap();
        let mut runtime = CudaRuntime::load_with(
            &graph,
            bindings,
            crate::ops::cuda_registry_without_cublaslt(),
        )
        .unwrap()
        .with_device(&device)
        .unwrap();
        runtime
            .search(
                &Default::default(),
                &Default::default(),
                &[
                    (state.id, vec![0f32; 4].into()),
                    (increment.id, vec![1f32; 4].into()),
                ]
                .into_iter()
                .collect(),
                &crate::harness_search_options(),
            )
            .unwrap();
        let mut state_memory =
            crate::test_memory::Allocation::new(device.stream().clone(), 16).unwrap();
        state_memory.write(0, &[0; 16]).unwrap();
        // SAFETY: the application retains this allocation until execution completes.
        unsafe { runtime.set_device_ptr(state_id, state_memory.ptr(), state_memory.bytes()) }
            .unwrap();
        let mut first = crate::test_memory::Allocation::new(
            runtime.cuda_stream().unwrap().clone(),
            runtime.arena_bytes().unwrap(),
        )
        .unwrap();
        let mut second = crate::test_memory::Allocation::new(
            runtime.cuda_stream().unwrap().clone(),
            runtime.arena_bytes().unwrap(),
        )
        .unwrap();
        let input_range = runtime.input_arena_range(increment.id).unwrap();
        let output_range = runtime.output_arena_range(previous.id).unwrap();
        let increment_data = HostBuffer::from(vec![1f32; 4]);
        first
            .write(input_range.offset, &increment_data.bytes)
            .unwrap();
        second
            .write(input_range.offset, &increment_data.bytes)
            .unwrap();
        let read_previous = |arena: &crate::test_memory::Allocation| {
            HostBuffer::new(
                luminal::dtype::PlanDtype::F32,
                arena.read(output_range.offset, output_range.bytes).unwrap(),
            )
            .unwrap()
            .as_f32()
            .unwrap()
        };
        let mut staging = runtime.allocate_staging().unwrap();
        runtime.execute(first.arena(), &mut staging).unwrap();
        assert_eq!(read_previous(&first), vec![0.]);
        runtime.execute(second.arena(), &mut staging).unwrap();
        assert_eq!(read_previous(&second), vec![4.]);
        let mut too_small =
            crate::test_memory::Allocation::new(device.stream().clone(), 1).unwrap();
        assert!(runtime.execute(too_small.arena(), &mut staging).is_err());
        let mut replacement_staging = runtime.allocate_staging().unwrap();
        runtime
            .execute(first.arena(), &mut replacement_staging)
            .unwrap();
        assert_eq!(read_previous(&first), vec![8.]);
    }
}
