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

#[derive(Default)]
pub struct CudaRuntime {
    native: Option<NativeParts>,
    /// Train 3: assemble/search with the cuBLASLt marker vocabulary.
    /// RULED always-on (2026-09-01); still OFF by default only until the
    /// search budget/sampler catches up with the collapse's
    /// re-description 2-cycle (see [`crate::ops::cuda_registry_with_cublaslt`]);
    /// enabled through [`CudaRuntime::load_with_cublaslt`].
    cublaslt: bool,
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
}

impl CudaRuntime {
    /// Record the graph's native program. Saturation happens in
    /// [`CudaRuntime::search`].
    pub fn load(graph: &graph::Graph) -> Result<Self> {
        let (pre_schedule, input_slots, output_slots, post_checks, labeled_checks) = graph
            .logical
            .bound_parts(&crate::bindings::CudaBindings)
            .map_err(|e| anyhow!(e))?;
        Ok(Self {
            native: Some(NativeParts {
                pre_schedule,
                input_slots,
                output_slots,
                post_checks,
                labeled_checks,
                binding_seeds: String::new(),
            }),
            ..Self::default()
        })
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
    pub fn load_with_cublaslt(graph: &graph::Graph) -> Result<Self> {
        let mut rt = Self::load(graph)?;
        rt.cublaslt = true;
        Ok(rt)
    }

    /// The matcher vocabulary this instance assembles/searches with.
    fn matchers(&self) -> Vec<Box<dyn luminal::layout_ir::OpMatcher>> {
        if self.cublaslt {
            crate::ops::cuda_matchers_with_cublaslt()
        } else {
            crate::ops::cuda_matchers()
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
        let native = self
            .native
            .as_mut()
            .ok_or_else(|| anyhow!("load before bind"))?;
        let name = var.into();
        native.binding_seeds.push_str(&format!(
            "(set (lower-bound-of (IntVar \"{name}\")) (bigint {lower}))\n\
             (set (upper-bound-of (IntVar \"{name}\")) (bigint {upper}))\n"
        ));
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
            luminal::egglog_snippet::assembled_program_for(&self.matchers()),
            program.text
        );
        let mut egraph = luminal::egglog_snippet::new_egraph();
        if let Err(err) = egraph.parse_and_run_program(None, &full) {
            // Name the door: re-saturate without checks, then probe each
            // labeled check alone.
            let mut doors = Vec::new();
            let unchecked = format!(
                "{}\n\n{}\n{}\n{}",
                luminal::egglog_snippet::assembled_program_for(&self.matchers()),
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

        // The caller's payloads are CHECKED here and go no further: this
        // backend's search ranks candidates with the device-free
        // heuristic (D6, 2026-09-03), which never runs anything, so
        // there is nothing to stage. The check stays because binding a
        // tensor that is not an input of the loaded program is a caller
        // bug either way.
        for tensor in input_data.keys() {
            assert!(
                program
                    .input_slots
                    .iter()
                    .any(|slot| slot.tensor == *tensor),
                "tensor {tensor:?} is not a bound input"
            );
        }

        // Own matchers, own allow list, own ranking: nothing in this
        // search touches another runtime.
        let outcome = crate::search::search_implementations(
            &serialized,
            &program,
            options,
            Some(if self.cublaslt {
                Self::allow_list_with_cublaslt()
            } else {
                Self::allow_list()
            }),
            self.matchers(),
        )?;

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
        self.plan = Some(outcome.best_plan.clone());
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
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| anyhow!("search before execute"))?;
        #[cfg(feature = "device")]
        {
            let outputs = crate::device::execute_plan(plan, &self.staged)?;
            self.outputs_host = outputs;
            Ok(())
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = plan;
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
