//! The native Metal runtime: load, search, bind device storage, execute.
//! Registry and search state belong to the runtime; core supplies the IR.

use crate::host_buffer::HostBuffer;
use anyhow::{Context, Result, anyhow, bail, ensure};
use luminal::bufferize::BufferIrGraph;

use crate::search::{CompileOptions, SearchOutcome};
use luminal::graph;
use luminal::layouts::DecodedLayout;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal::shape;

/// What `load` captured: the bound program (model text, this runtime's
/// boundary, the post-schedule checks) plus whatever the binding calls
/// accumulate before `search` assembles and saturates.
#[derive(Clone)]
struct NativeParts {
    bound: crate::bindings::BoundProgram,
    binding_seeds: String,
}

#[derive(Default)]
pub struct MetalRuntime {
    native: Option<NativeParts>,
    matchers: std::rc::Rc<Vec<Box<dyn luminal::layout_ir::OpMatcher>>>,
    allow: Vec<&'static str>,
    decoders: std::rc::Rc<luminal::egglog_utils::eclass::ConstructorRegistry>,
    plan: Option<BufferIrGraph<DecodedLayout>>,
    storage: Option<crate::arena::ArenaPlan>,
    /// Read by the device execute path, which exists only on macOS.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    externals: std::collections::BTreeSet<i64>,
    device_budget_bytes: Option<usize>,
    input_buffers: FxHashMap<NodeIndex, i64>,
    output_index: FxHashMap<NodeIndex, usize>,
    bounds: shape::DimensionBounds,
    dims: shape::DynMap,
    #[cfg(target_os = "macos")]
    device: Option<crate::device::MetalExecutable>,
}

impl MetalRuntime {
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
            output_index: self.output_index.clone(),
            externals: self.externals.clone(),
            device_budget_bytes: self.device_budget_bytes,
            #[cfg(target_os = "macos")]
            device: self
                .device
                .as_ref()
                .map(|device| device.fork())
                .transpose()?,
        })
    }

    /// LOAD under the default binding: every input read-only on its own
    /// buffer, every leaf read-write on its own.
    pub fn load(graph: &graph::Graph) -> Result<Self> {
        Self::load_with_registry(graph, crate::ops::metal_registry())
    }

    pub fn load_with_registry(
        graph: &graph::Graph,
        registry: Vec<crate::ops::RegisteredOp>,
    ) -> Result<Self> {
        Self::load_with(
            graph,
            crate::bindings::MetalBindings::leaves(&graph.logical),
            registry,
        )
    }

    /// LOAD under the caller's binding — which values enter and leave
    /// through which buffers, and which storage the application supplies. The tensor→buffer maps and the residency set are live
    /// available from here; all tensor transfers are performed by the application.
    pub fn load_with(
        graph: &graph::Graph,
        bindings: crate::bindings::MetalBindings,
        registry: Vec<crate::ops::RegisteredOp>,
    ) -> Result<Self> {
        let bound = bindings
            .bind(&graph.logical)
            .map_err(|reason| anyhow!("load refused: {reason}"))?;
        let allow = Self::allow_list_over(&registry);
        let matchers: Vec<Box<dyn luminal::layout_ir::OpMatcher>> =
            registry.into_iter().map(|entry| entry.matcher).collect();
        let decoders = luminal::egglog_snippet::decoder_registry_for(&matchers)?;
        let input_buffers = bound.inputs.iter().map(|b| (b.value, b.buffer)).collect();
        let output_index = bound
            .outputs
            .iter()
            .enumerate()
            .map(|(index, b)| (b.value, index))
            .collect();
        let externals = bound.externals.clone();
        Ok(Self {
            native: Some(NativeParts {
                bound,
                binding_seeds: String::new(),
            }),
            matchers: std::rc::Rc::new(matchers),
            allow,
            decoders: std::rc::Rc::new(decoders),
            input_buffers,
            output_index,
            externals,
            ..Self::default()
        })
    }

    fn matchers(&self) -> &[Box<dyn luminal::layout_ir::OpMatcher>] {
        &self.matchers
    }

    pub fn decoders(&self) -> &luminal::egglog_utils::eclass::ConstructorRegistry {
        &self.decoders
    }

    pub fn active_allow_list(&self) -> &[&'static str] {
        &self.allow
    }

    /// Use a shared device context/cache while keeping this program's state separate.
    #[cfg(target_os = "macos")]
    pub fn with_device(mut self, device: &crate::device::MetalDevice) -> Result<Self> {
        self.device = Some(device.executable());
        Ok(self)
    }

    /// Execution values never change a program's compiled domain.
    pub fn set_dim(&mut self, dim: impl Into<shape::Symbol>, value: usize) {
        self.dims.insert(dim.into(), value);
    }

    pub fn bounds(&self) -> &shape::DimensionBounds {
        &self.bounds
    }

    #[cfg(target_os = "macos")]
    pub fn graph_stats(&self) -> Option<crate::device::GraphStats> {
        self.device.as_ref().map(|d| d.stats())
    }

    pub fn allow_list() -> Vec<&'static str> {
        Self::allow_list_over(&crate::ops::metal_registry())
    }

    fn allow_list_over(registry: &[crate::ops::RegisteredOp]) -> Vec<&'static str> {
        registry
            .iter()
            .filter(|entry| {
                let prototype = entry.prototype.as_ref();
                if crate::plan_transparent(prototype) {
                    return true;
                }
                let dps = prototype.to_dps();
                let executable = dps.as_deref().unwrap_or(prototype);
                crate::as_kernel_op(executable).is_some()
            })
            .map(|entry| entry.matcher.egglog_constructor())
            .collect()
    }

    pub fn saturated_egraph(
        &self,
        bounds: &shape::DimensionBounds,
    ) -> Result<luminal::prelude::egraph_serialize::EGraph> {
        let (serialized, _program) = self.assemble_and_saturate(
            bounds,
            Some(crate::saturation::DEFAULT_ALGEBRA_MATCH_BUDGET),
        )?;
        Ok(serialized)
    }

    fn assemble_and_saturate(
        &self,
        bounds: &shape::DimensionBounds,
        algebra_match_budget: Option<usize>,
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
        if let Err(err) = crate::saturation::run_program(&mut egraph, &full, algebra_match_budget) {
            let mut doors = Vec::new();
            let unchecked = format!(
                "{}\n\n{}",
                luminal::egglog_snippet::assembled_program_for(self.matchers()),
                native.bound.text_unchecked_with_seeds(&seeds)
            );
            let mut probe = luminal::egglog_snippet::new_egraph();
            if crate::saturation::run_program(&mut probe, &unchecked, algebra_match_budget).is_ok()
            {
                for (label, text) in &native.bound.labeled_checks {
                    if probe.parse_and_run_program(None, text).is_err() {
                        doors.push(label.clone());
                    }
                }
            }
            if doors.is_empty() {
                return Err(err).context("Metal saturation failed");
            }
            bail!("shape contracts failed:\n  - {}", doors.join("\n  - "));
        }
        self.decoders.check(&egraph)?;
        let serialized = egraph.serialize(luminal::prelude::egglog::SerializeConfig::default());
        Ok((serialized.egraph, program))
    }

    pub fn search(
        &mut self,
        bounds: &shape::DimensionBounds,
        profile_dims: &shape::DynMap,
        input_data: &FxHashMap<NodeIndex, HostBuffer>,
        options: &CompileOptions,
    ) -> Result<SearchOutcome> {
        ensure!(
            cfg!(target_os = "macos"),
            "candidate search requires a Metal GPU"
        );
        #[cfg(target_os = "macos")]
        if let Some(device) = self.device.as_mut() {
            device.uninstall();
        }
        bounds.validate_values(profile_dims)?;
        let (mut serialized, program) =
            self.assemble_and_saturate(bounds, options.algebra_match_budget)?;
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
        #[cfg(target_os = "macos")]
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
        #[cfg(target_os = "macos")]
        if self.device.is_none() {
            self.device = Some(crate::device::MetalDevice::new()?.executable());
        }
        let mut evaluator = {
            #[cfg(target_os = "macos")]
            {
                crate::search::Evaluator::Device {
                    device: self.device.as_mut().expect("device initialized"),
                    staged: &staged,
                }
            }
            #[cfg(not(target_os = "macos"))]
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

    /// Memory required by this one program under its declared placements.
    pub fn arena_bytes(&self) -> Result<usize> {
        Ok(crate::storage::plan_storage(
            self.plan
                .as_ref()
                .ok_or_else(|| anyhow!("search before sizing memory"))?,
            &self.bounds.ranges(),
            &self.externals,
        )?
        .slab_bytes)
    }

    /// Native dependencies for application-managed storage and transfers.
    #[cfg(target_os = "macos")]
    pub fn metal_device(&self) -> Result<&metal::Device> {
        Ok(self
            .device
            .as_ref()
            .ok_or_else(|| anyhow!("attach a device or search first"))?
            .device())
    }
    #[cfg(target_os = "macos")]
    pub fn metal_queue(&self) -> Result<&metal::CommandQueue> {
        Ok(self
            .device
            .as_ref()
            .ok_or_else(|| anyhow!("attach a device or search first"))?
            .queue())
    }
    pub fn execute(&mut self, #[cfg(target_os = "macos")] arena: &metal::Buffer) -> Result<()> {
        self.execute_external(
            #[cfg(target_os = "macos")]
            arena,
            #[cfg(target_os = "macos")]
            &Default::default(),
        )
    }

    /// All device tensor storage is borrowed from the application for this call.
    pub fn execute_external(
        &mut self,
        #[cfg(target_os = "macos")] arena: &metal::Buffer,
        #[cfg(target_os = "macos")] external: &FxHashMap<i64, crate::device::ExternalBuffer>,
    ) -> Result<()> {
        self.bounds.validate_values(&self.dims)?;
        #[cfg(target_os = "macos")]
        {
            if self.device.is_none() {
                self.device = Some(crate::device::MetalDevice::new()?.executable());
            }
            anyhow::ensure!(self.plan.is_some(), "search before execute");
            ensure!(
                self.externals.iter().all(|id| external.contains_key(id)),
                "missing external buffer binding"
            );
            ensure!(
                external.keys().all(|id| self.externals.contains(id)),
                "unexpected external buffer binding"
            );
            let device = self.device.as_mut().unwrap();
            if !device.is_installed() {
                let plans = (self.plan.as_ref().unwrap().clone(), self.bounds.ranges());
                device.install_with_bindings(
                    plans,
                    self.externals.clone(),
                    self.device_budget_bytes,
                )?;
            }

            device.execute_external(arena, &self.dims, external)?;
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = self
                .plan()
                .ok_or_else(|| anyhow!("search before execute"))?;
            bail!(
                "Metal device execution is only available on macOS: plans can be \
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

    pub fn plan(&self) -> Option<&BufferIrGraph<DecodedLayout>> {
        self.plan.as_ref()
    }
}

#[cfg(all(test, target_os = "macos"))]
mod caller_memory_tests {
    use super::*;
    use luminal::layout_ir::{Access, FreedBy};
    use luminal::prelude::*;

    #[test]
    fn application_state_survives_replaced_execution_storage() {
        let mut graph = Graph::new();
        let state = graph.tensor(4, DType::F32);
        let increment = graph.tensor(4, DType::F32);
        let previous = state.sum(0);
        let next = state + increment;
        let mut bindings = crate::bindings::MetalBindings::new();
        let state_id = bindings.input_external(state.id);
        bindings.declare(state_id, Access::ReadWrite, FreedBy::Caller);
        bindings.input(increment.id);
        bindings.output(previous.id);
        bindings.output_on(next.id, state_id);
        let device = crate::device::MetalDevice::new().unwrap();
        let mut runtime = MetalRuntime::load_with(&graph, bindings, crate::ops::metal_registry())
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
            crate::test_memory::Allocation::new(device.device(), device.queue(), 16).unwrap();
        state_memory.write(0, &[0; 16]).unwrap();
        let external = [(
            state_id,
            crate::device::ExternalBuffer {
                buffer: state_memory.buffer().clone(),
                offset: 0,
                bytes: 16,
            },
        )]
        .into_iter()
        .collect();
        let mut first = crate::test_memory::Allocation::new(
            runtime.metal_device().unwrap(),
            runtime.metal_queue().unwrap(),
            runtime.arena_bytes().unwrap(),
        )
        .unwrap();
        let mut second = crate::test_memory::Allocation::new(
            runtime.metal_device().unwrap(),
            runtime.metal_queue().unwrap(),
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
        runtime.execute_external(first.buffer(), &external).unwrap();
        assert_eq!(read_previous(&first), vec![0.]);
        runtime
            .execute_external(second.buffer(), &external)
            .unwrap();
        assert_eq!(read_previous(&second), vec![4.]);
        let too_small =
            crate::test_memory::Allocation::new(device.device(), device.queue(), 1).unwrap();
        assert!(
            runtime
                .execute_external(too_small.buffer(), &external)
                .is_err()
        );
        runtime.execute_external(first.buffer(), &external).unwrap();
        assert_eq!(read_previous(&first), vec![8.]);
    }
}
