//! The native Metal runtime: load, bind, search, stage, execute, fetch.
//! Registry and search state belong to the runtime; core supplies the IR.

use crate::host_buffer::HostBuffer;
use anyhow::{Context, Result, anyhow, bail};
use luminal::bufferize::BufferIrGraph;

use crate::search::{CompileOptions, SearchOutcome};
use luminal::graph;
use luminal::layouts::DecodedLayout;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal::shape;

struct NativeParts {
    pre_schedule: String,
    input_slots: Vec<graph::InputSlot>,
    output_slots: Vec<graph::OutputSlot>,
    post_checks: String,
    labeled_checks: Vec<(String, String)>,
    binding_seeds: String,
}

#[derive(Default)]
pub struct MetalRuntime {
    native: Option<NativeParts>,
    matchers: Vec<Box<dyn luminal::layout_ir::OpMatcher>>,
    allow: Vec<&'static str>,
    decoders: luminal::egglog_utils::eclass::ConstructorRegistry,
    plan: Option<BufferIrGraph<DecodedLayout>>,
    staged: FxHashMap<i64, HostBuffer>,
    outputs_host: FxHashMap<usize, (HostBuffer, luminal::bufferize::OutputBinding<DecodedLayout>)>,
    input_buffers: FxHashMap<NodeIndex, i64>,
    output_index: FxHashMap<NodeIndex, usize>,
    dim_buckets: std::collections::BTreeMap<shape::Symbol, Vec<graph::DimBucket>>,
    bucket_plans: Vec<crate::search::BucketPlan>,
    selected_bucket: Option<usize>,
    dims: shape::DynMap,
    range_bound: std::collections::BTreeMap<shape::Symbol, (u64, u64)>,
    #[cfg(target_os = "macos")]
    device: Option<crate::device::MetalDevice>,
}

impl MetalRuntime {
    pub fn load(graph: &graph::Graph) -> Result<Self> {
        Self::load_with_registry(graph, crate::ops::metal_registry())
    }

    pub fn load_with_registry(
        graph: &graph::Graph,
        registry: Vec<crate::ops::RegisteredOp>,
    ) -> Result<Self> {
        let (pre_schedule, input_slots, output_slots, post_checks, labeled_checks) = graph
            .logical
            .bound_parts(&crate::bindings::MetalBindings)
            .map_err(|e| anyhow!(e))?;
        let allow = Self::allow_list_over(&registry);
        let matchers: Vec<Box<dyn luminal::layout_ir::OpMatcher>> =
            registry.into_iter().map(|entry| entry.matcher).collect();
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

    fn matchers(&self) -> &[Box<dyn luminal::layout_ir::OpMatcher>] {
        &self.matchers
    }

    pub fn decoders(&self) -> &luminal::egglog_utils::eclass::ConstructorRegistry {
        &self.decoders
    }

    pub fn active_allow_list(&self) -> &[&'static str] {
        &self.allow
    }

    fn invalidate_plans(&mut self) {
        self.plan = None;
        self.bucket_plans.clear();
        self.selected_bucket = None;
        self.outputs_host.clear();
        #[cfg(target_os = "macos")]
        if let Some(device) = &mut self.device {
            device.release_slab();
        }
    }

    pub fn bind_dyn_range(
        &mut self,
        var: impl Into<shape::Symbol>,
        lower: u64,
        upper: u64,
    ) -> Result<()> {
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
        self.range_bound.insert(name, (lower, upper));
        if lower == upper {
            self.dims.insert(name, lower as usize);
        }
        self.invalidate_plans();
        Ok(())
    }

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
        self.invalidate_plans();
        Ok(())
    }

    pub fn set_dim(&mut self, dim: impl Into<shape::Symbol>, value: usize) {
        self.dims.insert(dim.into(), value);
    }

    pub fn bucket_plans(&self) -> &[crate::search::BucketPlan] {
        &self.bucket_plans
    }

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

    pub fn saturated_egraph(&self) -> Result<luminal::prelude::egraph_serialize::EGraph> {
        let (serialized, _program) =
            self.assemble_and_saturate(Some(crate::saturation::DEFAULT_ALGEBRA_MATCH_BUDGET))?;
        Ok(serialized)
    }

    fn assemble_and_saturate(
        &self,
        algebra_match_budget: Option<usize>,
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
                crate::bindings::MetalBindings::SCHEDULE,
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
        if let Err(err) = crate::saturation::run_program(&mut egraph, &full, algebra_match_budget) {
            let mut doors = Vec::new();
            let unchecked = format!(
                "{}\n\n{}\n{}\n{}",
                luminal::egglog_snippet::assembled_program_for(self.matchers()),
                native.pre_schedule,
                native.binding_seeds,
                crate::bindings::MetalBindings::SCHEDULE
            );
            let mut probe = luminal::egglog_snippet::new_egraph();
            if crate::saturation::run_program(&mut probe, &unchecked, algebra_match_budget).is_ok()
            {
                for (label, text) in &native.labeled_checks {
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
        input_data: &FxHashMap<NodeIndex, HostBuffer>,
        options: &CompileOptions,
    ) -> Result<SearchOutcome> {
        self.invalidate_plans();
        let mut resolved_options = options.clone();
        #[cfg(target_os = "macos")]
        if let Some(device) = metal::Device::system_default() {
            let limit = usize::try_from(device.max_buffer_length())?;
            resolved_options.device_budget_bytes = Some(
                resolved_options
                    .device_budget_bytes
                    .map_or(limit, |requested| requested.min(limit)),
            );
        }

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

        for tensor in input_data.keys() {
            assert!(
                native.input_slots.iter().any(|slot| slot.tensor == *tensor),
                "tensor {tensor:?} is not a bound input"
            );
        }
        #[cfg(not(target_os = "macos"))]
        anyhow::ensure!(
            !options.profile_on_device,
            "device profiling requires macOS: this host can search by the heuristic, but a request to MEASURE \
             must not be answered with a prior"
        );

        #[cfg(target_os = "macos")]
        let staged_for_search: FxHashMap<i64, &HostBuffer> = if options.profile_on_device {
            native
                .input_slots
                .iter()
                .filter_map(|slot| input_data.get(&slot.tensor).map(|data| (slot.buffer, data)))
                .collect()
        } else {
            FxHashMap::default()
        };
        #[cfg(target_os = "macos")]
        if options.profile_on_device && self.device.is_none() {
            self.device = Some(crate::device::MetalDevice::new()?);
        }

        let base = if self.dim_buckets.is_empty() {
            Some(self.assemble_and_saturate(options.algebra_match_budget)?)
        } else {
            None
        };

        let allow = self.allow.clone();
        let matchers = &self.matchers;
        let mut evaluator = {
            #[cfg(target_os = "macos")]
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
            #[cfg(not(target_os = "macos"))]
            {
                crate::search::Evaluator::Heuristic
            }
        };

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
            let assembly = crate::search::BucketAssembly {
                assembled_program: &luminal::egglog_snippet::assembled_program_for(matchers),
                pre_schedule: &native.pre_schedule,
                binding_seeds: &native.binding_seeds,
                schedule: crate::bindings::MetalBindings::SCHEDULE,
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
            self.plan = Some(plan);
        } else {
            let _ = self.select_bucket_plan();
        }
        Ok(outcome)
    }

    pub fn set_data(&mut self, tensor: NodeIndex, data: impl Into<HostBuffer>) {
        let Some(&buffer) = self.input_buffers.get(&tensor) else {
            panic!("set_data on a tensor with no input binding");
        };
        self.staged.insert(buffer, data.into());
    }

    pub fn execute(&mut self) -> Result<()> {
        self.outputs_host.clear();
        if !self.bucket_plans.is_empty() {
            self.select_bucket_plan()?;
        }
        #[cfg(target_os = "macos")]
        {
            if self.device.is_none() {
                self.device = Some(crate::device::MetalDevice::new()?);
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
                device.install(plans)?;
            }
            let bucket = self.selected_bucket.unwrap_or(0);
            let staged = self.staged.iter().map(|(lit, data)| (*lit, data)).collect();
            let outputs = device.execute(bucket, &staged, &self.dims)?;
            self.outputs_host = outputs;
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = self
                .plan()
                .ok_or_else(|| anyhow!("search before execute"))?;
            bail!(
                "Metal execution is only available on macOS: plans can be \
                 searched and inspected but not executed on this host"
            )
        }
    }

    /// Return the backing bytes as f32. Use `fetch` and `layouts::dense_f32`
    /// to interpret a non-dense output through its returned layout.
    pub fn get_f32(&self, tensor: NodeIndex) -> Result<Vec<f32>> {
        let (payload, _) = self.fetch(tensor)?;
        payload.as_f32()
    }

    pub fn get_i32(&self, tensor: NodeIndex) -> Result<Vec<i32>> {
        let (payload, _) = self.fetch(tensor)?;
        payload.as_i32()
    }

    pub fn get_i64(&self, tensor: NodeIndex) -> Result<Vec<i64>> {
        let (payload, _) = self.fetch(tensor)?;
        payload.as_i64()
    }

    pub fn get_bool8(&self, tensor: NodeIndex) -> Result<&[u8]> {
        let (payload, _) = self.fetch(tensor)?;
        payload.as_bool8()
    }

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

    pub fn output_layout(
        &self,
        tensor: NodeIndex,
    ) -> Result<&luminal::bufferize::OutputBinding<DecodedLayout>> {
        Ok(self.fetch(tensor)?.1)
    }

    pub fn plan(&self) -> Option<&BufferIrGraph<DecodedLayout>> {
        self.selected_bucket
            .and_then(|i| self.bucket_plans.get(i))
            .map(|p| &p.plan)
            .or(self.plan.as_ref())
    }
}
