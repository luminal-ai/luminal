//! Profile candidates through the same Metal executor used by serving.
//! Compilation, explicit input transfers and warmup precede timing.
//! The profiling caller owns candidate storage and restores staged inputs per trial.

use anyhow::{Result, anyhow, ensure};
use metal::{Buffer, CommandQueue, Device, MTLCommandBufferStatus, MTLResourceOptions, NSRange};

use std::time::{Duration, Instant};

use luminal::bufferize::{BufferId, BufferIrGraph, BufferNode};
use luminal::layouts::DecodedLayout;
use luminal::prelude::{FxHashMap, FxHashSet};

use crate::arena::{ArenaPlan, ArenaSlice};
use crate::device::MetalExecutable;
use crate::host_buffer::HostBuffer;
use crate::search::early_stop_exceeded;

#[derive(Debug, Clone, Copy)]
pub enum Measurement {
    Timed {
        mean_nanos: u128,
        completed_trials: usize,
    },
    TimedOut {
        elapsed_nanos: u128,
        completed_trials: usize,
    },
}

#[derive(Debug)]
pub enum ProfileFailure {
    Prepare(anyhow::Error),
    Execute(anyhow::Error),
}

impl std::fmt::Display for ProfileFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileFailure::Prepare(err) => write!(f, "prepare: {err:#}"),
            ProfileFailure::Execute(err) => write!(f, "execute: {err:#}"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn profile_candidate_at(
    device: &mut MetalExecutable,
    plan: &BufferIrGraph<DecodedLayout>,
    staged: &FxHashMap<i64, &HostBuffer>,

    trials: usize,
    best_so_far: Option<u128>,
    candidate_timeout: Option<Duration>,
    shapes: &crate::symbolic::ShapeEnv,
    arena_budget: Option<usize>,
) -> Result<Measurement, ProfileFailure> {
    luminal::collective::ensure_local_profile(plan).map_err(ProfileFailure::Prepare)?;
    let staged = prepare_candidate(device, plan, staged, shapes, arena_budget)
        .map_err(ProfileFailure::Prepare)?;
    let mut arena = ProfilingMemory::new(device.device(), device.queue(), device.slab_bytes())
        .map_err(ProfileFailure::Prepare)?;
    initialize_inputs(device, plan, &mut arena, &staged, &shapes.values, None)
        .map_err(ProfileFailure::Prepare)?;
    let restore =
        trial_restore_buffers(plan, device.memory_plan().map_err(ProfileFailure::Prepare)?);
    device
        .execute(arena.buffer(), &shapes.values)
        .map_err(ProfileFailure::Prepare)?;

    let total = trials.max(1);
    let mut sum = 0u128;
    for trial in 0..total {
        // Restore any boundary buffer whose arena bytes can be modified during execution;
        // read-only buffers that are never written or overwritten by arena reuse stay resident.
        if !restore.is_empty() {
            initialize_inputs(
                device,
                plan,
                &mut arena,
                &staged,
                &shapes.values,
                Some(&restore),
            )
            .map_err(ProfileFailure::Prepare)?;
        }
        let start = Instant::now();
        device
            .execute(arena.buffer(), &shapes.values)
            .map_err(ProfileFailure::Execute)?;
        sum += start.elapsed().as_nanos();
        let completed = trial + 1;
        if completed == total {
            break;
        }
        if candidate_timeout.is_some_and(|budget| sum > budget.as_nanos()) {
            return Ok(Measurement::TimedOut {
                elapsed_nanos: sum,
                completed_trials: completed,
            });
        }
        if best_so_far.is_some_and(|best| early_stop_exceeded(sum / total as u128, best, 1.0)) {
            return Ok(Measurement::Timed {
                mean_nanos: sum / completed as u128,
                completed_trials: completed,
            });
        }
    }
    if candidate_timeout.is_some_and(|budget| sum > budget.as_nanos()) {
        return Ok(Measurement::TimedOut {
            elapsed_nanos: sum,
            completed_trials: total,
        });
    }
    Ok(Measurement::Timed {
        mean_nanos: sum / total as u128,
        completed_trials: total,
    })
}

pub fn profile_candidate(
    device: &mut MetalExecutable,
    plan: &BufferIrGraph<DecodedLayout>,
    staged: &FxHashMap<i64, &HostBuffer>,

    trials: usize,
    best: Option<u128>,
    timeout: Option<Duration>,
) -> Result<Measurement, ProfileFailure> {
    profile_candidate_at(
        device,
        plan,
        staged,
        trials,
        best,
        timeout,
        &Default::default(),
        None,
    )
}

fn prepare_candidate<'a>(
    device: &mut MetalExecutable,
    plan: &BufferIrGraph<DecodedLayout>,
    staged: &FxHashMap<i64, &'a HostBuffer>,

    shapes: &crate::symbolic::ShapeEnv,
    arena_budget: Option<usize>,
) -> anyhow::Result<FxHashMap<i64, std::borrow::Cow<'a, HostBuffer>>> {
    luminal::collective::ensure_local_profile(plan)?;
    let mut owned = FxHashMap::default();
    for buffer in plan.buffers.values() {
        if let Some(lit) = buffer.lit
            && let Some(data) = staged.get(&lit)
        {
            let bytes = crate::symbolic::bytes(&buffer.layout, &shapes.values)?;
            anyhow::ensure!(
                data.bytes.len() >= bytes,
                "profiling input {lit} has {} bytes but its declared profiling shape requires {bytes}",
                data.bytes.len()
            );
            let data = std::borrow::Cow::Borrowed(*data);
            owned.insert(lit, data);
        }
    }
    // Profiling supplies host stand-ins for external boundaries. Their storage
    // belongs to this profiling invocation, not the installed executable.
    let bindings = Default::default();
    device.install_with_bindings(
        (plan.clone(), shapes.bounds.clone()),
        bindings,
        arena_budget,
    )?;
    Ok(owned)
}

/// Boundary buffers whose arena bytes can be modified during execution and
/// therefore must be restored before each timed trial.
///
/// A boundary buffer is included if any DAG node writes directly to it
/// (`Compute` writes/results or `BufferCopy`/`BufferCopyValue` destinations),
/// or if its physical arena slice overlaps any written buffer, parameter block,
/// or workspace (for example, when a `FreedBy::Program` input's arena slot is
/// reused after its last read).
fn trial_restore_buffers(
    plan: &BufferIrGraph<DecodedLayout>,
    memory: &ArenaPlan,
) -> FxHashSet<BufferId> {
    let mut written_buffers = FxHashSet::default();

    for node in plan.dag.node_weights() {
        match node {
            BufferNode::BufferCopy { dst, .. } | BufferNode::BufferCopyValue { dst, .. } => {
                written_buffers.insert(dst.clone());
            }
            BufferNode::Compute {
                op,
                writes,
                result_info,
                ..
            } => {
                if op.as_any().is::<luminal::buffer_tensor_ir::BufferAlloc>()
                    || op.as_any().is::<luminal::buffer_tensor_ir::BufferFree>()
                {
                    continue;
                }
                for id in writes.iter().chain(result_info.iter().map(|s| &s.buffer)) {
                    written_buffers.insert(id.clone());
                }
            }
            BufferNode::BufferInput { .. } | BufferNode::BufferOutput { .. } => {}
        }
    }

    let slice_range = |slice: ArenaSlice| -> Option<(usize, usize)> {
        (slice.bytes > 0).then(|| (slice.offset, slice.offset.saturating_add(slice.reserved())))
    };

    let mut written_ranges = Vec::new();
    if let Some(range) = slice_range(memory.parameters) {
        written_ranges.push(range);
    }
    for &ws in memory.workspaces.values() {
        if let Some(range) = slice_range(ws) {
            written_ranges.push(range);
        }
    }
    for id in &written_buffers {
        if let Some(&slice) = memory.slices.get(id)
            && let Some(range) = slice_range(slice)
        {
            written_ranges.push(range);
        }
    }

    let mut restore = FxHashSet::default();
    for buffer in plan.buffers.values().filter(|b| b.lit.is_some()) {
        let Some(&home) = memory.slices.get(&buffer.id) else {
            continue;
        };
        let Some((start, end)) = slice_range(home) else {
            continue;
        };
        let clobbered = written_buffers.contains(&buffer.id)
            || written_ranges
                .iter()
                .any(|&(w_start, w_end)| start < w_end && w_start < end);
        if clobbered {
            restore.insert(buffer.id.clone());
        }
    }
    restore
}

/// Profiling owns data preparation. Missing payloads are explicit synthetic zeros
/// here; the executable never fabricates or uploads tensor data.
fn initialize_inputs(
    device: &MetalExecutable,
    plan: &BufferIrGraph<DecodedLayout>,
    arena: &mut ProfilingMemory,
    inputs: &FxHashMap<i64, std::borrow::Cow<'_, HostBuffer>>,
    dims: &luminal::shape::DynMap,
    only: Option<&FxHashSet<BufferId>>,
) -> anyhow::Result<()> {
    let slices = &device.memory_plan()?.slices;
    for buffer in plan.buffers.values().filter(|b| b.lit.is_some()) {
        if let Some(filter) = only
            && !filter.contains(&buffer.id)
        {
            continue;
        }
        let Some(home) = slices.get(&buffer.id) else {
            continue;
        };
        let bytes = crate::symbolic::bytes(&buffer.layout, dims)?;
        if let Some(data) = inputs.get(&buffer.lit.unwrap()) {
            anyhow::ensure!(
                Some(data.dtype) == buffer.layout.dtype,
                "profiling input dtype mismatch"
            );
            anyhow::ensure!(data.bytes.len() == bytes, "profiling input size mismatch");
            arena.write(home.offset, &data.bytes)?;
        } else {
            arena.zero(home.offset, bytes)?;
        }
    }
    Ok(())
}

/// Validate a finalist using a private profiling environment.
pub(crate) fn warmup_candidate(
    device: &mut MetalExecutable,
    plan: &BufferIrGraph<DecodedLayout>,
    inputs: &FxHashMap<i64, &HostBuffer>,
    shapes: &crate::symbolic::ShapeEnv,
    arena_budget: Option<usize>,
) -> anyhow::Result<()> {
    let inputs = prepare_candidate(device, plan, inputs, shapes, arena_budget)?;
    let mut arena = ProfilingMemory::new(device.device(), device.queue(), device.slab_bytes())?;
    initialize_inputs(device, plan, &mut arena, &inputs, &shapes.values, None)?;
    device.execute(arena.buffer(), &shapes.values)
}

struct ProfilingMemory {
    buffer: Buffer,
    device: Device,
    queue: CommandQueue,
}
impl ProfilingMemory {
    fn new(device: &Device, queue: &CommandQueue, bytes: usize) -> Result<Self> {
        ensure!(
            bytes as u64 <= device.max_buffer_length(),
            "allocation exceeds device limit"
        );
        Ok(Self {
            buffer: device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModePrivate),
            device: device.clone(),
            queue: queue.clone(),
        })
    }

    fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    fn write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| anyhow!("upload range overflow"))?;
        ensure!(
            end as u64 <= self.buffer.length(),
            "upload exceeds allocation"
        );
        for (index, chunk) in data.chunks(16 * 1024 * 1024).enumerate() {
            objc::rc::autoreleasepool(|| -> Result<()> {
                let staging = self.device.new_buffer_with_data(
                    chunk.as_ptr().cast(),
                    chunk.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let command = self.queue.new_command_buffer();
                let blit = command.new_blit_command_encoder();
                blit.copy_from_buffer(
                    &staging,
                    0,
                    &self.buffer,
                    (offset + index * 16 * 1024 * 1024) as u64,
                    chunk.len() as u64,
                );
                blit.end_encoding();
                command.commit();
                command.wait_until_completed();
                ensure!(
                    command.status() == MTLCommandBufferStatus::Completed,
                    "Metal upload failed"
                );
                Ok(())
            })?;
        }
        Ok(())
    }

    fn zero(&mut self, offset: usize, bytes: usize) -> Result<()> {
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("upload range overflow"))?;
        ensure!(
            end as u64 <= self.buffer.length(),
            "upload exceeds allocation"
        );
        if bytes == 0 {
            return Ok(());
        }
        objc::rc::autoreleasepool(|| -> Result<()> {
            let command = self.queue.new_command_buffer();
            let blit = command.new_blit_command_encoder();
            blit.fill_buffer(&self.buffer, NSRange::new(offset as u64, bytes as u64), 0);
            blit.end_encoding();
            command.commit();
            command.wait_until_completed();
            ensure!(
                command.status() == MTLCommandBufferStatus::Completed,
                "Metal zero-fill failed"
            );
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::MetalBindings;
    use crate::device::MetalDevice;
    use crate::runtime::MetalRuntime;
    use luminal::layout_ir::{Access, FreedBy};
    use luminal::prelude::*;

    fn read_back(device: &MetalDevice, buffer: &Buffer, offset: usize, bytes: usize) -> Vec<u8> {
        objc::rc::autoreleasepool(|| {
            let staging = device
                .device()
                .new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared);
            if bytes > 0 {
                let command = device.queue().new_command_buffer();
                let blit = command.new_blit_command_encoder();
                blit.copy_from_buffer(buffer, offset as u64, &staging, 0, bytes as u64);
                blit.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            }
            unsafe { std::slice::from_raw_parts(staging.contents().cast::<u8>(), bytes) }.to_vec()
        })
    }

    #[test]
    fn trial_restore_skips_readonly_inputs_and_restores_written_outputs() {
        let mut graph = Graph::new();
        let x = graph.tensor(vec![2, 4], DType::F32);
        let w = graph.tensor(vec![4, 3], DType::F32);
        let y = x.matmul(w);
        let mut runtime = MetalRuntime::load(&graph).unwrap();
        runtime
            .compile(&Default::default(), &Default::default())
            .unwrap();
        let plan = runtime.plan().unwrap();
        let memory = runtime.memory_plan().unwrap();
        let restore = trial_restore_buffers(plan, memory);

        let x_lit = runtime.input_buffer(x.id).unwrap();
        let w_lit = runtime.input_buffer(w.id).unwrap();
        let x_id = plan
            .buffers
            .values()
            .find(|b| b.lit == Some(x_lit))
            .unwrap()
            .id
            .clone();
        let w_id = plan
            .buffers
            .values()
            .find(|b| b.lit == Some(w_lit))
            .unwrap()
            .id
            .clone();
        let y_id = runtime.output_layout(y.id).unwrap().buffer;

        assert!(!restore.contains(&x_id));
        assert!(!restore.contains(&w_id));
        assert!(
            restore.contains(&y_id),
            "written output buffer must still be restored"
        );
    }

    #[test]
    fn trial_restore_keeps_mutated_input_and_skips_readonly_weight() {
        let mut graph = Graph::new();
        let state = graph.tensor(vec![4], DType::F32);
        let inc = graph.tensor(vec![4], DType::F32);
        let prev = state.sum(vec![0]);
        let next = state + inc;

        let mut bindings = MetalBindings::new();
        let state_lit = bindings.input(state.id);
        bindings.declare(state_lit, Access::ReadWrite, FreedBy::Caller);
        let inc_lit = bindings.input(inc.id);
        bindings.output(prev.id);
        bindings.output_on(next.id, state_lit);

        let mut runtime =
            MetalRuntime::load_with(&graph, bindings, crate::ops::metal_registry()).unwrap();
        runtime
            .compile(&Default::default(), &Default::default())
            .unwrap();
        let plan = runtime.plan().unwrap();
        let memory = runtime.memory_plan().unwrap();
        let restore = trial_restore_buffers(plan, memory);

        let state_id = plan
            .buffers
            .values()
            .find(|b| b.lit == Some(state_lit))
            .unwrap()
            .id
            .clone();
        let inc_id = plan
            .buffers
            .values()
            .find(|b| b.lit == Some(inc_lit))
            .unwrap()
            .id
            .clone();
        assert!(
            restore.contains(&state_id),
            "mutated input must be restored"
        );
        assert!(
            !restore.contains(&inc_id),
            "read-only input should not be restored"
        );
    }

    #[test]
    fn trial_restore_detects_arena_slice_overlap_for_readonly_input() {
        let mut graph = Graph::new();
        let x = graph.tensor(vec![4], DType::F32);
        let y = x + 1.0;

        let mut bindings = MetalBindings::new();
        let x_lit = bindings.input(x.id);
        bindings.declare(x_lit, Access::ReadOnly, FreedBy::Program);
        bindings.output(y.id);

        let mut runtime =
            MetalRuntime::load_with(&graph, bindings, crate::ops::metal_registry()).unwrap();
        runtime
            .compile(&Default::default(), &Default::default())
            .unwrap();
        let plan = runtime.plan().unwrap();
        let mut memory = runtime.memory_plan().unwrap().clone();

        let x_id = plan
            .buffers
            .values()
            .find(|b| b.lit == Some(x_lit))
            .unwrap()
            .id
            .clone();
        let out_id = runtime.output_layout(y.id).unwrap().buffer;
        // Simulate arena reuse placing the written output over the donated input's offset.
        let x_slice = memory.slices[&x_id];
        memory.slices.get_mut(&out_id).unwrap().offset = x_slice.offset;

        let restore = trial_restore_buffers(plan, &memory);
        assert!(
            restore.contains(&x_id),
            "input overlapped by a written arena slice must be restored"
        );
    }

    #[test]
    fn mutated_input_is_restored_between_executions_while_readonly_persists() {
        let mut graph = Graph::new();
        let state = graph.tensor(vec![4], DType::F32);
        let inc = graph.tensor(vec![4], DType::F32);
        let prev = state.sum(vec![0]);
        let next = state + inc;

        let mut bindings = MetalBindings::new();
        let state_lit = bindings.input(state.id);
        bindings.declare(state_lit, Access::ReadWrite, FreedBy::Caller);
        let inc_lit = bindings.input(inc.id);
        bindings.output(prev.id);
        bindings.output_on(next.id, state_lit);

        let device = MetalDevice::new().unwrap();
        let mut runtime = MetalRuntime::load_with(&graph, bindings, crate::ops::metal_registry())
            .unwrap()
            .with_device(&device)
            .unwrap();
        runtime
            .compile(&Default::default(), &Default::default())
            .unwrap();
        let plan = runtime.plan().unwrap().clone();
        let mut exec = device.executable();

        let state_buf = HostBuffer::from(vec![1.0f32, 2.0, 3.0, 4.0]);
        let inc_buf = HostBuffer::from(vec![10.0f32, 10.0, 10.0, 10.0]);
        let staged: FxHashMap<i64, &HostBuffer> = [(state_lit, &state_buf), (inc_lit, &inc_buf)]
            .into_iter()
            .collect();
        let shapes = crate::symbolic::ShapeEnv::default();
        let owned = prepare_candidate(&mut exec, &plan, &staged, &shapes, None).unwrap();
        let mut arena =
            ProfilingMemory::new(exec.device(), exec.queue(), exec.slab_bytes()).unwrap();
        initialize_inputs(&exec, &plan, &mut arena, &owned, &shapes.values, None).unwrap();
        let restore = trial_restore_buffers(&plan, exec.memory_plan().unwrap());
        let state_id = plan
            .buffers
            .values()
            .find(|b| b.lit == Some(state_lit))
            .unwrap()
            .id
            .clone();
        let inc_id = plan
            .buffers
            .values()
            .find(|b| b.lit == Some(inc_lit))
            .unwrap()
            .id
            .clone();
        assert!(restore.contains(&state_id));
        assert!(!restore.contains(&inc_id));

        let prev_slot = runtime.output_layout(prev.id).unwrap();
        let prev_slice = exec.memory_plan().unwrap().slices[&prev_slot.buffer];
        let state_slice = exec.memory_plan().unwrap().slices[&state_id];

        // Warmup mutates `state` in place from [1,2,3,4] to [11,12,13,14].
        exec.execute(arena.buffer(), &shapes.values).unwrap();
        let after_warmup = HostBuffer::new(
            luminal::dtype::PlanDtype::F32,
            read_back(&device, arena.buffer(), state_slice.offset, 16),
        )
        .unwrap()
        .as_f32()
        .unwrap();
        assert_eq!(after_warmup, vec![11.0, 12.0, 13.0, 14.0]);

        // Restoring only `restore` resets `state` while keeping read-only `inc` resident.
        for _ in 0..3 {
            initialize_inputs(
                &exec,
                &plan,
                &mut arena,
                &owned,
                &shapes.values,
                Some(&restore),
            )
            .unwrap();
            exec.execute(arena.buffer(), &shapes.values).unwrap();
            let prev_sum = HostBuffer::new(
                luminal::dtype::PlanDtype::F32,
                read_back(&device, arena.buffer(), prev_slice.offset, 4),
            )
            .unwrap()
            .as_f32()
            .unwrap();
            assert_eq!(prev_sum, vec![10.0]);
            let next_state = HostBuffer::new(
                luminal::dtype::PlanDtype::F32,
                read_back(&device, arena.buffer(), state_slice.offset, 16),
            )
            .unwrap()
            .as_f32()
            .unwrap();
            assert_eq!(next_state, vec![11.0, 12.0, 13.0, 14.0]);
        }
    }

    #[test]
    fn zero_fill_handles_odd_and_empty_ranges() {
        let device = MetalDevice::new().unwrap();
        let mut memory = ProfilingMemory::new(device.device(), device.queue(), 64).unwrap();
        memory.write(0, &[0xAAu8; 64]).unwrap();
        memory.zero(7, 1).unwrap();
        memory.zero(11, 7).unwrap();
        memory.zero(25, 0).unwrap();

        let got = read_back(&device, memory.buffer(), 0, 64);
        let mut expected = vec![0xAAu8; 64];
        expected[7] = 0;
        expected[11..18].fill(0);
        assert_eq!(got, expected);
    }

    #[test]
    fn mini_llama3_skipped_inputs_stay_byte_exact_across_trials() {
        use luminal::shape::IntExpr;
        use model_zoo::mini::llama3::MiniLlama3;

        let mut cx = Graph::new();
        let model = MiniLlama3::new(5, 8, 12, 4, 2, 1, &mut cx);
        let ids = cx.tensor(vec![1], DType::Int);
        let caches = vec![(
            cx.tensor(vec![4, 4], DType::F32),
            cx.tensor(vec![4, 4], DType::F32),
        )];
        let gather = cx.tensor(vec![2], DType::Int);
        let scatter = cx.tensor(vec![1], DType::Int);
        let (output, cache_outputs) =
            model.forward(ids, &caches, gather, scatter, IntExpr::from(1usize));
        let mut outputs = vec![output];
        outputs.extend(
            cache_outputs
                .into_iter()
                .flat_map(|(key, value)| [key, value]),
        );

        let overrides: FxHashMap<NodeIndex, HostBuffer> = [
            (ids.id, vec![3i32].into()),
            (gather.id, vec![0i32, 1].into()),
            (scatter.id, vec![1i32].into()),
        ]
        .into_iter()
        .collect();
        let input_data: FxHashMap<NodeIndex, HostBuffer> = cx
            .logical
            .input_specs()
            .into_iter()
            .enumerate()
            .map(|(seed, spec)| {
                let elements = spec
                    .dims
                    .iter()
                    .map(|dim| dim.to_usize().expect("static mini dimension"))
                    .product::<usize>();
                let value = if let Some(value) = overrides.get(&spec.id) {
                    value.clone()
                } else {
                    match spec.dtype {
                        DType::F32 => HostBuffer::from(
                            (0..elements)
                                .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
                                .collect::<Vec<_>>(),
                        ),
                        DType::Int => HostBuffer::from((0..elements as i32).collect::<Vec<_>>()),
                        DType::I64 => HostBuffer::from((0..elements as i64).collect::<Vec<_>>()),
                        other => panic!("unsupported mini input dtype {other:?}"),
                    }
                };
                (spec.id, value)
            })
            .collect();

        let read: Vec<NodeIndex> = outputs.iter().map(|t| t.id).collect();
        let device = MetalDevice::new().unwrap();
        let mut runtime = MetalRuntime::load_with(
            &cx,
            MetalBindings::dense(&cx.logical, &read),
            crate::ops::metal_registry(),
        )
        .unwrap()
        .with_device(&device)
        .unwrap();
        runtime
            .compile(&Default::default(), &Default::default())
            .unwrap();
        let plan = runtime.plan().unwrap().clone();
        let staged: FxHashMap<i64, &HostBuffer> = input_data
            .iter()
            .map(|(id, buf)| (runtime.input_buffer(*id).unwrap(), buf))
            .collect();

        let mut exec = device.executable();
        let shapes = crate::symbolic::ShapeEnv::default();
        let owned = prepare_candidate(&mut exec, &plan, &staged, &shapes, None).unwrap();
        let mut arena =
            ProfilingMemory::new(exec.device(), exec.queue(), exec.slab_bytes()).unwrap();
        initialize_inputs(&exec, &plan, &mut arena, &owned, &shapes.values, None).unwrap();
        let memory = exec.memory_plan().unwrap().clone();
        let restore = trial_restore_buffers(&plan, &memory);

        let skipped: Vec<(BufferId, ArenaSlice, Vec<u8>)> = plan
            .buffers
            .values()
            .filter(|b| b.lit.is_some() && !restore.contains(&b.id))
            .filter_map(|b| {
                let slice = *memory.slices.get(&b.id)?;
                let bytes = read_back(&device, arena.buffer(), slice.offset, slice.bytes);
                Some((b.id.clone(), slice, bytes))
            })
            .collect();
        assert!(
            !skipped.is_empty(),
            "MiniLlama3 read-only weights should be skipped between trials"
        );

        let output_slices: Vec<ArenaSlice> = outputs
            .iter()
            .map(|t| {
                let slot = runtime.output_layout(t.id).unwrap();
                memory.slices[&slot.buffer]
            })
            .collect();

        // Warmup + 3 trials restoring only `restore`.
        exec.execute(arena.buffer(), &shapes.values).unwrap();
        let mut first_outputs: Option<Vec<Vec<u8>>> = None;
        for _ in 0..3 {
            if !restore.is_empty() {
                initialize_inputs(
                    &exec,
                    &plan,
                    &mut arena,
                    &owned,
                    &shapes.values,
                    Some(&restore),
                )
                .unwrap();
            }
            exec.execute(arena.buffer(), &shapes.values).unwrap();
            let trial_outputs: Vec<Vec<u8>> = output_slices
                .iter()
                .map(|s| read_back(&device, arena.buffer(), s.offset, s.bytes))
                .collect();
            if let Some(expected) = &first_outputs {
                assert_eq!(&trial_outputs, expected, "trial outputs drifted");
            } else {
                first_outputs = Some(trial_outputs);
            }
        }

        for (id, slice, before) in &skipped {
            let after = read_back(&device, arena.buffer(), slice.offset, slice.bytes);
            assert_eq!(
                &after, before,
                "skipped buffer {id:?} was modified during profiling trials"
            );
        }
    }
}
