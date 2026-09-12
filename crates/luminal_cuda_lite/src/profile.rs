//! Fixed-input replay for CUDA search. Input snapshots live on the host; one
//! reusable set of device allocations is shared by all examples/candidates.
use super::*;
use luminal::search::{
    ProfileCase, ProfileMeasurement, assign_profile_cases, weighted_profile_cost,
};
use std::ops::Range;

/// Immutable backing bytes. Views of the same storage preserve aliasing and are
/// restored together, once per reset. Bytes use the graph input's physical ABI.
#[derive(Clone, Debug)]
// Retain the caller's allocation. Converting Vec into Arc<[u8]> copies every
// byte and physically commits lazily allocated zero pages for large states.
pub struct ProfileStorage(Arc<Vec<u8>>);
impl ProfileStorage {
    pub fn new(data: impl ToCudaInput) -> Self {
        Self(data.into_cuda_bytes().into())
    }
}

#[derive(Clone, Debug)]
struct InputSchema {
    id: NodeIndex,
    graph: usize,
    dtype: DType,
    bytes: Expression,
}
impl InputSchema {
    fn new(t: GraphTensor) -> Self {
        Self {
            id: t.id,
            graph: t.graph_ref as usize,
            dtype: t.dtype,
            bytes: (t.shape.physical_span() * t.dtype.bits() + 7) / 8,
        }
    }
    fn validate(&self, graph: &Graph) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.graph == graph as *const Graph as usize,
            "profile input belongs to another graph"
        );
        anyhow::ensure!(
            graph
                .input_meta
                .get(&self.id)
                .is_some_and(|(_, dtype)| *dtype == self.dtype),
            "profile binding {:?} is not an input with dtype {:?}",
            self.id,
            self.dtype
        );
        Ok(())
    }
    fn length(&self, dims: &DynMap) -> anyhow::Result<usize> {
        self.bytes
            .exec(dims)
            .ok_or_else(|| anyhow::anyhow!("missing dimensions for profile input {:?}", self.id))
    }
}
#[derive(Clone, Debug)]
struct Binding {
    schema: InputSchema,
    storage: ProfileStorage,
    range: Range<usize>,
    mirror: bool,
}

/// Frozen input bindings for one sample. All supplied bytes are restored, even
/// when a candidate aliases an output to an input. Omit large immutable inputs
/// only by explicitly declaring them shared on `ProfileWorkload`.
#[derive(Clone, Debug, Default)]
pub struct ProfileInputs {
    bindings: Vec<Binding>,
}
impl ProfileInputs {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn input(self, tensor: GraphTensor, data: impl ToCudaInput) -> Self {
        let storage = ProfileStorage::new(data);
        let len = storage.0.len();
        self.view(tensor, &storage, 0..len, false)
    }
    pub fn mirrored_input(self, tensor: GraphTensor, data: impl ToCudaInput) -> Self {
        let storage = ProfileStorage::new(data);
        let len = storage.0.len();
        self.view(tensor, &storage, 0..len, true)
    }
    /// Bind an input to a physical byte range; graph views/strides remain in the
    /// graph. Overlapping ranges must share the same `ProfileStorage` object.
    pub fn view(
        mut self,
        tensor: GraphTensor,
        storage: &ProfileStorage,
        range: Range<usize>,
        mirror: bool,
    ) -> Self {
        self.bindings.push(Binding {
            schema: InputSchema::new(tensor),
            storage: storage.clone(),
            range,
            mirror,
        });
        self
    }
    /// Select existing snapshots without copying their backing bytes.
    pub fn select(&self, tensors: &[GraphTensor]) -> anyhow::Result<Self> {
        let bindings = tensors
            .iter()
            .map(|tensor| {
                self.bindings
                    .iter()
                    .find(|b| {
                        b.schema.id == tensor.id && b.schema.graph == tensor.graph_ref as usize
                    })
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("missing snapshot input {:?}", tensor.id))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self { bindings })
    }
    fn groups(&self) -> Vec<ProfileStorage> {
        let mut groups: Vec<ProfileStorage> = vec![];
        for b in self.bindings.iter().sorted_by_key(|b| b.schema.id) {
            if !groups.iter().any(|s| Arc::ptr_eq(&s.0, &b.storage.0)) {
                groups.push(b.storage.clone());
            }
        }
        groups
    }
}

/// An explicit workload for any supported graph. Case weights describe global
/// invocation frequency. Shared inputs stay in the runtime and are never copied
/// per case; candidates that mutate them are invalid for this replay contract.
#[derive(Clone, Debug, Default)]
pub struct ProfileWorkload {
    cases: Vec<ProfileCase<ProfileInputs>>,
    shared: Vec<InputSchema>,
    timing_method: luminal::op::TimingMethod,
    device_snapshots: bool,
    warmup_trials: Option<usize>,
    reused_inputs: ProfileInputs,
    space_token: Option<Arc<Box<dyn luminal::op::EgglogOp>>>,
}
impl ProfileWorkload {
    pub fn new() -> Self {
        Self::default()
    }
    /// Device timestamps measure launched device work; host timing includes
    /// ordinary planning, parameter updates and dispatch through completion.
    /// Both exclude snapshot loading and untimed warmup. This is steady repeated
    /// invocation timing, not an application scheduler or multi-step trace metric.
    pub fn timing_method(mut self, method: luminal::op::TimingMethod) -> Self {
        self.timing_method = method;
        self
    }
    /// Untimed exact replays before measuring each candidate (default: one).
    /// Use additional replays when lazy initialization or managed-memory
    /// residency persists beyond the first invocation. These use the same
    /// case and restore its state before every invocation; no data is generated.
    pub fn warmup_trials(mut self, trials: usize) -> Self {
        assert!(trials > 0, "profile warmup count must be positive");
        self.warmup_trials = Some(trials);
        self
    }
    /// Cache immutable sample backing on the device. Repeated resets use device
    /// copies instead of host transfers, at the cost of extra device memory.
    /// Only the active case's backing stays resident; shared backing objects
    /// survive case switches. Uploads happen before candidate timing.
    pub fn device_snapshots(mut self, enabled: bool) -> Self {
        self.device_snapshots = enabled;
        self
    }
    /// Replay selected inputs in their existing owned device allocations. These
    /// inputs need no additional replay slots. `device_snapshots` also applies
    /// to their trial resets. On exit (including candidate failure), restore the supplied
    /// bytes and the original bindings. Capture `restore` immediately before
    /// search to preserve current application state. Each selected input must
    /// own a non-overlapping allocation and use a whole-storage snapshot.
    pub fn reuse_input_buffers(mut self, restore: ProfileInputs) -> Self {
        self.reused_inputs = restore;
        self
    }
    pub fn cases(&self) -> &[ProfileCase<ProfileInputs>] {
        &self.cases
    }
    pub fn shared_input(mut self, tensor: GraphTensor) -> Self {
        self.shared.push(InputSchema::new(tensor));
        self
    }
    pub fn case(
        mut self,
        id: impl Into<String>,
        dims: DynMap,
        inputs: ProfileInputs,
        weight: f64,
    ) -> Self {
        self.cases.push(ProfileCase::new(id, dims, inputs, weight));
        self
    }
}

/// One candidate's complete per-case measurements (direct or graph execution).
#[derive(Clone, Debug)]
pub struct ProfileEvaluation {
    pub bucket: usize,
    pub cuda_graph: bool,
    pub timing_method: luminal::op::TimingMethod,
    pub cases: Vec<ProfileMeasurement>,
    pub weighted_cost: Duration,
}

impl ProfileEvaluation {
    pub(crate) fn display(&self) -> String {
        format!(
            "weighted workload score={:?}; measured case latencies=[{}]",
            self.weighted_cost,
            self.cases
                .iter()
                .map(|case| format!("{}: {:?}", case.case_id, case.duration))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

pub(super) struct ReplaySession {
    assigned: Vec<Vec<usize>>,
    slots: Vec<Option<CudaSlice<u8>>>,
    group_ptrs: Vec<Vec<u64>>,
    reused: Vec<(u64, Binding)>,
    snapshots: FxHashMap<usize, CudaSlice<u8>>,
    active: Option<usize>,
    host_states: Vec<Box<dyn crate::host::ProfileState>>,
    saved_inputs: FxHashMap<NodeIndex, CudaInput>,
    saved_prepared: FxHashMap<NodeIndex, PreparedInput>,
    saved_prepared_unified: FxHashMap<NodeIndex, PreparedUnifiedOwner>,
    saved_mirrors: FxHashMap<NodeIndex, Vec<u8>>,
    saved_external: FxHashMap<NodeIndex, std::mem::ManuallyDrop<CudaSlice<u8>>>,
    saved_outputs: FxHashMap<NodeIndex, (u64, usize)>,
}

impl ReplaySession {
    pub(super) fn managed_capacity(&self) -> usize {
        self.saved_prepared_unified
            .values()
            .map(|o| o.buffer.len())
            .fold(0usize, usize::saturating_add)
    }
    pub(super) fn owned_device_bytes(&self) -> usize {
        self.saved_inputs
            .values()
            .filter_map(|input| match input {
                CudaInput::Buffer { buf, .. } => Some(buf.len()),
                CudaInput::Ptr(_) => None,
            })
            .chain(self.saved_prepared_unified.values().map(|o| o.buffer.len()))
            .chain(self.slots.iter().flatten().map(CudaSlice::len))
            .chain(self.snapshots.values().map(CudaSlice::len))
            .fold(0usize, usize::saturating_add)
    }
}

impl<O: IntoEgglogOp> CudaRuntimeImpl<O> {
    /// Validate and freeze a workload before `graph.search_with_rng`. Supplying
    /// no workload preserves legacy runtime-bound/synthetic profiling behavior.
    /// Each graph input must be declared in every case or explicitly shared.
    pub fn set_profile_workload(
        &mut self,
        graph: &Graph,
        mut workload: ProfileWorkload,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.profile_replay.is_none(),
            "cannot replace workload during replay"
        );
        anyhow::ensure!(!workload.cases.is_empty(), "profile workload has no cases");
        workload.space_token = graph
            .search_space()
            .ok_or_else(|| anyhow::anyhow!("build the search space before installing a workload"))?
            .ops
            .first()
            .cloned();
        anyhow::ensure!(
            workload
                .cases
                .iter()
                .map(|c| c.weight)
                .sum::<f64>()
                .is_finite(),
            "profile weight sum overflows"
        );
        let expected: FxHashSet<_> = graph.input_meta.keys().copied().collect();
        let mut shared = FxHashSet::default();
        for s in &workload.shared {
            s.validate(graph)?;
            anyhow::ensure!(shared.insert(s.id), "duplicate shared profile input");
        }
        anyhow::ensure!(
            self.prepared_inputs.keys().all(|id| shared.contains(id)),
            "prepared inputs must be shared by the representative workload"
        );
        let mut ids = FxHashSet::default();
        for case in &workload.cases {
            anyhow::ensure!(
                !case.id.is_empty() && ids.insert(&case.id),
                "empty or duplicate profile case ID"
            );
            anyhow::ensure!(
                case.weight.is_finite() && case.weight > 0.,
                "invalid profile weight"
            );
            let mut seen = shared.clone();
            for b in &case.inputs.bindings {
                b.schema.validate(graph)?;
                anyhow::ensure!(
                    seen.insert(b.schema.id),
                    "duplicate profile input {:?}",
                    b.schema.id
                );
                anyhow::ensure!(
                    b.range.start <= b.range.end && b.range.end <= b.storage.0.len(),
                    "invalid profile storage view"
                );
                let bits = b.schema.dtype.bits();
                anyhow::ensure!(
                    bits % 8 != 0 || b.range.start % (bits / 8) == 0,
                    "profile input view is not aligned to its dtype"
                );
                anyhow::ensure!(
                    b.range.len() == b.schema.length(&case.dims)?,
                    "case {} input {:?}: byte length does not match exact dimensions",
                    case.id,
                    b.schema.id
                );
            }
            anyhow::ensure!(
                seen == expected,
                "case {} must bind every graph input or declare it shared",
                case.id
            );
        }
        for b in &workload.reused_inputs.bindings {
            b.schema.validate(graph)?;
        }
        self.profile_workload = Some(workload);
        self.profile_evaluations.clear();
        Ok(())
    }
    pub fn clear_profile_workload(&mut self) {
        assert!(self.profile_replay.is_none());
        self.profile_workload = None;
    }
    pub fn profile_evaluations(&self) -> &[ProfileEvaluation] {
        &self.profile_evaluations
    }

    /// Snapshot selected boundary inputs using their current physical bindings.
    /// Capture immediately before execution, after installing coherent metadata.
    /// Overlapping device ranges are captured as one backing allocation. Explicit
    /// tensor state (including RNG state) uses this same mechanism.
    pub fn capture_profile_inputs(
        &self,
        tensors: &[GraphTensor],
        dims: &DynMap,
    ) -> anyhow::Result<ProfileInputs> {
        self.cuda_stream.synchronize()?;
        let mut ranges = vec![];
        for &tensor in tensors {
            anyhow::ensure!(
                !self.prepared_inputs.contains_key(&tensor.id),
                "prepared inputs must be shared, not snapshotted"
            );
            let (ptr, len) = self
                .current_hlir_device_binding(tensor.id)
                .ok_or_else(|| anyhow::anyhow!("missing capture input {:?}", tensor.id))?;
            anyhow::ensure!(
                len == InputSchema::new(tensor).length(dims)?,
                "capture input length does not match dimensions"
            );
            let end = ptr
                .checked_add(len as u64)
                .ok_or_else(|| anyhow::anyhow!("capture address overflow"))?;
            ranges.push((ptr, end, tensor));
        }
        ranges.sort_by_key(|r| r.0);
        let mut result_inputs = ProfileInputs::new();
        let mut i = 0;
        while i < ranges.len() {
            let start = ranges[i].0;
            let mut end = ranges[i].1;
            let mut j = i + 1;
            while j < ranges.len() && ranges[j].0 < end {
                end = end.max(ranges[j].1);
                j += 1;
            }
            let mut bytes = vec![0u8; (end - start) as usize];
            if !bytes.is_empty() {
                unsafe {
                    result::memcpy_dtoh_sync(&mut bytes, start)?;
                }
            }
            // A device read commits every destination page, including untouched
            // zero-filled state. A fresh zeroed allocation can retain lazy zero
            // pages instead; Arc<Vec<_>> then owns it without an eager copy.
            // The captured values and all overlapping views remain identical.
            if bytes.len() >= 1024 * 1024 && bytes.iter().all(|&byte| byte == 0) {
                bytes = vec![0; bytes.len()];
            }
            let storage = ProfileStorage(bytes.into());
            for &(ptr, end, tensor) in &ranges[i..j] {
                let range = (ptr - start) as usize..(end - start) as usize;
                let mirror = self.hlir_host_mirrors.get(&tensor.id);
                if let Some(mirror) = mirror {
                    anyhow::ensure!(
                        mirror.as_slice() == &storage.0[range.clone()],
                        "capture host mirror disagrees with device input"
                    );
                }
                result_inputs = result_inputs.view(tensor, &storage, range, mirror.is_some());
            }
            i = j;
        }
        Ok(result_inputs)
    }

    pub(crate) fn begin_profile_replay(
        &mut self,
        contexts: &[luminal::search::BucketContext<'_>],
    ) -> anyhow::Result<()> {
        let Some(workload) = &self.profile_workload else {
            return Ok(());
        };
        anyhow::ensure!(
            contexts.first().is_some_and(|c| c
                .space
                .ops
                .first()
                .zip(workload.space_token.as_ref())
                .is_some_and(|(a, b)| Arc::ptr_eq(a, b))),
            "profile workload belongs to another search space"
        );
        anyhow::ensure!(
            self.prepared_inputs
                .keys()
                .all(|id| workload.shared.iter().any(|s| s.id == *id)),
            "prepared inputs must be shared by the representative workload"
        );
        let mut reused = vec![];
        let mut reuse_ids = FxHashSet::default();
        for b in &workload.reused_inputs.bindings {
            let id = b.schema.id;
            anyhow::ensure!(reuse_ids.insert(id), "duplicate reused input");
            anyhow::ensure!(
                !workload.shared.iter().any(|s| s.id == id),
                "reused input cannot also be shared"
            );
            let Some(CudaInput::Buffer { buf, len }) = self.hlir_buffers.get(&id) else {
                anyhow::bail!("reused input must have owned device storage");
            };
            anyhow::ensure!(
                b.range == (0..b.storage.0.len()) && b.range.len() == *len,
                "reused input restore snapshot must cover its whole logical binding"
            );
            let ptr = buf.device_ptr(&self.cuda_stream).0;
            for &other in self.hlir_buffers.keys().filter(|&&other| other != id) {
                if let Some((other_ptr, other_len)) = self.current_hlir_device_binding(other) {
                    anyhow::ensure!(
                        !device_ranges_overlap(ptr, buf.len(), other_ptr, other_len),
                        "reused allocation overlaps another input"
                    );
                }
            }
            for case in &workload.cases {
                let sample = case
                    .inputs
                    .bindings
                    .iter()
                    .find(|s| s.schema.id == id)
                    .ok_or_else(|| anyhow::anyhow!("reused input missing from case"))?;
                anyhow::ensure!(
                    sample.range == (0..sample.storage.0.len())
                        && sample.range.len() <= buf.len()
                        && case
                            .inputs
                            .bindings
                            .iter()
                            .all(|other| other.schema.id == id
                                || !Arc::ptr_eq(&other.storage.0, &sample.storage.0)),
                    "reused case input must fit its owned allocation and have unaliased backing"
                );
            }
            reused.push((ptr, b.clone()));
        }
        let assigned = assign_profile_cases(&workload.cases, contexts)?;
        let mut capacities: Vec<usize> = vec![];
        for case in &workload.cases {
            for (i, group) in case.inputs.groups().iter().enumerate() {
                if capacities.len() <= i {
                    capacities.push(0);
                }
                if !case.inputs.bindings.iter().any(|b| {
                    reuse_ids.contains(&b.schema.id) && Arc::ptr_eq(&b.storage.0, &group.0)
                }) {
                    capacities[i] = capacities[i].max(group.0.len().max(1));
                }
            }
            for s in &workload.shared {
                let (_, len) = self
                    .current_hlir_device_binding(s.id)
                    .ok_or_else(|| anyhow::anyhow!("missing shared input {:?}", s.id))?;
                anyhow::ensure!(
                    len == s.length(&case.dims)?,
                    "shared input size varies or does not match sample dimensions"
                );
            }
        }
        // The previous program is about to be searched again. Its captured
        // graphs and high-water intermediate arena are disposable caches, and
        // keeping them alive while allocating replay storage can exhaust the
        // device before the first candidate is evaluated. Preserve the compiled
        // program and caller bindings so even an allocation failure can lazily
        // rematerialize the original program on its next execution.
        self.release_all_arenas();
        self.release_pooled_memory();
        let workload = self.profile_workload.as_ref().unwrap();
        let slots = capacities
            .into_iter()
            .enumerate()
            .map(|(i, n)| {
                if n == 0 {
                    return Ok(None);
                }
                self.cuda_stream.alloc_zeros(n).map(Some).map_err(|error| {
                    anyhow::anyhow!(
                        "profile replay slot {i} ({n} bytes): {error}; device free/total: {:?}",
                        self.cuda_stream.context().mem_get_info()
                    )
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let group_ptrs = workload
            .cases
            .iter()
            .map(|case| {
                case.inputs
                    .groups()
                    .iter()
                    .enumerate()
                    .map(|(index, group)| {
                        reused
                            .iter()
                            .find(|(_, restored)| {
                                case.inputs.bindings.iter().any(|b| {
                                    b.schema.id == restored.schema.id
                                        && Arc::ptr_eq(&b.storage.0, &group.0)
                                })
                            })
                            .map(|(ptr, _)| *ptr)
                            .unwrap_or_else(|| {
                                slots[index]
                                    .as_ref()
                                    .unwrap()
                                    .device_ptr(&self.cuda_stream)
                                    .0
                            })
                    })
                    .collect()
            })
            .collect();
        self.cuda_stream.synchronize()?;
        let shared_ids: Vec<_> = workload.shared.iter().map(|s| s.id).collect();
        self.release_all_bucket_cuda_graphs();
        let shared_bindings: Vec<_> = shared_ids
            .iter()
            .map(|&id| {
                let (ptr, len) = self.current_hlir_device_binding(id).unwrap();
                (
                    id,
                    ptr,
                    len,
                    self.hlir_host_mirrors.get(&id).cloned(),
                    self.prepared_inputs.get(&id).copied(),
                )
            })
            .collect();
        let session = ReplaySession {
            assigned,
            group_ptrs,
            reused,
            slots,
            snapshots: FxHashMap::default(),
            active: None,
            host_states: vec![],
            saved_inputs: std::mem::take(&mut self.hlir_buffers),
            saved_prepared: std::mem::take(&mut self.prepared_inputs),
            saved_prepared_unified: std::mem::take(&mut self.prepared_unified_owners),
            saved_mirrors: std::mem::take(&mut self.hlir_host_mirrors),
            saved_external: std::mem::take(&mut self.external_buffers),
            saved_outputs: std::mem::take(&mut self.output_ptr_registrations),
        };
        self.invalidate_output_registration_resolution();
        self.profile_replay = Some(session);
        for (id, ptr, len, mirror, prepared) in shared_bindings {
            unsafe {
                self.set_device_ptr(id, ptr, len);
            }
            if let Some(prepared) = prepared {
                self.prepared_inputs.insert(id, prepared);
            }
            if let Some(bytes) = mirror {
                self.hlir_host_mirrors.insert(id, bytes);
            }
        }
        self.profile_evaluations.clear();
        Ok(())
    }

    pub(crate) fn finish_profile_replay(&mut self) {
        if self.profile_replay.is_none() {
            return;
        }
        self.release_profile_op_states()
            .expect("profile operation state restoration failed");
        self.cuda_stream
            .synchronize()
            .expect("profile replay synchronization failed");
        self.release_all_bucket_cuda_graphs();
        // Owners are still retained by the session while restoring their bytes.
        for (ptr, binding) in &self.profile_replay.as_ref().unwrap().reused {
            if !binding.storage.0.is_empty() {
                unsafe { result::memcpy_htod_sync(*ptr, binding.storage.0.as_ref()) }
                    .expect("reused profile input restoration failed");
            }
        }
        let session = self.profile_replay.take().unwrap();
        self.hlir_buffers = session.saved_inputs;
        self.prepared_inputs = session.saved_prepared;
        self.prepared_unified_owners = session.saved_prepared_unified;
        self.hlir_host_mirrors = session.saved_mirrors;
        self.external_buffers = session.saved_external;
        self.output_ptr_registrations = session.saved_outputs;
        self.invalidate_output_registration_resolution();
        self.changed_hlir.extend(self.hlir_buffers.keys());
        for bucket in &mut self.compiled_buckets {
            bucket.hlir_synced = false;
            bucket.materialization_fully_dirty = true;
        }
        self.cancel_search_profile();
        // Slots drop only after graph releases and restoration of all bindings.
    }

    pub(crate) fn profile_warmup_trials(&self) -> usize {
        self.profile_workload
            .as_ref()
            .and_then(|w| w.warmup_trials)
            .unwrap_or(1)
    }

    pub(crate) fn profile_case_indices(&self, bucket: usize) -> Option<Vec<usize>> {
        self.profile_replay
            .as_ref()
            .map(|r| r.assigned[bucket].clone())
    }
    pub(crate) fn profile_case_dims(&self, case: usize) -> DynMap {
        self.profile_workload.as_ref().unwrap().cases[case]
            .dims
            .clone()
    }
    pub(crate) fn activate_profile_case(&mut self, case: usize) -> anyhow::Result<()> {
        let workload = self.profile_workload.as_ref().unwrap();
        let session = self.profile_replay.as_mut().unwrap();
        if workload.device_snapshots && session.active != Some(case) {
            let sample = &workload.cases[case];
            let groups = sample.inputs.groups();
            let keys: FxHashSet<_> = groups.iter().map(|g| Arc::as_ptr(&g.0) as usize).collect();
            // Release inactive storage before allocating its replacement, so
            // memory scales with the largest case rather than the case count.
            session.snapshots.retain(|key, _| keys.contains(key));
            for group in groups {
                let key = Arc::as_ptr(&group.0) as usize;
                if let std::collections::hash_map::Entry::Vacant(entry) =
                    session.snapshots.entry(key)
                {
                    let snapshot = self.cuda_stream.clone_htod(group.0.as_ref()).map_err(|error| {
                        anyhow::anyhow!("profile snapshot for case {:?} ({} bytes): {error}; device free/total: {:?}", sample.id, group.0.len(), self.cuda_stream.context().mem_get_info())
                    })?;
                    entry.insert(snapshot);
                }
            }
        }
        session.active = Some(case);
        self.restore_profile_inputs()
    }
    fn restore_profile_inputs(&mut self) -> anyhow::Result<()> {
        let Some(session) = &mut self.profile_replay else {
            return Ok(());
        };
        let Some(index) = session.active else {
            return Ok(());
        };
        let inputs = &self.profile_workload.as_ref().unwrap().cases[index].inputs;
        let groups = inputs.groups();
        for (group, &ptr) in groups.iter().zip(&session.group_ptrs[index]) {
            if !group.0.is_empty() {
                let mut destination = std::mem::ManuallyDrop::new(unsafe {
                    self.cuda_stream
                        .upgrade_device_ptr::<u8>(ptr, group.0.len())
                });
                if let Some(source) = session.snapshots.get(&(Arc::as_ptr(&group.0) as usize)) {
                    self.cuda_stream.memcpy_dtod(source, &mut *destination)?;
                } else {
                    self.cuda_stream
                        .memcpy_htod(group.0.as_ref(), &mut *destination)?;
                }
            }
        }
        let bindings: Vec<_> = inputs
            .bindings
            .iter()
            .map(|b| {
                let group = groups
                    .iter()
                    .position(|g| Arc::ptr_eq(&g.0, &b.storage.0))
                    .unwrap();
                let ptr = session.group_ptrs[index][group] + b.range.start as u64;
                (
                    b.schema.id,
                    ptr,
                    b.range.len(),
                    b.mirror.then(|| b.storage.0[b.range.clone()].to_vec()),
                )
            })
            .collect();
        for (id, ptr, len, mirror) in bindings {
            unsafe {
                self.set_device_ptr(id, ptr, len);
            }
            if let Some(bytes) = mirror {
                self.hlir_host_mirrors.insert(id, bytes);
            }
        }
        // End reset traffic before planning/timing (also supports borrowed streams).
        self.cuda_stream.synchronize()?;
        Ok(())
    }

    pub(crate) fn capture_profile_op_states(&mut self, llir: &LLIRGraph) -> anyhow::Result<()> {
        for node in llir.node_weights() {
            if let Some(host) = node.to_dialect::<dyn HostOp>() {
                let state = host.capture_profile_state(&self.cuda_stream)?;
                self.profile_replay
                    .as_mut()
                    .unwrap()
                    .host_states
                    .push(state);
            }
        }
        Ok(())
    }
    pub(crate) fn release_profile_op_states(&mut self) -> anyhow::Result<()> {
        let Some(session) = &mut self.profile_replay else {
            return Ok(());
        };
        // Keep snapshots alive if restoration fails, so outer cleanup can retry.
        for state in &session.host_states {
            state.restore(&self.cuda_stream)?;
        }
        self.cuda_stream.synchronize()?;
        session.host_states.clear();
        Ok(())
    }
    pub(crate) fn restore_profile_trial(&mut self) -> anyhow::Result<()> {
        if self.profile_replay.is_none() {
            return Ok(());
        }
        self.restore_profile_inputs()?;
        for state in &self.profile_replay.as_ref().unwrap().host_states {
            state.restore(&self.cuda_stream)?;
        }
        self.cuda_stream.synchronize()?;
        Ok(())
    }

    pub(crate) fn validate_profile_effects(&self, llir: &LLIRGraph) -> anyhow::Result<()> {
        let workload = self.profile_workload.as_ref().unwrap();
        let shared: FxHashSet<_> = workload.shared.iter().map(|s| s.id).collect();
        // Follow declared storage aliases, not op names or graph patterns.
        let incoming = |node| {
            llir.edges_directed(node, Direction::Incoming)
                .sorted_by_key(|e| e.id())
                .map(|e| e.source())
                .collect::<Vec<_>>()
        };
        let root = |mut node: NodeIndex| {
            for _ in 0..=llir.node_count() {
                if let Some(input) = llir[node].to_op::<Input>() {
                    return Some(NodeIndex::new(input.node));
                }
                let alias = llir[node]
                    .to_dialect::<dyn KernelOp>()?
                    .output_aliases_input()?;
                node = *incoming(node).get(alias)?;
            }
            None
        };
        for node in llir.node_indices() {
            let writes = if let Some(kernel) = llir[node].to_dialect::<dyn KernelOp>() {
                kernel
                    .output_aliases_input()
                    .filter(|_| kernel.mutates_aliased_input())
                    .into_iter()
                    .collect()
            } else if let Some(host) = llir[node].to_dialect::<dyn HostOp>() {
                host.profile_mutated_inputs()
            } else {
                vec![]
            };
            let inputs = incoming(node);
            for write in writes {
                let source = inputs
                    .get(write)
                    .ok_or_else(|| anyhow::anyhow!("invalid profile effect input index"))?;
                let Some(id) = root(*source) else {
                    continue;
                };
                anyhow::ensure!(
                    !shared.contains(&id),
                    "candidate writes shared read-only profile input {id:?}"
                );
                // Cross-input storage aliasing is invisible to e-graph exclusive-
                // use proofs. Reject destructive alternatives that would change
                // another input's logical value; ordinary out-of-place ops remain
                // legal. Views of a single graph input retain normal alias proofs.
                for case in &workload.cases {
                    if let Some(b) = case.inputs.bindings.iter().find(|b| b.schema.id == id) {
                        for other in &case.inputs.bindings {
                            anyhow::ensure!(
                                other.schema.id == id
                                    || !Arc::ptr_eq(&b.storage.0, &other.storage.0)
                                    || b.range.end <= other.range.start
                                    || other.range.end <= b.range.start,
                                "candidate writes overlapping profile input aliases"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn profile_timing_method(&self) -> luminal::op::TimingMethod {
        self.profile_workload
            .as_ref()
            .map(|w| w.timing_method)
            .unwrap_or_default()
    }

    pub(crate) fn record_profile_evaluation(
        &mut self,
        bucket: usize,
        cuda_graph: bool,
        timings: &[(usize, Duration)],
    ) -> anyhow::Result<Duration> {
        let workload = self.profile_workload.as_ref().unwrap();
        let total = workload.cases.iter().map(|c| c.weight).sum();
        let values: Vec<_> = timings
            .iter()
            .map(|&(i, time)| (workload.cases[i].weight, time))
            .collect();
        let cost = weighted_profile_cost(&values, total)?;
        self.profile_evaluations.push(ProfileEvaluation {
            bucket,
            cuda_graph,
            timing_method: workload.timing_method,
            weighted_cost: cost,
            cases: timings
                .iter()
                .map(|&(i, duration)| {
                    let c = &workload.cases[i];
                    ProfileMeasurement {
                        case_id: c.id.clone(),
                        dims: c.dims.clone(),
                        weight: c.weight,
                        duration,
                    }
                })
                .collect(),
        });
        Ok(cost)
    }
}

#[cfg(test)]
#[path = "profile_tests.rs"]
mod tests;
