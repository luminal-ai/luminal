//! Explicit, owned physical input representations. No pointer-keyed data cache.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PreparedInput {
    pub format: &'static str,
    pub capacity: usize,
}

/// Owns managed storage while runtime bindings expose non-owning device views.
pub(super) struct PreparedUnifiedOwner {
    pub buffer: cudarc::driver::UnifiedSlice<u8>,
    stream: Arc<CudaStream>,
}
impl Drop for PreparedUnifiedOwner {
    fn drop(&mut self) {
        // HostOps and captured kernels use raw pointers; their completion must
        // precede UnifiedSlice's synchronous free, independently of its events.
        self.stream
            .synchronize()
            .expect("prepared allocation synchronization failed");
    }
}

/// Resolve declared input writes through storage aliases, independent of op names.
pub(super) fn input_writes(llir: &LLIRGraph) -> anyhow::Result<FxHashSet<NodeIndex>> {
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
    let mut writes = FxHashSet::default();
    for node in llir.node_indices() {
        let inputs = incoming(node);
        let indices = if let Some(kernel) = llir[node].to_dialect::<dyn KernelOp>() {
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
        for index in indices {
            let source = inputs
                .get(index)
                .ok_or_else(|| anyhow::anyhow!("invalid declared input-write index"))?;
            if let Some(input) = root(*source) {
                writes.insert(input);
            }
        }
    }
    Ok(writes)
}

impl<O: IntoEgglogOp> CudaRuntimeImpl<O> {
    /// Inspect an input's current logical view and explicitly declared physical ABI.
    /// This non-owning view is valid only while its runtime binding is retained.
    pub fn input_buffer(&self, id: impl ToId) -> Option<DeviceBuffer> {
        Self::input_device_buffer(
            id.to_id(),
            &self.cuda_stream,
            &self.hlir_buffers,
            &self.external_buffers,
            &self.prepared_inputs,
        )
    }

    /// Install a read-only, explicitly formatted input allocation. The format is
    /// an operation-defined physical ABI, not a tensor dtype. `logical_bytes`
    /// bounds its primary view; HostOps can inspect the full allocation capacity
    /// and format to consume additional representations stored by that ABI.
    ///
    /// The caller must construct all representations coherently. Ordinary input
    /// setters clear this declaration. Graph writes and recurrent commits to a
    /// prepared input are rejected. Representative workloads must share it.
    pub fn set_prepared_buffer(
        &mut self,
        id: impl ToId,
        buffer: CudaSlice<u8>,
        logical_bytes: usize,
        format: &'static str,
    ) {
        let id = id.to_id();
        assert!(!format.is_empty() && logical_bytes <= buffer.len());
        assert!(
            self.compiled_buckets
                .iter()
                .all(|b| !b.input_writes.contains(&id)),
            "prepared input is written by a retained graph"
        );
        let capacity = buffer.len();
        let ptr = buffer.device_ptr(&self.cuda_stream).0;
        assert!(
            self.output_ptr_registrations
                .values()
                .all(|&(out, len)| !device_ranges_overlap(ptr, capacity, out, len)),
            "prepared input overlaps an external output"
        );
        self.set_buffer(id, buffer);
        let CudaInput::Buffer { len, .. } = self.hlir_buffers.get_mut(&id).unwrap() else {
            unreachable!()
        };
        *len = logical_bytes;
        self.prepared_inputs
            .insert(id, PreparedInput { format, capacity });
        self.changed_hlir.insert(id);
    }

    /// Install owned managed memory with the same read-only prepared ABI.
    /// CUDA may evict cold pages under device-memory pressure. The caller must
    /// warm representative workloads and account for migration in measurements.
    /// Captured pointers remain stable. Ordinary input replacement releases this
    /// owner only after stream completion; workloads must declare it shared.
    pub fn set_prepared_unified_buffer(
        &mut self,
        id: impl ToId,
        buffer: cudarc::driver::UnifiedSlice<u8>,
        logical_bytes: usize,
        format: &'static str,
    ) {
        let id = id.to_id();
        assert!(!format.is_empty() && logical_bytes <= buffer.len());
        assert!(
            self.compiled_buckets
                .iter()
                .all(|b| !b.input_writes.contains(&id)),
            "prepared input is written by a retained graph"
        );
        let capacity = buffer.len();
        let ptr = buffer.device_ptr(&self.cuda_stream).0;
        assert!(
            self.output_ptr_registrations
                .values()
                .all(|&(out, len)| !device_ranges_overlap(ptr, capacity, out, len)),
            "prepared input overlaps an external output"
        );
        unsafe {
            self.set_device_ptr(id, ptr, logical_bytes);
        }
        self.prepared_inputs
            .insert(id, PreparedInput { format, capacity });
        self.prepared_unified_owners.insert(
            id,
            PreparedUnifiedOwner {
                buffer,
                stream: self.cuda_stream.clone(),
            },
        );
        self.changed_hlir.insert(id);
    }

    pub(super) fn clear_prepared_input(&mut self, id: NodeIndex) {
        self.prepared_unified_owners.remove(&id);
        if self.prepared_inputs.remove(&id).is_some() {
            self.changed_hlir.insert(id);
        }
    }

    pub(super) fn validate_prepared_input_effects(&self, llir: &LLIRGraph) -> anyhow::Result<()> {
        if !self.prepared_inputs.is_empty() {
            anyhow::ensure!(
                input_writes(llir)?
                    .iter()
                    .all(|id| !self.prepared_inputs.contains_key(id)),
                "candidate writes a prepared read-only input"
            );
        }
        Ok(())
    }

    pub(super) fn input_device_buffer(
        id: NodeIndex,
        stream: &Arc<CudaStream>,
        inputs: &FxHashMap<NodeIndex, CudaInput>,
        external: &FxHashMap<NodeIndex, std::mem::ManuallyDrop<CudaSlice<u8>>>,
        prepared: &FxHashMap<NodeIndex, PreparedInput>,
    ) -> Option<DeviceBuffer> {
        let mut buffer = match inputs.get(&id)? {
            CudaInput::Buffer { buf, len } => {
                DeviceBuffer::new(buf.device_ptr(stream).0, *len).with_capacity(buf.len())
            }
            CudaInput::Ptr(ptr) => DeviceBuffer::new(*ptr, external.get(&id)?.len()),
        };
        if let Some(layout) = prepared.get(&id) {
            buffer = buffer
                .with_capacity(layout.capacity)
                .with_input_format(layout.format);
        }
        Some(buffer)
    }
}
