//! Share scratch within an ordered capture stream without making its lifetime
//! process-global. Owners retain immutable allocation generations; cache entries
//! are weak, so replacing or retiring a graph releases its unreferenced storage.

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, sys::CUstreamCaptureStatus};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock, Weak},
};

#[derive(Debug)]
pub struct Workspace {
    allocation: CudaSlice<u8>,
    ptr: u64,
    execution_stream: usize,
}
impl Workspace {
    pub fn ptr(&self) -> u64 {
        self.ptr
    }
    pub fn allocation(&self) -> &CudaSlice<u8> {
        &self.allocation
    }
}

type WorkspaceEntries = HashMap<(usize, usize), Weak<Workspace>>;

#[derive(Debug, Default)]
pub struct WorkspaceCache {
    entries: OnceLock<Mutex<WorkspaceEntries>>,
}
impl WorkspaceCache {
    pub const fn new() -> Self {
        Self {
            entries: OnceLock::new(),
        }
    }

    /// `ordered_stream` identifies scratch reuse. Allocate on the stream that
    /// will EXECUTE the graph, which may differ from its recording stream, so
    /// dropping the last owner enqueues a free after that graph's pending work.
    /// Every graph must retain the returned owner for its complete lifetime.
    /// Growth creates a new allocation; older captured pointers remain valid.
    pub fn acquire(
        &self,
        ordered_stream: &Arc<CudaStream>,
        execution_stream: &Arc<CudaStream>,
        bytes: usize,
    ) -> anyhow::Result<Arc<Workspace>> {
        anyhow::ensure!(
            ordered_stream.context().cu_ctx() == execution_stream.context().cu_ctx(),
            "workspace streams belong to different CUDA contexts"
        );
        let key = (
            ordered_stream.context().cu_ctx() as usize,
            ordered_stream.cu_stream() as usize,
        );
        let mut cache = self
            .entries
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .expect("workspace cache poisoned");
        cache.retain(|_, entry| entry.strong_count() != 0);
        if let Some(existing) = cache.get(&key).and_then(Weak::upgrade)
            && existing.allocation.len() >= bytes.max(1)
            && existing.execution_stream == execution_stream.cu_stream() as usize
        {
            return Ok(existing);
        }
        anyhow::ensure!(
            ordered_stream.capture_status()?
                == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
            "workspace must be prepared before capture"
        );
        let allocation = unsafe { execution_stream.alloc::<u8>(bytes.max(1))? };
        let ptr = allocation.device_ptr(execution_stream).0;
        let workspace = Arc::new(Workspace {
            allocation,
            ptr,
            execution_stream: execution_stream.cu_stream() as usize,
        });
        cache.insert(key, Arc::downgrade(&workspace));
        Ok(workspace)
    }
    /// Resolve an already-owned allocation while recording a graph. This never
    /// allocates and must not outlive the owner's captured graph generation.
    pub fn prepared(
        &self,
        stream: &Arc<CudaStream>,
        bytes: usize,
    ) -> anyhow::Result<Arc<Workspace>> {
        let key = (
            stream.context().cu_ctx() as usize,
            stream.cu_stream() as usize,
        );
        let workspace = self.entries.get().and_then(|entries| {
            entries
                .lock()
                .expect("workspace cache poisoned")
                .get(&key)
                .and_then(Weak::upgrade)
        });
        workspace
            .filter(|w| w.allocation.len() >= bytes.max(1))
            .ok_or_else(|| anyhow::anyhow!("workspace must have a live owner before capture"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    #[test]
    fn cache_is_weak_and_growth_preserves_owned_generations() {
        let context = CudaContext::new(0).unwrap();
        let capture = context.new_stream().unwrap();
        let execution = context.new_stream().unwrap();
        let other = context.new_stream().unwrap();
        let cache = WorkspaceCache::new();
        let first = cache.acquire(&capture, &execution, 64).unwrap();
        let again = cache.acquire(&capture, &execution, 32).unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        unsafe {
            cudarc::driver::result::memcpy_htod_sync(first.ptr(), &[23u8; 64]).unwrap();
        }
        let grown = cache.acquire(&capture, &execution, 1024).unwrap();
        assert_ne!(first.ptr(), grown.ptr());
        assert_eq!(
            execution.clone_dtoh(first.allocation()).unwrap(),
            vec![23; 64]
        );
        let distinct = cache.acquire(&capture, &other, 1024).unwrap();
        assert_ne!(
            grown.ptr(),
            distinct.ptr(),
            "unordered execution streams share scratch"
        );
        let weak = [
            Arc::downgrade(&first),
            Arc::downgrade(&grown),
            Arc::downgrade(&distinct),
        ];
        drop((first, again, grown, distinct));
        execution.synchronize().unwrap();
        other.synchronize().unwrap();
        assert!(weak.iter().all(|w| w.upgrade().is_none()));
        assert!(cache.prepared(&capture, 1).is_err());
        let stream_weak = Arc::downgrade(&execution);
        drop(execution);
        assert!(
            stream_weak.upgrade().is_none(),
            "cache roots retired streams"
        );
    }
}
