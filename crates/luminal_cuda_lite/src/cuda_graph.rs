//! CUDA graph ownership and updates. All graph work is serialized on one stream.
use crate::host::{CaptureCtx, PreparedHostOp};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaStream, sys as cu};
use std::{ptr, sync::Arc};

pub(crate) type Node = cu::CUgraphNode;
pub(crate) struct Graph {
    pub raw: cu::CUgraph,
    ctx: Arc<CudaContext>,
}
pub(crate) struct Executable {
    raw: cu::CUgraphExec,
    ctx: Arc<CudaContext>,
}
impl Graph {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        ctx.bind_to_thread()?;
        let mut raw = ptr::null_mut();
        unsafe {
            cu::cuGraphCreate(&mut raw, 0).result()?;
        }
        Ok(Self {
            raw,
            ctx: ctx.clone(),
        })
    }
    pub fn capture(stream: &Arc<CudaStream>, prepared: &dyn PreparedHostOp) -> Result<Self> {
        stream.context().bind_to_thread()?;
        unsafe {
            cu::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .result()?;
        }
        // End capture even on an error or unwind, so subsequent candidates can use the stream.
        struct Capture<'a>(&'a Arc<CudaStream>, bool);
        impl Drop for Capture<'_> {
            fn drop(&mut self) {
                if self.1 {
                    unsafe {
                        let mut graph = ptr::null_mut();
                        let _ = cu::cuStreamEndCapture(self.0.cu_stream(), &mut graph);
                        if !graph.is_null() {
                            let _ = cu::cuGraphDestroy(graph);
                        }
                    }
                }
            }
        }
        let mut capture = Capture(stream, true);
        let recorded = unsafe { prepared.record(&CaptureCtx { stream }) };
        let mut raw = ptr::null_mut();
        let ended = unsafe { cu::cuStreamEndCapture(stream.cu_stream(), &mut raw).result() };
        capture.1 = false;
        let graph = Self {
            raw,
            ctx: stream.context().clone(),
        };
        recorded.context("host op record")?;
        ended.context("end host op capture")?;
        ensure!(!raw.is_null(), "capture produced no graph");
        Ok(graph)
    }
    pub fn instantiate(&self) -> Result<Executable> {
        let mut raw = ptr::null_mut();
        unsafe {
            cu::cuGraphInstantiateWithFlags(&mut raw, self.raw, 0).result()?;
        }
        Ok(Executable {
            raw,
            ctx: self.ctx.clone(),
        })
    }
    pub fn copy(&self, deps: &[Node], params: &cu::CUDA_MEMCPY3D) -> Result<Node> {
        let mut node = ptr::null_mut();
        unsafe {
            cu::cuGraphAddMemcpyNode(
                &mut node,
                self.raw,
                deps.as_ptr(),
                deps.len(),
                params,
                self.ctx.cu_ctx(),
            )
            .result()?;
        }
        Ok(node)
    }
    pub fn kernel(&self, deps: &[Node], params: &cu::CUDA_KERNEL_NODE_PARAMS) -> Result<Node> {
        let mut node = ptr::null_mut();
        unsafe {
            cu::cuGraphAddKernelNode_v2(&mut node, self.raw, deps.as_ptr(), deps.len(), params)
                .result()?;
        }
        Ok(node)
    }
    pub fn child(&self, deps: &[Node], child: &Graph) -> Result<Node> {
        let mut node = ptr::null_mut();
        unsafe {
            cu::cuGraphAddChildGraphNode(&mut node, self.raw, deps.as_ptr(), deps.len(), child.raw)
                .result()?;
        }
        Ok(node)
    }
}
impl Executable {
    pub fn launch(&self, stream: &Arc<CudaStream>) -> Result<()> {
        self.ctx.bind_to_thread()?;
        unsafe {
            cu::cuGraphLaunch(self.raw, stream.cu_stream()).result()?;
        }
        Ok(())
    }
    pub fn copy(&self, node: Node, params: &cu::CUDA_MEMCPY3D) -> Result<()> {
        unsafe {
            cu::cuGraphExecMemcpyNodeSetParams(self.raw, node, params, self.ctx.cu_ctx())
                .result()?;
        }
        Ok(())
    }
    pub fn child(&self, node: Node, child: &Graph) -> Result<()> {
        unsafe {
            cu::cuGraphExecChildGraphNodeSetParams(self.raw, node, child.raw).result()?;
        }
        Ok(())
    }
    pub fn kernel(&self, node: Node, params: &cu::CUDA_KERNEL_NODE_PARAMS) -> Result<()> {
        unsafe {
            cu::cuGraphExecKernelNodeSetParams_v2(self.raw, node, params).result()?;
        }
        Ok(())
    }
    pub fn enable(&self, node: Node, enabled: bool) -> Result<()> {
        unsafe {
            cu::cuGraphNodeSetEnabled(self.raw, node, u32::from(enabled)).result()?;
        }
        Ok(())
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        if !self.raw.is_null() {
            unsafe {
                let _ = cu::cuGraphDestroy(self.raw);
            }
        }
    }
}
impl Drop for Executable {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = cu::cuGraphExecDestroy(self.raw);
        }
    }
}

/// Stable, pinned staging memory. Only touched by the CPU between synchronized launches.
pub(crate) struct Pinned {
    ptr: *mut u8,
    len: usize,
    ctx: Arc<CudaContext>,
}
impl Pinned {
    pub fn new(ctx: &Arc<CudaContext>, len: usize) -> Result<Self> {
        ctx.bind_to_thread()?;
        let mut ptr = ptr::null_mut();
        unsafe {
            cu::cuMemHostAlloc(&mut ptr, len.max(1), 0).result()?;
            std::ptr::write_bytes(ptr.cast::<u8>(), 0, len.max(1));
        }
        Ok(Self {
            ptr: ptr.cast(),
            len,
            ctx: ctx.clone(),
        })
    }
    pub fn ptr(&self) -> *mut std::ffi::c_void {
        self.ptr.cast()
    }
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl Drop for Pinned {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = cu::cuMemFreeHost(self.ptr.cast());
        }
    }
}

pub(crate) fn copy_params(src: u64, dst: u64, bytes: usize, kind: CopyKind) -> cu::CUDA_MEMCPY3D {
    let mut p = cu::CUDA_MEMCPY3D {
        srcXInBytes: 0,
        srcY: 0,
        srcZ: 0,
        srcLOD: 0,
        srcMemoryType: cu::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: ptr::null(),
        srcDevice: 0,
        srcArray: ptr::null_mut(),
        reserved0: ptr::null_mut(),
        srcPitch: 0,
        srcHeight: 0,
        dstXInBytes: 0,
        dstY: 0,
        dstZ: 0,
        dstLOD: 0,
        dstMemoryType: cu::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: ptr::null_mut(),
        dstDevice: 0,
        dstArray: ptr::null_mut(),
        reserved1: ptr::null_mut(),
        dstPitch: 0,
        dstHeight: 0,
        WidthInBytes: 0,
        Height: 1,
        Depth: 1,
    };
    p.srcMemoryType = cu::CUmemorytype::CU_MEMORYTYPE_DEVICE;
    p.dstMemoryType = cu::CUmemorytype::CU_MEMORYTYPE_DEVICE;
    p.srcDevice = src;
    p.dstDevice = dst;
    p.WidthInBytes = bytes.max(1);
    p.Height = 1;
    p.Depth = 1;
    match kind {
        CopyKind::HtoD => {
            p.srcMemoryType = cu::CUmemorytype::CU_MEMORYTYPE_HOST;
            p.srcHost = src as *const _;
            p.srcDevice = 0;
        }
        CopyKind::DtoH => {
            p.dstMemoryType = cu::CUmemorytype::CU_MEMORYTYPE_HOST;
            p.dstHost = dst as *mut _;
            p.dstDevice = 0;
        }
        CopyKind::DtoD => {}
    }
    p
}
#[derive(Clone, Copy)]
pub(crate) enum CopyKind {
    HtoD,
    DtoH,
    DtoD,
}

/// A bucket's view into the runtime's shared pinned staging allocation.
/// The runtime serializes launches and completes readback before reusing it.
pub(crate) trait PinnedRange {
    fn ptr(self, staging: &Pinned) -> u64;
    fn bytes(self, staging: &Pinned) -> &[u8];
    fn bytes_mut(self, staging: &mut Pinned) -> &mut [u8];
}
impl PinnedRange for crate::arena::ArenaSlice {
    fn ptr(self, staging: &Pinned) -> u64 {
        staging.ptr() as u64 + self.offset as u64
    }
    fn bytes(self, staging: &Pinned) -> &[u8] {
        &staging.bytes()[self.offset..self.offset + self.bytes]
    }
    fn bytes_mut(self, staging: &mut Pinned) -> &mut [u8] {
        &mut staging.bytes_mut()[self.offset..self.offset + self.bytes]
    }
}
