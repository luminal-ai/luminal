//! Caller-owned storage and explicit native device transfers.
#![allow(dead_code)]
use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use luminal_cuda_lite::CudaArena;
use std::sync::Arc;

pub struct Allocation {
    buffer: CudaSlice<u8>,
    stream: Arc<CudaStream>,
}
impl Allocation {
    pub fn new(stream: Arc<CudaStream>, bytes: usize) -> Result<Self> {
        // SAFETY: the caller initializes inputs; program operations initialize destinations.
        let buffer = unsafe { stream.alloc(bytes.max(1))? };
        Ok(Self { buffer, stream })
    }

    pub fn arena(&mut self) -> CudaArena<'_> {
        // SAFETY: this exclusive borrow keeps the allocation alive through execution.
        unsafe { CudaArena::from_raw(self.ptr(), self.bytes()) }
    }
    pub fn ptr(&self) -> u64 {
        self.buffer.device_ptr(&self.stream).0
    }
    pub fn bytes(&self) -> usize {
        self.buffer.len()
    }
    /// Explicit application readback after execution has completed.
    pub fn read(&self, offset: usize, bytes: usize) -> Result<Vec<u8>> {
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("readback range overflow"))?;
        ensure!(end <= self.bytes(), "readback exceeds allocation");
        let result = self.stream.clone_dtoh(&self.buffer.slice(offset..end))?;
        self.stream.synchronize()?;
        Ok(result)
    }
    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| anyhow!("upload range overflow"))?;
        ensure!(end <= self.bytes(), "upload exceeds allocation");
        self.stream
            .memcpy_htod(data, &mut self.buffer.slice_mut(offset..end))?;
        self.stream.synchronize()?;
        Ok(())
    }
}
