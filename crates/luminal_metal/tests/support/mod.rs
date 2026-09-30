//! Test-owned tensor transfers. Every upload/readback is an explicit test step.
#![allow(dead_code)]
#![cfg(target_os = "macos")]
use anyhow::{Result, ensure};
use luminal::{bufferize::OutputBinding, layouts::DecodedLayout, prelude::NodeIndex};
use luminal_metal::{HostBuffer, MetalRuntime};

mod memory;
pub use memory::Allocation;

pub trait TestTransfers {
    fn upload(
        &self,
        arena: &mut Allocation,
        tensor: NodeIndex,
        data: impl Into<HostBuffer>,
    ) -> Result<()>;
    fn download(
        &self,
        arena: &Allocation,
        tensor: NodeIndex,
    ) -> Result<(HostBuffer, OutputBinding<DecodedLayout>)>;
    fn read_f32(&self, arena: &Allocation, tensor: NodeIndex) -> Result<Vec<f32>> {
        self.download(arena, tensor)?.0.as_f32()
    }
    fn read_i32(&self, arena: &Allocation, tensor: NodeIndex) -> Result<Vec<i32>> {
        self.download(arena, tensor)?.0.as_i32()
    }
    fn read_i64(&self, arena: &Allocation, tensor: NodeIndex) -> Result<Vec<i64>> {
        self.download(arena, tensor)?.0.as_i64()
    }
    fn read_bool8(&self, arena: &Allocation, tensor: NodeIndex) -> Result<Vec<u8>> {
        Ok(self.download(arena, tensor)?.0.as_bool8()?.to_vec())
    }
}
impl TestTransfers for MetalRuntime {
    fn upload(
        &self,
        arena: &mut Allocation,
        tensor: NodeIndex,
        data: impl Into<HostBuffer>,
    ) -> Result<()> {
        let data = data.into();
        let home = self.input_arena_range(tensor)?;
        let lit = self.input_buffer(tensor)?;
        let buffer = self
            .plan()
            .unwrap()
            .buffers
            .values()
            .find(|b| b.lit == Some(lit))
            .unwrap();
        ensure!(
            Some(data.dtype) == buffer.layout.dtype,
            "input dtype mismatch"
        );
        ensure!(data.bytes.len() == home.bytes, "input byte count mismatch");
        arena.write(home.offset, &data.bytes)
    }
    fn download(
        &self,
        arena: &Allocation,
        tensor: NodeIndex,
    ) -> Result<(HostBuffer, OutputBinding<DecodedLayout>)> {
        let home = self.output_arena_range(tensor)?;
        let slot = self.output_layout(tensor)?;
        let dtype = self.plan().unwrap().buffers[&slot.buffer]
            .layout
            .dtype
            .unwrap();
        Ok((
            HostBuffer::new(dtype, arena.read(home.offset, home.bytes)?)?,
            slot,
        ))
    }
}

pub fn allocate_arena(runtime: &MetalRuntime) -> Result<Allocation> {
    Allocation::new(
        runtime.metal_device()?,
        runtime.metal_queue()?,
        runtime.arena_bytes()?,
    )
}
