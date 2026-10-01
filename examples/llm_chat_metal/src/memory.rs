//! Caller-owned storage and explicit native device transfers.
#![allow(dead_code)]
use anyhow::{Result, anyhow, ensure};
use metal::{Buffer, CommandQueue, Device, MTLCommandBufferStatus, MTLResourceOptions};

pub struct Allocation {
    buffer: Buffer,
    device: Device,
    queue: CommandQueue,
}
impl Allocation {
    pub fn new(device: &Device, queue: &CommandQueue, bytes: usize) -> Result<Self> {
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

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
    /// Explicit application readback after execution has completed.
    pub fn read(&self, offset: usize, bytes: usize) -> Result<Vec<u8>> {
        objc::rc::autoreleasepool(|| {
            let end = offset
                .checked_add(bytes)
                .ok_or_else(|| anyhow!("readback range overflow"))?;
            ensure!(
                end as u64 <= self.buffer.length(),
                "readback exceeds allocation"
            );
            if bytes == 0 {
                return Ok(vec![]);
            }
            let staging = self
                .device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            let command = self.queue.new_command_buffer();
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(&self.buffer, offset as u64, &staging, 0, bytes as u64);
            blit.end_encoding();
            command.commit();
            command.wait_until_completed();
            ensure!(
                command.status() == MTLCommandBufferStatus::Completed,
                "Metal readback failed"
            );
            Ok(
                unsafe { std::slice::from_raw_parts(staging.contents().cast::<u8>(), bytes) }
                    .to_vec(),
            )
        })
    }
    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
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
}
