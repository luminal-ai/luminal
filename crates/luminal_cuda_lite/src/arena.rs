//! Core physical arena planning, with backend dtype sizing.
use anyhow::{Result, anyhow};
pub use luminal::arena::*;
use luminal::bufferize::Buffer;

/// Size a resolved buffer from its carried layout and supported storage dtype.
pub fn buffer_bytes(buffer: &Buffer<luminal::layouts::DecodedLayout>) -> Result<usize> {
    let numel = buffer.layout.literal_span_elements().ok_or_else(|| {
        anyhow!(
            "buffer {:?} (backing {}) has no literal span — symbolic or \
                 undisclosed-reach layouts are not executable",
            buffer.label,
            buffer.backs
        )
    })?;
    let dtype = buffer.layout.dtype.ok_or_else(|| {
        anyhow!(
            "buffer {:?} (backing {}) carries no dtype fact",
            buffer.label,
            buffer.backs
        )
    })?;
    Ok(numel * crate::host_buffer::dtype_bytes(dtype)?)
}
