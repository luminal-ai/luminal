pub mod block;
pub mod kernel;
pub mod logical;
pub mod runtime;
use std::sync::Arc;

/// Tile size for matmul tiling. Used by TileMatmul.
pub const TILE_SIZE: u32 = 64;

pub use cudarc;

#[cfg(test)]
mod tests;

use cudarc::driver::CudaContext;
use luminal::op::DType;

fn cuda_dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::Bf16 => todo!(),
        DType::Int => "int",
    }
}

/// Returns the bandwidth of the device in GB/s
pub fn cuda_bandwidth_gbps(ctx: &Arc<CudaContext>) -> Option<usize> {
    Some(match ctx.name().unwrap().as_str() {
        "NVIDIA Thor" => 273,
        "NVIDIA H100 PCIe" => 2_000,
        "NVIDIA H100 SXM" => 3_350,
        _ => return None,
    })
}

/// Returns the bandwidth of the device in TFLOPs
pub fn cuda_compute_f32_tflops(ctx: &Arc<CudaContext>) -> Option<usize> {
    Some(match ctx.name().unwrap().as_str() {
        "NVIDIA Thor" => 125, // forced to use tf32 flops
        "NVIDIA H100 PCIe" => 756,
        "NVIDIA H100 SXM" => 989,
        _ => return None,
    })
}
