pub mod block;
pub mod host;
pub mod kernel;
pub mod logical;
pub mod runtime;
use std::sync::Arc;

pub use cudarc;

#[cfg(test)]
mod tests;

use cudarc::driver::CudaContext;
use luminal::op::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaComputeCapability {
    pub major: i32,
    pub minor: i32,
}

impl CudaComputeCapability {
    pub fn sm_arch(self) -> String {
        format!("sm_{}{}", self.major, self.minor)
    }

    pub fn supports_tf32_tensor_cores(self) -> bool {
        self.major >= 8
    }

    pub fn supports_tma(self) -> bool {
        self.major >= 9
    }
}

pub fn cuda_compute_capability(ctx: &Arc<CudaContext>) -> Option<CudaComputeCapability> {
    ctx.compute_capability()
        .ok()
        .map(|(major, minor)| CudaComputeCapability { major, minor })
}

pub fn cuda_nvrtc_arch(ctx: &Arc<CudaContext>) -> Option<String> {
    cuda_compute_capability(ctx).map(CudaComputeCapability::sm_arch)
}

pub fn cuda_supports_tf32_tensor_cores(ctx: &Arc<CudaContext>) -> bool {
    cuda_compute_capability(ctx)
        .map(CudaComputeCapability::supports_tf32_tensor_cores)
        .unwrap_or(false)
}

pub fn cuda_supports_tma(ctx: &Arc<CudaContext>) -> bool {
    cuda_compute_capability(ctx)
        .map(CudaComputeCapability::supports_tma)
        .unwrap_or(false)
}

fn cuda_dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::Bf16 => todo!(),
        DType::Int => "int",
        DType::Bool => "unsigned char",
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
