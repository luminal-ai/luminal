use cudarc::driver::{CudaContext, CudaDevice, CudaFunction, LaunchConfig};
use std::sync::Arc;

pub fn check_compute_capability(device: &CudaDevice) -> bool {
    // Get device properties
    if let Ok(props) = device.get_device_properties() {
        // Return true for Volta (7.0) and newer
        props.compute_capability_major >= 7
    } else {
        false
    }
}

pub fn get_optimal_implementation(device: &CudaDevice) -> &'static str {
    if check_compute_capability(device) {
        "quantized_matvec_int8_optimized" // Tensor Core optimized version
    } else {
        "quantized_matvec_int8_original" // Fallback version
    }
}

// Helper to initialize CUDA context with proper error handling
pub fn initialize_cuda() -> Option<(CudaDevice, Arc<CudaContext>)> {
    match CudaDevice::new(0) {
        Ok(device) => match device.context() {
            Ok(context) => Some((device, context)),
            Err(_) => None,
        },
        Err(_) => None,
    }
}

// Safe wrapper for kernel launch with proper error handling
pub fn launch_kernel_safely(
    device: &CudaDevice,
    kernel_name: &str,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    args: &[Box<dyn cudarc::driver::LaunchArg>],
) -> Result<(), Box<dyn std::error::Error>> {
    let config = LaunchConfig {
        grid_dim,
        block_dim,
        shared_mem_bytes: 0,
    };

    let function = device.get_or_load_func(kernel_name, kernel_name)?;
    unsafe {
        function.launch(config, args)?;
    }
    Ok(())
}
