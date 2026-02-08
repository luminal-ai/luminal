//! Metal backend implementation for benchmarking

use crate::{BenchmarkBackend, HardwareInfo};
use luminal::op::Runtime;
use luminal_metal::runtime::MetalRuntime;

/// Metal benchmark backend
pub struct MetalBenchmark;

impl BenchmarkBackend for MetalBenchmark {
    type Runtime = MetalRuntime;

    fn initialize() -> Self::Runtime {
        MetalRuntime::initialize(())
    }

    fn name() -> &'static str {
        "metal"
    }

    fn hardware_info() -> HardwareInfo {
        // Try to get device info from Metal
        let device = metal::Device::system_default().expect("No Metal device found");
        let device_name = device.name().to_string();

        // Estimate based on common Apple Silicon specs
        let (memory_gb, peak_bandwidth_gbps, peak_tflops) = estimate_device_specs(&device_name);

        HardwareInfo {
            device_name,
            memory_gb,
            peak_bandwidth_gbps: Some(peak_bandwidth_gbps),
            peak_tflops: Some(peak_tflops),
        }
    }
}

/// Estimate device specs based on device name
fn estimate_device_specs(device_name: &str) -> (f64, f64, f64) {
    // Memory (GB), Bandwidth (GB/s), FP32 TFLOPS
    if device_name.contains("M3 Max") {
        (128.0, 400.0, 14.0)
    } else if device_name.contains("M3 Pro") {
        (36.0, 200.0, 7.0)
    } else if device_name.contains("M3") {
        (24.0, 100.0, 3.5)
    } else if device_name.contains("M2 Max") {
        (96.0, 400.0, 13.6)
    } else if device_name.contains("M2 Pro") {
        (32.0, 200.0, 6.8)
    } else if device_name.contains("M2") {
        (24.0, 100.0, 3.6)
    } else if device_name.contains("M1 Max") {
        (64.0, 400.0, 10.4)
    } else if device_name.contains("M1 Pro") {
        (32.0, 200.0, 5.2)
    } else if device_name.contains("M1") {
        (16.0, 68.0, 2.6)
    } else {
        // Generic fallback
        (8.0, 50.0, 1.0)
    }
}
