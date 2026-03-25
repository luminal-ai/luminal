use cudarc::driver::CudaContext;
use luminal::prelude::*;
use tracing::{Level, enabled};

use crate::cuda_bandwidth_gbps;
use crate::runtime::CudaRuntime;

/// Test that measures bandwidth utilization for a large element-wise add kernel.
/// This demonstrates that KernelAdd can achieve reasonable bandwidth with large tensors.
#[test]
pub fn kernel_add_bandwidth_test() {
    // 64M elements = 256MB per tensor, 768MB total memory traffic (2 reads + 1 write)
    let size = 64 * 1024 * 1024;

    let mut cx = Graph::default();
    let a = cx.tensor(size).persist();
    let b = cx.tensor(size).persist();
    let output = (a + b).output();

    // Generate test data
    let data_a: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 * 0.001).collect();
    let data_b: Vec<f32> = (0..size)
        .map(|i| ((i + 500) % 1000) as f32 * 0.001)
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(a, data_a.clone());
    rt.set_data(b, data_b.clone());
    rt = cx.search(rt, 5);

    // Warm up
    rt.execute(&cx.dyn_map);

    // Run and measure
    rt.execute(&cx.dyn_map);

    // Print stats
    println!("\n=== Large KernelAdd Bandwidth Test ===");
    println!(
        "Tensor size: {} elements ({} MB per tensor)",
        size,
        size * 4 / 1024 / 1024
    );
    println!(
        "Total memory traffic: {} MB (2 reads + 1 write)",
        size * 4 * 3 / 1024 / 1024
    );
    if enabled!(Level::INFO) {
        rt.print_execution_stats();
    }

    // Verify correctness (spot check)
    let result = rt.get_f32(output);
    for i in [0, size / 2, size - 1] {
        let expected = data_a[i] + data_b[i];
        let got = result[i];
        assert!(
            (got - expected).abs() < 1e-5,
            "Mismatch at {}: expected {}, got {}",
            i,
            expected,
            got
        );
    }

    // Check bandwidth is reasonable (at least 50% of peak for large kernels)
    if let Some(peak_bw) = cuda_bandwidth_gbps(&ctx) {
        for stat in &rt.last_kernel_stats {
            let total_bytes = stat.bytes_loaded + stat.bytes_stored;
            if stat.name == "Add" && total_bytes > 0 {
                let utilization = stat.bandwidth_gbps / peak_bw as f64 * 100.0;
                println!(
                    "\nAdd kernel achieved {:.1} GB/s ({:.1}% of {:.0} GB/s peak)",
                    stat.bandwidth_gbps, utilization, peak_bw
                );
                println!(
                    "  Loaded: {} bytes, Stored: {} bytes",
                    stat.bytes_loaded, stat.bytes_stored
                );
                // Large adds should achieve decent bandwidth
                assert!(
                    utilization > 50.0,
                    "Bandwidth utilization too low: {:.1}%",
                    utilization
                );
            }
        }
    }
}
