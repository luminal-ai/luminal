#![allow(unused)]

//! Micro benchmark runner using criterion.
//!
//! Usage and output locations: see `crates/luminal_bench/README.md`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::time::Duration;

#[cfg(feature = "metal")]
use luminal_bench::{
    BenchMetrics, BenchMetricsMap, BenchResultCollector, BenchmarkBackend, BenchmarkPattern,
    HardwareSpec, MetalBenchmark, all_micro_patterns,
};

#[cfg(feature = "metal")]
use luminal::prelude::*;

#[cfg(feature = "metal")]
fn run_metal_pattern_benchmark(
    c: &mut Criterion,
    pattern: &dyn BenchmarkPattern,
    metrics_map: &mut BenchMetricsMap,
    collector: &BenchResultCollector,
) {
    use luminal::hlir::Input;
    use luminal::op::{Runtime, RuntimeStats};
    use luminal_metal::runtime::MetalRuntime;
    use rand::Rng;

    let backend_name = MetalBenchmark::name();
    let pattern_name = pattern.name();
    let group_name = format!("{}/{}", backend_name, pattern_name);

    let mut group = c.benchmark_group(&group_name);

    for size in pattern.sizes() {
        // Build graph and run search once per size; the benchmark loop only measures execution.
        let mut cx = Graph::default();
        pattern.build_graph(&mut cx, *size);

        cx.build_search_space::<MetalRuntime>();
        let mut rt = MetalRuntime::initialize(());

        let mut rng = rand::rng();
        for node in cx.graph.node_indices() {
            if let Some(Input { .. }) = (*cx.graph[node]).as_any().downcast_ref::<Input>() {
                let data: Vec<f32> = (0..size.value).map(|_| rng.random::<f32>()).collect();
                rt.set_data(node, &data);
            }
        }

        let mut rt = cx.search(rt, 5);
        rt.allocate_intermediate_buffers(&cx.dyn_map);

        let mut bench_metrics = None;
        if let Some(stats) = rt.execute_with_stats(&cx.dyn_map) {
            let metrics = BenchMetrics::new(stats.bytes_loaded, stats.bytes_stored, stats.flops);
            metrics_map.add(pattern_name, size.name, metrics.clone());
            bench_metrics = Some(metrics);
        }

        let dyn_map = cx.dyn_map.clone();

        group.bench_with_input(BenchmarkId::from_parameter(size.name), size, |b, size| {
            b.iter_custom(|iters| {
                let mut total_time = Duration::ZERO;

                for _ in 0..iters {
                    if let Some(stats) = rt.execute_with_stats(&dyn_map) {
                        total_time +=
                            Duration::from_secs_f64(stats.execution_time_us / 1_000_000.0);
                    }
                }

                if let Some(ref metrics) = bench_metrics {
                    let avg_time_us = total_time.as_secs_f64() * 1_000_000.0 / iters as f64;
                    collector.add(pattern_name, size.name, size.value, avg_time_us, metrics);
                }

                total_time
            });
        });
    }

    group.finish();
}

#[cfg(feature = "metal")]
fn metal_micro_benchmarks(c: &mut Criterion) {
    let hw = MetalBenchmark::hardware_info();

    println!("\n=== Metal Benchmark ===");
    println!("Device: {}", hw.device_name);
    println!("Memory: {:.1} GB", hw.memory_gb);
    if let Some(bw) = hw.peak_bandwidth_gbps {
        println!("Peak Bandwidth: {:.0} GB/s", bw);
    }
    if let Some(tf) = hw.peak_tflops {
        println!("Peak Compute: {:.1} TFLOPS", tf);
    }
    println!();

    let hardware_spec = HardwareSpec {
        device_name: hw.device_name.clone(),
        memory_gb: hw.memory_gb,
        peak_bandwidth_gbps: hw.peak_bandwidth_gbps.unwrap_or(100.0),
        peak_tflops: hw.peak_tflops.unwrap_or(1.0),
    };

    let mut metrics_map = BenchMetricsMap::new(hardware_spec.clone());
    let collector = BenchResultCollector::new(hardware_spec);

    for pattern in all_micro_patterns() {
        run_metal_pattern_benchmark(c, pattern.as_ref(), &mut metrics_map, &collector);
    }

    let metrics_path = std::path::Path::new("target/criterion/bench_metrics.json");
    if let Some(parent) = metrics_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = metrics_map.save(metrics_path) {
        eprintln!("Warning: Failed to save metrics mapping: {}", e);
    }

    let report = collector.into_report();
    report.print_summary();

    let report_path = std::path::Path::new("target/criterion/bench_report.json");
    if let Err(e) = report.save(report_path) {
        eprintln!("Warning: Failed to save full report: {}", e);
    } else {
        println!("\nReports saved to:");
        println!("  - {}", metrics_path.display());
        println!("  - {}", report_path.display());
    }
}

#[cfg(not(feature = "metal"))]
fn metal_micro_benchmarks(_c: &mut Criterion) {
    println!("Metal benchmarks disabled. Run with --features metal");
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2));
    targets = metal_micro_benchmarks
}

criterion_main!(benches);
