#![allow(unused)]

//! Pattern benchmark runner using criterion.
//!
//! Usage and output locations: see `crates/luminal_bench/README.md`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::time::Duration;

#[cfg(feature = "metal")]
use luminal_bench::{
    ATTENTION_SIZES, BenchMetrics, BenchMetricsMap, BenchResultCollector, BenchmarkBackend,
    HardwareSpec, MATMUL_SIZES, MetalBenchmark, TRANSFORMER_SIZES,
};

#[cfg(feature = "metal")]
use luminal::hlir::Input;

#[cfg(feature = "metal")]
use luminal::op::{Runtime, RuntimeStats};

#[cfg(feature = "metal")]
use luminal::prelude::*;

#[cfg(feature = "metal")]
use luminal_metal::runtime::MetalRuntime;

#[cfg(feature = "metal")]
use rand::Rng;

// ============================================================================
// Helper: Prepare runtime with graph and search (done once per size)
// ============================================================================

#[cfg(feature = "metal")]
struct PreparedBench {
    rt: MetalRuntime,
    dyn_map: luminal::prelude::FxHashMap<char, usize>,
    metrics: Option<BenchMetrics>,
}

#[cfg(feature = "metal")]
fn prepare_and_search(cx: &mut Graph, input_sizes: &[(NodeIndex, usize)]) -> Option<PreparedBench> {
    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());

    let mut rng = rand::rng();
    for (node, size) in input_sizes {
        let data: Vec<f32> = (0..*size).map(|_| rng.random::<f32>()).collect();
        rt.set_data(*node, &data);
    }

    let rt = cx.search(rt, 5);

    Some(PreparedBench {
        rt,
        dyn_map: cx.dyn_map.clone(),
        metrics: None,
    })
}

// ============================================================================
// MatMul Benchmark
// ============================================================================

#[cfg(feature = "metal")]
fn bench_matmul(
    c: &mut Criterion,
    metrics_map: &mut BenchMetricsMap,
    collector: &BenchResultCollector,
) {
    let mut group = c.benchmark_group("metal/matmul");

    for size in MATMUL_SIZES {
        let size_name = size.name;
        let (m, k, n) = (size.m, size.k, size.n);

        // Build graph and run search once per size; the benchmark loop only measures execution.
        let mut cx = Graph::default();
        let a = cx.tensor((m, k));
        let b_tensor = cx.tensor((k, n));
        let _ = a.matmul(b_tensor).output();

        let input_sizes: Vec<(NodeIndex, usize)> = cx
            .graph
            .node_indices()
            .filter_map(|node| {
                if (*cx.graph[node]).as_any().downcast_ref::<Input>().is_some() {
                    Some((node, m * k.max(k * n)))
                } else {
                    None
                }
            })
            .collect();

        let Some(mut prepared) = prepare_and_search(&mut cx, &input_sizes) else {
            println!("error:  Skipping matmul/{} - search failed", size_name);
            continue;
        };

        prepared.rt.allocate_intermediate_buffers(&prepared.dyn_map);

        if let Some(stats) = prepared.rt.execute_with_stats(&prepared.dyn_map) {
            let metrics = BenchMetrics::new(stats.bytes_loaded, stats.bytes_stored, stats.flops);
            metrics_map.add("matmul", size_name, metrics.clone());
            prepared.metrics = Some(metrics);
        }

        group.bench_with_input(BenchmarkId::from_parameter(size_name), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = Duration::ZERO;

                for _ in 0..iters {
                    if let Some(stats) = prepared.rt.execute_with_stats(&prepared.dyn_map) {
                        total_time +=
                            Duration::from_secs_f64(stats.execution_time_us / 1_000_000.0);
                    }
                }

                if let Some(ref metrics) = prepared.metrics {
                    let avg_time_us = total_time.as_secs_f64() * 1_000_000.0 / iters as f64;
                    collector.add("matmul", size_name, m * k * n, avg_time_us, metrics);
                }

                total_time
            });
        });
    }

    group.finish();
}

// ============================================================================
// Softmax Benchmark
// ============================================================================

#[cfg(feature = "metal")]
fn bench_softmax(
    c: &mut Criterion,
    metrics_map: &mut BenchMetricsMap,
    collector: &BenchResultCollector,
) {
    let mut group = c.benchmark_group("metal/softmax");

    for size in TRANSFORMER_SIZES {
        let size_name = size.name;
        let size_value = size.value;

        let dim = (size_value as f64).sqrt() as usize;
        let rows = size_value / dim;
        let cols = dim;

        let mut cx = Graph::default();
        let x = cx.tensor((rows, cols));
        let _ = x.softmax(1).output();

        let input_sizes: Vec<(NodeIndex, usize)> = cx
            .graph
            .node_indices()
            .filter_map(|node| {
                if (*cx.graph[node]).as_any().downcast_ref::<Input>().is_some() {
                    Some((node, size_value))
                } else {
                    None
                }
            })
            .collect();

        let Some(mut prepared) = prepare_and_search(&mut cx, &input_sizes) else {
            println!("error:  Skipping softmax/{} - search failed", size_name);
            continue;
        };

        prepared.rt.allocate_intermediate_buffers(&prepared.dyn_map);

        if let Some(stats) = prepared.rt.execute_with_stats(&prepared.dyn_map) {
            let metrics = BenchMetrics::new(stats.bytes_loaded, stats.bytes_stored, stats.flops);
            metrics_map.add("softmax", size_name, metrics.clone());
            prepared.metrics = Some(metrics);
        }

        group.bench_with_input(BenchmarkId::from_parameter(size_name), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = Duration::ZERO;

                for _ in 0..iters {
                    if let Some(stats) = prepared.rt.execute_with_stats(&prepared.dyn_map) {
                        total_time +=
                            Duration::from_secs_f64(stats.execution_time_us / 1_000_000.0);
                    }
                }

                if let Some(ref metrics) = prepared.metrics {
                    let avg_time_us = total_time.as_secs_f64() * 1_000_000.0 / iters as f64;
                    collector.add("softmax", size_name, size_value, avg_time_us, metrics);
                }

                total_time
            });
        });
    }

    group.finish();
}

// ============================================================================
// LayerNorm Benchmark
// ============================================================================

#[cfg(feature = "metal")]
fn bench_layer_norm(
    c: &mut Criterion,
    metrics_map: &mut BenchMetricsMap,
    collector: &BenchResultCollector,
) {
    let mut group = c.benchmark_group("metal/layer_norm");

    for size in TRANSFORMER_SIZES {
        let size_name = size.name;
        let size_value = size.value;

        // Typical shape: (batch * seq_len, hidden_dim)
        let hidden_dim = 128;
        let batch_seq = (size_value / hidden_dim).max(1);

        let mut cx = Graph::default();
        let x = cx.tensor((batch_seq, hidden_dim));
        // LayerNorm along last axis with epsilon
        let _ = x.layer_norm(1, 1e-5).output();

        let input_sizes: Vec<(NodeIndex, usize)> = cx
            .graph
            .node_indices()
            .filter_map(|node| {
                if (*cx.graph[node]).as_any().downcast_ref::<Input>().is_some() {
                    Some((node, batch_seq * hidden_dim))
                } else {
                    None
                }
            })
            .collect();

        let Some(mut prepared) = prepare_and_search(&mut cx, &input_sizes) else {
            println!("error:  Skipping layer_norm/{} - search failed", size_name);
            continue;
        };

        prepared.rt.allocate_intermediate_buffers(&prepared.dyn_map);

        if let Some(stats) = prepared.rt.execute_with_stats(&prepared.dyn_map) {
            let metrics = BenchMetrics::new(stats.bytes_loaded, stats.bytes_stored, stats.flops);
            metrics_map.add("layer_norm", size_name, metrics.clone());
            prepared.metrics = Some(metrics);
        }

        group.bench_with_input(BenchmarkId::from_parameter(size_name), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = Duration::ZERO;

                for _ in 0..iters {
                    if let Some(stats) = prepared.rt.execute_with_stats(&prepared.dyn_map) {
                        total_time +=
                            Duration::from_secs_f64(stats.execution_time_us / 1_000_000.0);
                    }
                }

                if let Some(ref metrics) = prepared.metrics {
                    let avg_time_us = total_time.as_secs_f64() * 1_000_000.0 / iters as f64;
                    collector.add(
                        "layer_norm",
                        size_name,
                        batch_seq * hidden_dim,
                        avg_time_us,
                        metrics,
                    );
                }

                total_time
            });
        });
    }

    group.finish();
}

// ============================================================================
// GeLU Benchmark
// ============================================================================

#[cfg(feature = "metal")]
fn bench_gelu(
    c: &mut Criterion,
    metrics_map: &mut BenchMetricsMap,
    collector: &BenchResultCollector,
) {
    let mut group = c.benchmark_group("metal/gelu");

    for size in TRANSFORMER_SIZES {
        let size_name = size.name;
        let size_value = size.value;

        let mut cx = Graph::default();
        let x = cx.tensor(size_value);
        let _ = x.gelu().output();

        let input_sizes: Vec<(NodeIndex, usize)> = cx
            .graph
            .node_indices()
            .filter_map(|node| {
                if (*cx.graph[node]).as_any().downcast_ref::<Input>().is_some() {
                    Some((node, size_value))
                } else {
                    None
                }
            })
            .collect();

        let Some(mut prepared) = prepare_and_search(&mut cx, &input_sizes) else {
            println!("error:  Skipping gelu/{} - search failed", size_name);
            continue;
        };

        prepared.rt.allocate_intermediate_buffers(&prepared.dyn_map);

        if let Some(stats) = prepared.rt.execute_with_stats(&prepared.dyn_map) {
            let metrics = BenchMetrics::new(stats.bytes_loaded, stats.bytes_stored, stats.flops);
            metrics_map.add("gelu", size_name, metrics.clone());
            prepared.metrics = Some(metrics);
        }

        group.bench_with_input(BenchmarkId::from_parameter(size_name), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = Duration::ZERO;

                for _ in 0..iters {
                    if let Some(stats) = prepared.rt.execute_with_stats(&prepared.dyn_map) {
                        total_time +=
                            Duration::from_secs_f64(stats.execution_time_us / 1_000_000.0);
                    }
                }

                if let Some(ref metrics) = prepared.metrics {
                    let avg_time_us = total_time.as_secs_f64() * 1_000_000.0 / iters as f64;
                    collector.add("gelu", size_name, size_value, avg_time_us, metrics);
                }

                total_time
            });
        });
    }

    group.finish();
}

// ============================================================================
// Attention Benchmark
// ============================================================================

#[cfg(feature = "metal")]
fn bench_attention(
    c: &mut Criterion,
    metrics_map: &mut BenchMetricsMap,
    collector: &BenchResultCollector,
) {
    let mut group = c.benchmark_group("metal/attention");

    for (seq_len, head_dim) in ATTENTION_SIZES {
        let size_name = format!("{}x{}", seq_len, head_dim);
        let seq_len = *seq_len;
        let head_dim = *head_dim;

        let mut cx = Graph::default();

        let q = cx.tensor((seq_len, head_dim));
        let k = cx.tensor((seq_len, head_dim));
        let v = cx.tensor((seq_len, head_dim));

        let scores = q.matmul(k.permute((1, 0)));
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scaled_scores = scores * scale;
        let attn_weights = scaled_scores.softmax(1);
        let _ = attn_weights.matmul(v).output();

        let input_sizes: Vec<(NodeIndex, usize)> = cx
            .graph
            .node_indices()
            .filter_map(|node| {
                if (*cx.graph[node]).as_any().downcast_ref::<Input>().is_some() {
                    Some((node, seq_len * head_dim))
                } else {
                    None
                }
            })
            .collect();

        let Some(mut prepared) = prepare_and_search(&mut cx, &input_sizes) else {
            println!("error:  Skipping attention/{} - search failed", size_name);
            continue;
        };

        prepared.rt.allocate_intermediate_buffers(&prepared.dyn_map);

        if let Some(stats) = prepared.rt.execute_with_stats(&prepared.dyn_map) {
            let metrics = BenchMetrics::new(stats.bytes_loaded, stats.bytes_stored, stats.flops);
            metrics_map.add("attention", &size_name, metrics.clone());
            prepared.metrics = Some(metrics);
        }

        let size_name_clone = size_name.clone();
        group.bench_with_input(
            BenchmarkId::from_parameter(&size_name),
            &(seq_len, head_dim),
            |b, _| {
                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        if let Some(stats) = prepared.rt.execute_with_stats(&prepared.dyn_map) {
                            total_time +=
                                Duration::from_secs_f64(stats.execution_time_us / 1_000_000.0);
                        }
                    }

                    if let Some(ref metrics) = prepared.metrics {
                        let avg_time_us = total_time.as_secs_f64() * 1_000_000.0 / iters as f64;
                        collector.add(
                            "attention",
                            &size_name_clone,
                            seq_len * head_dim,
                            avg_time_us,
                            metrics,
                        );
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Main Benchmark Entry
// ============================================================================

#[cfg(feature = "metal")]
fn metal_pattern_benchmarks(c: &mut Criterion) {
    let hw = MetalBenchmark::hardware_info();

    println!("\n=== Metal Pattern Benchmarks ===");
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

    bench_matmul(c, &mut metrics_map, &collector);
    bench_softmax(c, &mut metrics_map, &collector);
    bench_layer_norm(c, &mut metrics_map, &collector);
    bench_gelu(c, &mut metrics_map, &collector);
    bench_attention(c, &mut metrics_map, &collector);

    let metrics_path = std::path::Path::new("target/criterion/pattern_metrics.json");
    if let Some(parent) = metrics_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = metrics_map.save(metrics_path) {
        eprintln!("Warning: Failed to save metrics mapping: {}", e);
    }

    let report = collector.into_report();
    report.print_summary();

    let report_path = std::path::Path::new("target/criterion/pattern_report.json");
    if let Err(e) = report.save(report_path) {
        eprintln!("Warning: Failed to save full report: {}", e);
    } else {
        println!("\nReports saved to:");
        println!("  - {}", metrics_path.display());
        println!("  - {}", report_path.display());
    }
}

#[cfg(not(feature = "metal"))]
fn metal_pattern_benchmarks(_c: &mut Criterion) {
    println!("Metal benchmarks disabled. Run with --features metal");
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(3));
    targets = metal_pattern_benchmarks
}

criterion_main!(benches);
