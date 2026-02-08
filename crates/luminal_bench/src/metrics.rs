//! Benchmark metrics and mapping
//!
//! Provides a mapping from benchmark names to their constant metrics (bytes, flops).
//! Combined with Criterion's time measurements, this allows computing derived metrics
//! like throughput, MBU, and MFU.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Constant metrics for a single benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchMetrics {
    /// Total bytes transferred (loaded + stored)
    pub bytes: usize,
    /// Bytes loaded from memory
    pub bytes_loaded: usize,
    /// Bytes stored to memory
    pub bytes_stored: usize,
    /// Floating-point operations
    pub flops: usize,
}

impl BenchMetrics {
    pub fn new(bytes_loaded: usize, bytes_stored: usize, flops: usize) -> Self {
        Self {
            bytes: bytes_loaded + bytes_stored,
            bytes_loaded,
            bytes_stored,
            flops,
        }
    }

    /// Calculate throughput in GB/s given execution time in microseconds
    pub fn throughput_gbps(&self, time_us: f64) -> f64 {
        if time_us <= 0.0 {
            return 0.0;
        }
        self.bytes as f64 / time_us / 1000.0
    }

    /// Calculate TFLOPS given execution time in microseconds
    pub fn tflops(&self, time_us: f64) -> f64 {
        if time_us <= 0.0 {
            return 0.0;
        }
        self.flops as f64 / time_us / 1_000_000.0
    }

    /// Calculate MBU given execution time and peak bandwidth
    pub fn mbu(&self, time_us: f64, peak_bandwidth_gbps: f64) -> f64 {
        self.throughput_gbps(time_us) / peak_bandwidth_gbps * 100.0
    }

    /// Calculate MFU given execution time and peak TFLOPS
    pub fn mfu(&self, time_us: f64, peak_tflops: f64) -> f64 {
        self.tflops(time_us) / peak_tflops * 100.0
    }
}

/// Hardware specifications for a benchmark target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub device_name: String,
    pub memory_gb: f64,
    pub peak_bandwidth_gbps: f64,
    pub peak_tflops: f64,
}

/// Complete benchmark metrics mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchMetricsMap {
    /// Hardware specifications
    pub hardware: HardwareSpec,
    /// Mapping from "pattern/size" to metrics
    pub benchmarks: HashMap<String, BenchMetrics>,
}

impl BenchMetricsMap {
    pub fn new(hardware: HardwareSpec) -> Self {
        Self {
            hardware,
            benchmarks: HashMap::new(),
        }
    }

    /// Add metrics for a benchmark
    pub fn add(&mut self, pattern: &str, size: &str, metrics: BenchMetrics) {
        let key = format!("{}/{}", pattern, size);
        self.benchmarks.insert(key, metrics);
    }

    /// Get metrics for a benchmark
    pub fn get(&self, pattern: &str, size: &str) -> Option<&BenchMetrics> {
        let key = format!("{}/{}", pattern, size);
        self.benchmarks.get(&key)
    }

    /// Export to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Save to file
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = self.to_json();
        std::fs::write(path, json)
    }

    /// Load from file
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

// ============================================================================
// Legacy types (kept for compatibility)
// ============================================================================

/// Result of a single benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    /// Backend name (e.g., "metal", "cuda")
    pub backend: String,
    /// Benchmark pattern name (e.g., "add_vec")
    pub benchmark: String,
    /// Size label (e.g., "1m")
    pub size_label: String,
    /// Actual size value
    pub size_value: usize,
    /// Mean execution time in microseconds
    pub mean_us: f64,
    /// Standard deviation in microseconds
    pub std_us: f64,
    /// Throughput in GB/s (if applicable)
    pub throughput_gbps: Option<f64>,
    /// Memory Bandwidth Utilization (if peak bandwidth known)
    pub mbu: Option<f64>,
}

impl BenchResult {
    /// Calculate throughput given bytes transferred
    pub fn with_throughput(mut self, bytes: usize) -> Self {
        if self.mean_us > 0.0 {
            // bytes / microseconds = MB/s, then convert to GB/s
            self.throughput_gbps = Some((bytes as f64) / self.mean_us / 1000.0);
        }
        self
    }

    /// Calculate MBU given peak bandwidth
    pub fn with_mbu(mut self, peak_bandwidth_gbps: f64) -> Self {
        if let Some(throughput) = self.throughput_gbps {
            self.mbu = Some(throughput / peak_bandwidth_gbps * 100.0);
        }
        self
    }
}

/// Collection of benchmark results for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub backend: String,
    pub hardware: String,
    pub results: Vec<BenchResult>,
}

impl BenchReport {
    pub fn new(backend: &str, hardware: &str) -> Self {
        Self {
            backend: backend.to_string(),
            hardware: hardware.to_string(),
            results: Vec::new(),
        }
    }

    pub fn add_result(&mut self, result: BenchResult) {
        self.results.push(result);
    }

    /// Export to JSON (for CI integration)
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

// ============================================================================
// Full Report with Derived Metrics
// ============================================================================

/// Single benchmark result with all metrics (constant + derived)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullBenchResult {
    pub pattern: String,
    pub size: String,
    pub size_value: usize,
    /// Execution time in microseconds
    pub time_us: f64,
    /// Bytes transferred
    pub bytes: usize,
    /// Floating-point operations
    pub flops: usize,
    /// Throughput in GB/s
    pub throughput_gbps: f64,
    /// Memory Bandwidth Utilization (%)
    pub mbu_percent: f64,
    /// Compute in TFLOPS
    pub tflops: f64,
    /// Model FLOPs Utilization (%)
    pub mfu_percent: f64,
}

/// Full benchmark report with derived metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullBenchReport {
    pub hardware: HardwareSpec,
    pub timestamp: String,
    pub results: Vec<FullBenchResult>,
}

impl FullBenchReport {
    pub fn new(hardware: HardwareSpec) -> Self {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        Self {
            hardware,
            timestamp,
            results: Vec::new(),
        }
    }

    pub fn add_result(&mut self, result: FullBenchResult) {
        self.results.push(result);
    }

    /// Save to JSON file
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).unwrap_or_default();
        std::fs::write(path, json)
    }

    /// Print summary table to terminal
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(100));
        println!("BENCHMARK RESULTS - {}", self.hardware.device_name);
        println!(
            "Peak Bandwidth: {:.0} GB/s | Peak Compute: {:.1} TFLOPS",
            self.hardware.peak_bandwidth_gbps, self.hardware.peak_tflops
        );
        println!("{}", "=".repeat(100));
        println!(
            "{:<20} {:>8} {:>12} {:>10} {:>8} {:>10} {:>8}",
            "Pattern", "Size", "Time(μs)", "GB/s", "MBU%", "TFLOPS", "MFU%"
        );
        println!("{}", "-".repeat(100));

        for r in &self.results {
            println!(
                "{:<20} {:>8} {:>12.2} {:>10.2} {:>7.1}% {:>10.4} {:>7.1}%",
                r.pattern,
                r.size,
                r.time_us,
                r.throughput_gbps,
                r.mbu_percent,
                r.tflops,
                r.mfu_percent
            );
        }
        println!("{}", "=".repeat(100));
    }
}

/// Thread-safe collector for benchmark results
#[derive(Clone)]
pub struct BenchResultCollector {
    hardware: HardwareSpec,
    results: Arc<Mutex<Vec<FullBenchResult>>>,
}

impl BenchResultCollector {
    pub fn new(hardware: HardwareSpec) -> Self {
        Self {
            hardware,
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Add a benchmark result
    pub fn add(
        &self,
        pattern: &str,
        size: &str,
        size_value: usize,
        time_us: f64,
        metrics: &BenchMetrics,
    ) {
        let throughput_gbps = metrics.throughput_gbps(time_us);
        let tflops = metrics.tflops(time_us);
        let mbu_percent = metrics.mbu(time_us, self.hardware.peak_bandwidth_gbps);
        let mfu_percent = metrics.mfu(time_us, self.hardware.peak_tflops);

        let result = FullBenchResult {
            pattern: pattern.to_string(),
            size: size.to_string(),
            size_value,
            time_us,
            bytes: metrics.bytes,
            flops: metrics.flops,
            throughput_gbps,
            mbu_percent,
            tflops,
            mfu_percent,
        };

        self.results.lock().unwrap().push(result);
    }

    /// Generate full report
    pub fn into_report(self) -> FullBenchReport {
        let mut report = FullBenchReport::new(self.hardware);
        report.results = self.results.lock().unwrap().clone();
        // Sort by pattern name, then by size
        report.results.sort_by(|a, b| {
            a.pattern
                .cmp(&b.pattern)
                .then_with(|| a.size_value.cmp(&b.size_value))
        });
        report
    }
}
