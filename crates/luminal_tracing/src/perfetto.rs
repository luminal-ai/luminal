use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Duration,
};

use tracing_appender::non_blocking::{NonBlocking, WorkerGuard};
use tracing_perfetto_sdk_layer::NativeLayer;

// Re-export schema types for advanced users who need to post-process traces
// (e.g., merging GPU timing data)
pub use tracing_perfetto_sdk_schema as schema;

pub use tracing_perfetto_sdk_schema::{
    DataSourceConfig, TraceConfig, TrackEventConfig,
    trace_config::{BufferConfig, DataSource},
};

/// Creates a Perfetto tracing layer with default configuration.
///
/// Returns a tuple of `(layer, guard)`. The layer implements
/// [`Layer<S>`](tracing_subscriber::Layer) and can be composed with other layers
/// using [`tracing_subscriber::registry()`].
///
/// The guard must be kept alive and [`stop()`](PerfettoGuard::stop) should be called
/// before dropping to ensure all events are flushed.
///
/// # Example
///
/// ```rust,ignore
/// use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
///
/// let (perfetto, guard) = luminal_tracing::perfetto_layer("trace.pftrace");
///
/// tracing_subscriber::registry()
///     .with(EnvFilter::new("luminal=info"))
///     .with(perfetto)
///     .init();
///
/// // ... run your code ...
///
/// guard.stop();
///
/// // Optionally post-process the trace file (e.g., merge GPU timings)
/// // using guard.path
/// ```
///
/// # Composing with Other Layers
///
/// ```rust,ignore
/// use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
///
/// let (perfetto, guard) = luminal_tracing::perfetto_layer("trace.pftrace");
///
/// tracing_subscriber::registry()
///     .with(tracing_subscriber::fmt::layer())  // Console output
///     .with(perfetto)                          // Perfetto file
///     .with(my_custom_layer())                 // Your own layer
///     .init();
/// ```
pub fn perfetto_layer(path: impl AsRef<Path>) -> (PerfettoLayer, PerfettoGuard) {
    let config = default_perfetto_config();
    let path = path.as_ref().to_path_buf();
    let file = File::create(&path).expect("Failed to create Perfetto trace file");
    let (writer, writer_guard) = tracing_appender::non_blocking(file);

    let layer = NativeLayer::from_config(config, writer)
        .with_delay_slice_begin(true)
        .build()
        .expect("Failed to build Perfetto layer");

    let handle = layer.clone();

    let guard = PerfettoGuard {
        layer: handle,
        _writer_guard: writer_guard,
        path,
    };

    (layer, guard)
}

/// The concrete Perfetto layer type. This implements `Layer<S>` for any
/// subscriber `S` that satisfies the required bounds.
pub type PerfettoLayer = NativeLayer<NonBlocking>;

/// Guard that manages the lifecycle of a Perfetto tracing session.
///
/// This guard must be kept alive for the duration of tracing. Call [`stop`](Self::stop)
/// before dropping to ensure all trace events are flushed to disk.
///
/// The [`path`](Self::path) field provides access to the trace file path, which is
/// useful for post-processing (e.g., merging GPU timing data).
pub struct PerfettoGuard {
    layer: PerfettoLayer,
    _writer_guard: WorkerGuard,
    /// Path to the Perfetto trace file.
    pub path: PathBuf,
}

impl PerfettoGuard {
    /// Flush pending trace events to disk.
    ///
    /// This blocks until the flush completes or times out (5 seconds).
    pub fn flush(&self) {
        let _ = self
            .layer
            .flush(Duration::from_secs(5), Duration::from_secs(5));
    }

    /// Stop the tracing session and flush all pending events.
    ///
    /// This should be called before dropping the guard to ensure all trace
    /// data is written to disk.
    pub fn stop(&self) {
        self.flush();
        let _ = self.layer.stop();
    }

    /// Get a clone of the underlying layer handle.
    ///
    /// This can be useful if you need to interact with the layer directly.
    pub fn layer_handle(&self) -> PerfettoLayer {
        self.layer.clone()
    }
}

fn default_perfetto_config() -> TraceConfig {
    TraceConfig {
        buffers: vec![BufferConfig {
            size_kb: Some(256_000), // 256 MB
            ..Default::default()
        }],
        data_sources: vec![DataSource {
            config: Some(DataSourceConfig {
                name: Some("rust_tracing".into()),
                track_event_config: Some(TrackEventConfig {
                    filter_debug_annotations: Some(false),
                    enabled_categories: vec!["*".to_string()],
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    }
}
