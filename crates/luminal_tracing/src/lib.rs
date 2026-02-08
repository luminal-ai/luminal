use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Duration,
};

use tracing_appender::non_blocking::{NonBlocking, WorkerGuard};
use tracing_perfetto_sdk_layer::NativeLayer;
use tracing_perfetto_sdk_schema::{
    DataSourceConfig, TraceConfig, TrackEventConfig,
    trace_config::{BufferConfig, DataSource},
};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

pub struct TraceOptions {
    perfetto_file: Option<PathBuf>,
    pub env_filter: Option<String>,
}

/// This is a convenience tracing subscriber with some opinionated defaults.
pub fn subscriber() -> TraceOptions {
    TraceOptions {
        perfetto_file: None,
        env_filter: None,
    }
}

impl TraceOptions {
    pub fn perfetto(mut self, path: impl AsRef<Path>) -> Self {
        self.perfetto_file = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn env_filter(mut self, env_filter: impl ToString) -> Self {
        self.env_filter = Some(env_filter.to_string());
        self
    }

    /// Install as the global tracing subscriber
    pub fn init(self) -> TraceSession {
        let filter = self.env_filter.map(|f| {
            EnvFilter::builder()
                .parse(f)
                .expect("Invalid tracing env filter")
        });

        if let Some(file_path) = self.perfetto_file {
            let file = File::create(&file_path).expect("Failed to create trace file");
            let (writer, guard) = tracing_appender::non_blocking(file);
            let layer = NativeLayer::from_config(default_perfetto_config(), writer)
                .with_delay_slice_begin(true)
                .build()
                .expect("Failed to build perfetto layer");
            let handle = layer.clone();
            if let Some(f) = filter {
                tracing_subscriber::registry().with(f).with(layer).init();
            } else {
                tracing_subscriber::registry().with(layer).init();
            }
            TraceSession {
                perfetto_layer: Some(handle),
                _guard: Some(guard),
                perfetto_path: Some(file_path),
            }
        } else {
            if let Some(f) = filter {
                tracing_subscriber::registry()
                    .with(f)
                    .with(tracing_subscriber::fmt::layer())
                    .init();
            } else {
                tracing_subscriber::registry()
                    .with(tracing_subscriber::fmt::layer())
                    .init();
            }
            TraceSession {
                perfetto_layer: None,
                _guard: None,
                perfetto_path: None,
            }
        }
    }
}

pub struct TraceSession {
    perfetto_layer: Option<NativeLayer<NonBlocking>>,
    _guard: Option<WorkerGuard>,
    pub perfetto_path: Option<PathBuf>,
}

impl TraceSession {
    pub fn flush(&self) {
        if let Some(layer) = &self.perfetto_layer {
            let _ = layer.flush(Duration::from_secs(5), Duration::from_secs(5));
        }
    }

    pub fn stop(&self) {
        self.flush();
        if let Some(layer) = &self.perfetto_layer {
            let _ = layer.stop();
        }
    }
}

fn default_perfetto_config() -> TraceConfig {
    TraceConfig {
        buffers: vec![BufferConfig {
            size_kb: Some(32_000),
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
