//! Luminal tracing infrastructure.
//!
//! This crate provides a composable Perfetto tracing layer that integrates
//! with the `tracing-subscriber` ecosystem.
//!
//! # Example
//!
//! ```rust,ignore
//! use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
//!
//! let (perfetto, guard) = luminal_tracing::perfetto_layer("trace.pftrace");
//!
//! tracing_subscriber::registry()
//!     .with(tracing_subscriber::fmt::layer())  // Console output
//!     .with(luminal_tracing::luminal_filter()) // Default luminal filters
//!     .with(perfetto)                          // Perfetto trace file
//!     .init();
//!

use tracing_subscriber::filter::{LevelFilter, Targets};

mod perfetto;
pub use perfetto::*;

/// Sets some default crate filters: `luminal=info, egglog=off, everything_else=error`
pub fn luminal_filter() -> Targets {
    Targets::new()
        .with_default(LevelFilter::ERROR)
        .with_target("egglog", LevelFilter::OFF)
        .with_target("luminal", LevelFilter::INFO)
}
