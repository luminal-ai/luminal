//! Backend-neutral compact snapshots and direct export from egglog.
pub mod compact;
mod export;
pub use export::CompactSnapshotExt;
