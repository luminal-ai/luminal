//! Shared PyTorch bridge utilities: PT2 parsing, dtype tables, and the
//! ATen -> recorder-frontend translator.
//!
//! Runtime-specific Python packages (today `luminal_reference`) sit on top
//! of this crate; the eventual CUDA package reuses it unchanged and swaps
//! the runtime underneath.

pub mod dtype;
pub mod pt2_parser;
pub mod pt2_schema;
pub mod translate;

pub use dtype::TorchDType;
pub use pt2_parser::{InputKind, ParsedPT2, parse_pt2};
pub use translate::{TranslatedInput, TranslatedOutput, Translation, translate};
