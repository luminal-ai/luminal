//! Shared PyTorch bridge utilities: PT2 parsing, dtype tables, and the
//! ATen -> recorder-frontend translator.
//!
//! Runtime-specific Python packages (today `luminal_reference`) sit on top
//! of this crate; the eventual CUDA package reuses it unchanged and swaps
//! the runtime underneath.

pub mod declaration;
pub mod declared_dtype;
pub mod dim_range;
pub mod dtype;
pub mod pt2_parser;
pub mod pt2_schema;
pub mod torch_device;
pub mod torch_layout;
pub mod translate;

pub use declaration::{Declaration, Declarations, Placement, Rendered, declare};
pub use declared_dtype::{DeclaredDtype, declare_dtypes, declared_dtype_program};
pub use dim_range::{DimRange, declare_dim_ranges, dim_bucket, dim_range_program};
pub use dtype::TorchDType;
pub use pt2_parser::{InputKind, ParsedPT2, parse_pt2};
pub use torch_device::TorchDevice;
pub use torch_layout::TorchLayout;
pub use translate::{TranslatedInput, TranslatedOutput, Translation, translate};
