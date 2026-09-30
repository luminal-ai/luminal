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
pub use dim_range::{DimRange, declare_dim_ranges, dim_range_program};
pub use dtype::TorchDType;
pub use pt2_parser::{InputKind, ParsedPT2, parse_pt2};
pub use torch_device::TorchDevice;
pub use torch_layout::TorchLayout;
pub use translate::{TranslatedInput, TranslatedOutput, Translation, translate};

/// Preserve the exported domain for every translated dimension. Missing finite
/// endpoints retain PyTorch's signed shape range; profiling hints never narrow it.
pub fn dimension_bounds(
    translation: &Translation,
    parsed: &ParsedPT2,
) -> anyhow::Result<luminal::shape::DimensionBounds> {
    use anyhow::ensure;
    luminal::shape::DimensionBounds::from_ranges(
        translation
            .symbols
            .iter()
            .map(|(name, symbol)| {
                let range = parsed.program.range_constraints.get(name);
                let lo = range.and_then(|r| r.min_val).unwrap_or(0).max(0);
                let hi = range.and_then(|r| r.max_val).unwrap_or(i64::MAX - 1);
                ensure!(
                    hi >= lo,
                    "invalid exported dimension range for {name}: [{lo}, {hi}]"
                );
                Ok((*symbol, (usize::try_from(lo)?, usize::try_from(hi)?)))
            })
            .collect::<anyhow::Result<Vec<_>>>()?,
    )
}
