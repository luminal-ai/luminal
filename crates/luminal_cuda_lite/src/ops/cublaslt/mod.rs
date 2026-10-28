//! cuBLASLt production admission and execution.
//!
//! Both executable forms require shared approval. Logical and layout proofs join a
//! complete `CublasLtConfiguration` whitelist entry before either the separate
//! destination or in-place accumulating op is constructed. Extraction decodes
//! the elected descriptors; dispatch does not reinterpret them.
//!
//! F32 reduction reassociation and FMA contraction remain permitted. Explicit
//! casts retain their rounding boundaries; unproved epilogues remain generic.
pub mod accumulate_in_place;
pub mod approved;
#[cfg(feature = "device")]
pub mod device_call;
pub mod election;
pub mod exec;
/// Optional C and bias arguments of an approved complete call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CublasLtForm {
    Base,
    Bias,
    Accumulate,
    AccumulateBias,
}
impl CublasLtForm {
    pub const ALL: [Self; 4] = [
        Self::Base,
        Self::Bias,
        Self::Accumulate,
        Self::AccumulateBias,
    ];
    pub fn lit_arity(self) -> usize {
        2 + usize::from(self.has_c()) + usize::from(self.has_bias())
    }
    pub fn has_c(self) -> bool {
        matches!(self, Self::Accumulate | Self::AccumulateBias)
    }
    pub fn has_bias(self) -> bool {
        matches!(self, Self::Bias | Self::AccumulateBias)
    }
    pub fn operand_names(self) -> &'static [&'static str] {
        match self {
            Self::Base => &["a", "b"],
            Self::Bias => &["a", "b", "bias"],
            Self::Accumulate => &["a", "b", "c"],
            Self::AccumulateBias => &["a", "b", "bias", "c"],
        }
    }
}
