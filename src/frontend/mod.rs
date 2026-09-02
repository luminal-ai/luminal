// The high level interface implemented on GraphTensor. All of these ops get translated to primitive ops.
pub mod binary;
pub mod matmul;
pub mod movement;
pub mod other;
pub mod reduction;
pub mod tensor;
pub mod unary;

pub use movement::ViewChain;
pub use tensor::*;
