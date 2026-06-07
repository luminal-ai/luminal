mod attention;
mod rms_norm;
mod rope;
mod swiglu;

pub use attention::MetalFusedPostRopeAttention;
pub use rms_norm::MetalRmsNorm;
pub use rope::{MetalLlamaRope, MetalLlamaRope3D, MetalLlamaRope3DScatter};
pub use swiglu::{MetalFusedSwiGLUGemv, MetalSwiGLU};

pub type Ops = (
    MetalFusedPostRopeAttention,
    MetalLlamaRope,
    MetalLlamaRope3D,
    MetalLlamaRope3DScatter,
    MetalRmsNorm,
    MetalSwiGLU,
    MetalFusedSwiGLUGemv,
);
