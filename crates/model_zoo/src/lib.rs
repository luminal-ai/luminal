//! Backend-neutral logical model definitions built from `luminal` and
//! `luminal_nn`. Runtime crates own loading, compilation, and execution.
//!
//! Full models are grouped by family; [`mini`] contains small model fixtures.

pub mod checkpoint_layout;
pub mod mini;
pub mod model_support;

pub mod flux2;
pub mod gemma3;
pub mod gemma4_moe;
pub mod llama3;
pub mod llama3_1_fp8;
pub mod paged_llama3;
pub mod qwen3;
pub mod qwen3_moe;
pub mod whisper;
pub mod yolo_v11;
