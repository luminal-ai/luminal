//! Backend-neutral model, checkpoint, tokenizer, and session code shared by the chat applications.
pub mod app;
pub mod backend;
pub mod checkpoint;
pub mod graph;
pub mod hf;
pub mod sampling;
pub mod search;
pub mod session;
pub mod tokenizer;

use anyhow::{Result, bail};
use luminal::{
    dtype::DType,
    prelude::{FxHashMap, NodeIndex},
};

/// The application's host data; the backend owns the device representation.
#[derive(Clone, Debug)]
pub enum TensorData {
    F32(Vec<f32>),
    BF16(Vec<u16>),
    F16(Vec<u16>),
    I32(Vec<i32>),
}

impl TensorData {
    pub fn from_f32(dtype: DType, values: Vec<f32>) -> Result<Self> {
        Ok(match dtype {
            DType::F32 => Self::F32(values),
            DType::Bf16 => Self::BF16(
                values
                    .into_iter()
                    .map(|x| half::bf16::from_f32(x).to_bits())
                    .collect(),
            ),
            DType::F16 => Self::F16(
                values
                    .into_iter()
                    .map(|x| half::f16::from_f32(x).to_bits())
                    .collect(),
            ),
            dtype => bail!("chat has no floating-point host payload for {dtype:?}"),
        })
    }

    pub fn zeros(dtype: DType, elements: usize) -> Result<Self> {
        Self::from_f32(dtype, vec![0.; elements])
    }
}
pub type Inputs = FxHashMap<NodeIndex, TensorData>;
