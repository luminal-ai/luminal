//! Runtime adapters for the shared chat session. Each adapter owns its
//! runtime's binding statement and device representation.
use crate::Inputs;
use anyhow::Result;

pub trait Backend {
    fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>>;
    fn reset(&mut self) -> Result<()>;
}
