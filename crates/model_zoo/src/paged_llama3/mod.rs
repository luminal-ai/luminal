//! Logical, batched Llama 3 decode graph using a page-table cache layout.
//!
//! The page-table allocator and runtime execution loop belong in runtime
//! crates. This module specifies only the fixed-width logical graph.

pub use crate::llama3::{Llama3, Llama3Dims};
use crate::namespace::Namespace;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;
use luminal_nn::KvCachePool;

/// Fixed batch width of the step-invariant tick graph.
pub const ROWS: usize = 2;

pub struct BatchStep {
    pub cx: Graph,
    pub model: Llama3,
    pub tokens: GraphTensor,
    pub rope_cos: GraphTensor,
    pub rope_sin: GraphTensor,
    pub rope_rot: GraphTensor,
    pub gather_idx: GraphTensor,
    pub scatter_idx: GraphTensor,
    pub mask: GraphTensor,
    pub pool: KvCachePool,
    pub logits: GraphTensor,
    pub cache_outs: Vec<(GraphTensor, GraphTensor)>,
}

impl BatchStep {
    pub fn build(dims: &Llama3Dims, slots: usize) -> Self {
        let mut cx = Graph::new();
        let model = Llama3::init(&mut cx, dims);
        let tokens = cx.tensor(vec![ROWS], DType::Int);
        let rope_cos = cx.tensor(vec![ROWS, dims.head_dim], DType::F32);
        let rope_sin = cx.tensor(vec![ROWS, dims.head_dim], DType::F32);
        let rope_rot = cx.tensor(vec![dims.head_dim, dims.head_dim], DType::F32);
        let gather_idx = cx.tensor(vec![slots], DType::Int);
        let scatter_idx = cx.tensor(vec![ROWS], DType::Int);
        let mask = cx.tensor(vec![ROWS, slots], DType::F32);
        let pool = crate::namespace::named_kv_cache_pool(
            &mut cx,
            dims.layers,
            slots,
            dims.kv_dim(),
            DType::F32,
            &Namespace::root().child("cache"),
        );

        let mut x = luminal_nn::embedding(tokens, model.embed);
        let mut cache_outs = Vec::with_capacity(model.blocks.len());
        for (layer, block) in model.blocks.iter().enumerate() {
            let (next, k_cache, v_cache) = block.forward_rope_masked(
                x,
                pool.layers[layer].0,
                pool.layers[layer].1,
                gather_idx,
                scatter_idx,
                mask,
                rope_cos,
                rope_sin,
                rope_rot,
            );
            x = next;
            cache_outs.push((k_cache, v_cache));
        }
        let logits = luminal_nn::linear(
            luminal_nn::rms_norm(
                x,
                model.final_norm_weight,
                model.final_norm_bias,
                model.final_norm_epsilon,
            ),
            model.lm_head_weight,
            model.lm_head_bias,
        );

        Self {
            cx,
            model,
            tokens,
            rope_cos,
            rope_sin,
            rope_rot,
            gather_idx,
            scatter_idx,
            mask,
            pool,
            logits,
            cache_outs,
        }
    }
}
