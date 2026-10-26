//! google/gemma-4-26B-A4B (text tower) as pure logical ops — 100% of
//! THIS checkpoint (ruling 2026-08-12), the zoo's most heterogeneous
//! anatomy: 30 layers in the 5-sliding:1-full alternation where the
//! ROLES DIFFER STRUCTURALLY — sliding layers run head_dim 256 with 8
//! KV heads, theta 10k, full rotary, and their own v_proj; full layers
//! run head_dim 512 with 2 KV heads, theta 1M, PARTIAL rotary (0.25 —
//! zero-angle lanes pass through the pairing form untouched), and NO
//! v_proj (V is the K projection output). V takes a WEIGHTLESS
//! per-head RMS norm on every layer; q/k take learned QK-norms.
//! Attention scale is 1.0 (none). Seven learned norms per layer wrap a
//! PARALLEL dense+MoE FF stage; the whole residual stream multiplies a
//! learned per-layer scalar. The MoE router reads the RAW residual
//! (std-normed × router.scale × 1/√hidden) while experts read the
//! pre_ff_2-normed stream; top-8 weights renormalize then multiply the
//! learned per_expert_scale. Gating is the tanh-approx GELU in sigmoid
//! form everywhere. Logits softcap at 30. Embeddings tie with the
//! √hidden normalizer in-graph over the unscaled table.

use crate::namespace::Namespace;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;
use luminal_nn::{
    AttentionGeometry, CacheAccess, KvCache, KvCachePool, TopKRoutes, causal_bias, paged_attention,
    sliding_window_bias,
};

pub const SLIDING_WINDOW: usize = 1024;

#[derive(Clone)]
pub struct Gemma4Dims {
    pub vocab: usize,
    pub hidden: usize,
    pub dense_intermediate: usize,
    pub moe_intermediate: usize,
    pub experts: usize,
    pub top_k: usize,
    pub n_heads: usize,
    pub layers: usize,
    pub sliding_pattern: usize,
    pub window: usize,
    pub sliding_head_dim: usize,
    pub sliding_kv_heads: usize,
    pub full_head_dim: usize,
    pub full_kv_heads: usize,
    pub full_partial_rotary: f32,
    pub rms_eps: f32,
    pub logit_softcap: f32,
}

impl Gemma4Dims {
    pub fn gemma4_26b_a4b() -> Self {
        Self {
            vocab: 262_144,
            hidden: 2816,
            dense_intermediate: 2112,
            moe_intermediate: 704,
            experts: 128,
            top_k: 8,
            n_heads: 16,
            layers: 30,
            sliding_pattern: 6,
            window: SLIDING_WINDOW,
            sliding_head_dim: 256,
            sliding_kv_heads: 8,
            full_head_dim: 512,
            full_kv_heads: 2,
            full_partial_rotary: 0.25,
            rms_eps: 1e-6,
            logit_softcap: 30.0,
        }
    }

    pub fn tiny() -> Self {
        Self {
            vocab: 23,
            hidden: 8,
            dense_intermediate: 6,
            moe_intermediate: 4,
            experts: 4,
            top_k: 2,
            n_heads: 2,
            layers: 2, // layer 0 sliding, layer 1 full (pattern 2)
            sliding_pattern: 2,
            window: 3,
            sliding_head_dim: 4,
            sliding_kv_heads: 2,
            full_head_dim: 8,
            full_kv_heads: 1,
            full_partial_rotary: 0.25,
            rms_eps: 1e-6,
            logit_softcap: 30.0,
        }
    }

    #[allow(clippy::manual_is_multiple_of)] // Keep rust-version 1.85 compatibility.
    pub fn is_sliding(&self, layer: usize) -> bool {
        (layer + 1) % self.sliding_pattern != 0
    }

    pub fn head_dim(&self, layer: usize) -> usize {
        if self.is_sliding(layer) {
            self.sliding_head_dim
        } else {
            self.full_head_dim
        }
    }

    pub fn kv_heads(&self, layer: usize) -> usize {
        if self.is_sliding(layer) {
            self.sliding_kv_heads
        } else {
            self.full_kv_heads
        }
    }

    pub fn kv_dim(&self, layer: usize) -> usize {
        self.head_dim(layer) * self.kv_heads(layer)
    }

    /// Per-layer kv dims for the heterogeneous pool.
    pub fn kv_dims(&self) -> Vec<usize> {
        (0..self.layers).map(|l| self.kv_dim(l)).collect()
    }
}

/// The parallel-branch MoE (router on the RAW stream, experts on the
/// normed one, gemma-gelu gating, per-expert output scales).
pub struct Gemma4MoeFfn {
    pub router_proj_weight: GraphTensor,
    pub router_proj_bias: Option<GraphTensor>,
    pub router_scale: GraphTensor,
    pub per_expert_scale: GraphTensor,
    pub gate_up: GraphTensor,
    pub down: GraphTensor,
    pub hidden: usize,
    pub intermediate: usize,
    pub top_k: usize,
    pub rms_eps: f32,
}

impl Gemma4MoeFfn {
    fn new(ns: &Namespace, d: &Gemma4Dims, cx: &mut Graph) -> Self {
        let router = ns.child("router");
        let experts = ns.child("experts");
        Self {
            router_proj_weight: cx.named_tensor(
                router.child("proj").leaf("weight"),
                vec![d.hidden, d.experts],
                DType::F32,
            ),
            router_proj_bias: None,
            router_scale: cx.named_tensor(router.leaf("scale"), vec![d.hidden], DType::F32),
            per_expert_scale: cx.named_tensor(
                router.leaf("per_expert_scale"),
                vec![d.experts],
                DType::F32,
            ),
            gate_up: cx.named_tensor(
                experts.leaf("gate_up_proj"),
                vec![d.experts, 2 * d.moe_intermediate, d.hidden],
                DType::F32,
            ),
            down: cx.named_tensor(
                experts.leaf("down_proj"),
                vec![d.experts, d.hidden, d.moe_intermediate],
                DType::F32,
            ),
            hidden: d.hidden,
            intermediate: d.moe_intermediate,
            top_k: d.top_k,
            rms_eps: d.rms_eps,
        }
    }

    /// `raw` drives routing; `normed` feeds the experts.
    fn forward(&self, raw: GraphTensor, normed: GraphTensor) -> GraphTensor {
        let h = self.hidden;

        // Router: std-normed raw stream × router.scale × 1/sqrt(hidden).
        let scale = self.router_scale.expand_lhs(&raw.dims()[..1]);
        let router_hidden = raw.std_norm(vec![1], self.rms_eps) * scale * (h as f32).sqrt().recip();
        let probs = luminal_nn::linear(
            router_hidden,
            self.router_proj_weight,
            self.router_proj_bias,
        )
        .softmax(vec![1]);
        let expert_ids = probs.topk_indexes(self.top_k, 1);
        let routes = TopKRoutes::from_scores(probs, expert_ids).normalize();
        let scaled_weights = routes.weights() * routes.select(self.per_expert_scale);
        let routes = routes.with_weights(scaled_weights);

        let gate_up = routes.select(self.gate_up);
        let projected = routes
            .dispatch(normed)
            .expand_dim(2, 1)
            .matmul(gate_up.permute(vec![0, 1, 3, 2]))
            .squeeze(2);

        let gate = projected.slice_along(..self.intermediate, 2);
        let up = projected.slice_along(self.intermediate.., 2);
        let hidden_states = gemma_gelu(gate) * up;

        let down = routes.select(self.down);
        let routed_output = hidden_states
            .expand_dim(2, 1)
            .matmul(down.permute(vec![0, 1, 3, 2]))
            .squeeze(2);
        routes.combine(routed_output)
    }
}

/// The tanh-approx GELU in sigmoid form (fewer e-graph nodes).
#[allow(clippy::excessive_precision)]
fn gemma_gelu(x: GraphTensor) -> GraphTensor {
    x * (x * 1.595_769_1 * (x * x * 0.044715 + 1.0)).sigmoid()
}

pub struct Gemma4Block {
    pub input_norm_weight: Option<GraphTensor>,
    pub input_norm_bias: Option<GraphTensor>,
    pub input_norm_epsilon: f32,
    pub post_attn_norm_weight: Option<GraphTensor>,
    pub post_attn_norm_bias: Option<GraphTensor>,
    pub post_attn_norm_epsilon: f32,
    pub pre_ff_norm_weight: Option<GraphTensor>,
    pub pre_ff_norm_bias: Option<GraphTensor>,
    pub pre_ff_norm_epsilon: f32,
    pub post_ff_norm_weight: Option<GraphTensor>,
    pub post_ff_norm_bias: Option<GraphTensor>,
    pub post_ff_norm_epsilon: f32,
    pub post_ff_norm_1_weight: Option<GraphTensor>,
    pub post_ff_norm_1_bias: Option<GraphTensor>,
    pub post_ff_norm_1_epsilon: f32,
    pub pre_ff_norm_2_weight: Option<GraphTensor>,
    pub pre_ff_norm_2_bias: Option<GraphTensor>,
    pub pre_ff_norm_2_epsilon: f32,
    pub post_ff_norm_2_weight: Option<GraphTensor>,
    pub post_ff_norm_2_bias: Option<GraphTensor>,
    pub post_ff_norm_2_epsilon: f32,
    pub layer_scalar: GraphTensor,
    pub wq_weight: GraphTensor,
    pub wq_bias: Option<GraphTensor>,
    pub wk_weight: GraphTensor,
    pub wk_bias: Option<GraphTensor>,
    /// Sliding layers only — full layers take V from the K projection.
    pub wv: Option<(GraphTensor, Option<GraphTensor>)>,
    pub wo_weight: GraphTensor,
    pub wo_bias: Option<GraphTensor>,
    pub q_norm: GraphTensor,
    pub k_norm: GraphTensor,
    pub gate_weight: GraphTensor,
    pub gate_bias: Option<GraphTensor>,
    pub up_weight: GraphTensor,
    pub up_bias: Option<GraphTensor>,
    pub down_weight: GraphTensor,
    pub down_bias: Option<GraphTensor>,
    pub moe: Gemma4MoeFfn,
    pub sliding: bool,
    pub head_dim: usize,
    pub kv_heads: usize,
}

impl Gemma4Block {
    fn new(l: usize, d: &Gemma4Dims, cx: &mut Graph) -> Self {
        let sliding = d.is_sliding(l);
        let head_dim = d.head_dim(l);
        let kv_heads = d.kv_heads(l);
        let q_dim = d.n_heads * head_dim;
        let kv_dim = head_dim * kv_heads;
        let ns = Namespace::root()
            .child("model")
            .child("language_model")
            .child("layers")
            .index(l);
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        let rms = |segment: &str, cx: &mut Graph| {
            (
                Some(cx.named_tensor(ns.child(segment).leaf("weight"), vec![d.hidden], DType::F32)),
                None,
                d.rms_eps,
            )
        };
        {
            let (input_norm_weight, input_norm_bias, input_norm_epsilon) =
                rms("input_layernorm", cx);
            let (post_attn_norm_weight, post_attn_norm_bias, post_attn_norm_epsilon) =
                rms("post_attention_layernorm", cx);
            let (pre_ff_norm_weight, pre_ff_norm_bias, pre_ff_norm_epsilon) =
                rms("pre_feedforward_layernorm", cx);
            let (post_ff_norm_weight, post_ff_norm_bias, post_ff_norm_epsilon) =
                rms("post_feedforward_layernorm", cx);
            let (post_ff_norm_1_weight, post_ff_norm_1_bias, post_ff_norm_1_epsilon) =
                rms("post_feedforward_layernorm_1", cx);
            let (pre_ff_norm_2_weight, pre_ff_norm_2_bias, pre_ff_norm_2_epsilon) =
                rms("pre_feedforward_layernorm_2", cx);
            let (post_ff_norm_2_weight, post_ff_norm_2_bias, post_ff_norm_2_epsilon) =
                rms("post_feedforward_layernorm_2", cx);
            Self {
                input_norm_weight,
                input_norm_bias,
                input_norm_epsilon,
                post_attn_norm_weight,
                post_attn_norm_bias,
                post_attn_norm_epsilon,
                pre_ff_norm_weight,
                pre_ff_norm_bias,
                pre_ff_norm_epsilon,
                post_ff_norm_weight,
                post_ff_norm_bias,
                post_ff_norm_epsilon,
                post_ff_norm_1_weight,
                post_ff_norm_1_bias,
                post_ff_norm_1_epsilon,
                pre_ff_norm_2_weight,
                pre_ff_norm_2_bias,
                pre_ff_norm_2_epsilon,
                post_ff_norm_2_weight,
                post_ff_norm_2_bias,
                post_ff_norm_2_epsilon,
                layer_scalar: cx.named_tensor(ns.leaf("layer_scalar"), vec![1], DType::F32),
                wq_weight: cx.named_tensor(
                    attn.child("q_proj").leaf("weight"),
                    vec![d.hidden, q_dim],
                    DType::F32,
                ),
                wq_bias: None,
                wk_weight: cx.named_tensor(
                    attn.child("k_proj").leaf("weight"),
                    vec![d.hidden, kv_dim],
                    DType::F32,
                ),
                wk_bias: None,
                wv: sliding.then(|| {
                    (
                        cx.named_tensor(
                            attn.child("v_proj").leaf("weight"),
                            vec![d.hidden, kv_dim],
                            DType::F32,
                        ),
                        None,
                    )
                }),
                wo_weight: cx.named_tensor(
                    attn.child("o_proj").leaf("weight"),
                    vec![q_dim, d.hidden],
                    DType::F32,
                ),
                wo_bias: None,
                q_norm: cx.named_tensor(
                    attn.child("q_norm").leaf("weight"),
                    vec![head_dim],
                    DType::F32,
                ),
                k_norm: cx.named_tensor(
                    attn.child("k_norm").leaf("weight"),
                    vec![head_dim],
                    DType::F32,
                ),
                gate_weight: cx.named_tensor(
                    mlp.child("gate_proj").leaf("weight"),
                    vec![d.hidden, d.dense_intermediate],
                    DType::F32,
                ),
                gate_bias: None,
                up_weight: cx.named_tensor(
                    mlp.child("up_proj").leaf("weight"),
                    vec![d.hidden, d.dense_intermediate],
                    DType::F32,
                ),
                up_bias: None,
                down_weight: cx.named_tensor(
                    mlp.child("down_proj").leaf("weight"),
                    vec![d.dense_intermediate, d.hidden],
                    DType::F32,
                ),
                down_bias: None,
                moe: Gemma4MoeFfn::new(&ns, d, cx),
                sliding,
                head_dim,
                kv_heads,
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        d: &Gemma4Dims,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        q_pos: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let normed = luminal_nn::rms_norm(
            x,
            self.input_norm_weight,
            self.input_norm_bias,
            self.input_norm_epsilon,
        );
        let q_raw = luminal_nn::linear(normed, self.wq_weight, self.wq_bias);
        let k_raw = luminal_nn::linear(normed, self.wk_weight, self.wk_bias);
        let v_raw = match &self.wv {
            Some(wv) => luminal_nn::linear(normed, wv.0, wv.1),
            None => k_raw, // full layers: V IS the K projection output
        };
        let q = luminal_nn::rotary_apply(
            luminal_nn::rms_norm_heads(q_raw, self.head_dim, self.q_norm, d.rms_eps),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = luminal_nn::rotary_apply(
            luminal_nn::rms_norm_heads(k_raw, self.head_dim, self.k_norm, d.rms_eps),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let v = luminal_nn::rms_norm_heads_unweighted(v_raw, self.head_dim, d.rms_eps);
        let context_positions = q.graph().arange(gather_idx.dims1());
        let score_bias = if self.sliding {
            sliding_window_bias(q_pos, context_positions, d.window)
        } else {
            causal_bias(q_pos, context_positions)
        };
        let result = paged_attention(
            q,
            k,
            v,
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            score_bias,
            AttentionGeometry::with_scale(d.n_heads, self.kv_heads, self.head_dim, 1.0),
        );
        let (attn, k_cache, v_cache) = (result.output, result.cache.keys, result.cache.values);
        let x = x + luminal_nn::rms_norm(
            luminal_nn::linear(attn, self.wo_weight, self.wo_bias),
            self.post_attn_norm_weight,
            self.post_attn_norm_bias,
            self.post_attn_norm_epsilon,
        );

        let dense = luminal_nn::linear(
            gemma_gelu(luminal_nn::linear(
                luminal_nn::rms_norm(
                    x,
                    self.pre_ff_norm_weight,
                    self.pre_ff_norm_bias,
                    self.pre_ff_norm_epsilon,
                ),
                self.gate_weight,
                self.gate_bias,
            )) * luminal_nn::linear(
                luminal_nn::rms_norm(
                    x,
                    self.pre_ff_norm_weight,
                    self.pre_ff_norm_bias,
                    self.pre_ff_norm_epsilon,
                ),
                self.up_weight,
                self.up_bias,
            ),
            self.down_weight,
            self.down_bias,
        );
        let dense = luminal_nn::rms_norm(
            dense,
            self.post_ff_norm_1_weight,
            self.post_ff_norm_1_bias,
            self.post_ff_norm_1_epsilon,
        );
        let moe = self.moe.forward(
            x,
            luminal_nn::rms_norm(
                x,
                self.pre_ff_norm_2_weight,
                self.pre_ff_norm_2_bias,
                self.pre_ff_norm_2_epsilon,
            ),
        );
        let moe = luminal_nn::rms_norm(
            moe,
            self.post_ff_norm_2_weight,
            self.post_ff_norm_2_bias,
            self.post_ff_norm_2_epsilon,
        );
        let ff_out = luminal_nn::rms_norm(
            dense + moe,
            self.post_ff_norm_weight,
            self.post_ff_norm_bias,
            self.post_ff_norm_epsilon,
        );
        let x = x + ff_out;
        let scalar = self
            .layer_scalar
            .expand_lhs(&x.dims()[..1])
            .expand(x.dims());
        (x * scalar, k_cache, v_cache)
    }
}

pub struct Gemma4Moe {
    pub dims: Gemma4Dims,
    pub embed: GraphTensor,
    pub blocks: Vec<Gemma4Block>,
    pub final_norm_weight: Option<GraphTensor>,
    pub final_norm_bias: Option<GraphTensor>,
    pub final_norm_epsilon: f32,
}

impl Gemma4Moe {
    pub fn init(cx: &mut Graph, dims: &Gemma4Dims) -> Self {
        let text = Namespace::root().child("model").child("language_model");
        let blocks = (0..dims.layers)
            .map(|l| Gemma4Block::new(l, dims, cx))
            .collect();
        Self {
            dims: dims.clone(),
            embed: cx.named_tensor(
                text.child("embed_tokens").leaf("weight"),
                vec![dims.vocab, dims.hidden],
                DType::F32,
            ),
            blocks,
            final_norm_weight: Some(cx.named_tensor(
                text.child("norm").leaf("weight"),
                vec![dims.hidden],
                DType::F32,
            )),
            final_norm_bias: None,
            final_norm_epsilon: dims.rms_eps,
        }
    }

    /// Rope pairs per ROLE — (cos, sin, rot) for sliding (head_dim 256,
    /// theta 10k, full rotary) and full (head_dim 512, theta 1M,
    /// partial 0.25).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        tokens: GraphTensor,
        q_pos: GraphTensor,
        rope_sliding: (GraphTensor, GraphTensor, GraphTensor),
        rope_full: (GraphTensor, GraphTensor, GraphTensor),
        pool: &KvCachePool,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        assert_eq!(tokens.dtype, DType::Int);
        let mut x = luminal_nn::embedding(tokens, self.embed) * (self.dims.hidden as f32).sqrt();
        let mut caches_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let (cos, sin, rot) = if block.sliding {
                rope_sliding
            } else {
                rope_full
            };
            let (next, k_cache, v_cache) = block.forward(
                &self.dims,
                x,
                pool.layers[layer].0,
                pool.layers[layer].1,
                gather_idx,
                scatter_idx,
                q_pos,
                cos,
                sin,
                rot,
            );
            x = next;
            caches_out.push((k_cache, v_cache));
        }
        let logits = luminal_nn::embedding_projection(
            luminal_nn::rms_norm(
                x,
                self.final_norm_weight,
                self.final_norm_bias,
                self.final_norm_epsilon,
            ),
            self.embed,
        );
        let softcap = self.dims.logit_softcap;
        let logits = (logits * (1.0 / softcap)).tanh() * softcap;
        (logits, caches_out)
    }
}
