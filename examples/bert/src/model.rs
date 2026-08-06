use luminal::{dtype::DType, graph::Graph, prelude::GraphTensor};
use luminal_nn::LayerNorm;

// BERT-base hyperparams
pub const LAYERS: usize = 12;
pub const HIDDEN: usize = 768;
pub const INTERMEDIATE: usize = 3072;
pub const N_HEADS: usize = 12;
#[allow(dead_code)]
pub const HEAD_DIM: usize = HIDDEN / N_HEADS; // 64
#[allow(dead_code)]
pub const VOCAB_SIZE: usize = 30522;
pub const MAX_SEQ: usize = 512;
pub const TYPE_VOCAB_SIZE: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BertPrecision {
    F32,
    Bf16,
}

impl BertPrecision {
    fn weight_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Bf16 => DType::Bf16,
        }
    }

    fn act_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Bf16 => DType::Bf16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BertConfig {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub n_heads: usize,
    pub vocab_size: usize,
    pub max_seq: usize,
    pub type_vocab_size: usize,
}

impl Default for BertConfig {
    fn default() -> Self {
        Self {
            layers: LAYERS,
            hidden: HIDDEN,
            intermediate: INTERMEDIATE,
            n_heads: N_HEADS,
            vocab_size: VOCAB_SIZE,
            max_seq: MAX_SEQ,
            type_vocab_size: TYPE_VOCAB_SIZE,
        }
    }
}

fn persist(
    cx: &mut Graph,
    name: impl ToString,
    shape: impl luminal::prelude::ToShape,
) -> GraphTensor {
    cx.named_tensor(name, shape).persist()
}

fn linear_weight(
    cx: &mut Graph,
    prefix: impl ToString,
    shape: impl luminal::prelude::ToShape,
    precision: BertPrecision,
) -> GraphTensor {
    cx.named_tensor(format!("{}.weight", prefix.to_string()), shape)
        .as_dtype(precision.weight_dtype())
        .persist()
}

fn linear_bias(cx: &mut Graph, prefix: impl ToString, out: usize) -> GraphTensor {
    cx.named_tensor(format!("{}.bias", prefix.to_string()), out)
        .persist()
}

fn layer_linear_weight(
    cx: &mut Graph,
    layer: usize,
    suffix: &str,
    shape: impl luminal::prelude::ToShape,
    precision: BertPrecision,
) -> GraphTensor {
    linear_weight(
        cx,
        format!("bert.encoder.layer.{layer}.{suffix}"),
        shape,
        precision,
    )
}

fn layer_linear_bias(cx: &mut Graph, layer: usize, suffix: &str, out: usize) -> GraphTensor {
    linear_bias(cx, format!("bert.encoder.layer.{layer}.{suffix}"), out)
}

fn norm_in_f32(norm: &LayerNorm, x: GraphTensor, act: DType) -> GraphTensor {
    let x = if act == DType::F32 {
        x
    } else {
        x.cast(DType::F32)
    };
    // Manual LayerNorm: mean_norm + std_norm + weight + bias.
    // mean_norm is decomposed as x - x.mean() rather than calling
    // LayerNorm::forward with mean_norm=true, because the CUDA backend's
    // KernelRMSNorm egglog rule only matches the RMS norm pattern (no mean
    // subtraction). Writing the mean subtraction explicitly lets the std_norm
    // chain still fuse into a single kernel.
    let axis = x.shape.last_axis();
    let mean = x.mean(axis).expand_to_shape_on_axes(x.shape, axis);
    let h = x - mean;
    let h = h.std_norm(axis, 1e-12);
    let h = if let Some(w) = norm.weight {
        h * w.expand_lhs(&h.dims()[..h.dims().len() - 1])
    } else {
        h
    };
    let h = if let Some(b) = norm.bias {
        h + b.expand_lhs(&h.dims()[..h.dims().len() - 1])
    } else {
        h
    };
    if act == DType::F32 {
        h
    } else {
        h.cast(act)
    }
}

fn linear(input: GraphTensor, weight: GraphTensor, bias: Option<GraphTensor>) -> GraphTensor {
    let out = input.matmul(weight.t());
    if let Some(b) = bias {
        out + b.expand_lhs(&out.dims()[..out.dims().len() - 1])
    } else {
        out
    }
}

fn token_embedding(embedding: GraphTensor, token_ids: GraphTensor, hidden: usize) -> GraphTensor {
    let seq = token_ids.dims1();
    embedding.gather(
        (token_ids * hidden).expand_dim(1, hidden)
            + token_ids.graph().arange(hidden).expand_dim(0, seq),
    )
}

pub struct BertEmbeddings {
    token_embed: GraphTensor,
    type_embed: GraphTensor,
    position_embed: GraphTensor,
    norm: LayerNorm,
}

impl BertEmbeddings {
    pub fn init(cx: &mut Graph, config: &BertConfig, precision: BertPrecision) -> Self {
        let table_dtype = precision.act_dtype();
        Self {
            token_embed: persist(
                cx,
                "bert.embeddings.word_embeddings.weight",
                (config.vocab_size, config.hidden),
            )
            .as_dtype(table_dtype),
            type_embed: persist(
                cx,
                "bert.embeddings.token_type_embeddings.weight",
                (config.type_vocab_size, config.hidden),
            )
            .as_dtype(table_dtype),
            position_embed: persist(
                cx,
                "bert.embeddings.position_embeddings.weight",
                (config.max_seq, config.hidden),
            )
            .as_dtype(table_dtype),
            norm: LayerNorm::new(
                config.hidden,
                Some("bert.embeddings.LayerNorm.weight"),
                Some("bert.embeddings.LayerNorm.bias"),
                true,
                1e-12,
                cx,
            ),
        }
    }

    pub fn forward(
        &self,
        input_ids: GraphTensor,
        token_type_ids: GraphTensor,
        pos_ids: GraphTensor,
        act: DType,
        hidden: usize,
    ) -> GraphTensor {
        let tok = token_embedding(self.token_embed, input_ids, hidden);
        let typ = token_embedding(self.type_embed, token_type_ids, hidden);
        let pos = token_embedding(self.position_embed, pos_ids, hidden);
        norm_in_f32(&self.norm, tok + typ + pos, act)
    }
}

pub struct BertSelfAttention {
    q_weight: GraphTensor,
    q_bias: GraphTensor,
    k_weight: GraphTensor,
    k_bias: GraphTensor,
    v_weight: GraphTensor,
    v_bias: GraphTensor,
    o_weight: GraphTensor,
    o_bias: GraphTensor,
    head_dim: usize,
    n_heads: usize,
}

impl BertSelfAttention {
    pub fn init(
        cx: &mut Graph,
        layer: usize,
        config: &BertConfig,
        precision: BertPrecision,
    ) -> Self {
        let hidden = config.hidden;
        Self {
            q_weight: layer_linear_weight(
                cx,
                layer,
                "attention.self.query",
                (hidden, hidden),
                precision,
            ),
            q_bias: layer_linear_bias(cx, layer, "attention.self.query", hidden),
            k_weight: layer_linear_weight(
                cx,
                layer,
                "attention.self.key",
                (hidden, hidden),
                precision,
            ),
            k_bias: layer_linear_bias(cx, layer, "attention.self.key", hidden),
            v_weight: layer_linear_weight(
                cx,
                layer,
                "attention.self.value",
                (hidden, hidden),
                precision,
            ),
            v_bias: layer_linear_bias(cx, layer, "attention.self.value", hidden),
            o_weight: layer_linear_weight(
                cx,
                layer,
                "attention.output.dense",
                (hidden, hidden),
                precision,
            ),
            o_bias: layer_linear_bias(cx, layer, "attention.output.dense", hidden),
            head_dim: config.hidden / config.n_heads,
            n_heads: config.n_heads,
        }
    }

    pub fn forward(&self, x: GraphTensor, mask: GraphTensor, _act: DType) -> GraphTensor {
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let q = linear(x, self.q_weight, Some(self.q_bias));
        let k = linear(x, self.k_weight, Some(self.k_bias));
        let v = linear(x, self.v_weight, Some(self.v_bias));

        let q = q.split_dims(1, self.head_dim).transpose(0, 1);
        let k = k.split_dims(1, self.head_dim).transpose(0, 1);
        let v = v.split_dims(1, self.head_dim).transpose(0, 1);

        let scores = q.matmul(k.permute((0, 2, 1))) * scale;
        let masked = scores + mask.expand_dim(0, self.n_heads);
        let weights = masked.softmax(2);

        let out = weights.matmul(v).transpose(0, 1).merge_dims(1, 2);
        linear(out, self.o_weight, Some(self.o_bias))
    }
}

pub struct BertLayer {
    attention: BertSelfAttention,
    attn_norm: LayerNorm,
    intermediate_weight: GraphTensor,
    intermediate_bias: GraphTensor,
    output_weight: GraphTensor,
    output_bias: GraphTensor,
    output_norm: LayerNorm,
}

impl BertLayer {
    pub fn init(
        cx: &mut Graph,
        layer: usize,
        config: &BertConfig,
        precision: BertPrecision,
    ) -> Self {
        let hidden = config.hidden;
        let intermediate = config.intermediate;
        Self {
            attention: BertSelfAttention::init(cx, layer, config, precision),
            attn_norm: {
                let w = format!("bert.encoder.layer.{layer}.attention.output.LayerNorm.weight");
                let b = format!("bert.encoder.layer.{layer}.attention.output.LayerNorm.bias");
                LayerNorm::new(hidden, Some(w.as_str()), Some(b.as_str()), true, 1e-12, cx)
            },
            intermediate_weight: layer_linear_weight(
                cx,
                layer,
                "intermediate.dense",
                (intermediate, hidden),
                precision,
            ),
            intermediate_bias: layer_linear_bias(cx, layer, "intermediate.dense", intermediate),
            output_weight: layer_linear_weight(
                cx,
                layer,
                "output.dense",
                (hidden, intermediate),
                precision,
            ),
            output_bias: layer_linear_bias(cx, layer, "output.dense", hidden),
            output_norm: {
                let w = format!("bert.encoder.layer.{layer}.output.LayerNorm.weight");
                let b = format!("bert.encoder.layer.{layer}.output.LayerNorm.bias");
                LayerNorm::new(hidden, Some(w.as_str()), Some(b.as_str()), true, 1e-12, cx)
            },
        }
    }

    pub fn forward(&self, x: GraphTensor, mask: GraphTensor, act: DType) -> GraphTensor {
        let attn = self.attention.forward(x, mask, act);
        let h = norm_in_f32(&self.attn_norm, x + attn, act);
        let inter = linear(h, self.intermediate_weight, Some(self.intermediate_bias)).gelu();
        let out = linear(inter, self.output_weight, Some(self.output_bias));
        norm_in_f32(&self.output_norm, h + out, act)
    }
}

pub struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn init(cx: &mut Graph, config: &BertConfig, precision: BertPrecision) -> Self {
        Self {
            layers: (0..config.layers)
                .map(|l| BertLayer::init(cx, l, config, precision))
                .collect(),
        }
    }

    pub fn forward(&self, x: GraphTensor, mask: GraphTensor, act: DType) -> GraphTensor {
        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(h, mask, act);
        }
        h
    }
}

pub struct BertLMPredictionHead {
    dense_weight: GraphTensor,
    dense_bias: GraphTensor,
    norm: LayerNorm,
    decoder_bias: GraphTensor,
}

impl BertLMPredictionHead {
    pub fn init(cx: &mut Graph, config: &BertConfig, precision: BertPrecision) -> Self {
        let hidden = config.hidden;
        Self {
            dense_weight: linear_weight(
                cx,
                "cls.predictions.transform.dense",
                (hidden, hidden),
                precision,
            ),
            dense_bias: linear_bias(cx, "cls.predictions.transform.dense", hidden),
            norm: LayerNorm::new(
                hidden,
                Some("cls.predictions.transform.LayerNorm.weight"),
                Some("cls.predictions.transform.LayerNorm.bias"),
                true,
                1e-12,
                cx,
            ),
            decoder_bias: persist(cx, "cls.predictions.bias", config.vocab_size),
        }
    }

    pub fn forward(
        &self,
        x: GraphTensor,
        embedding_weight: GraphTensor,
        act: DType,
    ) -> GraphTensor {
        let h = linear(x, self.dense_weight, Some(self.dense_bias)).gelu();
        let h = norm_in_f32(&self.norm, h, act);
        // Weight tying: decoder weight = embedding weight
        let logits = h.matmul(embedding_weight.t());
        logits
            + self
                .decoder_bias
                .expand_lhs(&logits.dims()[..logits.dims().len() - 1])
    }
}

pub struct BertForMaskedLM {
    config: BertConfig,
    precision: BertPrecision,
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    head: BertLMPredictionHead,
}

impl BertForMaskedLM {
    pub fn init_f32(cx: &mut Graph) -> Self {
        Self::init_with_precision(cx, BertConfig::default(), BertPrecision::F32)
    }

    pub fn init_bf16(cx: &mut Graph) -> Self {
        Self::init_with_precision(cx, BertConfig::default(), BertPrecision::Bf16)
    }

    pub fn init_with_precision(
        cx: &mut Graph,
        config: BertConfig,
        precision: BertPrecision,
    ) -> Self {
        Self {
            config,
            precision,
            embeddings: BertEmbeddings::init(cx, &config, precision),
            encoder: BertEncoder::init(cx, &config, precision),
            head: BertLMPredictionHead::init(cx, &config, precision),
        }
    }

    pub fn forward(
        &self,
        input_ids: GraphTensor,
        token_type_ids: GraphTensor,
        pos_ids: GraphTensor,
        mask: GraphTensor,
    ) -> GraphTensor {
        let act = self.precision.act_dtype();
        let emb =
            self.embeddings
                .forward(input_ids, token_type_ids, pos_ids, act, self.config.hidden);
        let encoded = self.encoder.forward(emb, mask, act);
        self.head.forward(encoded, self.embeddings.token_embed, act)
    }

    // pub fn parameter_tensors(&self) -> Vec<GraphTensor> {
    //     let mut tensors = vec![
    //         self.embeddings.token_embed,
    //         self.embeddings.type_embed,
    //         self.embeddings.position_embed,
    //     ];
    //     if let Some(w) = self.embeddings.norm.weight {
    //         tensors.push(w);
    //     }
    //     if let Some(b) = self.embeddings.norm.bias {
    //         tensors.push(b);
    //     }
    //     for layer in &self.encoder.layers {
    //         tensors.push(layer.attention.q_weight);
    //         tensors.push(layer.attention.q_bias);
    //         tensors.push(layer.attention.k_weight);
    //         tensors.push(layer.attention.k_bias);
    //         tensors.push(layer.attention.v_weight);
    //         tensors.push(layer.attention.v_bias);
    //         tensors.push(layer.attention.o_weight);
    //         tensors.push(layer.attention.o_bias);
    //         if let Some(w) = layer.attn_norm.weight {
    //             tensors.push(w);
    //         }
    //         if let Some(b) = layer.attn_norm.bias {
    //             tensors.push(b);
    //         }
    //         tensors.push(layer.intermediate_weight);
    //         tensors.push(layer.intermediate_bias);
    //         tensors.push(layer.output_weight);
    //         tensors.push(layer.output_bias);
    //         if let Some(w) = layer.output_norm.weight {
    //             tensors.push(w);
    //         }
    //         if let Some(b) = layer.output_norm.bias {
    //             tensors.push(b);
    //         }
    //     }
    //     tensors.push(self.head.dense_weight);
    //     tensors.push(self.head.dense_bias);
    //     if let Some(w) = self.head.norm.weight {
    //         tensors.push(w);
    //     }
    //     if let Some(b) = self.head.norm.bias {
    //         tensors.push(b);
    //     }
    //     tensors.push(self.head.decoder_bias);
    //     tensors
    // }
}

// #[cfg(test)]
// mod tests {
//     use luminal::prelude::*;

//     use super::*;

//     fn run_forward(seq_len: usize) {
//         let mut cx = Graph::default();
//         let input_ids = cx.named_tensor("input_ids", 's').as_dtype(DType::Int);
//         let token_type_ids = cx.named_tensor("token_type_ids", 's').as_dtype(DType::Int);
//         let pos_ids = cx.named_tensor("pos_ids", 's').as_dtype(DType::Int);
//         let mask = cx.named_tensor("mask", ('s', 's'));

//         let bert = BertForMaskedLM::init_f32(&mut cx);
//         let logits = bert
//             .forward(input_ids, token_type_ids, pos_ids, mask)
//             .output();

//         cx.set_dim('s', seq_len);
//         let mut rt: ReferenceRuntime = cx.compile(
//             ReferenceRuntime::default(),
//             CompileOptions::default().search_graph_limit(1),
//         );

//         // Set input data
//         rt.set_data(input_ids, vec![0i32; seq_len]);
//         rt.set_data(token_type_ids, vec![0i32; seq_len]);
//         rt.set_data(pos_ids, (0..seq_len as i32).collect::<Vec<_>>());
//         rt.set_data(mask, vec![0f32; seq_len * seq_len]);

//         // Set zero data for all weight tensors
//         for t in bert.parameter_tensors() {
//             let n: usize = t.shape.dims.iter().map(|e| e.to_usize().unwrap()).product();
//             match t.dtype {
//                 DType::F32 => rt.set_data(t, vec![0f32; n]),
//                 DType::Bf16 => rt.set_data(t, vec![half::bf16::ZERO; n]),
//                 _ => {}
//             }
//         }

//         rt.execute(&cx.dyn_map);
//         let out = rt.get_f32(logits);
//         assert_eq!(out.len(), seq_len * VOCAB_SIZE);
//     }

//     // #[test]
//     // fn forward_seq_1() {
//     //     run_forward(1);
//     // }

//     #[test]
//     fn forward_seq_4() {
//         run_forward(4);
//     }

//     // #[test]
//     // fn forward_seq_8() {
//     //     run_forward(8);
//     // }
// }
