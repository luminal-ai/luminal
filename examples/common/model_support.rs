#![allow(dead_code)]
#![allow(unused_imports)]

use luminal::prelude::*;
pub use luminal_nn::*;

/// Example-side helper for spelling checkpoint paths. Naming intentionally
/// lives outside `luminal_nn` because checkpoint layout is model definition.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Namespace {
    path: String,
}

impl Namespace {
    pub fn root() -> Self {
        Self::default()
    }

    pub fn child(&self, segment: impl AsRef<str>) -> Self {
        let segment = segment.as_ref();
        assert!(!segment.is_empty(), "namespace segment must not be empty");
        assert!(
            !segment.contains('.'),
            "namespace segments must not contain dots: {segment:?}"
        );
        let path = if self.path.is_empty() {
            segment.to_owned()
        } else {
            format!("{}.{}", self.path, segment)
        };
        Self { path }
    }

    pub fn index(&self, index: usize) -> Self {
        self.child(index.to_string())
    }

    pub fn leaf(&self, name: impl AsRef<str>) -> String {
        let name = name.as_ref();
        assert!(!name.is_empty(), "tensor name must not be empty");
        assert!(
            !name.contains('.'),
            "tensor leaf names must not contain dots: {name:?}"
        );
        if self.path.is_empty() {
            name.to_owned()
        } else {
            format!("{}.{}", self.path, name)
        }
    }
}

/// Model-side parameter bundle used by the examples.
pub struct Linear {
    pub weight: GraphTensor,
    pub bias: Option<GraphTensor>,
}

impl Linear {
    pub fn new(
        inp: usize,
        out: usize,
        bias: bool,
        dtype: DType,
        ns: &Namespace,
        cx: &mut Graph,
    ) -> Self {
        Self {
            weight: cx.named_tensor(ns.leaf("weight"), (inp, out), dtype),
            bias: bias.then(|| cx.named_tensor(ns.leaf("bias"), out, dtype)),
        }
    }

    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        luminal_nn::linear(input, self.weight, self.bias)
    }
}

/// Model-side embedding parameter bundle used by the examples.
pub struct Embedding {
    pub weight: GraphTensor,
}

impl Embedding {
    pub fn new(
        n_embeddings: usize,
        embedding_dim: usize,
        dtype: DType,
        ns: &Namespace,
        cx: &mut Graph,
    ) -> Self {
        Self {
            weight: cx.named_tensor(ns.leaf("weight"), (n_embeddings, embedding_dim), dtype),
        }
    }

    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        luminal_nn::embedding(input, self.weight)
    }

    pub fn reverse(&self, input: GraphTensor) -> GraphTensor {
        luminal_nn::embedding_projection(input, self.weight)
    }
}

/// Model-side normalization parameter bundle used by the examples.
pub struct LayerNorm {
    pub weight: Option<GraphTensor>,
    pub bias: Option<GraphTensor>,
    mean_norm: bool,
    epsilon: f32,
    unit_offset: bool,
}

impl LayerNorm {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        weight: bool,
        bias: bool,
        mean_norm: bool,
        epsilon: f32,
        dtype: DType,
        ns: &Namespace,
        cx: &mut Graph,
    ) -> Self {
        Self {
            weight: weight.then(|| cx.named_tensor(ns.leaf("weight"), dim, dtype)),
            bias: bias.then(|| cx.named_tensor(ns.leaf("bias"), dim, dtype)),
            mean_norm,
            epsilon,
            unit_offset: false,
        }
    }

    pub fn with_unit_offset(mut self) -> Self {
        self.unit_offset = true;
        self
    }

    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        let weight = self.weight.map(|weight| {
            if self.unit_offset {
                weight + 1.0
            } else {
                weight
            }
        });
        if self.mean_norm {
            luminal_nn::layer_norm(input, weight, self.bias, self.epsilon)
        } else {
            luminal_nn::rms_norm(input, weight, self.bias, self.epsilon)
        }
    }
}

/// Model-side convolution parameter bundle used by examples and training tests.
pub struct ConvND {
    pub weight: GraphTensor,
    pub bias: Option<GraphTensor>,
    config: luminal_nn::ConvNdConfig,
    ch_in: usize,
    ch_out: usize,
}

impl ConvND {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ch_in: usize,
        ch_out: usize,
        kernel: impl AsRef<[usize]>,
        stride: impl AsRef<[usize]>,
        dilation: impl AsRef<[usize]>,
        padding: impl AsRef<[usize]>,
        bias: bool,
        dtype: DType,
        ns: &Namespace,
        cx: &mut Graph,
    ) -> Self {
        let kernel = kernel.as_ref().to_vec();
        let kernel_product = kernel.iter().product::<usize>();
        Self {
            weight: cx.named_tensor(ns.leaf("weight"), (ch_out, ch_in * kernel_product), dtype),
            bias: bias.then(|| cx.named_tensor(ns.leaf("bias"), ch_out, dtype)),
            config: luminal_nn::ConvNdConfig::new(kernel, stride, dilation, padding),
            ch_in,
            ch_out,
        }
    }

    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        luminal_nn::conv_nd(input, self.weight, self.bias, &self.config)
    }

    pub fn infer_output_shape(&self, input: &[usize]) -> Vec<usize> {
        self.config
            .infer_output_shape(input, self.ch_in, self.ch_out)
    }
}

/// Model-side parameter bundle for the fixed per-tensor FP8 formulation.
pub struct Fp8Linear {
    pub weight: GraphTensor,
    pub input_scale: GraphTensor,
    pub weight_scale: GraphTensor,
    out: usize,
}

impl Fp8Linear {
    pub fn new(inp: usize, out: usize, ns: &Namespace, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor(ns.leaf("weight"), (out, inp), DType::F8E4M3),
            input_scale: cx.named_tensor(ns.leaf("input_scale"), (), DType::F32),
            weight_scale: cx.named_tensor(ns.leaf("weight_scale"), (), DType::F32),
            out,
        }
    }

    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        luminal_nn::fp8_linear(input, self.weight, self.input_scale, self.weight_scale)
    }

    pub fn out_features(&self) -> usize {
        self.out
    }
}

pub fn named_kv_cache_pool(
    cx: &mut Graph,
    layers: usize,
    slots: usize,
    kv_dim: usize,
    dtype: DType,
    ns: &Namespace,
) -> luminal_nn::KvCachePool {
    named_heterogeneous_kv_cache_pool(cx, slots, &vec![kv_dim; layers], dtype, ns)
}

pub fn named_heterogeneous_kv_cache_pool(
    cx: &mut Graph,
    slots: usize,
    kv_dims: &[usize],
    dtype: DType,
    ns: &Namespace,
) -> luminal_nn::KvCachePool {
    let layers = kv_dims.iter().enumerate().map(|(layer, kv_dim)| {
        let layer_ns = ns.index(layer);
        luminal_nn::KvCache::new(
            cx.named_tensor(layer_ns.leaf("k_cache"), (slots, *kv_dim), dtype),
            cx.named_tensor(layer_ns.leaf("v_cache"), (slots, *kv_dim), dtype),
        )
    });
    luminal_nn::KvCachePool::from_layers(layers)
}
