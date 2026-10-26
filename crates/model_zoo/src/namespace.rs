use luminal::prelude::*;

/// Model-side helper for spelling checkpoint paths. Naming intentionally
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
            cx.named_tensor(layer_ns.leaf("k_cache"), vec![slots, *kv_dim], dtype),
            cx.named_tensor(layer_ns.leaf("v_cache"), vec![slots, *kv_dim], dtype),
        )
    });
    luminal_nn::KvCachePool::from_layers(layers)
}
