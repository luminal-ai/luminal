//! Functional graph-side key/value-cache primitives over caller-supplied tensors.

use luminal::prelude::{DType, GraphTensor};

use crate::{gather_rows, scatter_rows};

/// One layer's graph-side key/value cache value.
///
/// The first axis is a flat physical slot space. Allocation and page-table
/// policy are deliberately left to the caller.
#[derive(Clone, Copy)]
pub struct KvCache {
    pub keys: GraphTensor,
    pub values: GraphTensor,
}

impl KvCache {
    pub fn new(keys: GraphTensor, values: GraphTensor) -> Self {
        assert_eq!(keys.rank(), 2, "key cache must have shape [slots, width]");
        assert_eq!(
            values.rank(),
            2,
            "value cache must have shape [slots, width]"
        );
        assert_eq!(keys.dims(), values.dims(), "key/value cache shapes differ");
        assert_eq!(keys.dtype, values.dtype, "key/value cache dtypes differ");
        Self { keys, values }
    }

    /// Return a new cache value with the supplied rows overwritten.
    pub fn write(self, slots: GraphTensor, new_keys: GraphTensor, new_values: GraphTensor) -> Self {
        assert_eq!(slots.dtype, DType::Int);
        assert_eq!(slots.rank(), 1, "cache write slots must be rank one");
        assert_eq!(new_keys.dims(), new_values.dims(), "new K/V shapes differ");
        assert_eq!(new_keys.dtype, new_values.dtype, "new K/V dtypes differ");
        assert_eq!(new_keys.dtype, self.keys.dtype, "new/cache K dtypes differ");
        assert_eq!(
            new_values.dtype, self.values.dtype,
            "new/cache V dtypes differ"
        );
        Self {
            keys: scatter_rows(new_keys, slots, self.keys),
            values: scatter_rows(new_values, slots, self.values),
        }
    }

    /// Read cache rows in the caller's desired logical context order.
    pub fn read(self, slots: GraphTensor) -> KvContext {
        assert_eq!(slots.dtype, DType::Int);
        assert_eq!(slots.rank(), 1, "cache read slots must be rank one");
        KvContext {
            keys: gather_rows(self.keys, slots),
            values: gather_rows(self.values, slots),
        }
    }
}

/// Keys and values gathered into logical context order.
#[derive(Clone, Copy)]
pub struct KvContext {
    pub keys: GraphTensor,
    pub values: GraphTensor,
}

/// Physical cache slots used by one paged-attention operation.
#[derive(Clone, Copy)]
pub struct CacheAccess {
    pub write_slots: GraphTensor,
    pub read_slots: GraphTensor,
}

impl CacheAccess {
    pub fn new(write_slots: GraphTensor, read_slots: GraphTensor) -> Self {
        assert_eq!(write_slots.dtype, DType::Int);
        assert_eq!(read_slots.dtype, DType::Int);
        assert_eq!(write_slots.rank(), 1, "cache write slots must be rank one");
        assert_eq!(read_slots.rank(), 1, "cache read slots must be rank one");
        Self {
            write_slots,
            read_slots,
        }
    }
}

/// A validated collection of caller-supplied key/value cache tensors.
pub struct KvCachePool {
    pub layers: Vec<(GraphTensor, GraphTensor)>,
    pub kv_dims: Vec<usize>,
    pub slots: usize,
    pub dtype: DType,
}

impl KvCachePool {
    pub fn from_layers(layers: impl IntoIterator<Item = KvCache>) -> Self {
        let layers: Vec<_> = layers
            .into_iter()
            .map(|cache| (cache.keys, cache.values))
            .collect();
        assert!(
            !layers.is_empty(),
            "a cache pool requires at least one layer"
        );
        let slots = layers[0].0.dims()[0]
            .to_usize()
            .expect("cache slot count must be static");
        let dtype = layers[0].0.dtype;
        let kv_dims = layers
            .iter()
            .map(|(keys, values)| {
                let cache = KvCache::new(*keys, *values);
                assert_eq!(cache.keys.dtype, dtype, "cache layer dtype mismatch");
                assert_eq!(
                    cache.keys.dims()[0],
                    luminal::shape::IntExpr::from(slots),
                    "cache layer slot count mismatch"
                );
                cache.keys.dims()[1]
                    .to_usize()
                    .expect("cache width must be static")
            })
            .collect();
        Self {
            layers,
            kv_dims,
            slots,
            dtype,
        }
    }

    pub fn layer(&self, index: usize) -> KvCache {
        let (keys, values) = self.layers[index];
        KvCache::new(keys, values)
    }
}

#[cfg(test)]
mod tests {
    use super::{KvCache, KvCachePool};
    use luminal::prelude::{DType, Graph};

    #[test]
    fn cache_pool_wraps_caller_supplied_tensors() {
        let mut cx = Graph::new();
        let layers = (0..2).map(|_| {
            KvCache::new(
                cx.tensor((8, 4), DType::Bf16),
                cx.tensor((8, 4), DType::Bf16),
            )
        });
        let pool = KvCachePool::from_layers(layers);
        assert_eq!(pool.dtype, DType::Bf16);
        for (keys, values) in pool.layers {
            assert_eq!(keys.dtype, DType::Bf16);
            assert_eq!(values.dtype, DType::Bf16);
        }
    }
}
