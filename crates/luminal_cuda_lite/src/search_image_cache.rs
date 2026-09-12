//! Bucket-scoped compiled images survive GPU module eviction. The compilation
//! environment is held fixed for a search bucket, as with its source-keyed
//! CUDA module cache; this cache does not outlive that scope.
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
};

#[derive(Default)]
struct Images {
    bytes: usize,
    entries: HashMap<String, Vec<u8>>,
    order: VecDeque<String>,
}
impl Images {
    fn insert(&mut self, key: String, image: &[u8], limit: usize) {
        if image.len() > limit || self.entries.contains_key(&key) {
            return;
        }
        while self.bytes + image.len() > limit {
            let old = self.order.pop_front().unwrap();
            self.bytes -= self.entries.remove(&old).unwrap().len();
        }
        self.bytes += image.len();
        self.order.push_back(key.clone());
        self.entries.insert(key, image.to_vec());
    }
}
thread_local! {
    static IMAGES: RefCell<Option<Images>> = const { RefCell::new(None) };
}
/// Restores a surrounding scope even when candidate compilation panics.
pub(crate) struct SearchImageCache(Option<Images>);
impl SearchImageCache {
    pub(crate) fn enter() -> Self {
        Self(IMAGES.with(|cache| cache.replace(Some(Images::default()))))
    }
}
impl Drop for SearchImageCache {
    fn drop(&mut self) {
        IMAGES.with(|cache| cache.replace(self.0.take()));
    }
}
// Architecture and every compiler option are included. NVRTC is process-loaded
// and cannot change within a bucket. A bucket never reuses another bucket's
// images, and standalone/custom compilation outside search remains unchanged.
pub(crate) fn key(source_key: &str, options: &[String]) -> String {
    crate::artifact::module_key(&format!("{source_key}:{options:?}"))
}
pub(crate) fn lookup(key: &str) -> Option<Vec<u8>> {
    IMAGES.with(|cache| cache.borrow().as_ref()?.entries.get(key).cloned())
}
pub(crate) fn record(key: String, image: &[u8]) {
    IMAGES.with(|cache| {
        if let Some(cache) = cache.borrow_mut().as_mut() {
            cache.insert(key, image, 64 * 1024 * 1024);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_images_survive_handle_eviction_without_leaking_across_scopes() {
        assert!(lookup("a").is_none());
        {
            let _scope = SearchImageCache::enter();
            record("a".into(), &[1, 2, 3]);
            assert_eq!(lookup("a"), Some(vec![1, 2, 3]));
            let result = std::panic::catch_unwind(|| {
                let _nested = SearchImageCache::enter();
                assert!(lookup("a").is_none());
                record("b".into(), &[4]);
                panic!("failed candidate");
            });
            assert!(result.is_err());
            assert_eq!(lookup("a"), Some(vec![1, 2, 3]));
            assert!(lookup("b").is_none());
        }
        assert!(lookup("a").is_none());
        let mut cache = Images::default();
        cache.insert("a".into(), &[1, 2, 3], 5);
        cache.insert("b".into(), &[4, 5, 6], 5);
        cache.insert("large".into(), &[0; 6], 5);
        assert_eq!(cache.bytes, 3);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.entries["b"], [4, 5, 6]);
    }
    #[test]
    fn compiler_options_and_source_identity_separate_images() {
        let a = key("source-a", &["--gpu-architecture=sm_90".into()]);
        assert_ne!(a, key("source-b", &["--gpu-architecture=sm_90".into()]));
        assert_ne!(a, key("source-a", &["--gpu-architecture=sm_80".into()]));
        assert_ne!(
            a,
            key(
                "source-a",
                &["--gpu-architecture=sm_90".into(), "--use_fast_math".into()]
            )
        );
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    #[test]
    fn cached_images_recreate_released_modules_and_preserve_compile_guards() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut output = stream.alloc_zeros::<f32>(32).unwrap();
        let src = "extern \"C\" __global__ void cached_image(float* out) { out[threadIdx.x] = threadIdx.x + 17.0f; }";
        let _scope = SearchImageCache::enter();
        let mut times = Vec::new();
        for _ in 0..4 {
            let start = std::time::Instant::now();
            let image = crate::compile_module_image_for_current_device(&ctx, src).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
            let module = ctx.load_module(image).unwrap();
            let function = module.load_function("cached_image").unwrap();
            unsafe {
                stream
                    .launch_builder(&function)
                    .arg(&mut output)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            stream.synchronize().unwrap();
            // Both function and module owners are released before the next compile.
        }
        assert_eq!(
            stream.clone_dtoh(&output).unwrap(),
            (17..49).map(|x| x as f32).collect::<Vec<_>>()
        );
        assert_eq!(
            IMAGES.with(|cache| cache.borrow().as_ref().unwrap().entries.len()),
            1
        );
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::with_kernel_source_limit(Some(1), || {
                crate::compile_module_image_for_current_device(&ctx, src)
            })
            .unwrap();
        }));
        assert!(
            failure.is_err(),
            "a cached image cannot bypass the source budget"
        );
        let artifact = crate::artifact::module_artifact_session(&ctx, None).unwrap();
        crate::artifact::finish_module_artifact_capture(&artifact);
        crate::artifact::with_module_artifact_session(artifact, || {
            assert!(
                crate::compile_module_image_for_current_device(&ctx, src).is_err(),
                "strict artifact misses cannot fall back to cached search images"
            );
        });
        println!("source-to-image milliseconds after releasing module each time: {times:?}");
    }
}
