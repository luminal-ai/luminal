#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!("Enable only one backend feature for the gemma example");

#[cfg(not(any(feature = "cuda", feature = "metal")))]
compile_error!("Enable one backend feature for the gemma example");

use crate::model::{KVCache, HEAD_DIM, LAYERS, N_KV_HEADS};
use luminal::prelude::*;
use std::path::Path;

#[cfg(feature = "metal")]
use half::{bf16, f16};
#[cfg(feature = "metal")]
use luminal::hlir::Input;
#[cfg(feature = "metal")]
use memmap2::MmapOptions;
#[cfg(feature = "metal")]
use safetensors::SafeTensors;
#[cfg(feature = "metal")]
use std::fs::File;

#[cfg(feature = "cuda")]
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
#[cfg(feature = "metal")]
use luminal_metal::runtime::MetalRuntime;

#[cfg(feature = "cuda")]
pub type ExampleRuntime = CudaRuntime;
#[cfg(feature = "metal")]
pub type ExampleRuntime = MetalRuntime;

pub fn build_search_space(cx: &mut Graph) {
    cx.build_search_space::<ExampleRuntime>();
}

#[cfg(feature = "cuda")]
pub fn init_runtime() -> ExampleRuntime {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    CudaRuntime::initialize(stream)
}

#[cfg(feature = "metal")]
pub fn init_runtime() -> ExampleRuntime {
    MetalRuntime::initialize(())
}

#[cfg(feature = "cuda")]
pub fn load_weights(runtime: &mut ExampleRuntime, cx: &Graph, weights_path: &Path) {
    runtime.load_safetensors(cx, weights_path.to_str().unwrap());
}

#[cfg(feature = "metal")]
pub fn load_weights(runtime: &mut ExampleRuntime, cx: &Graph, weights_path: &Path) {
    let file = File::open(weights_path).unwrap();
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let st = SafeTensors::deserialize(&mmap).unwrap();

    for node in cx.graph.node_indices() {
        if let Some(Input { label, .. }) = (*cx.graph[node]).as_any().downcast_ref::<Input>() {
            if let Ok(tensor) = st.tensor(label) {
                match tensor.dtype() {
                    safetensors::Dtype::F32 => {
                        let data: &[f32] = bytemuck::cast_slice(tensor.data());
                        runtime.set_data(node, data.to_vec());
                    }
                    safetensors::Dtype::F16 => {
                        let data: &[f16] = bytemuck::cast_slice(tensor.data());
                        runtime
                            .set_data(node, data.iter().map(|v| v.to_f32()).collect::<Vec<f32>>());
                    }
                    safetensors::Dtype::BF16 => {
                        let data: &[bf16] = bytemuck::cast_slice(tensor.data());
                        runtime
                            .set_data(node, data.iter().map(|v| v.to_f32()).collect::<Vec<f32>>());
                    }
                    dtype => panic!("Unsupported dtype for Metal gemma example: {dtype:?}"),
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub fn init_kv_cache(runtime: &mut ExampleRuntime, kv_cache: &KVCache, max_seq_len: usize) {
    let cache_bytes = N_KV_HEADS * max_seq_len * HEAD_DIM * std::mem::size_of::<f32>();
    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }
}

#[cfg(feature = "metal")]
pub fn init_kv_cache(runtime: &mut ExampleRuntime, kv_cache: &KVCache, max_seq_len: usize) {
    let cache_elems = N_KV_HEADS * max_seq_len * HEAD_DIM;
    let zeros = vec![0.0f32; cache_elems];
    for i in 0..LAYERS {
        runtime.set_data(kv_cache.k_caches[i], zeros.clone());
        runtime.set_data(kv_cache.v_caches[i], zeros.clone());
    }
}

pub fn roundtrip_kv_cache(
    runtime: &mut ExampleRuntime,
    kv_cache: &KVCache,
    cache_outputs: &[(GraphTensor, GraphTensor)],
) {
    for (layer_idx, (k_out, v_out)) in cache_outputs.iter().enumerate() {
        let k_buf = runtime.remove_buffer(*k_out);
        let v_buf = runtime.remove_buffer(*v_out);
        runtime.set_buffer(kv_cache.k_caches[layer_idx], k_buf);
        runtime.set_buffer(kv_cache.v_caches[layer_idx], v_buf);
    }
}
