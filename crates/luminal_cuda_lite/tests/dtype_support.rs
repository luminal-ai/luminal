#![cfg(feature = "device")]
//! Device tests for the half/bfloat/double dtype paths. These exercise the
//! NVRTC include plumbing (`cuda_fp16.h`/`cuda_bf16.h` are not built into
//! NVRTC) and the `recip` half-overload workaround, which a graph or a
//! PyTorch model over these dtypes depends on.

use half::{bf16, f16};
use luminal::dtype::PlanDtype;
use luminal::prelude::*;
use luminal_cuda_lite::{CudaRuntime, HostBuffer, harness_search_options};
use rustc_hash::FxHashMap;

fn host(dtype: PlanDtype, bytes: Vec<u8>) -> HostBuffer {
    HostBuffer::new(dtype, bytes).expect("well-formed dtype payload")
}

/// f16 values as little-endian storage bytes.
fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| f16::from_f32(*v).to_bits().to_ne_bytes())
        .collect()
}

/// bf16 values as little-endian storage bytes.
fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_bits().to_ne_bytes())
        .collect()
}

/// `a * b` on `dtype`, returned as raw output storage bytes.
fn mul_device(dtype: DType, a: HostBuffer, b: HostBuffer) -> Vec<u8> {
    let mut cx = Graph::new();
    let ta = cx.tensor((4,), dtype);
    let tb = cx.tensor((4,), dtype);
    let out = ta * tb;
    let data: FxHashMap<_, _> = [(ta.id, a.clone()), (tb.id, b.clone())]
        .into_iter()
        .collect();
    let mut rt = CudaRuntime::load(&cx).expect("load");
    rt.search(&data, &harness_search_options())
        .expect("search dtype mul");
    rt.set_data(ta.id, a).unwrap();
    rt.set_data(tb.id, b).unwrap();
    rt.execute().expect("execute");
    rt.fetch(out.id).expect("fetch").0.bytes.clone()
}

#[test]
fn f16_multiply_round_trips() {
    let a = host(PlanDtype::F16, f16_bytes(&[1.5, 1.5, 1.5, 1.5]));
    let b = host(PlanDtype::F16, f16_bytes(&[2.0, 2.0, 2.0, 2.0]));
    let got = mul_device(DType::F16, a, b);
    assert_eq!(got, f16_bytes(&[3.0, 3.0, 3.0, 3.0]));
}

#[test]
fn bfloat16_multiply_round_trips() {
    let a = host(PlanDtype::Bf16, bf16_bytes(&[1.5, 1.5, 1.5, 1.5]));
    let b = host(PlanDtype::Bf16, bf16_bytes(&[2.0, 2.0, 2.0, 2.0]));
    let got = mul_device(DType::Bf16, a, b);
    assert_eq!(got, bf16_bytes(&[3.0, 3.0, 3.0, 3.0]));
}

#[test]
fn f64_reciprocal_round_trips() {
    let a = host(
        PlanDtype::F64,
        [4.0f64, 2.0, 1.0, 8.0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect(),
    );
    let mut cx = Graph::new();
    let t = cx.tensor((4,), DType::F64);
    let out = t.reciprocal();
    let data: FxHashMap<_, _> = [(t.id, a.clone())].into_iter().collect();
    let mut rt = CudaRuntime::load(&cx).expect("load");
    rt.search(&data, &harness_search_options())
        .expect("search f64 reciprocal");
    rt.set_data(t.id, a).unwrap();
    rt.execute().expect("execute");
    let bytes = rt.fetch(out.id).expect("fetch").0.bytes.clone();
    let got: Vec<f64> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_ne_bytes(*c))
        .collect();
    assert_eq!(got, vec![0.25, 0.5, 1.0, 0.125]);
}

#[test]
fn f16_reciprocal_round_trips() {
    // The `1.0f / __half` ambiguity regression: half recip must compile.
    let a = host(PlanDtype::F16, f16_bytes(&[4.0, 2.0, 1.0, 8.0]));
    let mut cx = Graph::new();
    let t = cx.tensor((4,), DType::F16);
    let out = t.reciprocal();
    let data: FxHashMap<_, _> = [(t.id, a.clone())].into_iter().collect();
    let mut rt = CudaRuntime::load(&cx).expect("load");
    rt.search(&data, &harness_search_options())
        .expect("search f16 reciprocal");
    rt.set_data(t.id, a).unwrap();
    rt.execute().expect("execute");
    let bytes = rt.fetch(out.id).expect("fetch").0.bytes.clone();
    assert_eq!(bytes, f16_bytes(&[0.25, 0.5, 1.0, 0.125]));
}

/// The build embeds the toolkit's half/bfloat header closure, so a runtime
/// machine needs no CUDA toolkit. Prove it is self-sufficient: NVRTC compiles
/// both headers with ONLY the embedded directory on the include path.
#[test]
fn embedded_headers_compile_without_a_toolkit() {
    let dir = luminal_cuda_lite::device::embedded_cuda_headers_dir()
        .expect("the device build embeds the CUDA half headers");
    assert!(dir.join("cuda_fp16.h").is_file(), "embedded cuda_fp16.h");
    assert!(dir.join("cuda_bf16.h").is_file(), "embedded cuda_bf16.h");
    let src = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
extern "C" __global__ void k(const __half* a, const __nv_bfloat16* b, __half* o) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    o[i] = __float2half(__half2float(a[i]) + __bfloat162float(b[i]));
}"#;
    let opts = cudarc::nvrtc::CompileOptions {
        include_paths: vec![dir.to_string_lossy().into_owned()],
        ..Default::default()
    };
    cudarc::nvrtc::compile_ptx_with_opts(src, opts)
        .expect("embedded headers alone compile half/bfloat kernels");
}
