#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!("features `cuda` and `metal` are mutually exclusive");

#[cfg(all(feature = "cuda", feature = "metal"))]
fn main() {}

#[cfg(all(feature = "cuda", not(feature = "metal")))]
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
#[cfg(all(feature = "metal", not(feature = "cuda"), target_vendor = "apple"))]
use luminal_metal::MetalRuntime;
#[cfg(any(
    all(feature = "cuda", not(feature = "metal")),
    all(feature = "metal", not(feature = "cuda"), target_vendor = "apple")
))]
use qwen::{QwenRunConfig, Runtime, run_qwen};

#[cfg(all(feature = "cuda", not(feature = "metal")))]
fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    run_qwen(CudaRuntime::initialize(stream), QwenRunConfig::default()).unwrap();
}

#[cfg(all(feature = "metal", not(feature = "cuda"), target_vendor = "apple"))]
fn main() {
    run_qwen(MetalRuntime::initialize(()), QwenRunConfig::default()).unwrap();
}

#[cfg(all(feature = "metal", not(feature = "cuda"), not(target_vendor = "apple")))]
fn main() {
    eprintln!("qwen --features metal requires an Apple target with Metal support.");
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn main() {
    eprintln!("select exactly one backend with `--features cuda` or `--features metal`.");
    std::process::exit(2);
}
