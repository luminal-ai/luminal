//! CUDA chat application and its device storage.
#![cfg(any(target_os = "linux", target_os = "windows"))]
pub mod backend;
mod memory;

pub fn run() -> anyhow::Result<()> {
    llm_chat::app::main("CUDA", |graph, weights, search| {
        backend::CudaBackend::compile(
            graph,
            weights,
            &luminal_cuda_lite::CompileOptions {
                search_log: true,
                generations: search.generations,
                generation_size: search.population,
                seed: search.seed,
                device_budget_bytes: search.device_budget_bytes,
                ..Default::default()
            },
        )
    })
}
