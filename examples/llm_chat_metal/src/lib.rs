//! Metal chat application and its device storage.
#![cfg(target_os = "macos")]
pub mod backend;
mod memory;

pub fn run() -> anyhow::Result<()> {
    llm_chat::app::main("Metal", |graph, weights, search| {
        backend::MetalBackend::compile(
            graph,
            weights,
            &luminal_metal::CompileOptions {
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
