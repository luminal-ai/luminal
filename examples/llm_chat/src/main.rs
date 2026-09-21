fn main() -> anyhow::Result<()> {
    #[cfg(any(feature = "luminal_cuda_lite", feature = "metal"))]
    return llm_chat::app::main();

    #[cfg(not(any(feature = "luminal_cuda_lite", feature = "metal")))]
    anyhow::bail!("enable a backend: --features cuda or --features metal")
}
