#[cfg(any(feature = "cuda_lite", feature = "metal"))]
mod app;

fn main() -> anyhow::Result<()> {
    #[cfg(any(feature = "cuda_lite", feature = "metal"))]
    {
        app::main()
    }
    #[cfg(not(any(feature = "cuda_lite", feature = "metal")))]
    anyhow::bail!("enable exactly one backend: --features cuda_lite or --features metal")
}
