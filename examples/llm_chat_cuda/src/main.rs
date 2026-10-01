fn main() -> anyhow::Result<()> {
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    return llm_chat_cuda::run();
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    anyhow::bail!("llm_chat_cuda requires Linux or Windows")
}
