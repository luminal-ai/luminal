fn main() -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    return llm_chat_metal::run();
    #[cfg(not(target_os = "macos"))]
    anyhow::bail!("llm_chat_metal requires macOS")
}
