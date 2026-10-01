#[cfg(any(target_os = "linux", target_os = "windows"))]
#[path = "support/validate.rs"]
mod application;

fn main() -> anyhow::Result<()> {
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    return application::main();
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    anyhow::bail!("llm_chat_cuda requires Linux or Windows")
}
