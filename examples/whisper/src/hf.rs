use hf_hub::api::sync::Api;
use std::path::PathBuf;

/// Downloads whisper model files (tokenizer.json + model.safetensors) from HuggingFace.
/// Returns the path of the cache directory containing both files.
pub fn prepare_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());

    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_dir = tokenizer_path.parent().unwrap().to_path_buf();

    repo.get("model.safetensors")?;
    Ok(model_dir)
}
