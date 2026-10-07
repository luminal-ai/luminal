//! Hugging Face Hub downloads for checkpoints that are not already on disk.
//!
//! The chat application consumes a checkpoint directory. When the user does not
//! provide one, these helpers fetch the files the loader needs from a Hub repo:
//! configuration and tokenizer metadata plus the safetensors weights, whether
//! they are a single file or a sharded index. Only ordinary checkpoint files
//! are downloaded; no remote model code is executed and no weights are
//! synthesized.
use anyhow::{Context, Result};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::{Repo, RepoType};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::File,
    path::{Path, PathBuf},
};

/// Metadata every chat checkpoint must provide.
const REQUIRED_FILES: &[&str] = &["config.json", "tokenizer.json", "tokenizer_config.json"];
/// Metadata the chat runtime uses when the repo provides it. Many repos omit
/// some of these, so a missing file is not an error.
const OPTIONAL_FILES: &[&str] = &["chat_template.jinja", "generation_config.json"];

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

/// Download a model checkpoint into the Hugging Face cache and return its
/// snapshot directory, which the local checkpoint loader can read directly.
///
/// `revision` pins a branch, tag, or commit. Authentication, the cache
/// location, and the endpoint come from the standard Hugging Face environment
/// (`HF_TOKEN`, `HF_HOME`, `HF_ENDPOINT`).
pub fn download_checkpoint(repo_id: &str, revision: Option<&str>) -> Result<PathBuf> {
    // `Api::new` ignores `HF_HOME`/`HF_ENDPOINT`, so build from the environment
    // explicitly. hf-hub reads the stored token file but not `HF_TOKEN`.
    let mut builder = ApiBuilder::from_env();
    if let Ok(token) = std::env::var("HF_TOKEN") {
        builder = builder.with_token(Some(token));
    }
    let api = builder
        .build()
        .context("initialize the Hugging Face Hub client")?;
    let repo = match revision {
        Some(revision) => api.repo(Repo::with_revision(
            repo_id.to_owned(),
            RepoType::Model,
            revision.to_owned(),
        )),
        None => api.model(repo_id.to_owned()),
    };
    // `config.json` determines the snapshot directory shared by every file.
    let config = repo
        .get("config.json")
        .with_context(|| format!("download {repo_id}/config.json"))?;
    let directory = config.parent().map(Path::to_path_buf).ok_or_else(|| {
        anyhow::anyhow!("Hugging Face returned no directory for {repo_id}/config.json")
    })?;
    for file in REQUIRED_FILES {
        repo.get(file)
            .with_context(|| format!("download {repo_id}/{file}"))?;
    }
    for file in OPTIONAL_FILES {
        let _ = repo.get(file);
    }
    download_safetensors(&repo, repo_id)?;
    Ok(directory)
}

/// Fetch the single-shard weights, or every shard named by the safetensors
/// index when the repo is sharded.
fn download_safetensors(repo: &ApiRepo, repo_id: &str) -> Result<()> {
    if repo.get("model.safetensors").is_ok() {
        return Ok(());
    }
    let index = repo.get("model.safetensors.index.json").with_context(|| {
        format!("{repo_id} has neither model.safetensors nor model.safetensors.index.json")
    })?;
    let index: SafetensorsIndex = serde_json::from_reader(
        File::open(&index).with_context(|| format!("open {}", index.display()))?,
    )
    .with_context(|| format!("parse {repo_id}/model.safetensors.index.json"))?;
    for shard in shard_files(&index.weight_map) {
        repo.get(&shard)
            .with_context(|| format!("download {repo_id}/{shard}"))?;
    }
    Ok(())
}

/// Shard paths named by a safetensors weight map, sorted and deduplicated.
fn shard_files(weight_map: &BTreeMap<String, String>) -> Vec<String> {
    let mut shards: Vec<String> = weight_map.values().cloned().collect();
    shards.sort();
    shards.dedup();
    shards
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_files_sort_and_deduplicate() {
        let weight_map = BTreeMap::from([
            (
                "b".to_owned(),
                "model-00002-of-00002.safetensors".to_owned(),
            ),
            (
                "a".to_owned(),
                "model-00001-of-00002.safetensors".to_owned(),
            ),
            (
                "c".to_owned(),
                "model-00002-of-00002.safetensors".to_owned(),
            ),
        ]);
        assert_eq!(
            shard_files(&weight_map),
            vec![
                "model-00001-of-00002.safetensors".to_owned(),
                "model-00002-of-00002.safetensors".to_owned(),
            ]
        );
    }

    #[test]
    fn index_weight_map_parses_into_shards() {
        let index: SafetensorsIndex = serde_json::from_str(
            r#"{"weight_map":{"a":"one.safetensors","b":"two.safetensors","c":"one.safetensors"}}"#,
        )
        .unwrap();
        assert_eq!(
            shard_files(&index.weight_map),
            vec!["one.safetensors".to_owned(), "two.safetensors".to_owned()]
        );
    }
}
