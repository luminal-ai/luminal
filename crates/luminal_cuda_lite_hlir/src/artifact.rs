//! CUDA module artifact capture, serialization, and loading.

use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, Mutex},
};

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use cudarc::driver::CudaContext;
use luminal::op::IntoEgglogOp;
use luminal::prelude::Graph;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{cuda_nvrtc_compile_options, loaded_nvrtc_version, runtime::CudaRuntimeImpl};

thread_local! {
    static MODULE_ARTIFACT_SESSION: RefCell<Option<Arc<Mutex<ModuleArtifact>>>> =
        const { RefCell::new(None) };
}

const MODULE_ARTIFACT_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
struct ModuleArtifactSignature {
    target_arch: String,
    nvrtc_options: Vec<String>,
    nvrtc_version: Option<u32>,
}

#[derive(Deserialize, Serialize)]
struct SerializedModuleArtifact {
    version: u32,
    signature: ModuleArtifactSignature,
    images: HashMap<String, String>,
}

pub(crate) struct ModuleArtifact {
    signature: ModuleArtifactSignature,
    images: HashMap<String, Vec<u8>>,
    loading: bool,
    capturing: bool,
}

pub(crate) enum ModuleImageLookup {
    Compile,
    Hit(Vec<u8>),
    Missing { available: usize },
}

pub(crate) fn module_artifact_session(
    ctx: &Arc<CudaContext>,
    data: Option<&str>,
) -> Result<Arc<Mutex<ModuleArtifact>>, String> {
    let signature = module_artifact_signature(ctx)?;
    let artifact = if let Some(data) = data {
        deserialize_module_artifact(data, signature)?
    } else {
        ModuleArtifact {
            signature,
            images: HashMap::new(),
            loading: false,
            capturing: false,
        }
    };
    Ok(Arc::new(Mutex::new(artifact)))
}

pub(crate) fn with_module_artifact_session<T>(
    session: Arc<Mutex<ModuleArtifact>>,
    run: impl FnOnce() -> T,
) -> T {
    struct SessionGuard(Option<Arc<Mutex<ModuleArtifact>>>);
    impl Drop for SessionGuard {
        fn drop(&mut self) {
            MODULE_ARTIFACT_SESSION.with(|current| current.replace(self.0.take()));
        }
    }

    let previous = MODULE_ARTIFACT_SESSION.with(|current| current.replace(Some(session)));
    let _guard = SessionGuard(previous);
    run()
}

pub(crate) fn serialize_module_artifact(session: &Mutex<ModuleArtifact>) -> String {
    let artifact = session.lock().unwrap();
    let serialized = SerializedModuleArtifact {
        version: MODULE_ARTIFACT_VERSION,
        signature: artifact.signature.clone(),
        images: artifact
            .images
            .iter()
            .map(|(key, image)| (key.clone(), BASE64.encode(image)))
            .collect(),
    };
    serde_json::to_string(&serialized).unwrap()
}

pub(crate) fn module_artifact_is_loading(artifact: &Mutex<ModuleArtifact>) -> bool {
    artifact.lock().unwrap().loading
}

pub(crate) fn begin_module_artifact_capture(artifact: &Mutex<ModuleArtifact>) {
    let mut artifact = artifact.lock().unwrap();
    artifact.images.clear();
    artifact.loading = false;
    artifact.capturing = true;
}

pub(crate) fn finish_module_artifact_capture(artifact: &Mutex<ModuleArtifact>) {
    let mut artifact = artifact.lock().unwrap();
    artifact.loading = true;
    artifact.capturing = false;
}

/// Recompile the selected schedule into a fresh capture so rejected search
/// candidates are not included in the saved artifact.
pub(crate) fn capture_selected_schedule<O: IntoEgglogOp + 'static>(
    graph: &mut Graph,
    runtime: &mut CudaRuntimeImpl<O>,
    artifact: &Mutex<ModuleArtifact>,
) -> Result<(), String> {
    if module_artifact_is_loading(artifact) {
        return Ok(());
    }

    begin_module_artifact_capture(artifact);
    runtime.clear_kernel_cache();
    graph.load_selected_schedule(runtime)?;
    finish_module_artifact_capture(artifact);
    Ok(())
}

pub(crate) fn module_key(source: &str) -> String {
    format!("{:x}", Sha256::digest(source.as_bytes()))
}

pub(crate) fn lookup_module_image(key: &str) -> ModuleImageLookup {
    MODULE_ARTIFACT_SESSION.with(|current| {
        let session = current.borrow();
        let Some(session) = session.as_ref() else {
            return ModuleImageLookup::Compile;
        };
        let artifact = session.lock().unwrap();
        match artifact.images.get(key) {
            Some(image) => ModuleImageLookup::Hit(image.clone()),
            None if artifact.loading => ModuleImageLookup::Missing {
                available: artifact.images.len(),
            },
            None => ModuleImageLookup::Compile,
        }
    })
}

pub(crate) fn record_module_image(key: &str, image: &[u8]) {
    MODULE_ARTIFACT_SESSION.with(|current| {
        if let Some(session) = current.borrow().as_ref() {
            let mut artifact = session.lock().unwrap();
            if artifact.capturing {
                artifact.images.insert(key.to_owned(), image.to_vec());
            }
        }
    });
}

fn module_artifact_signature(ctx: &Arc<CudaContext>) -> Result<ModuleArtifactSignature, String> {
    let (major, minor) = ctx
        .compute_capability()
        .map_err(|error| error.to_string())?;
    let target_arch = format!("sm_{major}{minor}");
    Ok(ModuleArtifactSignature {
        nvrtc_options: cuda_nvrtc_compile_options(&target_arch),
        target_arch,
        nvrtc_version: loaded_nvrtc_version(),
    })
}

fn deserialize_module_artifact(
    data: &str,
    signature: ModuleArtifactSignature,
) -> Result<ModuleArtifact, String> {
    let serialized: SerializedModuleArtifact =
        serde_json::from_str(data).map_err(|error| error.to_string())?;
    if serialized.version != MODULE_ARTIFACT_VERSION {
        return Err(format!(
            "unsupported CUDA module artifact version {}, expected {}",
            serialized.version, MODULE_ARTIFACT_VERSION
        ));
    }
    if serialized.signature != signature {
        return Err("CUDA module artifact is incompatible with this device".to_string());
    }
    let images = serialized
        .images
        .into_iter()
        .map(|(key, image)| {
            BASE64
                .decode(image)
                .map(|image| (key, image))
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    Ok(ModuleArtifact {
        signature,
        images,
        loading: true,
        capturing: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signature() -> ModuleArtifactSignature {
        ModuleArtifactSignature {
            target_arch: "sm_90".to_string(),
            nvrtc_options: vec!["--gpu-architecture=sm_90".to_string()],
            nvrtc_version: Some(12080),
        }
    }

    #[test]
    fn module_artifact_round_trip() {
        let session = Mutex::new(ModuleArtifact {
            signature: signature(),
            images: HashMap::from([("kernel".to_string(), vec![1, 2, 3])]),
            loading: false,
            capturing: false,
        });

        let data = serialize_module_artifact(&session);
        let loaded = deserialize_module_artifact(&data, signature()).unwrap();

        assert!(loaded.loading);
        assert_eq!(loaded.images["kernel"], [1, 2, 3]);
    }

    #[test]
    fn module_artifact_rejects_incompatible_device() {
        let session = Mutex::new(ModuleArtifact {
            signature: signature(),
            images: HashMap::new(),
            loading: false,
            capturing: false,
        });
        let data = serialize_module_artifact(&session);
        let mut incompatible = signature();
        incompatible.target_arch = "sm_80".to_string();

        assert!(deserialize_module_artifact(&data, incompatible).is_err());
    }

    #[test]
    fn loaded_artifact_does_not_fall_back_to_compilation() {
        let session = Arc::new(Mutex::new(ModuleArtifact {
            signature: signature(),
            images: HashMap::new(),
            loading: true,
            capturing: false,
        }));

        with_module_artifact_session(session, || {
            assert!(matches!(
                lookup_module_image("missing"),
                ModuleImageLookup::Missing { available: 0 }
            ));
        });
    }

    #[test]
    fn selected_capture_discards_search_candidates() {
        let session = Mutex::new(ModuleArtifact {
            signature: signature(),
            images: HashMap::from([("candidate".to_string(), vec![1])]),
            loading: false,
            capturing: false,
        });

        begin_module_artifact_capture(&session);
        {
            let artifact = session.lock().unwrap();
            assert!(artifact.images.is_empty());
            assert!(artifact.capturing);
        }

        finish_module_artifact_capture(&session);
        let artifact = session.lock().unwrap();
        assert!(artifact.loading);
        assert!(!artifact.capturing);
    }
}
