//! Architectural boundary: model_zoo defines graphs; runtime crates
//! own executable applications.

use std::fs;
use std::path::{Path, PathBuf};

fn collect_named(root: &Path, name: &str, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).expect("read model zoo directory") {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            collect_named(&path, name, out);
        } else if path.file_name().is_some_and(|file| file == name) {
            out.push(path);
        }
    }
}

#[test]
fn model_zoo_does_not_depend_on_runtimes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/model_zoo");
    let mut manifests = Vec::new();
    collect_named(&root, "Cargo.toml", &mut manifests);
    assert!(!manifests.is_empty(), "no model zoo manifest found");

    for manifest in manifests {
        let text = fs::read_to_string(&manifest).expect("read model zoo manifest");
        for runtime in [
            "luminal_reference",
            "luminal_cuda",
            "luminal_cuda_lite",
            "luminal_metal",
        ] {
            assert!(
                !text.contains(runtime),
                "{} depends on runtime crate {runtime}",
                manifest.display()
            );
        }
    }
}

#[test]
fn neural_network_building_blocks_have_no_runtime_dependency() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join("luminal_nn")
        .join("Cargo.toml");
    let text = fs::read_to_string(&manifest).expect("read luminal_nn manifest");
    let normal_dependencies = text
        .split("[dev-dependencies]")
        .next()
        .expect("normal dependency section");
    assert!(
        !normal_dependencies.contains("luminal_reference"),
        "luminal_nn normal dependencies include a runtime"
    );
}

#[test]
fn model_zoo_is_library_only() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/model_zoo");
    let mut mains = Vec::new();
    collect_named(&root, "main.rs", &mut mains);
    assert!(
        mains.is_empty(),
        "runtime/application entry points belong in runtime crates: {mains:?}"
    );
}
