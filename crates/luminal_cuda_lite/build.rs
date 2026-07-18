// Luminal API-threshold cfgs are documented in `src/kernel/cuda_graph.rs`.
// Luminal's version ladder is separate from cudarc's binding-release table.
mod cuda_version {
    include!("build_support/cuda_version.rs");
}

use std::process::Command;

const RERUN_ENV_VARS: [&str; 6] = [
    "CUDARC_CUDA_VERSION",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_ROOT",
    "CUDA_TOOLKIT_ROOT_DIR",
    "PATH",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_support/cuda_version.rs");
    for var in RERUN_ENV_VARS {
        println!("cargo:rerun-if-env-changed={var}");
    }

    for &(_, _, cfg) in cuda_version::API_LADDER {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
    }

    let (major, minor) = detect_cuda_version();
    println!("cargo:rustc-env=LUMINAL_CUDA_MAJOR_VERSION={major}");
    println!("cargo:rustc-env=LUMINAL_CUDA_MINOR_VERSION={minor}");

    if let Some(warning) = cuda_version::maybe_warn_newer_major(major) {
        println!("cargo:warning={warning}");
    }

    for cfg in cuda_version::threshold_cfgs(major, minor) {
        println!("cargo:rustc-cfg={cfg}");
    }
}

fn detect_cuda_version() -> (usize, usize) {
    if let Ok(version) = std::env::var("CUDARC_CUDA_VERSION") {
        let parsed = cuda_version::parse_cudarc_env(version.trim()).unwrap_or_else(|| {
            panic!(
                "Malformed `$CUDARC_CUDA_VERSION={}` (expected format like `12030` for CUDA 12.3)",
                version.trim()
            );
        });
        validate_version(parsed);
        return parsed;
    }

    if let Some(parsed) = detect_cuda_version_from_nvcc() {
        validate_version(parsed);
        return parsed;
    }

    let fallback = cuda_version::FALLBACK_CUDA_VERSION;
    println!(
        "cargo:warning=Failed to run `nvcc --version`. Using Luminal fallback CUDA {fallback:?} for API threshold cfgs."
    );
    fallback
}

fn validate_version((major, minor): (usize, usize)) {
    if let Err(message) = cuda_version::ensure_supported_version(major, minor) {
        panic!("{message}");
    }
}

fn detect_cuda_version_from_nvcc() -> Option<(usize, usize)> {
    let output = Command::new("nvcc").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    cuda_version::parse_nvcc_stdout(&String::from_utf8_lossy(&output.stdout))
}
