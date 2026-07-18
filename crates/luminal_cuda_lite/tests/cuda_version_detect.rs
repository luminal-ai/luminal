mod cuda_version {
    include!("../build_support/cuda_version.rs");
}

use cuda_version::{
    FALLBACK_CUDA_VERSION, cudarc_env_format, ensure_supported_version, parse_cudarc_env,
    parse_nvcc_stdout, threshold_cfgs,
};

#[test]
fn parse_cudarc_env_known_versions() {
    assert_eq!(parse_cudarc_env("12030"), Some((12, 3)));
    assert_eq!(parse_cudarc_env("13030"), Some((13, 3)));
    assert_eq!(parse_cudarc_env("11080"), Some((11, 8)));
    assert_eq!(parse_cudarc_env("13040"), Some((13, 4)));
    assert_eq!(parse_cudarc_env("12000"), Some((12, 0)));
}

#[test]
fn parse_cudarc_env_roundtrip_format() {
    assert_eq!(cudarc_env_format(12, 3), "12030");
    assert_eq!(parse_cudarc_env(&cudarc_env_format(12, 0)), Some((12, 0)));
}

#[test]
fn parse_cudarc_env_rejects_malformed() {
    assert_eq!(parse_cudarc_env("99999"), None);
    assert_eq!(parse_cudarc_env(""), None);
    assert_eq!(parse_cudarc_env("abc"), None);
}

#[test]
fn parse_nvcc_stdout_fixture_12_8() {
    let stdout = "nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 12.8, V12.8.89
";
    assert_eq!(parse_nvcc_stdout(stdout), Some((12, 8)));
}

#[test]
fn parse_nvcc_stdout_newer_versions() {
    let make = |ver: &str| {
        format!(
            "nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on ...
Cuda compilation tools, release {ver}, V{ver}.0
"
        )
    };
    assert_eq!(parse_nvcc_stdout(&make("13.4")), Some((13, 4)));
    assert_eq!(parse_nvcc_stdout(&make("14.0")), Some((14, 0)));
    assert_eq!(parse_nvcc_stdout(&make("12.7")), Some((12, 7)));
}

#[test]
fn threshold_cfgs_by_version() {
    assert!(threshold_cfgs(11, 8).is_empty());
    assert_eq!(threshold_cfgs(12, 2), vec!["luminal_cuda_ge_12_0"]);
    assert_eq!(
        threshold_cfgs(12, 3),
        vec!["luminal_cuda_ge_12_0", "luminal_cuda_ge_12_3"]
    );
    assert_eq!(
        threshold_cfgs(12, 7),
        vec!["luminal_cuda_ge_12_0", "luminal_cuda_ge_12_3"]
    );
    assert_eq!(
        threshold_cfgs(12, 8),
        vec![
            "luminal_cuda_ge_12_0",
            "luminal_cuda_ge_12_3",
            "luminal_cuda_ge_12_8"
        ]
    );
    assert_eq!(threshold_cfgs(13, 4), threshold_cfgs(13, 3));
}

#[test]
fn ensure_supported_version_floor() {
    assert!(ensure_supported_version(11, 4).is_ok());
    assert!(ensure_supported_version(12, 0).is_ok());
    let err = ensure_supported_version(11, 2).unwrap_err();
    assert!(err.contains("below Luminal's minimum"));
}

#[test]
fn fallback_is_top_ladder_rung() {
    assert_eq!(FALLBACK_CUDA_VERSION, (12, 8));
    assert_eq!(
        threshold_cfgs(FALLBACK_CUDA_VERSION.0, FALLBACK_CUDA_VERSION.1),
        threshold_cfgs(13, 3)
    );
}
