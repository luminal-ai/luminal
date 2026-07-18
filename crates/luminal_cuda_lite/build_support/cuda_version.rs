// Luminal CUDA API-threshold ladder for #[cfg] gating (see `cuda_graph` module docs).
// This is separate from cudarc's binding-release table.

/// Minimum CUDA toolkit version Luminal supports (below this, graph APIs are unavailable).
pub const MIN_SUPPORTED_CUDA: (usize, usize) = (11, 4);

/// Last major toolkit version Luminal has validated; newer majors emit a build warning only.
#[allow(dead_code)] // used by `build.rs`, not unit tests
pub const MAX_VALIDATED_MAJOR: usize = 13;

/// API change points where `cuda_graph.rs` branches on `#[cfg(luminal_cuda_ge_*)]`.
/// Add a rung only when compilation reveals a new driver API variant.
pub const API_LADDER: &[(usize, usize, &str)] = &[
    (12, 0, "luminal_cuda_ge_12_0"),
    (12, 3, "luminal_cuda_ge_12_3"),
    (12, 8, "luminal_cuda_ge_12_8"),
];

/// Used when `nvcc` is unavailable; top ladder rung (same cfgs as any version >= 12.8).
pub const FALLBACK_CUDA_VERSION: (usize, usize) = (12, 8);

#[allow(dead_code)] // used by unit tests and for documenting the CUDARC_CUDA_VERSION encoding
pub fn cudarc_env_format(major: usize, minor: usize) -> String {
    format!("{major}0{minor}0")
}

/// Parse `CUDARC_CUDA_VERSION` (`{major}0{minor}0`) without validating against cudarc releases.
pub fn parse_cudarc_env(version: &str) -> Option<(usize, usize)> {
    let version = version.trim();
    if !version.ends_with('0') || version.len() < 3 {
        return None;
    }
    let inner = &version[..version.len() - 1];
    let sep = inner.find('0')?;
    if sep == 0 {
        return None;
    }
    let major: usize = inner[..sep].parse().ok()?;
    let minor: usize = if inner[sep + 1..].is_empty() {
        0
    } else {
        inner[sep + 1..].parse().ok()?
    };
    Some((major, minor))
}

fn parse_major_minor(version_number: &str) -> Option<(usize, usize)> {
    let mut parts = version_number.split('.');
    let major: usize = parts.next()?.parse().ok()?;
    let minor: usize = parts.next()?.parse().ok()?;
    Some((major, minor))
}

/// Parse `nvcc --version` stdout using cudarc's line-3 layout; accept any `major.minor`.
pub fn parse_nvcc_stdout(stdout: &str) -> Option<(usize, usize)> {
    let version_line = stdout.lines().nth(3)?;
    let release_section = version_line.split(", ").nth(1)?;
    let version_number = release_section.split(' ').nth(1)?;
    parse_major_minor(version_number)
}

pub fn version_at_least(
    major: usize,
    minor: usize,
    req_major: usize,
    req_minor: usize,
) -> bool {
    major > req_major || (major == req_major && minor >= req_minor)
}

/// Returns `Err` with a message when `(major, minor)` is below [`MIN_SUPPORTED_CUDA`].
pub fn ensure_supported_version(major: usize, minor: usize) -> Result<(), String> {
    let (min_major, min_minor) = MIN_SUPPORTED_CUDA;
    if version_at_least(major, minor, min_major, min_minor) {
        Ok(())
    } else {
        Err(format!(
            "CUDA {major}.{minor} is below Luminal's minimum supported toolkit ({min_major}.{min_minor})"
        ))
    }
}

pub fn threshold_cfgs(major: usize, minor: usize) -> Vec<&'static str> {
    API_LADDER
        .iter()
        .filter(|&&(req_major, req_minor, _)| version_at_least(major, minor, req_major, req_minor))
        .map(|&(_, _, cfg)| cfg)
        .collect()
}

#[allow(dead_code)] // used by `build.rs`, not unit tests
pub fn maybe_warn_newer_major(major: usize) -> Option<String> {
    if major > MAX_VALIDATED_MAJOR {
        Some(format!(
            "CUDA {major}.x is newer than the last validated major ({MAX_VALIDATED_MAJOR}); \
             graph API struct layouts and driver symbols are assumed compatible — verify."
        ))
    } else {
        None
    }
}
