//! Embed the CUDA toolkit's `cuda_fp16.h`/`cuda_bf16.h` header closure so
//! NVRTC can compile half/bfloat kernels on a machine that has only the CUDA
//! driver, not the toolkit.
//!
//! NVRTC ships neither half header, so a kernel over F16/BF16 needs an include
//! tree. Building the `device` feature already requires a CUDA toolkit
//! (cudarc's `cuda-version-from-build-system`), so this script locates that
//! toolkit's headers at build time and embeds them into the binary. At runtime
//! the device module materializes them once into a temp directory and puts it
//! on NVRTC's include path as a fallback after any real runtime toolkit.
//!
//! Point `LUMINAL_CUDA_HEADERS_DIR` at an include tree to use a pinned or
//! vendored copy instead of the build machine's toolkit.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

/// Include directives that NVRTC resolves from its own built-in headers do not
/// exist under these roots and are skipped.
fn cuda_roots() -> Vec<PathBuf> {
    if let Ok(explicit) = std::env::var("LUMINAL_CUDA_HEADERS_DIR") {
        let path = PathBuf::from(explicit);
        if path.is_dir() {
            return vec![path];
        }
        panic!(
            "LUMINAL_CUDA_HEADERS_DIR={} is not a directory",
            path.display()
        );
    }
    let mut roots = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CONDA_PREFIX"] {
        if let Ok(root) = std::env::var(var) {
            let include = PathBuf::from(root).join("include");
            if include.is_dir() && !roots.contains(&include) {
                roots.push(include);
            }
        }
    }
    let system_include = PathBuf::from("/usr/local/cuda/include");
    if system_include.is_dir() && !roots.contains(&system_include) {
        roots.push(system_include);
    }
    roots
}

/// Where libcu++ headers (`nv/...`) live when they are not under a CUDA root.
/// `cuda_fp16.h` includes `<nv/target>`; NVRTC cannot resolve it built-in.
fn nv_roots(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = roots.to_vec();
    for extra in ["/usr/include", "/usr/local/include"] {
        let path = PathBuf::from(extra);
        if path.is_dir() && !out.contains(&path) {
            out.push(path);
        }
    }
    out
}

fn resolve(include: &str, cuda_roots: &[PathBuf], nv_roots: &[PathBuf]) -> Option<PathBuf> {
    for root in cuda_roots {
        let candidate = root.join(include);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    // libcu++ headers are not toolkit files; only chase the namespaces they
    // use so system glibc headers (<stdlib.h>, ...) are left to NVRTC.
    if (include.starts_with("nv/") || include.starts_with("cuda/"))
        && let Some(root) = nv_roots.iter().find(|root| root.join(include).is_file())
    {
        return Some(root.join(include));
    }
    None
}

/// Walk `#include` directives from the two entry headers, collecting every file
/// that resolves under the CUDA/libcu++ roots.
fn closure(
    entries: &[&str],
    cuda_roots: &[PathBuf],
    nv_roots: &[PathBuf],
) -> Vec<(String, PathBuf)> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut queue: Vec<(String, PathBuf)> = Vec::new();
    for entry in entries {
        let path = resolve(entry, cuda_roots, nv_roots).unwrap_or_else(|| {
            panic!("CUDA header {entry} not found; set LUMINAL_CUDA_HEADERS_DIR")
        });
        if seen.insert((*entry).to_string()) {
            queue.push(((*entry).to_string(), path));
        }
    }
    let mut out = Vec::new();
    while let Some((name, path)) = queue.pop() {
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read CUDA header {}: {e}", path.display()));
        for line in text.lines() {
            let Some(rest) = line.trim_start().strip_prefix("#include") else {
                continue;
            };
            let rest = rest.trim_start();
            let close = match rest.as_bytes().first() {
                Some(b'<') => '>',
                Some(b'"') => '"',
                _ => continue,
            };
            let Some(end) = rest[1..].find(close) else {
                continue;
            };
            let include = &rest[1..1 + end];
            if let Some(resolved) = resolve(include, cuda_roots, nv_roots)
                && seen.insert(include.to_string())
            {
                queue.push((include.to_string(), resolved));
            }
        }
        out.push((name, path));
    }
    out
}

fn content_tag(files: &[(String, PathBuf)]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for (name, path) in files {
        for byte in name.bytes().chain(fs::read(path).unwrap_or_default()) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for var in [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_ROOT",
        "CONDA_PREFIX",
        "LUMINAL_CUDA_HEADERS_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let generated = out_dir.join("cuda_headers.rs");
    let device = std::env::var_os("CARGO_FEATURE_DEVICE").is_some();
    if !device {
        // The include! is cfg(feature = "device"); nothing needs embedding.
        fs::write(
            &generated,
            "pub static HEADERS: &[(&str, &[u8])] = &[];\npub const TAG: &str = \"none\";\n",
        )
        .expect("write empty cuda_headers.rs");
        return;
    }

    let roots = cuda_roots();
    let nv = nv_roots(&roots);
    if roots.is_empty() {
        panic!(
            "the `device` feature needs the CUDA toolkit headers (cuda_fp16.h/cuda_bf16.h); \
             none of CUDA_HOME/CUDA_PATH/CUDA_ROOT/CONDA_PREFIX/{} contained them. \
             Set LUMINAL_CUDA_HEADERS_DIR to an include tree.",
            "/usr/local/cuda/include"
        );
    }
    let files = closure(&["cuda_fp16.h", "cuda_bf16.h"], &roots, &nv);
    let tag = content_tag(&files);

    let mut code = String::new();
    code.push_str("// @generated by build.rs: embedded CUDA half/bfloat headers.\n");
    code.push_str("pub static HEADERS: &[(&str, &[u8])] = &[\n");
    for (name, path) in &files {
        let dest = out_dir.join("cuda_include").join(name);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).expect("create embedded header dir");
        }
        fs::copy(path, &dest)
            .unwrap_or_else(|e| panic!("copy {} -> {}: {e}", path.display(), dest.display()));
        code.push_str(&format!(
            "    ({name:?}, include_bytes!({:?})),\n",
            dest.to_string_lossy()
        ));
    }
    code.push_str("];\n");
    code.push_str(&format!("pub const TAG: &str = {tag:?};\n"));
    fs::write(&generated, code).expect("write cuda_headers.rs");
}
