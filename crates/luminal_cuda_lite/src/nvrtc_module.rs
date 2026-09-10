//! A process-wide NVRTC module cache for HOST OPS that launch their own
//! hand-written kernels (the fused serving ops: paged attention, the
//! MXFP4 MoE halves).
//!
//! The executor's `KernelCache` compiles the codegen'd `k` entry of a
//! `KernelOp`; a host op instead carries a fixed CUDA source with one or
//! more named entries, compiled ONCE per process for the current
//! device's compute capability and shared by every op instance (ops are
//! cloned freely during the search, so the cache cannot live on the
//! instance).

use anyhow::{Context, Result, anyhow};
use cudarc::driver::{CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

type Cache = Mutex<HashMap<(u64, String), (Arc<CudaModule>, CudaFunction)>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Functions by a caller-chosen KEY that determines the source (an op
/// name plus whatever the op bakes into its kernel text): the per-launch
/// cost is one short-string lookup, not a hash of the whole source.
type KeyedCache = Mutex<HashMap<(String, String), CudaFunction>>;

fn keyed_cache() -> &'static KeyedCache {
    static CACHE: OnceLock<KeyedCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The entry `entry` of the kernel source `source()` builds, compiled
/// once per `key` — the caller's promise is that equal keys mean equal
/// sources. Host ops that launch every tick use this so a decode tick
/// pays no source hashing or formatting per launch.
pub fn kernel_function_keyed(
    stream: &Arc<CudaStream>,
    key: &str,
    entry: &str,
    source: impl FnOnce() -> String,
) -> Result<CudaFunction> {
    {
        let cache = keyed_cache()
            .lock()
            .map_err(|_| anyhow!("keyed module cache poisoned"))?;
        if let Some(function) = cache.get(&(key.to_string(), entry.to_string())) {
            return Ok(function.clone());
        }
    }
    let function = kernel_function(stream, &source(), entry)?;
    keyed_cache()
        .lock()
        .map_err(|_| anyhow!("keyed module cache poisoned"))?
        .insert((key.to_string(), entry.to_string()), function.clone());
    Ok(function)
}

fn source_key(source: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut hasher);
    hasher.finish()
}

/// The `--gpu-architecture` NVRTC needs for the stream's device — PTX
/// for its own compute capability, so `__shfl_sync` and friends are
/// available and the driver JITs nothing surprising.
fn arch_for(stream: &Arc<CudaStream>) -> Result<&'static str> {
    let (major, minor) = stream
        .context()
        .compute_capability()
        .context("querying compute capability")?;
    // NVRTC option strings are `&'static str` in cudarc; one leaked
    // string per distinct capability per process is the cost.
    static ARCHES: OnceLock<Mutex<HashMap<(i32, i32), &'static str>>> = OnceLock::new();
    let mut arches = ARCHES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| anyhow!("arch cache poisoned"))?;
    Ok(*arches
        .entry((major, minor))
        .or_insert_with(|| Box::leak(format!("compute_{major}{minor}").into_boxed_str())))
}

/// Compile `source` (once) for the stream's device and return its
/// `entry` function. Errors carry the NVRTC log.
pub fn kernel_function(
    stream: &Arc<CudaStream>,
    source: &str,
    entry: &str,
) -> Result<CudaFunction> {
    let key = (source_key(source), entry.to_string());
    let mut cache = cache()
        .lock()
        .map_err(|_| anyhow!("module cache poisoned"))?;
    if let Some((_, function)) = cache.get(&key) {
        return Ok(function.clone());
    }
    // One module per source: compile it and hoist every entry the
    // caller will ask for lazily.
    let module = match cache
        .iter()
        .find(|((hash, _), _)| *hash == key.0)
        .map(|(_, (module, _))| module.clone())
    {
        Some(module) => module,
        None => {
            let opts = CompileOptions {
                arch: Some(arch_for(stream)?),
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(source, opts)
                .map_err(|e| anyhow!("NVRTC failed for host-op kernel `{entry}`: {e:?}"))?;
            stream
                .context()
                .load_module(ptx)
                .with_context(|| format!("loading host-op module for `{entry}`"))?
        }
    };
    let function = module
        .load_function(entry)
        .with_context(|| format!("host-op entry `{entry}` missing from its module"))?;
    cache.insert(key, (module, function.clone()));
    Ok(function)
}
