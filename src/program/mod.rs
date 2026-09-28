//! Prebuilt programs.
//!
//! A search ends with one genome per bucket: a choice of e-node per e-class
//! of that bucket's saturated e-graph. Everything the runtime executes is
//! derived from those choices by the extractor, deterministically, so the
//! choices plus the e-graphs they index are the whole program. This module
//! writes them to a directory and reads them back into the same per-bucket
//! LLIR a fresh search would have produced, without saturating or searching.
//!
//! What is stored per bucket is the *whole* e-graph, not the chosen terms:
//! extractors legitimately read e-nodes the genome did not select (an input
//! lookup scanning its e-class, a fallback source), so a trimmed e-graph
//! would extract a different program or none.
//!
//! The HLIR the program was selected for is identified by the hash of its
//! egglog text, taken after the loop-rolling prepass. LLIR `Input`/`Output`
//! nodes carry HLIR node indices, so the loading graph must be built by the
//! same code, in the same order: a different graph is refused, never
//! silently rebound.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::egglog_utils::{ClassId, LlirExtractor, NodeId, SerializedEGraph, hlir_to_egglog};
use crate::graph::{BucketLLIR, CompileOptions, DimBucket, Graph};
use crate::op::{EgglogOp, IntoEgglogOp, LLIROp, Runtime};
use crate::search::{LlirFingerprint, unroll_packed_llir};
use crate::shape::{DynDimIntervals, DynMap, Symbol};

/// Bump when the on-disk layout or the meaning of any field changes.
pub const PROGRAM_FORMAT_VERSION: u32 = 1;
pub const MANIFEST_FILE: &str = "manifest.json";

/// One bucket's chosen program: which e-node every e-class resolves to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BucketSelection {
    /// Index into `SearchSpace::buckets` at search time.
    pub space_bucket_index: usize,
    /// dim → index into the bucket list of that dim.
    pub bucket_indices: DynMap,
    /// The dyn map the finalist was validated at and the runtime prepares at.
    pub representative_dyn_map: DynMap,
    /// Interval facts saturation assumed for the bucket (informational; the
    /// runtime re-derives them from the buckets).
    pub intervals: DynDimIntervals,
    pub choices: Vec<(ClassId, NodeId)>,
    /// Fingerprint of the packed extraction of `choices`; a re-extraction
    /// that does not reproduce it is refused.
    pub fingerprint: LlirFingerprint,
}

/// The selection a search made, one entry per bucket in bucket order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProgramSelection {
    pub buckets: Vec<BucketSelection>,
}

/// `manifest.json`: everything a load checks before extracting, plus opaque
/// `backend` and `app` blocks the runtime and the application fill with their
/// own identity (device, toolchain, model, weights) and check themselves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramManifest {
    pub format_version: u32,
    pub luminal_version: String,
    /// FNV-1a of the HLIR egglog program text plus root, after loop rolling.
    pub hlir_hash: u64,
    pub hlir_node_count: usize,
    pub custom_op_count: usize,
    /// Sort names of the registered egglog ops, in registration order.
    pub op_names: Vec<String>,
    pub dim_buckets: FxHashMap<Symbol, Vec<DimBucket>>,
    pub search_dims: DynMap,
    pub buckets: Vec<BucketSelection>,
    #[serde(default)]
    pub backend: serde_json::Value,
    #[serde(default)]
    pub app: serde_json::Value,
}

#[derive(Debug)]
pub enum ProgramError {
    Io {
        path: PathBuf,
        err: std::io::Error,
    },
    Json {
        path: PathBuf,
        err: serde_json::Error,
    },
    FormatVersion {
        saved: u32,
        supported: u32,
    },
    /// The graph being loaded into is not the graph the program was selected
    /// for.
    HlirMismatch {
        field: &'static str,
        saved: String,
        current: String,
    },
    OpSetMismatch {
        saved: Vec<String>,
        current: Vec<String>,
    },
    /// Dim buckets or search dims differ from the compile options.
    OptionsMismatch {
        detail: String,
    },
    /// A stored choice names an e-class or e-node the e-graph does not have.
    ChoiceMismatch {
        bucket: usize,
        msg: String,
    },
    FingerprintMismatch {
        bucket: usize,
        saved: LlirFingerprint,
        current: LlirFingerprint,
    },
    Extraction {
        bucket: usize,
        msg: String,
    },
    /// `save_program` needs the search space the selection was made in.
    NoSearchSpace,
    /// A selection bucket does not correspond to the search-space bucket it
    /// names.
    SelectionMismatch {
        bucket: usize,
        detail: String,
    },
}

impl std::fmt::Display for ProgramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, err } => write!(f, "{}: {err}", path.display()),
            Self::Json { path, err } => write!(f, "{}: {err}", path.display()),
            Self::FormatVersion { saved, supported } => write!(
                f,
                "program format version {saved} is not the supported {supported}"
            ),
            Self::HlirMismatch {
                field,
                saved,
                current,
            } => write!(
                f,
                "the graph is not the one the program was built for: {field} saved {saved}, current {current}"
            ),
            Self::OpSetMismatch { saved, current } => write!(
                f,
                "registered egglog ops differ: saved {saved:?}, current {current:?}"
            ),
            Self::OptionsMismatch { detail } => write!(f, "compile options differ: {detail}"),
            Self::ChoiceMismatch { bucket, msg } => {
                write!(f, "bucket {bucket}: stored choice does not fit the e-graph: {msg}")
            }
            Self::FingerprintMismatch {
                bucket,
                saved,
                current,
            } => write!(
                f,
                "bucket {bucket}: re-extraction produced {current:?}, the program was {saved:?}"
            ),
            Self::Extraction { bucket, msg } => write!(f, "bucket {bucket}: extraction failed: {msg}"),
            Self::NoSearchSpace => write!(f, "the graph has no search space to save"),
            Self::SelectionMismatch { bucket, detail } => {
                write!(f, "selection bucket {bucket} does not match the search space: {detail}")
            }
        }
    }
}

impl std::error::Error for ProgramError {}

/// What a load hands the runtime: the manifest and one LLIR per bucket, in
/// the same form `Runtime::compile`'s search would have selected.
pub struct LoadedProgram {
    pub manifest: ProgramManifest,
    pub bucket_llirs: Vec<BucketLLIR>,
}

/// `(hash, node count)` of the graph as saturation sees it.
pub fn hlir_identity(graph: &Graph) -> (u64, usize) {
    let (program, root) = hlir_to_egglog(graph);
    let mut hash = fnv1a(0xcbf2_9ce4_8422_2325, program.as_bytes());
    hash = fnv1a(hash, b"\0");
    hash = fnv1a(hash, root.as_bytes());
    (hash, graph.graph.node_count())
}

/// FNV-1a: stable across processes and toolchains, unlike `DefaultHasher`.
fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn bucket_file(index: usize) -> String {
    format!("bucket-{index:02}.egraph.json")
}

fn io_err(path: &Path) -> impl FnOnce(std::io::Error) -> ProgramError + '_ {
    move |err| ProgramError::Io {
        path: path.to_path_buf(),
        err,
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), ProgramError> {
    let bytes = serde_json::to_vec(value).map_err(|err| ProgramError::Json {
        path: path.to_path_buf(),
        err,
    })?;
    // Write beside the target and rename so a reader never sees a partial file.
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    std::fs::write(&tmp, bytes).map_err(io_err(&tmp))?;
    std::fs::rename(&tmp, path).map_err(io_err(path))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, ProgramError> {
    let bytes = std::fs::read(path).map_err(io_err(path))?;
    serde_json::from_slice(&bytes).map_err(|err| ProgramError::Json {
        path: path.to_path_buf(),
        err,
    })
}

/// Rewrite `manifest.json` in place, for a caller that fills `backend` or
/// `app` after `save_program`.
pub fn write_manifest(dir: &Path, manifest: &ProgramManifest) -> Result<(), ProgramError> {
    write_json(&dir.join(MANIFEST_FILE), manifest)
}

pub fn read_manifest(dir: &Path) -> Result<ProgramManifest, ProgramError> {
    let manifest: ProgramManifest = read_json(&dir.join(MANIFEST_FILE))?;
    if manifest.format_version != PROGRAM_FORMAT_VERSION {
        return Err(ProgramError::FormatVersion {
            saved: manifest.format_version,
            supported: PROGRAM_FORMAT_VERSION,
        });
    }
    Ok(manifest)
}

/// The egglog ops a runtime extracts with, as `build_search_space` registers
/// them: backend ops first, then HLIR.
fn program_ops<Rt: Runtime>() -> Vec<Arc<Box<dyn EgglogOp>>> {
    let mut ops = Rt::Ops::into_vec();
    ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
    ops
}

fn op_names(ops: &[Arc<Box<dyn EgglogOp>>]) -> Vec<String> {
    ops.iter().map(|op| op.sort().name.clone()).collect()
}

/// Write `selection` and the e-graphs it indexes under `dir`. The graph must
/// still hold the search space the selection was made in (it does after
/// `compile`). `options` are the compile options of that search; their
/// `search_dims` are recorded so a load can apply them.
pub fn save_program(
    dir: &Path,
    graph: &Graph,
    selection: &ProgramSelection,
    options: &CompileOptions,
    backend: serde_json::Value,
    app: serde_json::Value,
) -> Result<ProgramManifest, ProgramError> {
    let space = graph.search_space().ok_or(ProgramError::NoSearchSpace)?;
    for (i, bucket) in selection.buckets.iter().enumerate() {
        let Some(space_bucket) = space.buckets.get(bucket.space_bucket_index) else {
            return Err(ProgramError::SelectionMismatch {
                bucket: i,
                detail: format!(
                    "space bucket {} of {}",
                    bucket.space_bucket_index,
                    space.buckets.len()
                ),
            });
        };
        if space_bucket.bucket_indices != bucket.bucket_indices {
            return Err(ProgramError::SelectionMismatch {
                bucket: i,
                detail: format!(
                    "bucket indices {:?} vs space {:?}",
                    bucket.bucket_indices, space_bucket.bucket_indices
                ),
            });
        }
    }
    std::fs::create_dir_all(dir).map_err(io_err(dir))?;
    let (hlir_hash, hlir_node_count) = hlir_identity(graph);
    let manifest = ProgramManifest {
        format_version: PROGRAM_FORMAT_VERSION,
        luminal_version: env!("CARGO_PKG_VERSION").to_string(),
        hlir_hash,
        hlir_node_count,
        custom_op_count: graph.custom_ops.len(),
        op_names: op_names(&space.ops),
        dim_buckets: space.dim_buckets.clone(),
        search_dims: options.search_dims.clone(),
        buckets: selection.buckets.clone(),
        backend,
        app,
    };
    for (i, bucket) in selection.buckets.iter().enumerate() {
        let egraph = &space.buckets[bucket.space_bucket_index].egraph;
        write_json(&dir.join(bucket_file(i)), egraph)?;
    }
    // Last, so a directory with a manifest is a complete program.
    write_json(&dir.join(MANIFEST_FILE), &manifest)?;
    Ok(manifest)
}

impl Graph {
    /// Rebuild the per-bucket LLIR of the program saved under `dir` for this
    /// graph, which must have been built by the same code as the graph the
    /// program was selected for. Replays the loop-rolling prepass and applies
    /// the saved `search_dims`, exactly as `compile` would have before
    /// searching, then extracts. `options` must agree with the saved buckets
    /// where they say anything.
    pub fn load_program<Rt: Runtime>(
        &mut self,
        dir: &Path,
        options: &CompileOptions,
    ) -> Result<LoadedProgram, ProgramError> {
        let manifest = read_manifest(dir)?;
        self.run_auto_loop_rolling_prepass(options);
        for (&dim, &value) in &manifest.search_dims {
            self.set_dim(dim, value);
        }
        self.extract_program::<Rt>(dir, manifest, options)
    }

    /// Re-extract the program saved under `dir` from this graph as it is:
    /// already rolled, dims already set. What `save_program`'s caller uses to
    /// prove the saved selection reproduces the program it just installed.
    pub fn extract_saved_program<Rt: Runtime>(
        &self,
        dir: &Path,
        options: &CompileOptions,
    ) -> Result<LoadedProgram, ProgramError> {
        let manifest = read_manifest(dir)?;
        self.extract_program::<Rt>(dir, manifest, options)
    }

    fn extract_program<Rt: Runtime>(
        &self,
        dir: &Path,
        manifest: ProgramManifest,
        options: &CompileOptions,
    ) -> Result<LoadedProgram, ProgramError> {
        let (hlir_hash, hlir_node_count) = hlir_identity(self);
        if hlir_hash != manifest.hlir_hash {
            return Err(ProgramError::HlirMismatch {
                field: "hlir hash",
                saved: format!("{:016x}", manifest.hlir_hash),
                current: format!("{hlir_hash:016x}"),
            });
        }
        if hlir_node_count != manifest.hlir_node_count {
            return Err(ProgramError::HlirMismatch {
                field: "node count",
                saved: manifest.hlir_node_count.to_string(),
                current: hlir_node_count.to_string(),
            });
        }
        if self.custom_ops.len() != manifest.custom_op_count {
            return Err(ProgramError::HlirMismatch {
                field: "custom op count",
                saved: manifest.custom_op_count.to_string(),
                current: self.custom_ops.len().to_string(),
            });
        }
        let ops = program_ops::<Rt>();
        let current_names = op_names(&ops);
        if current_names != manifest.op_names {
            return Err(ProgramError::OpSetMismatch {
                saved: manifest.op_names.clone(),
                current: current_names,
            });
        }
        if !options.dim_buckets.is_empty() && options.dim_buckets != manifest.dim_buckets {
            return Err(ProgramError::OptionsMismatch {
                detail: format!(
                    "dim buckets saved {:?}, requested {:?}",
                    manifest.dim_buckets, options.dim_buckets
                ),
            });
        }
        if !options.search_dims.is_empty() && options.search_dims != manifest.search_dims {
            return Err(ProgramError::OptionsMismatch {
                detail: format!(
                    "search dims saved {:?}, requested {:?}",
                    manifest.search_dims, options.search_dims
                ),
            });
        }
        let custom_ops: Vec<LLIROp> = self.custom_ops.iter().map(|op| op.to_llir_op()).collect();

        let mut bucket_llirs = Vec::with_capacity(manifest.buckets.len());
        for (i, bucket) in manifest.buckets.iter().enumerate() {
            let egraph: SerializedEGraph = read_json(&dir.join(bucket_file(i)))?;
            let mut extractor = LlirExtractor::new(&egraph, &ops);
            let genome = extractor
                .index_owned_choices(&bucket.choices)
                .map_err(|msg| ProgramError::ChoiceMismatch { bucket: i, msg })?;
            let extracted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let packed = extractor.extract_indexed_packed(&genome, &custom_ops);
                let fingerprint = packed.fingerprint();
                (fingerprint, unroll_packed_llir(packed))
            }));
            let (fingerprint, llir) = extracted.map_err(|payload| ProgramError::Extraction {
                bucket: i,
                msg: crate::mask_events::panic_payload(payload.as_ref()),
            })?;
            if fingerprint != bucket.fingerprint {
                return Err(ProgramError::FingerprintMismatch {
                    bucket: i,
                    saved: bucket.fingerprint,
                    current: fingerprint,
                });
            }
            bucket_llirs.push((
                bucket.bucket_indices.clone(),
                bucket.representative_dyn_map.clone(),
                llir,
            ));
        }
        Ok(LoadedProgram {
            manifest,
            bucket_llirs,
        })
    }
}

#[cfg(test)]
mod tests;
