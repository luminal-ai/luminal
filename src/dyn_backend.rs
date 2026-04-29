//! Dynamic backend trait and factory-based compilation.
//!
//! This module provides:
//! - [`DynBackend`]: an object-safe trait for dynamic backend dispatch
//! - [`compile_backend`]: generic helper that handles the full compilation pipeline
//! - [`BackendFactory`]: function pointer type for backend factories
//! - [`NativeDynBackend`]: the reference implementation for CPU

use std::collections::HashMap;

use half::{bf16, f16};
use petgraph::stable_graph::NodeIndex;
use rustc_hash::FxHashMap;

use crate::dtype::DType;
use crate::graph::Graph;
use crate::hlir::{NativeData, NativeRuntime, Output};
use crate::op::Runtime;

// ---------------------------------------------------------------------------
// DynBackend trait
// ---------------------------------------------------------------------------

/// Object-safe backend trait for dynamic dispatch.
///
/// Wraps a concrete [`Runtime`] implementor, providing a uniform interface
/// for `luminal_python` (and other dynamic consumers) without requiring
/// generic type parameters.
pub trait DynBackend {
    fn name(&self) -> &str;

    /// The device type this backend operates on (e.g. "cpu", "cuda").
    /// Used by the Python frontend to decide input tensor placement.
    fn device_type(&self) -> &str {
        "cpu"
    }

    fn set_data_bytes(&mut self, node: NodeIndex, bytes: Vec<u8>, dtype: DType);
    fn set_data_f32(&mut self, node: NodeIndex, data: Vec<f32>);
    fn get_output_f32(&self, node: NodeIndex) -> Vec<f32>;
    fn get_output_i32(&self, _node: NodeIndex) -> Vec<i32> {
        panic!("get_output_i32 not supported by '{}'", self.name());
    }
    fn get_output_bool(&self, _node: NodeIndex) -> Vec<bool> {
        panic!("get_output_bool not supported by '{}'", self.name());
    }
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>);

    // --- Optional device pointer support (GPU backends) --------------------

    fn supports_device_ptrs(&self) -> bool {
        false
    }
    /// # Safety
    /// Device pointer must be valid and point to at least `n_bytes` bytes.
    unsafe fn set_device_ptr(&mut self, _node: NodeIndex, _ptr: u64, _n_bytes: usize) {
        panic!("set_device_ptr not supported by '{}'", self.name());
    }
    /// # Safety
    /// Device pointer must remain valid through the next `execute()` call.
    unsafe fn set_output_device_ptr(&mut self, _node: NodeIndex, _ptr: u64, _n_bytes: usize) {
        panic!("set_output_device_ptr not supported by '{}'", self.name());
    }
    fn output_is_zero_copy(&self, _node: NodeIndex) -> bool {
        false
    }
    /// # Safety
    /// `dest_ptr` must be a valid device allocation with at least `n_bytes`.
    unsafe fn copy_output_to_device_ptr(&self, _node: NodeIndex, _dest_ptr: u64, _n_bytes: usize) {
        panic!(
            "copy_output_to_device_ptr not supported by '{}'",
            self.name()
        );
    }
}

// ---------------------------------------------------------------------------
// BackendCompileArgs + BackendFactory + Registry
// ---------------------------------------------------------------------------

/// Arguments passed to a backend factory during compilation.
pub struct BackendCompileArgs {
    pub search_iters: usize,
    pub weights: Vec<(String, Vec<u8>, DType)>,
    pub tensor_sizes: HashMap<String, usize>,
    pub device_ptrs: HashMap<String, (u64, usize)>,
}

/// Canonical PyCapsule name for [`BackendFactory`] function-pointer capsules.
///
/// Value MUST remain `"luminal.backend_factory"` for compatibility with
/// external plugin producers built against older versions of this crate.
pub const BACKEND_FACTORY_CAPSULE_NAME: &std::ffi::CStr = c"luminal.backend_factory";

/// A factory function that compiles a [`Graph`] into a ready-to-execute [`DynBackend`].
pub type BackendFactory = fn(&mut Graph, BackendCompileArgs) -> Result<Box<dyn DynBackend>, String>;

/// Compile a graph using a factory function directly.
pub fn compile_backend_from_factory(
    factory: BackendFactory,
    graph: &mut Graph,
    args: BackendCompileArgs,
) -> Result<Box<dyn DynBackend>, String> {
    factory(graph, args)
}

// ---------------------------------------------------------------------------
// compile_backend — generic compilation helper
// ---------------------------------------------------------------------------

/// Optional callback for uploading a device pointer + byte count to a node.
pub type SetDevicePtrFn<'a, Rt> = &'a dyn Fn(&mut Rt, NodeIndex, u64, usize);

/// Generic compilation pipeline shared by all backends.
///
/// Handles: build search space → init runtime → set device ptrs → set dummy
/// data → search → load weights → wrap as `Box<dyn DynBackend>`.
///
/// Backend-specific behavior is injected via callbacks:
/// - `init`: create the concrete runtime
/// - `set_raw`: upload raw bytes + dtype to a node
/// - `set_device_ptr`: optional zero-copy device pointer setter
/// - `wrap`: wrap the final runtime in a `Box<dyn DynBackend>`
pub fn compile_backend<Rt: Runtime + 'static>(
    graph: &mut Graph,
    args: BackendCompileArgs,
    init: impl FnOnce() -> Result<Rt, String>,
    set_raw: impl Fn(&mut Rt, NodeIndex, Vec<u8>, DType),
    set_device_ptr: Option<SetDevicePtrFn<'_, Rt>>,
    wrap: impl FnOnce(Rt) -> Box<dyn DynBackend>,
) -> Result<Box<dyn DynBackend>, String> {
    // Build label map from input_meta (plain data — no downcast needed,
    // survives cross-binary type identity mismatches with external plugins).
    let label_map = build_label_map(graph);

    graph.build_search_space::<Rt>();

    let mut rt = init()?;

    // Set device pointers for zero-copy weights (GPU backends)
    let mut device_ptr_nodes = rustc_hash::FxHashSet::default();
    if let Some(set_ptr) = set_device_ptr {
        for (label, &(ptr, n_bytes)) in &args.device_ptrs {
            if let Some(&node_id) = label_map.get(label) {
                set_ptr(&mut rt, node_id, ptr, n_bytes);
                device_ptr_nodes.insert(node_id);
            }
        }
    }

    // Set dummy ones for Input nodes (required for search profiling).
    // Must use 1, NOT 0 — zero inputs cause NaN in many ops.
    for (&node_id, (label, dtype)) in &graph.input_meta {
        if device_ptr_nodes.contains(&node_id) {
            continue;
        }
        if let Some(&n) = args.tensor_sizes.get(label) {
            if n > 0 {
                set_raw(&mut rt, node_id, make_ones_bytes(n, *dtype), *dtype);
            }
        }
    }

    // Search
    let mut rt = graph.search(rt, args.search_iters);

    // Rebuild label map after search (graph may have changed)
    let label_map = build_label_map(graph);

    // Load real weights post-search (skip device-ptr weights)
    for (label, bytes, dtype) in &args.weights {
        if !args.device_ptrs.contains_key(label) {
            if let Some(&node_id) = label_map.get(label) {
                set_raw(&mut rt, node_id, bytes.clone(), *dtype);
            }
        }
    }

    Ok(wrap(rt))
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/// Build a `label → NodeIndex` map for all Input nodes in the graph.
///
/// Uses `graph.input_meta` (plain data) rather than downcasting, so it works
/// correctly when the graph was built by a different compilation unit (e.g.
/// an external backend plugin compiled as a separate wheel).
pub fn build_label_map(graph: &Graph) -> HashMap<String, NodeIndex> {
    graph
        .input_meta
        .iter()
        .map(|(&node_id, (label, _))| (label.clone(), node_id))
        .collect()
}

/// Create a byte buffer of `n_elements` ones for the given dtype.
///
/// IMPORTANT: Must use 1, NOT 0 — zero inputs cause NaN in many ops
/// (fmod, recip, log, etc.) during search profiling.
pub fn make_ones_bytes(n_elements: usize, dtype: DType) -> Vec<u8> {
    // Safety: all source types have defined bit representations; we just
    // reinterpret the backing Vec<u8> without changing the allocation.
    unsafe fn as_bytes<T>(v: Vec<T>) -> Vec<u8> {
        let mut v = std::mem::ManuallyDrop::new(v);
        let ptr = v.as_mut_ptr() as *mut u8;
        let len = v.len() * std::mem::size_of::<T>();
        unsafe { Vec::from_raw_parts(ptr, len, len) }
    }
    match dtype {
        DType::F32 | DType::TF32 => unsafe { as_bytes(vec![1.0f32; n_elements]) },
        DType::F64 => unsafe { as_bytes(vec![1.0f64; n_elements]) },
        DType::F16 => unsafe { as_bytes(vec![f16::from_f32(1.0); n_elements]) },
        DType::Bf16 => unsafe { as_bytes(vec![bf16::from_f32(1.0); n_elements]) },
        DType::Int => unsafe { as_bytes(vec![1i32; n_elements]) },
        DType::I16 => unsafe { as_bytes(vec![1i16; n_elements]) },
        DType::U16 => unsafe { as_bytes(vec![1u16; n_elements]) },
        _ => vec![1u8; n_elements], // I8, U8, Bool, sub-byte types
    }
}

/// Convert raw bytes + [`DType`] to [`NativeData`].
pub fn bytes_to_native_data(bytes: Vec<u8>, dtype: DType) -> NativeData {
    // Safety: source bytes are from a valid typed buffer; we reinterpret.
    unsafe fn from_bytes<T: Copy>(bytes: Vec<u8>) -> Vec<T> {
        let n = bytes.len() / std::mem::size_of::<T>();
        let mut bytes = std::mem::ManuallyDrop::new(bytes);
        unsafe { Vec::from_raw_parts(bytes.as_mut_ptr() as *mut T, n, n) }
    }
    match dtype {
        DType::F32 | DType::TF32 => NativeData::F32(unsafe { from_bytes(bytes) }),
        DType::F64 => {
            let f64s: Vec<f64> = unsafe { from_bytes(bytes) };
            NativeData::F32(f64s.into_iter().map(|v| v as f32).collect())
        }
        DType::F16 => NativeData::F16(unsafe { from_bytes(bytes) }),
        DType::Bf16 => NativeData::Bf16(unsafe { from_bytes(bytes) }),
        DType::Int => NativeData::Int(unsafe { from_bytes(bytes) }),
        DType::Bool => NativeData::Bool(bytes.into_iter().map(|b| b != 0).collect()),
        DType::I8 => NativeData::Int(bytes.iter().map(|&b| b as i8 as i32).collect()),
        DType::U8 => NativeData::Int(bytes.iter().map(|&b| b as i32).collect()),
        DType::I16 => {
            let i16s: Vec<i16> = unsafe { from_bytes(bytes) };
            NativeData::Int(i16s.into_iter().map(|v| v as i32).collect())
        }
        DType::U16 => {
            let u16s: Vec<u16> = unsafe { from_bytes(bytes) };
            NativeData::Int(u16s.into_iter().map(|v| v as i32).collect())
        }
        _ => NativeData::F32(vec![]),
    }
}

// ---------------------------------------------------------------------------
// NativeDynBackend
// ---------------------------------------------------------------------------

/// [`DynBackend`] wrapper for the native (CPU) runtime.
pub struct NativeDynBackend {
    pub runtime: NativeRuntime,
}

impl DynBackend for NativeDynBackend {
    fn name(&self) -> &str {
        "native"
    }

    fn set_data_bytes(&mut self, node: NodeIndex, bytes: Vec<u8>, dtype: DType) {
        self.runtime
            .set_data(node, bytes_to_native_data(bytes, dtype));
    }

    fn set_data_f32(&mut self, node: NodeIndex, data: Vec<f32>) {
        self.runtime.set_data(node, data);
    }

    fn get_output_f32(&self, node: NodeIndex) -> Vec<f32> {
        let data = self.output_buffer(node);
        (0..data.len()).map(|i| data.f32(i)).collect()
    }

    fn get_output_i32(&self, node: NodeIndex) -> Vec<i32> {
        let data = self.output_buffer(node);
        (0..data.len()).map(|i| data.i32(i)).collect()
    }

    fn get_output_bool(&self, node: NodeIndex) -> Vec<bool> {
        let data = self.output_buffer(node);
        (0..data.len()).map(|i| data.bool(i)).collect()
    }

    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) {
        self.runtime.execute(dyn_map);
    }
}

impl NativeDynBackend {
    fn output_buffer(&self, node: NodeIndex) -> &NativeData {
        let output_id = self
            .runtime
            .graph
            .node_indices()
            .find(|n| {
                (**self.runtime.graph[*n])
                    .as_any()
                    .downcast_ref::<Output>()
                    .is_some_and(|out| out.node == node.index())
            })
            .unwrap_or_else(|| panic!("No output node found for {:?}", node));
        self.runtime
            .buffers
            .get(&output_id)
            .unwrap_or_else(|| panic!("No buffer data for output {:?}", node))
    }
}

pub fn native_factory(
    graph: &mut Graph,
    args: BackendCompileArgs,
) -> Result<Box<dyn DynBackend>, String> {
    compile_backend::<NativeRuntime>(
        graph,
        args,
        || Ok(NativeRuntime::default()),
        // NativeRuntime::set_data requires the LLIR graph to be loaded (it searches
        // for Input nodes in the LLIR). Before search, the LLIR is empty. We guard
        // against that: if rt.graph is empty, skip (dummy data isn't needed for
        // native since its profile is a no-op).
        |rt, node, bytes, dtype| {
            if rt.graph.node_count() > 0 {
                rt.set_data(node, bytes_to_native_data(bytes, dtype));
            }
        },
        None,
        |rt| Box::new(NativeDynBackend { runtime: rt }),
    )
}
