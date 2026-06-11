pub mod find_indptrs;
pub mod jit;

use std::sync::{Arc, Mutex, OnceLock};

use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{EXPRESSION, OP_KIND},
        extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::{
        tracing::{Level, span},
        *,
    },
};

use crate::{
    cudarc::driver::{CudaSlice, CudaStream, DevicePtr, result},
    host::{DeviceBuffer, HostOp},
};

/// FlashInfer attention op (batch decode, fp32).
///
/// Replaces the full paged-GQA attention pattern (gather → broadcast → Q*K^T →
/// scale → mask → softmax → *V) with a single FlashInfer fused kernel.
///
/// Runtime graph inputs: Q, K_pool, V_pool, compact gather_idx, and optionally
/// qo_indptr + kv_indptr for standalone multi-sequence execution. The additive
/// causal mask is only an egglog proof anchor and is not a runtime dependency.
#[derive(Debug)]
pub struct FlashInferAttention {
    pub num_qo_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub page_size: usize,
    pub batch_dim: Expression,

    pub plan_info: Mutex<Vec<i64>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlashInferDecodeSpec {
    total_q_tokens: usize,
    batch_size: usize,
    c: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    head_dim: usize,
    kv_dim: usize,
    max_kv_pages: usize,
    kv_indptr_host: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashInferDecodePointers {
    q: u64,
    k_cache: u64,
    v_cache: u64,
    gather_idx: u64,
    output: u64,
    pub(crate) explicit_kv_indptr: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlashInferDecodeCaptureSignature {
    pub(crate) spec: FlashInferDecodeSpec,
    pub(crate) ptrs: FlashInferDecodePointers,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlashInferResolvedDecode {
    spec: FlashInferDecodeSpec,
    ptrs: FlashInferDecodePointers,
}

impl FlashInferResolvedDecode {
    pub(crate) fn has_explicit_indptr(&self) -> bool {
        self.ptrs.explicit_kv_indptr.is_some()
    }

    pub(crate) fn current_c(&self) -> usize {
        self.spec.c
    }

    pub(crate) fn graph_plan_capacity(&self, existing_capacity: Option<usize>) -> usize {
        if self.has_explicit_indptr() || self.spec.total_q_tokens != 1 {
            return self.spec.c;
        }
        if let Some(capacity) = existing_capacity
            && self.spec.c <= capacity
        {
            return capacity;
        }
        flashinfer_graph_plan_capacity(self.spec.c, self.spec.max_kv_pages)
    }

    pub(crate) fn signature_for_graph_plan(
        &self,
        plan_c: usize,
    ) -> FlashInferDecodeCaptureSignature {
        let mut spec = self.spec.clone();
        let ptrs = self.ptrs;
        if !self.has_explicit_indptr() && spec.total_q_tokens == 1 {
            spec.c = plan_c;
            spec.kv_indptr_host = vec![0, plan_c as i32];
        }
        FlashInferDecodeCaptureSignature { spec, ptrs }
    }
}

pub(crate) struct PreparedFlashInferDecode {
    lib: &'static jit::FlashInferLib,
    spec: FlashInferDecodeSpec,
    plan_info: Vec<i64>,
    _float_workspace: &'static CudaSlice<u8>,
    float_workspace_ptr: u64,
    _int_workspace: &'static CudaSlice<u8>,
    int_workspace_ptr: u64,
    _page_locked_workspace_ptr: *mut u8,
    _owned_kv_indptr: Option<CudaSlice<i32>>,
    owned_kv_indptr_ptr: Option<u64>,
    current_c: Option<Mutex<CudaSlice<i32>>>,
    current_c_ptr: Option<u64>,
    _indices: CudaSlice<i32>,
    indices_ptr: u64,
    _last_page_len: CudaSlice<i32>,
    last_page_len_ptr: u64,
    _temp_output: CudaSlice<f32>,
    temp_output_ptr: u64,
}

// SAFETY: PreparedFlashInferDecode owns CUDA device allocations and a process-lifetime
// library handle. Calls are serialized through the CUDA stream that captures or
// launches the graph.
unsafe impl Send for PreparedFlashInferDecode {}
unsafe impl Sync for PreparedFlashInferDecode {}

// SAFETY: PAGE_LOCKED_WORKSPACE holds a raw pointer to page-locked CUDA memory
// allocated once and serialized via the CUDA stream that owns it.
unsafe impl Send for FlashInferAttention {}
unsafe impl Sync for FlashInferAttention {}

const FLOAT_WORKSPACE_SIZE: usize = 128 * 1024 * 1024; // 128 MiB
const INT_WORKSPACE_SIZE: usize = 8 * 1024 * 1024; // 8 MiB

static PAGE_LOCKED_WORKSPACE: OnceLock<PageLockedPtr> = OnceLock::new();

struct PageLockedPtr(*mut u8);

// SAFETY: The pointer is page-locked CUDA memory allocated once via
// posix_memalign + cudaHostRegister and only mutated during OnceLock
// initialization.
unsafe impl Send for PageLockedPtr {}
unsafe impl Sync for PageLockedPtr {}

impl std::fmt::Debug for PageLockedPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PageLockedPtr({:p})", self.0)
    }
}

impl Default for FlashInferAttention {
    fn default() -> Self {
        Self {
            num_qo_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            page_size: 0,
            batch_dim: Expression::default(),
            plan_info: Mutex::new(Vec::new()),
        }
    }
}

impl EgglogOp for FlashInferAttention {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FlashInferAttention",
            &[
                ("num_qo_heads", EXPRESSION),
                ("num_kv_heads", EXPRESSION),
                ("head_dim", EXPRESSION),
                ("page_size", EXPRESSION),
                ("batch_dim", EXPRESSION),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        // Q, K_pool, V_pool, compact gather_idx. The egglog rules still use
        // flat gather indices and masks as structural proof anchors, but
        // extract() returns only the runtime inputs FlashInfer actually uses.
        5
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(include_str!["flashinfer_attention.egg"])]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let num_qo_heads = extract_expr(egraph, kind_children[0], expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let num_kv_heads = extract_expr(egraph, kind_children[1], expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let head_dim = extract_expr(egraph, kind_children[2], expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let page_size = extract_expr(egraph, kind_children[3], expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let batch_dim = extract_expr(egraph, kind_children[4], expr_cache).unwrap();

        let extracted = Self {
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            batch_dim,
            plan_info: Mutex::new(Vec::new()),
        };

        // Trigger JIT compilation (or .so cache hit) at extract time, not at
        // first execute. Pays the ~30s cold-cache nvcc cost during compile
        // rather than during the GA profiling loop, where it would dominate
        // the candidate's measured runtime and make the GA reject FlashInfer.
        let _ = jit::ensure_compiled(head_dim);

        let flat_idx_node = input_enodes[3];
        let gather_idx = find_indptrs::try_find_compact_gather_idx(egraph, flat_idx_node)
            .expect("FlashInferAttention matched a gather without recoverable compact gather_idx");
        let final_inputs = vec![
            input_enodes[0],
            input_enodes[1],
            input_enodes[2],
            gather_idx,
        ];

        let op = LLIROp::new::<dyn HostOp>(Box::new(extracted) as Box<dyn HostOp>);
        (op, final_inputs)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl FlashInferAttention {
    pub(crate) fn graph_inputs(&self) -> usize {
        4
    }

    pub(crate) fn resolve_for_graph(
        &self,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<FlashInferResolvedDecode> {
        let total_q_tokens = self
            .batch_dim
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FlashInferAttention batch_dim is unresolved"))?;
        let c = *dyn_map
            .get(&'c')
            .ok_or_else(|| anyhow::anyhow!("FlashInferAttention requires dynamic dim 'c'"))?;
        if inputs.len() != 4 && inputs.len() != 6 {
            anyhow::bail!(
                "FlashInferAttention expects 4 inputs (derived causal decode) or 6 inputs (explicit indptrs), got {}",
                inputs.len()
            );
        }

        let get_buf = |name: &str, node: NodeIndex| -> anyhow::Result<DeviceBuffer> {
            buffers.get(&node).copied().ok_or_else(|| {
                anyhow::anyhow!("FlashInferAttention missing {name} buffer for {node:?}")
            })
        };

        let q_buf = get_buf("Q", inputs[0])?;
        let k_buf = get_buf("K_cache", inputs[1])?;
        let v_buf = get_buf("V_cache", inputs[2])?;
        let gather_idx_buf = get_buf("gather_idx", inputs[3])?;
        let out_buf = get_buf("output", self_node)?;

        let kv_dim = self.num_kv_heads * self.head_dim;
        let kv_bytes = kv_dim * std::mem::size_of::<f32>();
        let max_kv_pages = k_buf
            .len()
            .checked_div(kv_bytes)
            .zip(v_buf.len().checked_div(kv_bytes))
            .map(|(k_pages, v_pages)| k_pages.min(v_pages).max(c))
            .unwrap_or(c);
        let (kv_indptr_host, batch_size, explicit_kv_indptr) = if inputs.len() >= 6 {
            let r = *dyn_map
                .get(&'r')
                .ok_or_else(|| anyhow::anyhow!("FlashInferAttention requires dynamic dim 'r'"))?;
            let kv_indptr_buf = get_buf("kv_indptr", inputs[5])?;
            // Host contents are read during prepare, where stream synchronization
            // is allowed. The pointer is part of the capture signature.
            (
                Vec::with_capacity(r),
                r.saturating_sub(1),
                Some(kv_indptr_buf.ptr()),
            )
        } else {
            (Vec::new(), total_q_tokens, None)
        };

        Ok(FlashInferResolvedDecode {
            spec: FlashInferDecodeSpec {
                total_q_tokens,
                batch_size,
                c,
                num_qo_heads: self.num_qo_heads,
                num_kv_heads: self.num_kv_heads,
                page_size: self.page_size,
                head_dim: self.head_dim,
                kv_dim,
                max_kv_pages,
                kv_indptr_host,
            },
            ptrs: FlashInferDecodePointers {
                q: q_buf.ptr(),
                k_cache: k_buf.ptr(),
                v_cache: v_buf.ptr(),
                gather_idx: gather_idx_buf.ptr(),
                output: out_buf.ptr(),
                explicit_kv_indptr,
            },
        })
    }

    pub(crate) fn prepare_resolved_for_graph(
        &self,
        stream: &Arc<CudaStream>,
        mut resolved: FlashInferResolvedDecode,
        enable_cuda_graph: bool,
    ) -> anyhow::Result<PreparedFlashInferDecode> {
        let lib = jit::ensure_compiled(self.head_dim);
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;
        let spec = &mut resolved.spec;

        let mut current_c = None;
        let mut current_c_ptr = None;

        let (owned_kv_indptr, owned_kv_indptr_ptr) = if let Some(kv_indptr_ptr) =
            resolved.ptrs.explicit_kv_indptr
        {
            let r = spec.batch_size + 1;
            let mut kv_indptr_host_bytes = vec![0u8; r * std::mem::size_of::<i32>()];
            unsafe {
                result::memcpy_dtoh_async(
                    &mut kv_indptr_host_bytes,
                    kv_indptr_ptr,
                    stream.cu_stream(),
                )?;
            }
            stream.synchronize()?;
            spec.kv_indptr_host = bytes_to_i32_vec(kv_indptr_host_bytes);
            (None, None)
        } else if enable_cuda_graph && spec.total_q_tokens == 1 {
            let actual_c = spec.c;
            let plan_c = flashinfer_graph_plan_capacity(actual_c, spec.max_kv_pages);
            spec.c = plan_c;
            spec.kv_indptr_host = vec![0, plan_c as i32];
            let dev = stream.clone_htod(&[0i32, actual_c as i32])?;
            let ptr = dev.device_ptr(stream).0;
            let current = stream.clone_htod(&[actual_c as i32])?;
            current_c_ptr = Some(current.device_ptr(stream).0);
            current_c = Some(Mutex::new(current));
            (Some(dev), Some(ptr))
        } else {
            if enable_cuda_graph {
                anyhow::bail!(
                    "FlashInfer graph capture for derived causal decode currently requires s=1, got s={}",
                    spec.total_q_tokens
                );
            }
            if spec.total_q_tokens != 1 {
                anyhow::bail!(
                    "FlashInfer derived causal decode without explicit indptrs requires s=1, got s={}",
                    spec.total_q_tokens
                );
            }
            spec.kv_indptr_host = vec![0, spec.c as i32];
            let dev = stream.clone_htod(&spec.kv_indptr_host)?;
            let ptr = dev.device_ptr(stream).0;
            (Some(dev), Some(ptr))
        };

        let total_pages = spec.kv_indptr_host.last().copied().unwrap_or_default();
        if total_pages < 0 || total_pages as usize > spec.c {
            anyhow::bail!(
                "FlashInfer derived decode describes {total_pages} KV pages, but compact gather_idx has only {}; fp32 FlashInfer currently supports decode-style causal masks, not single-sequence prefill",
                spec.c
            );
        }

        let indices = unsafe { stream.alloc::<i32>(spec.c.max(1))? };
        let indices_ptr = indices.device_ptr(stream).0;
        let last_page_len_host = vec![1i32; spec.batch_size];
        let last_page_len = if last_page_len_host.is_empty() {
            unsafe { stream.alloc::<i32>(1)? }
        } else {
            stream.clone_htod(&last_page_len_host)?
        };
        let last_page_len_ptr = last_page_len.device_ptr(stream).0;
        let temp_output = unsafe {
            stream.alloc::<f32>((spec.total_q_tokens * spec.num_qo_heads * spec.head_dim).max(1))?
        };
        let temp_output_ptr = temp_output.device_ptr(stream).0;

        let (float_workspace, float_workspace_ptr, int_workspace, int_workspace_ptr) =
            flashinfer_workspaces(stream);
        let page_locked_workspace = PAGE_LOCKED_WORKSPACE.get_or_init(|| unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let status = libc::posix_memalign(&mut ptr, 4096, INT_WORKSPACE_SIZE);
            assert_eq!(status, 0, "Failed to allocate page-locked workspace");
            let cuda_status = cuda_pin_memory(ptr, INT_WORKSPACE_SIZE);
            assert_eq!(cuda_status, 0, "Failed to pin memory");
            PageLockedPtr(ptr as *mut u8)
        });

        let mut plan_info_buf = [0i64; 16];
        let mut plan_info_len: i32 = 0;
        let plan_ret = unsafe {
            (lib.plan)(
                float_workspace_ptr as *mut std::ffi::c_void,
                FLOAT_WORKSPACE_SIZE,
                int_workspace_ptr as *mut std::ffi::c_void,
                INT_WORKSPACE_SIZE,
                page_locked_workspace.0 as *mut std::ffi::c_void,
                spec.kv_indptr_host.as_mut_ptr(),
                spec.batch_size as i32,
                spec.num_qo_heads as i32,
                spec.num_kv_heads as i32,
                spec.page_size as i32,
                spec.head_dim as i32,
                enable_cuda_graph,
                cu_stream,
                plan_info_buf.as_mut_ptr(),
                &mut plan_info_len,
            )
        };
        if plan_ret != 0 {
            return Err(anyhow::anyhow!(
                "FlashInfer decode plan failed with error code {plan_ret}"
            ));
        }

        Ok(PreparedFlashInferDecode {
            lib,
            spec: spec.clone(),
            plan_info: plan_info_buf[..plan_info_len as usize].to_vec(),
            _float_workspace: float_workspace,
            float_workspace_ptr,
            _int_workspace: int_workspace,
            int_workspace_ptr,
            _page_locked_workspace_ptr: page_locked_workspace.0,
            _owned_kv_indptr: owned_kv_indptr,
            owned_kv_indptr_ptr,
            current_c,
            current_c_ptr,
            _indices: indices,
            indices_ptr,
            _last_page_len: last_page_len,
            last_page_len_ptr,
            _temp_output: temp_output,
            temp_output_ptr,
        })
    }
}

impl PreparedFlashInferDecode {
    pub(crate) fn plan_c(&self) -> usize {
        self.spec.c
    }

    pub(crate) fn update_current_c(
        &self,
        stream: &Arc<CudaStream>,
        c: usize,
    ) -> anyhow::Result<()> {
        if let Some(current_c) = &self.current_c {
            let mut current_c = current_c
                .lock()
                .map_err(|_| anyhow::anyhow!("FlashInfer current_c lock poisoned"))?;
            stream.memcpy_htod(&[c as i32], &mut *current_c)?;
        }
        Ok(())
    }

    pub(crate) fn enqueue(
        &self,
        stream: &Arc<CudaStream>,
        ptrs: FlashInferDecodePointers,
    ) -> anyhow::Result<()> {
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;
        let kv_indptr_ptr = ptrs
            .explicit_kv_indptr
            .or(self.owned_kv_indptr_ptr)
            .ok_or_else(|| anyhow::anyhow!("FlashInfer decode is missing kv_indptr pointer"))?;

        let mut plan_info = self.plan_info.clone();
        if let Some(current_c_ptr) = self.current_c_ptr {
            unsafe {
                (self.lib.prepare_decode_metadata)(
                    self.int_workspace_ptr as *mut std::ffi::c_void,
                    plan_info.as_mut_ptr(),
                    plan_info.len() as i32,
                    current_c_ptr as *const i32,
                    ptrs.gather_idx as *const i32,
                    self.indices_ptr as *mut i32,
                    kv_indptr_ptr as *mut i32,
                    self.spec.c as i32,
                    self.spec.kv_dim as i32,
                    cu_stream,
                );
            }
        } else if self.spec.c > 0 {
            unsafe {
                (self.lib.extract_slot_indices)(
                    ptrs.gather_idx as *const i32,
                    self.indices_ptr as *mut i32,
                    self.spec.c as i32,
                    self.spec.kv_dim as i32,
                    cu_stream,
                );
            }
        }

        let run_ret = unsafe {
            (self.lib.run)(
                self.float_workspace_ptr as *mut std::ffi::c_void,
                FLOAT_WORKSPACE_SIZE,
                self.int_workspace_ptr as *mut std::ffi::c_void,
                plan_info.as_mut_ptr(),
                plan_info.len() as i32,
                ptrs.q as *mut f32,
                ptrs.k_cache as *mut f32,
                ptrs.v_cache as *mut f32,
                kv_indptr_ptr as *mut i32,
                self.indices_ptr as *mut i32,
                self.last_page_len_ptr as *mut i32,
                self.temp_output_ptr as *mut f32,
                self.spec.batch_size as i32,
                self.spec.num_qo_heads as i32,
                self.spec.num_kv_heads as i32,
                self.spec.page_size as i32,
                self.spec.head_dim as i32,
                cu_stream,
            )
        };

        if run_ret != 0 {
            return Err(anyhow::anyhow!(
                "FlashInfer decode run failed with error code {run_ret}"
            ));
        }

        unsafe {
            (self.lib.transpose_output)(
                self.temp_output_ptr as *const f32,
                ptrs.output as *mut f32,
                self.spec.total_q_tokens as i32,
                self.spec.num_qo_heads as i32,
                self.spec.head_dim as i32,
                cu_stream,
            );
        }

        Ok(())
    }
}

pub(crate) fn flashinfer_graph_plan_capacity(actual_c: usize, max_kv_pages: usize) -> usize {
    let required = actual_c.max(1);
    if let Some(capacity) = std::env::var("LUMINAL_FLASHINFER_DECODE_GRAPH_CAPACITY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
    {
        return capacity.max(required);
    }
    max_kv_pages.max(required)
}

impl HostOp for FlashInferAttention {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let resolved = self.resolve_for_graph(self_node, inputs, buffers, dyn_map)?;
        let ptrs = resolved.ptrs;
        let prepared = self.prepare_resolved_for_graph(stream, resolved, false)?;

        let _span = span!(
            Level::TRACE,
            "FlashInferAttention",
            prepared.spec.total_q_tokens,
            prepared.spec.batch_size,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        .entered();
        prepared.enqueue(stream, ptrs)
    }

    fn output_size(&self) -> Expression {
        self.batch_dim * self.num_qo_heads * self.head_dim
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("FlashInferAttention")
    }
}

/// Pin host memory for CUDA async memcpy.
///
/// `cudaHostRegister` lives in libcudart, which cudarc doesn't link to our
/// binary. Resolve it via `dlopen`/`dlsym` so we don't need a build script or
/// a `#[link]` directive — keeping the crate buildable without any nvcc-side
/// dependencies.
fn flashinfer_workspaces(
    stream: &Arc<CudaStream>,
) -> (&'static CudaSlice<u8>, u64, &'static CudaSlice<u8>, u64) {
    static FLOAT_WORKSPACE: OnceLock<CudaSlice<u8>> = OnceLock::new();
    static INT_WORKSPACE: OnceLock<CudaSlice<u8>> = OnceLock::new();

    let float_ws = FLOAT_WORKSPACE
        .get_or_init(|| unsafe { stream.alloc::<u8>(FLOAT_WORKSPACE_SIZE).unwrap() });
    let int_ws =
        INT_WORKSPACE.get_or_init(|| unsafe { stream.alloc::<u8>(INT_WORKSPACE_SIZE).unwrap() });
    let float_ptr = float_ws.device_ptr(stream).0;
    let int_ptr = int_ws.device_ptr(stream).0;
    (float_ws, float_ptr, int_ws, int_ptr)
}

fn bytes_to_i32_vec(bytes: Vec<u8>) -> Vec<i32> {
    let len = bytes.len() / std::mem::size_of::<i32>();
    let mut bytes = std::mem::ManuallyDrop::new(bytes);
    unsafe { Vec::from_raw_parts(bytes.as_mut_ptr() as *mut i32, len, len) }
}

unsafe fn cuda_pin_memory(ptr: *mut std::ffi::c_void, size: usize) -> i32 {
    type HostRegisterFn = unsafe extern "C" fn(*mut std::ffi::c_void, usize, u32) -> i32;
    static FN: OnceLock<usize> = OnceLock::new();

    let raw = *FN.get_or_init(|| unsafe {
        let lib = [
            "libcudart.so",
            "libcudart.so.13",
            "libcudart.so.12",
            "libcudart.so.11",
        ]
        .iter()
        .find_map(|n| libloading::Library::new(*n).ok())
        .expect("FlashInfer: could not dlopen libcudart for cudaHostRegister");
        let sym: libloading::Symbol<HostRegisterFn> = lib
            .get(b"cudaHostRegister\0")
            .expect("FlashInfer: libcudart missing cudaHostRegister symbol");
        let ptr = *sym as *const () as usize;
        // Keep libcudart resident for the process lifetime so the function
        // pointer remains valid.
        std::mem::forget(lib);
        ptr
    });
    let f: HostRegisterFn = unsafe { std::mem::transmute(raw) };
    // cudaHostRegisterDefault = 0
    unsafe { f(ptr, size, 0) }
}
