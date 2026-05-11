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
/// Graph inputs (7): Q, K_pool, V_pool, flat_gather_idx, mask, qo_indptr, kv_indptr.
/// The egglog rule captures the first 5; `extract()` appends qo/kv indptrs after
/// walking the e-graph from the mask. `batch_size` is derived at runtime from the
/// indptr length (= num_sequences + 1).
#[derive(Debug)]
pub struct FlashInferAttention {
    pub num_qo_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub page_size: usize,
    pub batch_dim: Expression,

    pub plan_info: Mutex<Vec<i64>>,
}

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
        // Q, K_pool, V_pool, flat_gather_idx, mask (egglog IList).
        // extract() appends qo_indptr + kv_indptr → 7 actual inputs at runtime.
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

        // Walk the mask e-graph chain to recover qo_indptr / kv_indptr Input nodes.
        // input_enodes: [Q, K_cache, V_cache, gather_idx, mask]
        let mask_node = input_enodes[4];
        let indptrs = find_indptrs::find_indptr_inputs(egraph, mask_node);

        // Build final inputs: [Q, K_cache, V_cache, gather_idx, mask, qo_indptr, kv_indptr]
        let mut final_inputs = input_enodes;
        final_inputs.push(indptrs.qo_indptr);
        final_inputs.push(indptrs.kv_indptr);

        let op = LLIROp::new::<dyn HostOp>(Box::new(extracted) as Box<dyn HostOp>);
        (op, final_inputs)
    }

    fn cleanup(&self) -> bool {
        false
    }
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
        let lib = jit::ensure_compiled(self.head_dim);

        let total_q_tokens = self
            .batch_dim
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FlashInferAttention batch_dim is unresolved"))?;
        let c = *dyn_map
            .get(&'c')
            .ok_or_else(|| anyhow::anyhow!("FlashInferAttention requires dynamic dim 'c'"))?;
        let r = *dyn_map
            .get(&'r')
            .ok_or_else(|| anyhow::anyhow!("FlashInferAttention requires dynamic dim 'r'"))?;

        if inputs.len() < 7 {
            anyhow::bail!(
                "FlashInferAttention expects 7 inputs (Q, K, V, flat_idx, mask, qo_indptr, kv_indptr), got {}",
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
        let flat_idx_buf = get_buf("flat_gather_idx", inputs[3])?;
        // inputs[4] = mask (unused by FlashInfer — indptrs replace it)
        let kv_indptr_buf = get_buf("kv_indptr", inputs[6])?;
        let out_buf = get_buf("output", self_node)?;

        // Derive batch_size (num sequences) from r = indptr length.
        let batch_size = r.saturating_sub(1);

        let _span = span!(
            Level::TRACE,
            "FlashInferAttention",
            total_q_tokens,
            batch_size,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        .entered();

        let kv_dim = self.num_kv_heads * self.head_dim;
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

        // Extract slot indices (one per context page) from the flat gather index.
        let indices_buf = unsafe { stream.alloc::<u8>(c.max(1) * std::mem::size_of::<i32>())? };
        let (indices_ptr, _idx_guard) = indices_buf.device_ptr(stream);

        if c > 0 {
            unsafe {
                (lib.extract_slot_indices)(
                    flat_idx_buf.ptr() as *const i32,
                    indices_ptr as *mut i32,
                    c as i32,
                    kv_dim as i32,
                    cu_stream,
                );
            }
        }

        // Read kv_indptr to host for the plan phase.
        let kv_indptr_bytes = r * 4;
        let mut kv_indptr_host_bytes = vec![0u8; kv_indptr_bytes];
        unsafe {
            result::memcpy_dtoh_async(
                &mut kv_indptr_host_bytes,
                kv_indptr_buf.ptr(),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()?;
        let kv_indptr_host: Vec<i32> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(kv_indptr_host_bytes);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut i32, r, r)
        };

        // kv_last_page_len = [1; batch_size] when page_size=1.
        let last_page_host: Vec<i32> = vec![1; batch_size];
        let last_page_dev: CudaSlice<u8> = if batch_size > 0 {
            stream.clone_htod(unsafe {
                std::slice::from_raw_parts(
                    last_page_host.as_ptr() as *const u8,
                    last_page_host.len() * std::mem::size_of::<i32>(),
                )
            })?
        } else {
            unsafe { stream.alloc::<u8>(1)? }
        };
        let (last_page_ptr, _lp_guard) = last_page_dev.device_ptr(stream);

        // Global shared workspaces (allocated once across all op instances to
        // amortize the ~4ms first-allocation cost during GA profiling).
        static FLOAT_WORKSPACE: OnceLock<CudaSlice<u8>> = OnceLock::new();
        static INT_WORKSPACE: OnceLock<CudaSlice<u8>> = OnceLock::new();
        let float_ws = FLOAT_WORKSPACE
            .get_or_init(|| unsafe { stream.alloc::<u8>(FLOAT_WORKSPACE_SIZE).unwrap() });
        let int_ws = INT_WORKSPACE
            .get_or_init(|| unsafe { stream.alloc::<u8>(INT_WORKSPACE_SIZE).unwrap() });
        let page_locked_ws = PAGE_LOCKED_WORKSPACE.get_or_init(|| unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let status = libc::posix_memalign(&mut ptr, 4096, INT_WORKSPACE_SIZE);
            assert_eq!(status, 0, "Failed to allocate page-locked workspace");
            let cuda_status = cuda_pin_memory(ptr, INT_WORKSPACE_SIZE);
            assert_eq!(cuda_status, 0, "Failed to pin memory");
            PageLockedPtr(ptr as *mut u8)
        });

        let (float_ws_ptr, _fws_guard) = float_ws.device_ptr(stream);
        let (int_ws_ptr, _iws_guard) = int_ws.device_ptr(stream);

        // FlashInfer decode writes (total_q_tokens, heads, dim);
        // luminal expects (heads, total_q_tokens, dim) — transpose at the end.
        let output_elems = total_q_tokens * self.num_qo_heads * self.head_dim;
        let temp_out_buf =
            unsafe { stream.alloc::<u8>(output_elems * std::mem::size_of::<f32>())? };
        let (temp_out_ptr, _tmp_guard) = temp_out_buf.device_ptr(stream);

        // PrefillPlanInfo has 15 entries, DecodePlanInfo fewer — 16 is enough.
        let mut plan_info_buf = [0i64; 16];
        let mut plan_info_len: i32 = 0;

        // ── BatchDecode path ──
        // Prefill kernels require fp16/bf16 tensor-core MMA; the C API returns -1
        // when called from the fp32 pipeline. We only use decode here.
        let plan_ret = unsafe {
            (lib.plan)(
                float_ws_ptr as *mut std::ffi::c_void,
                FLOAT_WORKSPACE_SIZE,
                int_ws_ptr as *mut std::ffi::c_void,
                INT_WORKSPACE_SIZE,
                page_locked_ws.0 as *mut std::ffi::c_void,
                kv_indptr_host.as_ptr() as *mut i32,
                batch_size as i32,
                self.num_qo_heads as i32,
                self.num_kv_heads as i32,
                self.page_size as i32,
                self.head_dim as i32,
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

        let mut plan_info = self.plan_info.lock().unwrap();
        plan_info.clear();
        plan_info.extend_from_slice(&plan_info_buf[..plan_info_len as usize]);

        let run_ret = unsafe {
            (lib.run)(
                float_ws_ptr as *mut std::ffi::c_void,
                FLOAT_WORKSPACE_SIZE,
                int_ws_ptr as *mut std::ffi::c_void,
                plan_info.as_mut_ptr(),
                plan_info.len() as i32,
                q_buf.ptr() as *mut f32,
                k_buf.ptr() as *mut f32,
                v_buf.ptr() as *mut f32,
                kv_indptr_buf.ptr() as *mut i32,
                indices_ptr as *mut i32,
                last_page_ptr as *mut i32,
                temp_out_ptr as *mut f32,
                batch_size as i32,
                self.num_qo_heads as i32,
                self.num_kv_heads as i32,
                self.page_size as i32,
                self.head_dim as i32,
                cu_stream,
            )
        };
        drop(plan_info);

        if run_ret != 0 {
            return Err(anyhow::anyhow!(
                "FlashInfer decode run failed with error code {run_ret}"
            ));
        }

        // Transpose (total_q_tokens, heads, dim) → (heads, total_q_tokens, dim)
        unsafe {
            (lib.transpose_output)(
                temp_out_ptr as *const f32,
                out_buf.ptr() as *mut f32,
                total_q_tokens as i32,
                self.num_qo_heads as i32,
                self.head_dim as i32,
                cu_stream,
            );
        }

        Ok(())
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
