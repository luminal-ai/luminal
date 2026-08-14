pub mod find_indptrs;
pub mod jit;
pub mod sink_attention;

use std::sync::{Arc, Mutex, OnceLock};

use luminal::{
    dtype::DType,
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, EXPRESSION, F64, OP_KIND},
        extract_dtype, extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::{
        tracing::{Level, span},
        *,
    },
};

use jit::FlashInferDType;

use crate::{
    cudarc::driver::{CudaSlice, CudaStream, DevicePtr, result},
    host::{DeviceBuffer, HostOp},
    resource::{HostDeviceMemoryPlan, ResourceViolation, SharedDeviceMemoryAllocation},
};

/// FlashInfer attention op (batch decode for f32/f16/bf16, batch prefill for
/// f16/bf16).
///
/// Replaces the full paged-GQA attention pattern (gather → broadcast → Q*K^T →
/// scale → mask → softmax → *V) with a single FlashInfer fused kernel. Decode
/// (one q token per sequence) works in every dtype; prefill (multiple q tokens
/// per sequence) requires tensor cores and is therefore 16-bit only.
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
    /// Total tokens attended over. Captured from the matched shape by the
    /// rules, so it does not depend on what the graph calls this dim.
    pub context_dim: Expression,
    pub dtype: DType,
    /// Softmax scale; 0.0 = default `1/sqrt(head_dim)`.
    pub sm_scale: f64,
    /// Sliding-window size in FlashInfer's `window_left` convention
    /// (number of previous kv positions visible); -1 = no window.
    pub window_left: i64,

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
    dtype: FlashInferDType,
    /// f32 bits of the softmax scale actually passed to the kernel.
    sm_scale_bits: u32,
    /// FlashInfer window_left; -1 = no sliding window. Part of the spec so
    /// the shared prepare cache and capture signatures distinguish kernel
    /// variants.
    window_left: i32,
    kv_indptr_host: Vec<i32>,
    qo_indptr_host: Vec<i32>,
}

impl FlashInferDecodeSpec {
    /// Prefill = more q tokens than sequences. Requires 16-bit dtype (the
    /// prepare path rejects f32 prefill before this matters).
    fn is_prefill(&self) -> bool {
        self.total_q_tokens > self.batch_size
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashInferDecodePointers {
    q: u64,
    k_cache: u64,
    v_cache: u64,
    gather_idx: u64,
    output: u64,
    pub(crate) explicit_qo_indptr: Option<u64>,
    pub(crate) explicit_kv_indptr: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlashInferDecodeCaptureSignature {
    pub(crate) spec: FlashInferDecodeSpec,
    pub(crate) ptrs: FlashInferDecodePointers,
}

/// Pointer-free proof that two derived-decode islands may reuse one prepared
/// allocation. Equal shapes are not enough: the mutable metadata buffers are
/// populated from `gather_idx`, so the producer node is part of the key.
/// Explicit-indptr plans deliberately have no key because their device
/// contents can change without their node identities changing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlashInferPrepareKey {
    spec: FlashInferDecodeSpec,
    gather_idx: NodeIndex,
}

impl FlashInferPrepareKey {
    pub(crate) fn for_inputs(spec: FlashInferDecodeSpec, inputs: &[NodeIndex]) -> Option<Self> {
        (inputs.len() == 4).then(|| Self {
            spec,
            gather_idx: inputs[3],
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FlashInferDeviceResourceSpec {
    /// `None` means explicit indptr contents cannot be proven equal without a
    /// forbidden preflight device read, so this plan must not be deduplicated.
    pub(crate) cache_key: Option<FlashInferPrepareKey>,
    spec: FlashInferDecodeSpec,
    explicit_qo_indptr: bool,
    explicit_kv_indptr: bool,
    enable_cuda_graph: bool,
}

impl FlashInferDeviceResourceSpec {
    pub(crate) fn prepared_device_bytes(&self) -> Result<usize, ResourceViolation> {
        prepared_device_bytes(
            &self.spec,
            self.explicit_qo_indptr,
            self.explicit_kv_indptr,
            self.enable_cuda_graph,
        )
    }
}

fn prepared_device_bytes(
    spec: &FlashInferDecodeSpec,
    explicit_qo_indptr: bool,
    explicit_kv_indptr: bool,
    enable_cuda_graph: bool,
) -> Result<usize, ResourceViolation> {
    let is_prefill = spec.is_prefill();
    let mut allocations = Vec::with_capacity(6);

    if !explicit_kv_indptr {
        // Derived prefill owns [0, c]. CUDA-graph decode owns [0, c]
        // plus the one-element current-c update buffer.
        allocations.push(2usize * std::mem::size_of::<i32>());
        if enable_cuda_graph && !is_prefill && spec.total_q_tokens == 1 {
            allocations.push(std::mem::size_of::<i32>());
        }
    }
    if is_prefill && !explicit_qo_indptr {
        allocations.push(2usize * std::mem::size_of::<i32>());
    }
    allocations.push(
        spec.c
            .max(1)
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or(ResourceViolation::ArithmeticOverflow {
                resource: "FlashInfer indices",
            })?,
    );
    allocations.push(
        spec.batch_size
            .max(1)
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or(ResourceViolation::ArithmeticOverflow {
                resource: "FlashInfer last-page lengths",
            })?,
    );
    allocations.push(
        spec.total_q_tokens
            .checked_mul(spec.num_qo_heads)
            .and_then(|elements| elements.checked_mul(spec.head_dim))
            .and_then(|elements| elements.checked_mul(spec.dtype.size_of()))
            .map(|bytes| bytes.max(1))
            .ok_or(ResourceViolation::ArithmeticOverflow {
                resource: "FlashInfer temporary output",
            })?,
    );
    allocations
        .into_iter()
        .try_fold(0usize, |sum, bytes| sum.checked_add(bytes))
        .ok_or(ResourceViolation::ArithmeticOverflow {
            resource: "FlashInfer prepared device memory",
        })
}

/// Rows in a CSR index pointer, from the buffer's length.
fn indptr_rows(
    buffer_lengths: &FxHashMap<NodeIndex, usize>,
    node: NodeIndex,
) -> Result<usize, ResourceViolation> {
    buffer_lengths
        .get(&node)
        .map(|bytes| bytes / std::mem::size_of::<i32>())
        .ok_or(ResourceViolation::HostResourcePlanning {
            name: "FlashInfer indptr length",
        })
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
    _owned_qo_indptr: Option<CudaSlice<i32>>,
    owned_qo_indptr_ptr: Option<u64>,
    current_c: Option<Mutex<CudaSlice<i32>>>,
    current_c_ptr: Option<u64>,
    _indices: CudaSlice<i32>,
    indices_ptr: u64,
    _last_page_len: CudaSlice<i32>,
    last_page_len_ptr: u64,
    _temp_output: CudaSlice<u8>,
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
static FLOAT_WORKSPACE: OnceLock<CudaSlice<u8>> = OnceLock::new();
static INT_WORKSPACE: OnceLock<CudaSlice<u8>> = OnceLock::new();

pub(crate) fn shared_device_memory_allocation() -> SharedDeviceMemoryAllocation {
    SharedDeviceMemoryAllocation {
        key: "flashinfer-global-workspaces",
        bytes: FLOAT_WORKSPACE_SIZE + INT_WORKSPACE_SIZE,
    }
}

/// Process-global workspaces survive after the graph that first requested
/// them is discarded. Resource planning must therefore keep charging them to
/// later candidates, including candidates with no FlashInfer op of their own.
pub(crate) fn resident_shared_device_memory_allocations() -> Vec<SharedDeviceMemoryAllocation> {
    (FLOAT_WORKSPACE.get().is_some() || INT_WORKSPACE.get().is_some())
        .then(shared_device_memory_allocation)
        .into_iter()
        .collect()
}

static PAGE_LOCKED_WORKSPACE: OnceLock<PageLockedPtr> = OnceLock::new();

/// Process-wide page-locked (pinned) staging buffer for FlashInfer's
/// host-side plan writes. Allocated once, never freed; shared by the FA2 and
/// FA3 (SinkAttention) plan paths.
pub(crate) fn page_locked_workspace() -> &'static PageLockedPtr {
    PAGE_LOCKED_WORKSPACE.get_or_init(|| unsafe {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = libc::posix_memalign(&mut ptr, 4096, INT_WORKSPACE_SIZE);
        assert_eq!(status, 0, "Failed to allocate page-locked workspace");
        let cuda_status = cuda_pin_memory(ptr, INT_WORKSPACE_SIZE);
        assert_eq!(cuda_status, 0, "Failed to pin memory");
        PageLockedPtr(ptr as *mut u8)
    })
}

pub(crate) struct PageLockedPtr(pub(crate) *mut u8);

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
            context_dim: Expression::default(),
            dtype: DType::F32,
            sm_scale: 0.0,
            window_left: -1,
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
                ("context_dim", EXPRESSION),
                ("dtype", DTYPE),
                ("sm_scale", F64),
                ("window_left", F64),
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
        // FlashInfer requires Ampere+ (sm_80; the kernels use cp.async /
        // async-copy). On older arches — e.g. a T4 (sm_75) — emit no rules so
        // the search never selects FlashInfer and attention stays on the HLIR
        // chain. Without this the rule fires, the search picks it, and the
        // JIT'd kernel's symbol is absent on launch (CUDA_ERROR_NOT_FOUND).
        // All of this egg's relations (const_like, fi_*, flashinfer_*) are
        // self-contained, so dropping it leaves no dangling references.
        if crate::device_compute_major() < 8 {
            return vec![];
        }
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
        let context_dim = extract_expr(egraph, kind_children[5], expr_cache).unwrap();
        let dtype = extract_dtype(egraph, kind_children[6]);
        let sm_scale: f64 = egraph.enodes[kind_children[7]]
            .0
            .replace('"', "")
            .parse()
            .unwrap();
        let window_left = egraph.enodes[kind_children[8]]
            .0
            .replace('"', "")
            .parse::<f64>()
            .unwrap()
            .round() as i64;
        assert!(
            FlashInferDType::from_dtype(dtype).is_some(),
            "FlashInferAttention extracted with unsupported dtype {dtype:?}"
        );

        let extracted = Self {
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            batch_dim,
            context_dim,
            dtype,
            sm_scale,
            window_left,
            plan_info: Mutex::new(Vec::new()),
        };

        // Trigger JIT compilation (or .so cache hit) at extract time, not at
        // first execute. Pays the ~30s cold-cache nvcc cost during compile
        // rather than during the GA profiling loop, where it would dominate
        // the candidate's measured runtime and make the GA reject FlashInfer.
        let _ = jit::ensure_compiled(head_dim, window_left >= 0);

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

    /// Resolve only dimensions and allocation sizes. No pointer is fabricated
    /// and no explicit-indptr contents are copied from the device during
    /// preflight.
    pub(crate) fn device_resource_spec(
        &self,
        inputs: &[NodeIndex],
        buffer_lengths: &FxHashMap<NodeIndex, usize>,
        dyn_map: &DynMap,
        enable_cuda_graph: bool,
    ) -> Result<FlashInferDeviceResourceSpec, ResourceViolation> {
        let total_q_tokens =
            self.batch_dim
                .exec(dyn_map)
                .ok_or(ResourceViolation::UnresolvedExpression {
                    resource: "FlashInfer total query tokens",
                })?;
        let c = self
            .context_dim
            .exec(dyn_map)
            .ok_or(ResourceViolation::UnresolvedExpression {
                resource: "FlashInfer context length",
            })?;
        if inputs.len() != 4 && inputs.len() != 6 {
            return Err(ResourceViolation::HostResourcePlanning {
                name: "FlashInferAttention input arity",
            });
        }
        let dtype = FlashInferDType::from_dtype(self.dtype).ok_or(
            ResourceViolation::HostResourcePlanning {
                name: "FlashInferAttention dtype",
            },
        )?;
        let kv_dim = self.num_kv_heads.checked_mul(self.head_dim).ok_or(
            ResourceViolation::ArithmeticOverflow {
                resource: "FlashInfer KV width",
            },
        )?;
        let kv_bytes =
            kv_dim
                .checked_mul(dtype.size_of())
                .ok_or(ResourceViolation::ArithmeticOverflow {
                    resource: "FlashInfer KV row bytes",
                })?;
        let k_bytes = buffer_lengths.get(&inputs[1]).copied().ok_or(
            ResourceViolation::HostResourcePlanning {
                name: "FlashInfer K-cache length",
            },
        )?;
        let v_bytes = buffer_lengths.get(&inputs[2]).copied().ok_or(
            ResourceViolation::HostResourcePlanning {
                name: "FlashInfer V-cache length",
            },
        )?;
        let max_kv_pages = k_bytes
            .checked_div(kv_bytes)
            .zip(v_bytes.checked_div(kv_bytes))
            .map(|(k_pages, v_pages)| k_pages.min(v_pages).max(c))
            .unwrap_or(c);
        let explicit_indptr = inputs.len() == 6;
        let batch_size = if explicit_indptr {
            // A CSR index pointer has one entry per sequence plus a terminator,
            // so the row count is the buffer's own length. Asking dyn_map for it
            // by name meant the caller had to declare a dim describing an input
            // the op is already holding.
            indptr_rows(buffer_lengths, inputs[5])?.saturating_sub(1)
        } else if total_q_tokens > 1 && dtype.supports_prefill() {
            1
        } else {
            total_q_tokens
        };
        let sm_scale = if self.sm_scale == 0.0 {
            1.0 / (self.head_dim as f32).sqrt()
        } else {
            self.sm_scale as f32
        };
        let mut spec = FlashInferDecodeSpec {
            total_q_tokens,
            batch_size,
            c,
            num_qo_heads: self.num_qo_heads,
            num_kv_heads: self.num_kv_heads,
            page_size: self.page_size,
            head_dim: self.head_dim,
            kv_dim,
            max_kv_pages,
            dtype,
            sm_scale_bits: sm_scale.to_bits(),
            window_left: self.window_left as i32,
            kv_indptr_host: Vec::new(),
            qo_indptr_host: Vec::new(),
        };
        if enable_cuda_graph && !explicit_indptr && total_q_tokens == 1 {
            let plan_c = flashinfer_graph_plan_capacity(c, max_kv_pages);
            spec.c = plan_c;
            spec.kv_indptr_host = vec![0, plan_c as i32];
        }
        Ok(FlashInferDeviceResourceSpec {
            cache_key: FlashInferPrepareKey::for_inputs(spec.clone(), inputs),
            spec,
            explicit_qo_indptr: explicit_indptr,
            explicit_kv_indptr: explicit_indptr,
            enable_cuda_graph,
        })
    }

    pub(crate) fn resolve_for_graph(
        &self,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<FlashInferResolvedDecode> {
        let total_q_tokens = self
            .batch_dim
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FlashInferAttention batch_dim is unresolved"))?;
        let c = self
            .context_dim
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FlashInferAttention context_dim is unresolved"))?;
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

        let dtype = FlashInferDType::from_dtype(self.dtype).ok_or_else(|| {
            anyhow::anyhow!(
                "FlashInferAttention does not support dtype {:?}",
                self.dtype
            )
        })?;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let kv_bytes = kv_dim * dtype.size_of();
        let max_kv_pages = k_buf
            .len()
            .checked_div(kv_bytes)
            .zip(v_buf.len().checked_div(kv_bytes))
            .map(|(k_pages, v_pages)| k_pages.min(v_pages).max(c))
            .unwrap_or(c);
        let (kv_indptr_host, batch_size, explicit_qo_indptr, explicit_kv_indptr) =
            if inputs.len() >= 6 {
                let qo_indptr_buf = get_buf("qo_indptr", inputs[4])?;
                let kv_indptr_buf = get_buf("kv_indptr", inputs[5])?;
                let r = kv_indptr_buf.len() / std::mem::size_of::<i32>();
                anyhow::ensure!(
                    r >= 2,
                    "FlashInferAttention: kv_indptr holds {r} rows, need at least 2 \
                 (one per sequence plus a terminator)"
                );
                // Host contents are read during prepare, where stream synchronization
                // is allowed. The pointers are part of the capture signature.
                (
                    Vec::with_capacity(r),
                    r.saturating_sub(1),
                    Some(qo_indptr_buf.ptr()),
                    Some(kv_indptr_buf.ptr()),
                )
            } else {
                // Derived causal path: one q token per sequence is decode
                // (batch == total_q_tokens). Multiple q tokens with a 16-bit
                // dtype is single-sequence prefill (batch = 1). f32 keeps the
                // decode fiction and is rejected in prepare, preserving the
                // old behavior.
                let batch_size = if total_q_tokens > 1 && dtype.supports_prefill() {
                    1
                } else {
                    total_q_tokens
                };
                (Vec::new(), batch_size, None, None)
            };

        let sm_scale = if self.sm_scale == 0.0 {
            1.0 / (self.head_dim as f32).sqrt()
        } else {
            self.sm_scale as f32
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
                dtype,
                sm_scale_bits: sm_scale.to_bits(),
                window_left: self.window_left as i32,
                kv_indptr_host,
                qo_indptr_host: Vec::new(),
            },
            ptrs: FlashInferDecodePointers {
                q: q_buf.ptr(),
                k_cache: k_buf.ptr(),
                v_cache: v_buf.ptr(),
                gather_idx: gather_idx_buf.ptr(),
                output: out_buf.ptr(),
                explicit_qo_indptr,
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
        let lib = jit::ensure_compiled(self.head_dim, self.window_left >= 0);
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;
        let spec = &mut resolved.spec;
        let is_prefill = spec.is_prefill();
        if is_prefill && !spec.dtype.supports_prefill() {
            anyhow::bail!(
                "FlashInfer prefill requires f16/bf16 (tensor core MMA); got {:?} with s={} batch={}",
                spec.dtype,
                spec.total_q_tokens,
                spec.batch_size
            );
        }

        let mut current_c = None;
        let mut current_c_ptr = None;

        let read_device_i32s = |ptr: u64, n: usize| -> anyhow::Result<Vec<i32>> {
            let mut host_bytes = vec![0u8; n * std::mem::size_of::<i32>()];
            unsafe {
                result::memcpy_dtoh_async(&mut host_bytes, ptr, stream.cu_stream())?;
            }
            stream.synchronize()?;
            Ok(bytes_to_i32_vec(host_bytes))
        };

        let (owned_kv_indptr, owned_kv_indptr_ptr) = if let Some(kv_indptr_ptr) =
            resolved.ptrs.explicit_kv_indptr
        {
            let r = spec.batch_size + 1;
            spec.kv_indptr_host = read_device_i32s(kv_indptr_ptr, r)?;
            (None, None)
        } else if is_prefill {
            // Single-sequence prefill: s q tokens attending causally to a
            // c-token context whose last s slots are the q tokens themselves.
            spec.kv_indptr_host = vec![0, spec.c as i32];
            let dev = stream.clone_htod(&spec.kv_indptr_host)?;
            let ptr = dev.device_ptr(stream).0;
            (Some(dev), Some(ptr))
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
                    "FlashInfer derived causal decode without explicit indptrs requires s=1, got s={} (f32 prefill is unsupported — tensor cores are 16-bit)",
                    spec.total_q_tokens
                );
            }
            spec.kv_indptr_host = vec![0, spec.c as i32];
            let dev = stream.clone_htod(&spec.kv_indptr_host)?;
            let ptr = dev.device_ptr(stream).0;
            (Some(dev), Some(ptr))
        };

        // Prefill also needs qo_indptr (host for plan, device for run).
        let (owned_qo_indptr, owned_qo_indptr_ptr) = if is_prefill {
            if let Some(qo_indptr_ptr) = resolved.ptrs.explicit_qo_indptr {
                let r = spec.batch_size + 1;
                spec.qo_indptr_host = read_device_i32s(qo_indptr_ptr, r)?;
                (None, None)
            } else {
                spec.qo_indptr_host = vec![0, spec.total_q_tokens as i32];
                let dev = stream.clone_htod(&spec.qo_indptr_host)?;
                let ptr = dev.device_ptr(stream).0;
                (Some(dev), Some(ptr))
            }
        } else {
            (None, None)
        };

        let total_pages = spec.kv_indptr_host.last().copied().unwrap_or_default();
        if total_pages < 0 || total_pages as usize > spec.c {
            anyhow::bail!(
                "FlashInfer describes {total_pages} KV pages, but compact gather_idx has only {}",
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
        let temp_output_bytes =
            (spec.total_q_tokens * spec.num_qo_heads * spec.head_dim * spec.dtype.size_of()).max(1);
        let temp_output = unsafe { stream.alloc::<u8>(temp_output_bytes)? };
        let temp_output_ptr = temp_output.device_ptr(stream).0;

        let (float_workspace, float_workspace_ptr, int_workspace, int_workspace_ptr) =
            flashinfer_workspaces(stream);
        let page_locked_workspace = page_locked_workspace();

        let mut plan_info_buf = [0i64; 16];
        let mut plan_info_len: i32 = 0;
        let plan_ret = if is_prefill {
            unsafe {
                (lib.prefill_plan)(
                    float_workspace_ptr as *mut std::ffi::c_void,
                    FLOAT_WORKSPACE_SIZE,
                    int_workspace_ptr as *mut std::ffi::c_void,
                    INT_WORKSPACE_SIZE,
                    page_locked_workspace.0 as *mut std::ffi::c_void,
                    spec.qo_indptr_host.as_mut_ptr(),
                    spec.kv_indptr_host.as_mut_ptr(),
                    spec.total_q_tokens as i32,
                    spec.batch_size as i32,
                    spec.num_qo_heads as i32,
                    spec.num_kv_heads as i32,
                    spec.page_size as i32,
                    spec.head_dim as i32,
                    spec.dtype as i32,
                    spec.window_left,
                    cu_stream,
                    plan_info_buf.as_mut_ptr(),
                    &mut plan_info_len,
                )
            }
        } else {
            unsafe {
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
                    spec.dtype as i32,
                    enable_cuda_graph,
                    cu_stream,
                    plan_info_buf.as_mut_ptr(),
                    &mut plan_info_len,
                )
            }
        };
        if plan_ret != 0 {
            return Err(anyhow::anyhow!(
                "FlashInfer {} plan failed with error code {plan_ret}",
                if is_prefill { "prefill" } else { "decode" }
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
            _owned_qo_indptr: owned_qo_indptr,
            owned_qo_indptr_ptr,
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

    /// Enqueue the attention kernels onto `stream`.
    ///
    /// `include_metadata` controls whether the kv-metadata preparation kernel
    /// (indices/indptr/valid-mask derived from `gather_idx` + `current_c`) is
    /// launched first. CUDA-graph callers that share prepared allocations set
    /// this for every user: sharing is allowed only between dependency-ordered
    /// users, so each refresh completes before that user consumes the buffers.
    pub(crate) fn enqueue(
        &self,
        stream: &Arc<CudaStream>,
        ptrs: FlashInferDecodePointers,
        include_metadata: bool,
    ) -> anyhow::Result<()> {
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;
        let kv_indptr_ptr = ptrs
            .explicit_kv_indptr
            .or(self.owned_kv_indptr_ptr)
            .ok_or_else(|| anyhow::anyhow!("FlashInfer decode is missing kv_indptr pointer"))?;

        let mut plan_info = self.plan_info.clone();
        if !include_metadata {
            // A preceding dependency has already populated the metadata.
        } else if let Some(current_c_ptr) = self.current_c_ptr {
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

        // At total_q_tokens == 1 the (batch, heads, dim) → (heads, batch, dim)
        // output transpose is a byte-identity, so the kernel writes the real
        // output buffer directly and the transpose launch is skipped — one
        // fewer graph node per attention island per step.
        let direct_output = self.spec.total_q_tokens == 1;
        let run_output_ptr = if direct_output {
            ptrs.output
        } else {
            self.temp_output_ptr
        };

        let run_ret = if self.spec.is_prefill() {
            let qo_indptr_ptr = ptrs
                .explicit_qo_indptr
                .or(self.owned_qo_indptr_ptr)
                .ok_or_else(|| anyhow::anyhow!("FlashInfer prefill is missing qo_indptr"))?;
            unsafe {
                (self.lib.prefill_run)(
                    self.float_workspace_ptr as *mut std::ffi::c_void,
                    FLOAT_WORKSPACE_SIZE,
                    self.int_workspace_ptr as *mut std::ffi::c_void,
                    plan_info.as_mut_ptr(),
                    plan_info.len() as i32,
                    ptrs.q as *mut std::ffi::c_void,
                    ptrs.k_cache as *mut std::ffi::c_void,
                    ptrs.v_cache as *mut std::ffi::c_void,
                    qo_indptr_ptr as *mut i32,
                    kv_indptr_ptr as *mut i32,
                    self.indices_ptr as *mut i32,
                    self.last_page_len_ptr as *mut i32,
                    run_output_ptr as *mut std::ffi::c_void,
                    self.spec.total_q_tokens as i32,
                    self.spec.batch_size as i32,
                    self.spec.num_qo_heads as i32,
                    self.spec.num_kv_heads as i32,
                    self.spec.page_size as i32,
                    self.spec.head_dim as i32,
                    self.spec.dtype as i32,
                    f32::from_bits(self.spec.sm_scale_bits),
                    self.spec.window_left,
                    cu_stream,
                )
            }
        } else {
            unsafe {
                (self.lib.run)(
                    self.float_workspace_ptr as *mut std::ffi::c_void,
                    FLOAT_WORKSPACE_SIZE,
                    self.int_workspace_ptr as *mut std::ffi::c_void,
                    plan_info.as_mut_ptr(),
                    plan_info.len() as i32,
                    ptrs.q as *mut std::ffi::c_void,
                    ptrs.k_cache as *mut std::ffi::c_void,
                    ptrs.v_cache as *mut std::ffi::c_void,
                    kv_indptr_ptr as *mut i32,
                    self.indices_ptr as *mut i32,
                    self.last_page_len_ptr as *mut i32,
                    run_output_ptr as *mut std::ffi::c_void,
                    self.spec.batch_size as i32,
                    self.spec.num_qo_heads as i32,
                    self.spec.num_kv_heads as i32,
                    self.spec.page_size as i32,
                    self.spec.head_dim as i32,
                    self.spec.dtype as i32,
                    f32::from_bits(self.spec.sm_scale_bits),
                    self.spec.window_left,
                    cu_stream,
                )
            }
        };

        if run_ret != 0 {
            return Err(anyhow::anyhow!(
                "FlashInfer {} run failed with error code {run_ret}",
                if self.spec.is_prefill() {
                    "prefill"
                } else {
                    "decode"
                }
            ));
        }

        if !direct_output {
            unsafe {
                (self.lib.transpose_output)(
                    self.temp_output_ptr as *const std::ffi::c_void,
                    ptrs.output as *mut std::ffi::c_void,
                    self.spec.total_q_tokens as i32,
                    self.spec.num_qo_heads as i32,
                    self.spec.head_dim as i32,
                    self.spec.dtype as i32,
                    cu_stream,
                );
            }
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
    // Tiered capacity instead of the full KV pool: planning at the pool size
    // (e.g. 4096) makes the decode kernel split KV into a padded grid that is
    // mostly invalid blocks at short contexts (measured 6.3µs decode + 2.3µs
    // merge per layer at c≈200 planned for 4096). Plan at the next power of
    // two of the current context (min 256); when c outgrows the tier, the
    // signature changes and the island recaptures with the next tier — a few
    // recaptures per sequence instead of µs lost on every step.
    required
        .next_power_of_two()
        .max(256)
        .min(max_kv_pages.max(required))
}

impl HostOp for FlashInferAttention {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
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
        prepared.enqueue(stream, ptrs, true)
    }

    fn output_size(&self) -> Expression {
        self.batch_dim * self.num_qo_heads * self.head_dim
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn device_memory_plan(
        &self,
        _self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffer_lengths: &FxHashMap<NodeIndex, usize>,
        dyn_map: &DynMap,
    ) -> Result<HostDeviceMemoryPlan, ResourceViolation> {
        let resource_spec = self.device_resource_spec(inputs, buffer_lengths, dyn_map, false)?;
        Ok(HostDeviceMemoryPlan {
            transient_peak_bytes: resource_spec.prepared_device_bytes()?,
            shared_allocations: vec![shared_device_memory_allocation()],
            ..Default::default()
        })
    }

    fn resource_buffer_nodes(&self, inputs: &[NodeIndex]) -> Vec<NodeIndex> {
        // device_resource_spec derives max_kv_pages from the logical lengths
        // of K and V. Q, gather indices, and explicit indptr contents do not
        // alter the allocation plan.
        inputs.get(1..3).unwrap_or_default().to_vec()
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

#[cfg(test)]
mod resource_tests {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn resource_signature_depends_only_on_kv_lengths() {
        let attention = FlashInferAttention::default();
        let inputs = (0..6).map(NodeIndex::new).collect_vec();

        assert_eq!(
            attention.resource_buffer_nodes(&inputs),
            vec![inputs[1], inputs[2]]
        );
        assert!(attention.resource_buffer_nodes(&inputs[..2]).is_empty());
    }

    #[test]
    fn device_resource_spec_is_pointer_free_and_capacity_adjusted() {
        let attention = FlashInferAttention {
            num_qo_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            page_size: 1,
            batch_dim: 1.into(),
            context_dim: Expression::from('c'),
            dtype: DType::F16,
            sm_scale: 0.0,
            window_left: -1,
            plan_info: Mutex::new(Vec::new()),
        };
        let inputs = (0..4).map(NodeIndex::new).collect_vec();
        let kv_row_bytes = 2 * 64 * 2;
        let lengths = FxHashMap::from_iter([
            (inputs[1], 4096 * kv_row_bytes),
            (inputs[2], 4096 * kv_row_bytes),
        ]);
        let dyn_map = FxHashMap::from_iter([(Symbol::from('c'), 100)]);
        let resident_before = resident_shared_device_memory_allocations();

        let spec = attention
            .device_resource_spec(&inputs, &lengths, &dyn_map, true)
            .unwrap();
        let mut other_metadata_inputs = inputs.clone();
        other_metadata_inputs[3] = NodeIndex::new(99);
        let other_spec = attention
            .device_resource_spec(&other_metadata_inputs, &lengths, &dyn_map, true)
            .unwrap();

        assert_eq!(resident_shared_device_memory_allocations(), resident_before);
        assert!(spec.cache_key.is_some());
        assert_ne!(spec.cache_key, other_spec.cache_key);
        // Capacity tier 256: kv indptr (8), current-c (4), indices
        // (1024), last-page length (4), and temporary output (512).
        assert_eq!(spec.prepared_device_bytes().unwrap(), 1_552);
        assert_eq!(shared_device_memory_allocation().bytes, 136 * 1024 * 1024);
    }

    #[test]
    fn device_resource_spec_uses_context_for_uninstalled_cache_sentinels() {
        let attention = FlashInferAttention {
            num_qo_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            page_size: 1,
            batch_dim: 1.into(),
            context_dim: Expression::from('c'),
            dtype: DType::F16,
            sm_scale: 0.0,
            window_left: -1,
            plan_info: Mutex::new(Vec::new()),
        };
        let inputs = (0..4).map(NodeIndex::new).collect_vec();
        let dyn_map = FxHashMap::from_iter([(Symbol::from('c'), 100)]);
        let uninstalled_lengths = FxHashMap::from_iter([(inputs[1], 0), (inputs[2], 0)]);

        let spec = attention
            .device_resource_spec(&inputs, &uninstalled_lengths, &dyn_map, true)
            .unwrap();

        assert_eq!(spec.spec.max_kv_pages, 100);
        assert_eq!(spec.spec.c, 100);
        assert_eq!(spec.prepared_device_bytes().unwrap(), 928);
        assert_eq!(
            attention
                .device_resource_spec(&inputs, &FxHashMap::default(), &dyn_map, true)
                .unwrap_err(),
            ResourceViolation::HostResourcePlanning {
                name: "FlashInfer K-cache length"
            }
        );
    }

    #[test]
    fn explicit_indptr_resource_specs_do_not_assume_cache_sharing() {
        let attention = FlashInferAttention {
            num_qo_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            page_size: 1,
            batch_dim: 2.into(),
            context_dim: Expression::from('c'),
            dtype: DType::F16,
            sm_scale: 0.0,
            window_left: -1,
            plan_info: Mutex::new(Vec::new()),
        };
        let inputs = (0..6).map(NodeIndex::new).collect_vec();
        let kv_row_bytes = 2 * 64 * 2;
        let lengths = FxHashMap::from_iter([
            (inputs[1], 4096 * kv_row_bytes),
            (inputs[2], 4096 * kv_row_bytes),
            // 3 CSR rows -> batch of 2. Supplied as the buffer's length, which
            // is where the op reads it from.
            (inputs[5], 3 * std::mem::size_of::<i32>()),
        ]);
        let dyn_map = FxHashMap::from_iter([(Symbol::from('c'), 100)]);

        let spec = attention
            .device_resource_spec(&inputs, &lengths, &dyn_map, true)
            .unwrap();

        assert!(spec.cache_key.is_none());
        // No owned indptrs: indices (400), two last-page lengths (8),
        // and temporary output (1024).
        assert_eq!(spec.prepared_device_bytes().unwrap(), 1_432);
    }
}
